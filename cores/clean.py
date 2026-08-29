import os
import time
import json
import numpy as np
import cv2
import torch

from util import data, util, ffmpeg, filt
from util import image_processing as impro
from restoration.service import RestorationService, composite
from .init import video_init
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

torch.set_float32_matmul_precision('high')



# ----------------------------------------------------------------------
# Device handling for the mosaic-position detector (BiSeNet)
# ----------------------------------------------------------------------
def detector_device(opt):
    """Resolve the torch device for the position detector from --device."""
    from restoration.device_manager import DeviceManager

    dm = DeviceManager(getattr(opt, 'device', 'auto'), quiet=True)
    return dm


# ----------------------------------------------------------------------
# Step 2: find mosaic positions (+ scene cuts)
# ----------------------------------------------------------------------
def get_mosaic_positions(opt, netM, imagepaths, savemask=True):
    """Detect per-frame (x, y, size) and save masks.

    Also records scene cuts (frame indices where the content changes
    abruptly). Cuts are used later to bound restoration clips so temporal
    propagation never crosses a scene change. Results are checkpointed to
    allow resume.
    """
    dm = detector_device(opt)
    try:
        netM.to(dm.info.device)
        netM.eval()
    except Exception as e:
        print(f"Error moving netM to {dm.info.type}: {e}, using CPU")
        netM = netM.cpu()

    # resume support
    continue_flag = False
    resume_frame = 0
    pre_positions = None
    pre_cuts = []
    if os.path.isfile(os.path.join(opt.temp_dir, 'step.json')):
        step = util.loadjson(os.path.join(opt.temp_dir, 'step.json'))
        resume_frame = int(step['frame'])
        if int(step['step']) > 2:
            return np.load(os.path.join(opt.temp_dir, 'mosaic_positions.npy')), \
                _load_scene_cuts(opt)
        if int(step['step']) >= 2 and resume_frame > 0:
            pre_positions = np.load(os.path.join(opt.temp_dir, 'mosaic_positions.npy'))
            pre_cuts = _load_scene_cuts(opt)
            continue_flag = True
            imagepaths = imagepaths[resume_frame:]

    positions = []
    cuts = []
    batch_size = getattr(opt, 'position_batch_size', 4)

    dev_name = dm.info.type if not hasattr(dm.info, 'name') or not dm.info.name else dm.info.name
    print(f'Step:2/4 -- Find mosaic location ({dm.info.type})')

    if not opt.no_preview:
        cv2.namedWindow('mosaic mask', cv2.WINDOW_NORMAL)

    t1 = time.time()
    scene_threshold = float(getattr(opt, 'scene_threshold', 32))
    prev_thumb = None

    import queue as _queue
    prefetch_q = _queue.Queue(maxsize=3)
    mask_executor = ThreadPoolExecutor(max_workers=2) if savemask else None

    def save_mask_batch(mask_data):
        for mask, path in mask_data:
            try:
                cv2.imwrite(os.path.join(opt.temp_dir, 'mosaic_mask', path), mask)
            except Exception as e:
                print(f"Error saving mask for {path}: {e}")

    def _image_loader():
        for b_start in range(0, len(imagepaths), batch_size):
            b_end = min(b_start + batch_size, len(imagepaths))
            batch_paths = imagepaths[b_start:b_end]
            loaded = []
            with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, len(batch_paths))) as ex:
                futs = [ex.submit(impro.imread, os.path.join(opt.temp_dir, 'video2image', p))
                        for p in batch_paths]
                for fut in futs:
                    try:
                        loaded.append(fut.result(timeout=15))
                    except Exception as e:
                        print(f"Error loading image: {e}")
                        loaded.append(None)
            prefetch_q.put((b_start, b_end, batch_paths, loaded))
        prefetch_q.put(None)

    loader_thread = Thread(target=_image_loader, daemon=True)
    loader_thread.start()

    processed = 0
    while True:
        item = prefetch_q.get()
        if item is None:
            break
        batch_start, batch_end, batch_paths, batch_images = item

        batch_positions = []
        batch_masks = []

        for i, (img, path) in enumerate(zip(batch_images, batch_paths)):
            abs_idx = resume_frame + batch_start + i
            if img is None:
                batch_positions.append([0, 0, 0])
                prev_thumb = None
                continue

            # ---- scene cut detection (cheap thumbnail diff) ----
            thumb = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev_thumb is not None:
                diff = float(np.mean(np.abs(gray - prev_thumb)))
                if diff > scene_threshold and abs_idx > 0:
                    cuts.append(abs_idx)
            prev_thumb = gray

            # Try multi detection when all_mosaic_area is set or auto detects multi
            # We always try multi first; it falls back to single
            try:
                mask_all, boxes = _detect_all_one(img, netM, opt)
                if mask_all is not None and boxes:
                    # For backward compat, batch_positions stores the largest box
                    # But we also save the full multi mask for later multi handling
                    # Choose largest as representative for positions array
                    largest = max(boxes, key=lambda b: b[2])
                    x, y, size = largest
                    mask = mask_all
                    # Save multi info for later if needed (store boxes count in separate file)
                    # We encode multi boxes in a sidecar JSON per frame if multi
                    if len(boxes) > 1:
                        try:
                            import json as _json
                            multi_path = os.path.join(opt.temp_dir, 'mosaic_mask', path + '.json')
                            # store boxes relative to full frame
                            _boxes = [[int(b[0]), int(b[1]), int(b[2])] for b in boxes]
                            # Write JSON sidecar (best-effort)
                            with open(multi_path, 'w') as _f:
                                _json.dump(_boxes, _f)
                        except Exception:
                            pass
                    batch_positions.append([x, y, size])
                    if mask is not None:
                        batch_masks.append((mask, path))
                else:
                    mask, x, y, size = _detect_one(img, netM, opt)
                    batch_positions.append([x, y, size])
                    if mask is not None:
                        batch_masks.append((mask, path))
            except Exception as e:
                # Fallback to single
                try:
                    mask, x, y, size = _detect_one(img, netM, opt)
                    batch_positions.append([x, y, size])
                    if mask is not None:
                        batch_masks.append((mask, path))
                except Exception as e2:
                    print(f"Error processing image {path}: {e2}")
                    batch_positions.append([0, 0, 0])
                if not opt.no_preview and mask is not None \
                        and isinstance(mask, np.ndarray) and mask.size > 0 and i % 2 == 0:
                    try:
                        cv2.imshow('mosaic mask', mask)
                        cv2.waitKey(1)
                    except Exception:
                        pass
            except Exception as e:
                print(f"Error processing image {path}: {e}")
                batch_positions.append([0, 0, 0])

        positions.extend(batch_positions)
        processed += len(batch_positions)

        if savemask and batch_masks:
            mask_executor.submit(save_mask_batch, list(batch_masks))

        current_frame = min(batch_end + resume_frame, len(imagepaths) + resume_frame)
        total = len(imagepaths) + resume_frame
        if current_frame % 1000 == 0:
            save_positions = np.array(positions)
            if continue_flag:
                save_positions = np.concatenate((pre_positions, save_positions), axis=0)
            np.save(os.path.join(opt.temp_dir, 'mosaic_positions.npy'), save_positions)
            util.savejson(os.path.join(opt.temp_dir, 'step.json'), {'step': 2, 'frame': current_frame})

        t2 = time.time()
        print(f'\r{current_frame}/{total} '
              f'{util.get_bar(100*current_frame/total, num=35)} '
              f'{util.counttime(t1, t2, current_frame, total)}', end='')

    loader_thread.join()
    if mask_executor is not None:
        # Restoration reads these files immediately after this function. Do not
        # let asynchronous mask I/O race the compositor.
        mask_executor.shutdown(wait=True)

    if not opt.no_preview:
        cv2.destroyAllWindows()

    print('\nOptimize mosaic locations...')
    positions = np.array(positions)
    if continue_flag and pre_positions is not None:
        positions = np.concatenate((pre_positions, positions), axis=0)

    for i in range(3):
        positions[:, i] = filt.medfilt(positions[:, i], opt.medfilt_num)

    # ---- auto-adapt min_mosaic_size / min_mosaic_area from observed distribution ----
    if not bool(getattr(opt, 'no_auto_adapt', False)):
        try:
            valid_sizes = positions[positions[:, 2] > 0, 2]
            if len(valid_sizes) >= 3:
                # Report observed distribution
                median = float(np.median(valid_sizes))
                p25 = float(np.percentile(valid_sizes, 25))
                p75 = float(np.percentile(valid_sizes, 75))
                print(f"[auto-adapt] observed sizes: median={median:.1f} p25={p25:.1f} p75={p75:.1f} (n={len(valid_sizes)})")
                # Adaptive threshold: keep most detections, but filter tiny speckles
                # Use 0.5 * median but bounded to [20, 80]; also respect p25
                adapted = int(max(18, min(80, median * 0.5, p25 * 0.8)))
                # Only lower the threshold, never raise it above user request when detection is sparse
                user_min = int(getattr(opt, 'min_mosaic_size', 40))
                if len(valid_sizes) / max(1, len(positions)) < 0.08:
                    # Sparse detection: be permissive
                    adapted = min(adapted, 20)
                # If user explicitly set a very permissive value, keep it
                # Otherwise auto-lower if our adapted is lower
                if adapted < user_min:
                    print(f"[auto-adapt] lowering min_mosaic_size {user_min} -> {adapted}")
                    opt.min_mosaic_size = adapted
                # Also adapt min_mosaic_area based on median size
                # area ~ (2*size)^2, use 0.2 * median_area as threshold
                median_area = float(np.median(valid_sizes**2 * 4))
                adapted_area = int(max(30, min(300, median_area * 0.15)))
                user_area = int(getattr(opt, 'min_mosaic_area', 150))
                if adapted_area < user_area and len(valid_sizes) >= 5:
                    print(f"[auto-adapt] lowering min_mosaic_area {user_area} -> {adapted_area}")
                    opt.min_mosaic_area = adapted_area
        except Exception as e:
            print(f"[auto-adapt] size adaptation failed: {e}")

    all_cuts = sorted(set(pre_cuts) | set(cuts)) if continue_flag else sorted(set(cuts))
    step = {'step': 3, 'frame': 0}
    util.savejson(os.path.join(opt.temp_dir, 'step.json'), step)
    np.save(os.path.join(opt.temp_dir, 'mosaic_positions.npy'), positions)
    util.savejson(os.path.join(opt.temp_dir, 'scene_cuts.json'),
                  {'cuts': [int(c) for c in all_cuts], 'threshold': scene_threshold})
    return positions, all_cuts


def _detect_one(img, netM, opt):
    """Single-frame detection; returns (mask, x, y, size) in that order."""
    from models import runmodel
    return runmodel.get_mosaic_position(img, netM, opt)


def _detect_all_one(img, netM, opt):
    """Multi-mosaic detection; returns (mask_all, boxes) where boxes is list of (x,y,size)."""
    from models import runmodel
    return runmodel.get_mosaic_position_multi(img, netM, opt)


def _load_scene_cuts(opt):
    p = os.path.join(opt.temp_dir, 'scene_cuts.json')
    if os.path.isfile(p):
        try:
            return list(util.loadjson(p)['cuts'])
        except Exception:
            pass
    return []



# ----------------------------------------------------------------------
# Static image (also used by the HTTP server)
# ----------------------------------------------------------------------
def cleanmosaic_img(opt, service):
    """Clean a single image through the unified service.

    Note: get_mosaic_position returns (mask, x, y, size); older code
    unpacked it in the wrong order here, which broke static-image cleaning.
    """
    path = opt.media_path
    print('Clean Mosaic:', path)
    img_origin = impro.imread(path)
    img_result = service.clean_image(img_origin, opt)
    impro.imwrite(os.path.join(opt.result_dir,
                 os.path.splitext(os.path.basename(path))[0] + '_clean.jpg'), img_result)
    return img_result


def cleanmosaic_img_server(opt, img_origin, service):
    return service.clean_image(img_origin, opt)


# ----------------------------------------------------------------------
# Run grouping: scene-bounded temporal segments to restore
# ----------------------------------------------------------------------
def group_runs(positions, cuts, min_size, gap_tolerance=3, max_run=400,
               max_union_ratio=3.0):
    """Split frame indices into contiguous runs of active frames.

    - runs never cross scene cuts
    - gaps of <= gap_tolerance inactive frames are absorbed (their boxes are
      interpolated from neighbours) so clips stay temporally coherent
    - a run is split when its length exceeds max_run or when the union box
      would grow beyond max_union_ratio * the largest member half-size
    Returns a list of runs; each run is a list of (frame_idx, x, y, size)
    with size > 0 for every entry (interpolated where needed).
    """
    n = len(positions)
    cut_set = set(int(c) for c in cuts)

    active = [s > min_size for (_, _, s) in positions]

    runs = []
    cur = []

    def flush():
        nonlocal cur
        if len(cur) >= 2:
            runs.append(cur)
        elif len(cur) == 1:
            runs.append(cur)
        cur = []

    def union_too_big(cur, x, y, size):
        if not cur:
            return False
        xs = [b[1] for b in cur] + [x]
        ys = [b[2] for b in cur] + [y]
        ss = [b[3] for b in cur] + [size]
        side = 2 * (max(max(xs) - min(xs), max(ys) - min(ys)) / 2 + max(ss))
        return side > max_union_ratio * max(ss) * 2

    i = 0
    while i < n:
        if active[i]:
            x, y, s = positions[i]
            if cur and (i in cut_set or len(cur) >= max_run or union_too_big(cur, x, y, s)):
                flush()
            cur.append((i, int(x), int(y), int(s)))
        else:
            # gap absorption: look ahead
            j = i
            while j < n and not active[j] and (j - i) < gap_tolerance and j not in cut_set:
                j += 1
            if cur and j < n and active[j] and j not in cut_set \
                    and not union_too_big(cur, positions[j][0], positions[j][1], positions[j][2]):
                # interpolate boxes across the gap
                prev = cur[-1]
                nxt = (j, int(positions[j][0]), int(positions[j][1]), int(positions[j][2]))
                span = nxt[0] - prev[0]
                for k in range(1, span):
                    t = k / span
                    gx = int(prev[1] * (1 - t) + nxt[1] * t)
                    gy = int(prev[2] * (1 - t) + nxt[2] * t)
                    gs = int(prev[3] * (1 - t) + nxt[3] * t)
                    if prev[0] + k in cut_set:
                        continue
                    cur.append((prev[0] + k, gx, gy, gs))
                i = j
                continue
            else:
                flush()
        i += 1
    flush()
    return runs


def run_union_box(run):
    """Square box (cx, cy, halfsize) covering all member boxes."""
    xs = [b[1] for b in run]
    ys = [b[2] for b in run]
    ss = [b[3] for b in run]
    cx = (min(xs) + max(xs)) // 2
    cy = (min(ys) + max(ys)) // 2
    half = max(max(xs) - min(xs), max(ys) - min(ys)) // 2 + max(ss)
    return int(cx), int(cy), int(half)



# ----------------------------------------------------------------------
# Unified video cleaning: detect -> group -> clip restore -> composite
# ----------------------------------------------------------------------
def _crop_for_backend(img, cx, cy, half, max_side=640):
    """Square crop around (cx,cy); downscaled if the box is huge."""
    if img is None:
        return None
    h, w = img.shape[:2]
    x0, x1 = max(0, cx - half), min(w, cx + half)
    y0, y1 = max(0, cy - half), min(h, cy + half)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    ch, cw = crop.shape[:2]
    side = max(ch, cw)
    if side > max_side:
        scale = max_side / side
        crop = cv2.resize(crop, (max(4, int(cw * scale)), max(4, int(ch * scale))),
                          interpolation=cv2.INTER_AREA)
    return crop


def _extract_member_crop(restored_union, cx, cy, half, x, y, size,
                         frame_shape):
    """Map a tracked member box back into its restored fixed-union crop.

    Temporal backends need one stable union crop across the run. Compositing the
    whole union into each moving member box stretches and shifts content, so we
    slice the corresponding subregion first (including border clipping and any
    max-side downscale performed by ``_crop_for_backend``).
    """
    h, w = frame_shape[:2]
    ux0, ux1 = max(0, cx - half), min(w, cx + half)
    uy0, uy1 = max(0, cy - half), min(h, cy + half)
    mx0, mx1 = max(0, x - size), min(w, x + size)
    my0, my1 = max(0, y - size), min(h, y + size)
    if restored_union is None or restored_union.size == 0 \
            or ux1 <= ux0 or uy1 <= uy0 or mx1 <= mx0 or my1 <= my0:
        return restored_union

    rh, rw = restored_union.shape[:2]
    sx, sy = rw / float(ux1 - ux0), rh / float(uy1 - uy0)
    rx0 = int(round((mx0 - ux0) * sx))
    rx1 = int(round((mx1 - ux0) * sx))
    ry0 = int(round((my0 - uy0) * sy))
    ry1 = int(round((my1 - uy0) * sy))
    rx0, rx1 = max(0, rx0), min(rw, rx1)
    ry0, ry1 = max(0, ry0), min(rh, ry1)
    member = restored_union[ry0:ry1, rx0:rx1]
    return member if member.size else restored_union


def cleanmosaic_video(opt, service, netM):
    """Single unified video path for every backend.

    Replaces the old by-frame and BVDNet-fusion loops: frames are grouped
    into scene-bounded runs, each run is restored as a temporal clip through
    the active RestorationBackend, then composited and encoded.
    """
    path = opt.media_path
    fps, imagepaths, height, width = video_init(opt, path)
    start_frame = int(imagepaths[0][7:13])
    positions, cuts = get_mosaic_positions(opt, netM, imagepaths, savemask=True)
    positions = positions[(start_frame - 1):]
    cuts = [int(c) - (start_frame - 1) for c in cuts if int(c) >= (start_frame - 1)]

    n = len(imagepaths)
    # ---- Multi-mosaic detection: count frames with multi sidecars ----
    multi_frame_count = 0
    try:
        import glob as _glob, json as _js
        # Sample up to 20 frames to estimate multi ratio
        sample = list(range(0, n, max(1, n//20)))[:20]
        for idx in sample:
            mp = os.path.join(opt.temp_dir, 'mosaic_mask', imagepaths[idx] + '.json')
            if os.path.exists(mp):
                try:
                    with open(mp, 'r') as _f:
                        _boxes = _js.load(_f)
                        if isinstance(_boxes, list) and len(_boxes) > 1:
                            multi_frame_count += 1
                except Exception:
                    pass
        multi_ratio = multi_frame_count / max(1, len(sample))
        if multi_ratio > 0.3:
            print(f"[multi] detected multi-mosaic in {multi_frame_count}/{len(sample)} sampled frames (ratio {multi_ratio:.2f}) -> per-frame multi restoration")
    except Exception:
        multi_ratio = 0

    # If multi, use per-frame per-component restoration instead of temporal grouping
    if multi_ratio > 0.3:
        # Per-frame multi path
        replace_dir = os.path.join(opt.temp_dir, 'replace_mosaic')
        video2image_dir = os.path.join(opt.temp_dir, 'video2image')
        mosaic_mask_dir = os.path.join(opt.temp_dir, 'mosaic_mask')
        write_queue = Queue(maxsize=16)
        preview_queue = Queue(maxsize=8)
        def writer():
            import shutil
            while True:
                item = write_queue.get()
                if item is None:
                    break
                save_path, img, delete_path = item
                try:
                    if img is None:
                        shutil.copy2(delete_path, save_path)
                    else:
                        cv2.imwrite(save_path, img)
                    if delete_path and os.path.exists(delete_path) and not getattr(opt, 'keep_frames', False):
                        os.remove(delete_path)
                except Exception as e:
                    print(f"Writer error: {e}")
                finally:
                    write_queue.task_done()
        def previewer():
            while True:
                item = preview_queue.get()
                if item is None:
                    break
                try:
                    cv2.imshow('clean', item)
                    cv2.waitKey(1)
                except Exception:
                    pass
        writer_thread = Thread(target=writer, daemon=True)
        writer_thread.start()
        preview_thread = None
        if not opt.no_preview:
            preview_thread = Thread(target=previewer, daemon=True)
            preview_thread.start()
        t1 = time.time()
        done_frames = 0
        for idx in range(n):
            src_path = os.path.join(video2image_dir, imagepaths[idx])
            dst_path = os.path.join(replace_dir, imagepaths[idx])
            img = impro.imread(src_path)
            if img is None:
                write_queue.put((dst_path, None, src_path))
                continue
            mask_path = os.path.join(mosaic_mask_dir, imagepaths[idx])
            mask = cv2.imread(mask_path, 0) if os.path.exists(mask_path) else None
            multi_path = mask_path + '.json'
            multi_boxes = None
            if os.path.exists(multi_path):
                try:
                    import json as _json
                    with open(multi_path, 'r') as _f:
                        multi_boxes = _json.load(_f)
                except Exception:
                    multi_boxes = None
            if mask is None or mask.size == 0 or not multi_boxes or len(multi_boxes) <= 1:
                # Fallback to single handling for this frame (if single)
                if mask is not None and mask.size > 0 and multi_boxes and len(multi_boxes)==1:
                    x,y,s = multi_boxes[0]
                    if s > 12:
                        crop = img[max(0,y-s):y+s, max(0,x-s):x+s]
                        if crop.size>0:
                            restored = service.restore_region_sequence([crop])[0]
                            img = composite(img, restored, mask, x, y, s, opt)
                # Even if no mosaic, just copy
                # Check if this frame was supposed to be in a run (single mosaic case)
                # For multi video, we treat as per-frame, so just write img (possibly cleaned)
                write_queue.put((dst_path, img if mask is not None and multi_boxes else None, src_path))
            else:
                # Multi: restore each component independently
                try:
                    result_img = img
                    # Need per-component masks
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
                    min_area = int(getattr(opt, 'min_mosaic_area', 150))
                    # Build list of component masks
                    for lbl in range(1, num_labels):
                        if stats[lbl, cv2.CC_STAT_AREA] < min_area:
                            continue
                        comp_mask = np.zeros_like(mask, dtype=np.uint8)
                        comp_mask[labels == lbl] = 255
                        ys, xs = np.where(comp_mask > 127)
                        if len(xs)==0:
                            continue
                        cx = int(np.mean(xs)); cy = int(np.mean(ys))
                        x_min, x_max = int(np.min(xs)), int(np.max(xs))
                        y_min, y_max = int(np.min(ys)), int(np.max(ys))
                        s = max(x_max - x_min, y_max - y_min)//2
                        if s < 12:
                            continue
                        # Find closest box
                        best = min(multi_boxes, key=lambda b: np.hypot(b[0]-cx, b[1]-cy)) if multi_boxes else (cx,cy,s)
                        x_b, y_b, s_b = int(best[0]), int(best[1]), int(best[2])
                        # Use the box's size for cropping, but use comp_mask for feathering
                        crop = result_img[max(0,y_b-s_b):y_b+s_b, max(0,x_b-s_b):x_b+s_b]
                        if crop.size==0:
                            continue
                        restored = service.restore_region_sequence([crop])[0]
                        result_img = composite(result_img, restored, comp_mask, x_b, y_b, s_b, opt)
                    img = result_img
                except Exception as e:
                    print(f"multi per-frame fallback: {e}")
                write_queue.put((dst_path, img, src_path))
            done_frames += 1
            if done_frames % 10 == 0 or done_frames == n:
                t2 = time.time()
                print(f'\r{done_frames}/{n} {util.get_bar(100*done_frames/max(n,1), num=35)} {util.counttime(t1, t2, done_frames, max(n,1))}', end='')
            if not opt.no_preview and preview_queue.qsize() < 4:
                try:
                    preview_queue.put(img.copy())
                except Exception:
                    pass
        write_queue.join()
        write_queue.put(None)
        writer_thread.join()
        if preview_thread:
            preview_queue.put(None)
            preview_thread.join(timeout=5)
        if not opt.no_preview:
            cv2.destroyAllWindows()
        print('\nStep:4/4 -- Convert images to video')
        ffmpeg.image2video(
            fps,
            os.path.join(replace_dir, f'output_%06d.{opt.tempimage_type}'),
            os.path.join(opt.temp_dir, 'voice_tmp.mp3'),
            os.path.join(opt.result_dir, os.path.splitext(os.path.basename(path))[0] + '_clean.mp4')
        )
        return
    # ---- Single-mosaic path (original) ----
    min_size = int(getattr(opt, 'min_mosaic_size', 40))
    runs = group_runs(positions, cuts, min_size,
                      gap_tolerance=int(getattr(opt, 'run_gap_tolerance', 3)),
                      max_run=int(getattr(opt, 'max_clip_run', 300)))

    covered = set()
    for run in runs:
        covered.update(b[0] for b in run)

    print(f'\nStep:3/4 -- Clean Mosaic ({service.backend.name}): '
          f'{len(runs)} segment(s), {len(covered)}/{n} frames to restore')

    replace_dir = os.path.join(opt.temp_dir, 'replace_mosaic')
    video2image_dir = os.path.join(opt.temp_dir, 'video2image')
    mosaic_mask_dir = os.path.join(opt.temp_dir, 'mosaic_mask')

    write_queue = Queue(maxsize=16)
    preview_queue = Queue(maxsize=8)

    def writer():
        import shutil
        while True:
            item = write_queue.get()
            if item is None:
                break
            save_path, img, delete_path = item
            try:
                if img is None:
                    shutil.copy2(delete_path, save_path)
                else:
                    cv2.imwrite(save_path, img)
                if delete_path and os.path.exists(delete_path) \
                        and not getattr(opt, 'keep_frames', False):
                    os.remove(delete_path)
            except Exception as e:
                print(f"Writer error: {e}")
            finally:
                write_queue.task_done()

    def previewer():
        while True:
            item = preview_queue.get()
            if item is None:
                break
            try:
                cv2.imshow('clean', item)
                cv2.waitKey(1)
            except Exception:
                pass

    writer_thread = Thread(target=writer, daemon=True)
    writer_thread.start()
    preview_thread = None
    if not opt.no_preview:
        preview_thread = Thread(target=previewer, daemon=True)
        preview_thread.start()

    t1 = time.time()
    done_frames = 0

    # 1) restore runs
    for run in runs:
        cx, cy, half = run_union_box(run)
        # clamp union box to frame bounds
        cx = int(np.clip(cx, 0, width - 1))
        cy = int(np.clip(cy, 0, height - 1))
        half = int(np.clip(half, 4, max(width, height)))

        crops, valid = [], []
        max_restore_side = int(getattr(opt, 'max_restore_side', 0) or 640)
        for (idx, x, y, s) in run:
            img = impro.imread(os.path.join(video2image_dir, imagepaths[idx]))
            crop = _crop_for_backend(
                img, cx, cy, half, max_side=max_restore_side)
            crops.append(crop if crop is not None else np.zeros((8, 8, 3), np.uint8))
            valid.append(crop is not None)

        restored = service.restore_region_sequence(crops)

        for (idx, x, y, s), fake, ok in zip(run, restored, valid):
            src_path = os.path.join(video2image_dir, imagepaths[idx])
            dst_path = os.path.join(replace_dir, imagepaths[idx])
            img_result = None
            if ok:
                img = impro.imread(src_path)
                mask_path = os.path.join(mosaic_mask_dir, imagepaths[idx])
                mask = cv2.imread(mask_path, 0) if os.path.exists(mask_path) else None
                # Check for multi sidecar
                multi_path = mask_path + '.json'
                multi_boxes = None
                if os.path.exists(multi_path):
                    try:
                        import json as _json
                        with open(multi_path, 'r') as _f:
                            multi_boxes = _json.load(_f)
                    except Exception:
                        multi_boxes = None
                if img is not None and mask is not None and mask.size > 0:
                    # If multi_boxes exists and has >1 entry, composite each component
                    if multi_boxes and len(multi_boxes) > 1:
                        # Multi-mosaic per-frame handling
                        # For multi, we need to restore each component separately
                        # However we have only one 'fake' for the union. We will
                        # composite each component using the same fake but with
                        # its own per-component mask.
                        # Extract per-component masks from the combined mask
                        try:
                            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
                            min_area = int(getattr(opt, 'min_mosaic_area', 150))
                            result_img = img
                            for lbl in range(1, num_labels):
                                if stats[lbl, cv2.CC_STAT_AREA] < min_area:
                                    continue
                                comp_mask = np.zeros_like(mask, dtype=np.uint8)
                                comp_mask[labels == lbl] = 255
                                # Find its box
                                ys, xs = np.where(comp_mask > 127)
                                if len(xs) == 0:
                                    continue
                                cx_c = int(np.mean(xs)); cy_c = int(np.mean(ys))
                                x_min, x_max = int(np.min(xs)), int(np.max(xs))
                                y_min, y_max = int(np.min(ys)), int(np.max(ys))
                                s_c = max(x_max - x_min, y_max - y_min)//2
                                if s_c < 12:
                                    continue
                                # Find corresponding multi_box (closest)
                                best = min(multi_boxes, key=lambda b: np.hypot(b[0]-cx_c, b[1]-cy_c)) if multi_boxes else (cx_c, cy_c, s_c)
                                # For multi, we need to restore each component's crop individually
                                # To avoid re-running the backend for each component per frame (which would be slow),
                                # we reuse the union restoration but extract per-component member crops
                                fake_member_c = _extract_member_crop(
                                    fake, cx, cy, half, best[0], best[1], best[2], img.shape)
                                result_img = composite(result_img, fake_member_c, comp_mask, best[0], best[1], best[2], opt)
                            img_result = result_img
                        except Exception as e:
                            print(f"multi composite fallback: {e}")
                            fake_member = _extract_member_crop(
                                fake, cx, cy, half, x, y, s, img.shape)
                            img_result = composite(img, fake_member, mask, x, y, s, opt)
                    else:
                        fake_member = _extract_member_crop(
                            fake, cx, cy, half, x, y, s, img.shape)
                        img_result = composite(img, fake_member, mask, x, y, s, opt)
                    if not opt.no_preview and preview_queue.qsize() < 4 and img_result is not None:
                        preview_queue.put(img_result.copy())
            write_queue.put((dst_path, img_result, src_path))
            done_frames += 1
            if done_frames % 5 == 0 or done_frames == len(covered):
                t2 = time.time()
                print(f'\r{done_frames}/{len(covered)} '
                      f'{util.get_bar(100*done_frames/max(len(covered),1), num=35)} '
                      f'{util.counttime(t1, t2, done_frames, max(len(covered),1))}', end='')

    # 2) untouched frames: plain copy
    for idx in range(n):
        if idx in covered:
            continue
        src_path = os.path.join(video2image_dir, imagepaths[idx])
        dst_path = os.path.join(replace_dir, imagepaths[idx])
        write_queue.put((dst_path, None, src_path))

    # Do not begin encoding until every output frame is durably written.
    write_queue.join()
    write_queue.put(None)
    writer_thread.join()
    if preview_thread:
        preview_queue.put(None)
        preview_thread.join(timeout=5)
    if not opt.no_preview:
        cv2.destroyAllWindows()

    print('\nStep:4/4 -- Convert images to video')
    ffmpeg.image2video(
        fps,
        os.path.join(replace_dir, f'output_%06d.{opt.tempimage_type}'),
        os.path.join(opt.temp_dir, 'voice_tmp.mp3'),
        os.path.join(opt.result_dir, os.path.splitext(os.path.basename(path))[0] + '_clean.mp4')
    )


# ----------------------------------------------------------------------
# Backwards-compatible aliases (old entry points now use the unified path)
# ----------------------------------------------------------------------
def cleanmosaic_video_byframe(opt, netG, netM):       # pragma: no cover
    raise NotImplementedError(
        "cleanmosaic_video_byframe was replaced by cleanmosaic_video(opt, service, netM); "
        "per-frame backends are selected via --model legacy.")


def cleanmosaic_video_fusion(opt, netG, netM):        # pragma: no cover
    raise NotImplementedError(
        "cleanmosaic_video_fusion was replaced by cleanmosaic_video(opt, service, netM); "
        "BVDNet checkpoints are selected via --model <video checkpoint path>.")
