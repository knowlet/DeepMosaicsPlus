import cv2
import sys
sys.path.append("..")
import util.image_processing as impro
from util import mosaic
from util import data
import torch
import numpy as np

torch.set_float32_matmul_precision('high')

def run_segment(img,net,size = 360,gpu_id = '-1'):
    img = impro.resize(img,size)
    img = data.im2tensor(img, gpu_id = gpu_id, bgr2rgb = False, is0_1 = True)
    # Make sure the tensor is on the same device as the model
    img = img.to(next(net.parameters()).device).float()
    mask = net(img)
    mask = data.tensor2im(mask, gray=True, is0_1 = True)
    return mask

def run_pix2pix(img, net, opt):
    if opt.netG == 'HD':
        img = impro.resize(img, 512)
    else:
        img = impro.resize(img, 128)

    device = next(net.parameters()).device
    
    # DirectML-specific tensor handling
    if 'directml' in str(device):
        try:
            # Force CPU tensor creation first for DirectML
            img_tensor = data.im2tensor(img, device=torch.device('cpu'))
            img_tensor = img_tensor.contiguous()
            
            # Move to DirectML device only when needed
            img_tensor = img_tensor.to(device)
            
            # Clear any cached tensors before inference
            if hasattr(torch, 'directml') and hasattr(torch.directml, 'empty_cache'):
                torch.directml.empty_cache()
            
            # Ensure model is in eval mode
            net.eval()
            
            with torch.inference_mode():
                img_fake = net(img_tensor)
                # Immediately move to CPU and clone to break DirectML tensor chain
                img_fake = img_fake.detach().cpu().clone()
                
            # Clear DirectML cache after inference
            if hasattr(torch, 'directml') and hasattr(torch.directml, 'empty_cache'):
                torch.directml.empty_cache()
                
            img_fake = data.tensor2im(img_fake)
            return img_fake
            
        except Exception as e:
            print(f"DirectML error, falling back to CPU: {e}")
            # Complete fallback to CPU processing
            try:
                net_cpu = net.cpu()
                img_tensor = data.im2tensor(img, device=torch.device('cpu'))
                net_cpu.eval()
                with torch.inference_mode():
                    img_fake = net_cpu(img_tensor)
                img_fake = data.tensor2im(img_fake)
                return img_fake
            except Exception as e2:
                print(f"CPU fallback also failed: {e2}")
                # Return original image if all else fails
                return img
    else:
        # Normal CUDA/CPU path
        img_tensor = data.im2tensor(img, device=device)
        with torch.inference_mode():
            img_fake = net(img_tensor)
        img_fake = data.tensor2im(img_fake)
        return img_fake


def traditional_cleaner(img,opt):
    h,w = img.shape[:2]
    img = cv2.blur(img, (opt.tr_blur,opt.tr_blur))
    img = img[::opt.tr_down,::opt.tr_down,:]
    img = cv2.resize(img, (w,h),interpolation=cv2.INTER_LANCZOS4)
    return img

def run_styletransfer(opt, net, img):

    if opt.output_size != 0:
        if 'resize' in opt.preprocess and 'resize_scale_width' not in opt.preprocess:
            img = impro.resize(img,opt.output_size)
        elif 'resize_scale_width' in opt.preprocess:
            img = cv2.resize(img, (opt.output_size,opt.output_size))
        img = img[0:4*int(img.shape[0]/4),0:4*int(img.shape[1]/4),:]

    if 'edges' in opt.preprocess:
        if opt.canny > 100:
            canny_low = opt.canny-50
            canny_high = np.clip(opt.canny+50,0,255)
        elif opt.canny < 50:
            canny_low = np.clip(opt.canny-25,0,255)
            canny_high = opt.canny+25
        else:
            canny_low = opt.canny-int(opt.canny/2)
            canny_high = opt.canny+int(opt.canny/2)
        img = cv2.Canny(img,canny_low,canny_high)
        if opt.only_edges:
            return img
        img = data.im2tensor(img,gpu_id=opt.gpu_id,gray=True)
    else:    
        img = data.im2tensor(img,gpu_id=opt.gpu_id)
    img = net(img)
    img = data.tensor2im(img)
    return img

def get_ROI_position(img,net,opt,keepsize=True):
    mask = run_segment(img,net,size=360,gpu_id = opt.gpu_id)
    mask = impro.mask_threshold(mask,opt.mask_extend,opt.mask_threshold)
    if keepsize:
        mask = impro.resize_like(mask, img)
    x,y,halfsize,area = impro.boundingSquare(mask, 1)
    return mask,x,y,halfsize,area

def _process_mask_to_roi(mask, opt, all_mosaic=False):
    """Common helper: dilate, filter small components, find largest ROI, compute (mask, x, y, size)."""
    h, w = mask.shape[:2]
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    try:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        min_area = int(getattr(opt, 'min_mosaic_area', 150))
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                mask[labels == i] = 0
    except Exception as e:
        print(f"Connected components error: {e}")
        return None, 0, 0, 0
    if not getattr(opt, 'all_mosaic_area', False) and not all_mosaic:
        mask = impro.find_mostlikely_ROI(mask)
    try:
        mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        area = int(np.sum(mask_binary > 0))
        if area == 0:
            return None, 0, 0, 0
        y_coords, x_coords = np.where(mask_binary > 0)
        if len(x_coords) == 0 or len(y_coords) == 0:
            return None, 0, 0, 0
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))
        x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
        y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
        size = max(x_max - x_min, y_max - y_min) // 2
        return mask, center_x, center_y, size
    except Exception as e:
        print(f"Mask processing error: {e}")
        return None, 0, 0, 0


def _extract_all_boxes(mask, opt):
    """Extract all valid mosaic boxes from a binary mask.

    Returns list of (x, y, size, area, component_mask) for each connected
    component that passes min_mosaic_area. Each component_mask is a binary
    image with only that component white (for per-component compositing).
    The returned mask per component is the full-size binary mask for that
    single component (same shape as input mask).
    """
    try:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    except Exception as e:
        print(f"Connected components error: {e}")
        return []
    min_area = int(getattr(opt, 'min_mosaic_area', 150))
    boxes = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        # Create component mask
        comp = np.zeros_like(mask, dtype=np.uint8)
        comp[labels == i] = 255
        # Find its bounding square
        try:
            ys, xs = np.where(comp > 127)
            if len(xs) == 0:
                continue
            cx = int(np.mean(xs))
            cy = int(np.mean(ys))
            x_min, x_max = int(np.min(xs)), int(np.max(xs))
            y_min, y_max = int(np.min(ys)), int(np.max(ys))
            size = max(x_max - x_min, y_max - y_min) // 2
            if size < 12 or size > 500:
                continue
            boxes.append((cx, cy, size, area, comp))
        except Exception:
            continue
    # Sort by area descending (largest first)
    boxes.sort(key=lambda b: b[3], reverse=True)
    return boxes


def get_mosaic_position_multi(img_origin, net_mosaic_pos, opt):
    """Multi-mosaic version: returns (mask_all, boxes) where boxes is list of (x,y,size).

    mask_all is the binary mask with all detected components (or None).
    boxes is list of (x, y, size) for each component, sorted by area.
    If no mosaic found, returns (None, []).
    Uses the same adaptive sweep as get_mosaic_position but scores by total
    area and number of components, and falls back to tiled detection when
    the full-frame response is weak.
    """
    h, w = img_origin.shape[:2]
    try:
        raw = run_segment(img_origin, net_mosaic_pos, size=360, gpu_id=opt.gpu_id)
    except Exception as e:
        print(f"Segmentation error: {e}")
        return None, []
    if raw is None or raw.size == 0:
        return None, []
    if not isinstance(raw, np.ndarray):
        return None, []
    try:
        raw_resized = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LANCZOS4)
    except Exception as e:
        print(f"Mask resize error: {e}")
        return None, []

    auto = not bool(getattr(opt, 'no_auto_adapt', False))
    ex_mun = int(min(h, w) / 20)
    base_th = int(getattr(opt, 'mask_threshold', 48))
    base_area = int(getattr(opt, 'min_mosaic_area', 150))

    if not auto:
        mask = impro.mask_threshold(raw_resized, ex_mun=ex_mun, threshold=base_th)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        boxes = _extract_all_boxes(mask, opt)
        if not boxes:
            return None, []
        # Build combined mask with all components
        combined = np.zeros_like(mask, dtype=np.uint8)
        for _, _, _, _, comp in boxes:
            combined = cv2.bitwise_or(combined, comp)
        # If not all_mosaic_area, keep only largest
        if not getattr(opt, 'all_mosaic_area', False):
            # largest is first after sort
            cx, cy, s, _, _ = boxes[0]
            # Return single largest as mask
            largest_mask = np.zeros_like(mask, dtype=np.uint8)
            largest_mask[combined>127] = 0
            # Actually use the component mask of largest
            largest_mask = boxes[0][4]
            return largest_mask, [(boxes[0][0], boxes[0][1], boxes[0][2])]
        # all_mosaic: return combined and all boxes
        return combined, [(b[0], b[1], b[2]) for b in boxes]

    # Auto-adaptive sweep
    thresholds = []
    for t in [base_th, 40, 32, 24, 16, 12, 8, 5, 64]:
        if 5 <= t <= 96 and t not in thresholds:
            thresholds.append(t)
    thresholds = sorted(set(thresholds), reverse=True)
    area_candidates = sorted(set([base_area, 200, 150, 100, 50, 30]), reverse=True)

    best_mask = None
    best_boxes = []
    best_score = -1
    best_th = None

    for th in thresholds:
        for min_a in area_candidates:
            class _Proxy: pass
            p = _Proxy()
            p.__dict__.update(opt.__dict__)
            p.min_mosaic_area = min_a
            mask_th = impro.mask_threshold(raw_resized, ex_mun=ex_mun, threshold=th)
            mask_th = cv2.dilate(mask_th, np.ones((3, 3), np.uint8), iterations=1)
            boxes = _extract_all_boxes(mask_th, p)
            if not boxes:
                continue
            # Filter by size already in _extract_all_boxes
            total_area = sum(b[3] for b in boxes)
            num = len(boxes)
            # Score: total area, plus bonus for multiple components when they are well-sized
            # Penalize thresholds far from base
            th_penalty = abs(th - base_th) * 8
            # Prefer at least 1 component, but if multi, ensure they are not tiny speckles
            # For multi, require total area > num * 800 (approx 28x28)
            if num > 1 and total_area < num * 800:
                continue
            score = total_area - th_penalty + num * 200  # small bonus for multi
            if score > best_score:
                best_score = score
                best_th = th
                # Build combined mask
                combined = np.zeros_like(mask_th, dtype=np.uint8)
                for _, _, _, _, comp in boxes:
                    combined = cv2.bitwise_or(combined, comp)
                best_mask = combined
                best_boxes = [(b[0], b[1], b[2]) for b in boxes]
        if best_score > 0 and th == base_th:
            try:
                if int(raw_resized.max()) >= base_th:
                    break
            except Exception:
                pass

    if best_mask is not None and best_boxes:
        # If not all_mosaic_area and single largest is desired but we have multi,
        # we have a choice: if auto and multi detected with good scores, return all
        # to enable multi-mosaic cleaning. Check if multi is plausible.
        if not getattr(opt, 'all_mosaic_area', False):
            # Auto-enable multi if we detected >=2 components with reasonable size
            # and total area is large enough
            if len(best_boxes) >= 2 and best_score > 2000:
                # Return all for multi-mosaic
                return best_mask, best_boxes
            # Otherwise return only largest
            # Re-extract largest component mask
            # Find largest box's component
            # We already have best_mask with all, but for single we should return only largest component
            # To get largest component mask, we need to re-extract with best_th
            # Simplify: find contours of best_mask and keep largest
            largest = impro.find_mostlikely_ROI(best_mask.copy())
            # Recompute its box
            ys, xs = np.where(largest > 127)
            if len(xs) > 0:
                cx = int(np.mean(xs)); cy = int(np.mean(ys))
                x_min, x_max = int(np.min(xs)), int(np.max(xs))
                y_min, y_max = int(np.min(ys)), int(np.max(ys))
                size = max(x_max - x_min, y_max - y_min)//2
                return largest, [(cx, cy, size)]
            return best_mask, [best_boxes[0]]
        return best_mask, best_boxes

    # Fallback: tiled detection when full-frame fails (raw max is low)
    # This helps for multi-mosaic where full-frame confidence is diluted
    try:
        raw_max = int(raw_resized.max())
    except Exception:
        raw_max = 0
    if raw_max < 30 and raw_max >= 5:
        # Try 2x2 tiling with overlap
        tile_boxes = []
        tile_masks = []
        # Tiles: 4 quadrants with 20% overlap
        tile_w, tile_h = w // 2, h // 2
        overlaps = [(0, 0), (w - tile_w, 0), (0, h - tile_h), (w - tile_w, h - tile_h)]
        # Also add center tile for 720p
        overlaps.append((w//4, h//4))
        for ox, oy in overlaps:
            # Ensure tile within bounds
            x0, y0 = max(0, ox), max(0, oy)
            x1, y1 = min(w, x0 + tile_w + w//5), min(h, y0 + tile_h + h//5)
            tile = img_origin[y0:y1, x0:x1]
            if tile.size == 0:
                continue
            # Run detection on tile (reuse raw tiling would be more efficient, but we run net again for simplicity)
            try:
                raw_tile = run_segment(tile, net_mosaic_pos, size=360, gpu_id=opt.gpu_id)
                raw_tile_resized = cv2.resize(raw_tile, (tile.shape[1], tile.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                # Use low threshold for tile
                mask_tile = impro.mask_threshold(raw_tile_resized, ex_mun=6, threshold=max(8, int(raw_tile.max()*0.4) if raw_tile.max()>0 else 12))
                mask_tile = cv2.dilate(mask_tile, np.ones((3,3),np.uint8), iterations=1)
                boxes_tile = _extract_all_boxes(mask_tile, opt)
                for cx, cy, s, _, comp in boxes_tile:
                    # Map back to full frame coordinates
                    gx, gy = cx + x0, cy + y0
                    # Create full-size component mask
                    full_comp = np.zeros((h, w), dtype=np.uint8)
                    full_comp[y0:y1, x0:x1] = cv2.resize(comp, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
                    tile_boxes.append((gx, gy, s, full_comp))
            except Exception:
                continue
        if tile_boxes:
            # Merge overlapping boxes via NMS-ish: keep largest, suppress nearby
            # Simple: sort by area (size), keep if not overlapping heavily with kept
            tile_boxes.sort(key=lambda b: b[2], reverse=True)
            kept = []
            kept_masks = []
            for gx, gy, s, comp in tile_boxes:
                overlap = False
                for kx, ky, ks, _ in kept:
                    dist = np.hypot(gx - kx, gy - ky)
                    if dist < (s + ks) * 0.7:
                        overlap = True
                        break
                if not overlap:
                    kept.append((gx, gy, s, comp))
                    kept_masks.append(comp)
            if kept:
                combined = np.zeros((h, w), dtype=np.uint8)
                for _, _, _, comp in kept:
                    combined = cv2.bitwise_or(combined, comp)
                return combined, [(b[0], b[1], b[2]) for b in kept]
    # Final fallback: most permissive
    mask = impro.mask_threshold(raw_resized, ex_mun=ex_mun, threshold=thresholds[-1])
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    boxes = _extract_all_boxes(mask, opt)
    if boxes:
        combined = np.zeros_like(mask, dtype=np.uint8)
        for _, _, _, _, comp in boxes:
            combined = cv2.bitwise_or(combined, comp)
        if not getattr(opt, 'all_mosaic_area', False) and len(boxes) > 1:
            # For backward compat, if single mode but we found multi via permissive, still return all if auto
            # Check score
            if len(boxes) >= 2:
                return combined, [(b[0], b[1], b[2]) for b in boxes]
        return combined, [(b[0], b[1], b[2]) for b in boxes]
    return None, []


def get_mosaic_position(img_origin, net_mosaic_pos, opt):
    h, w = img_origin.shape[:2]
    try:
        raw = run_segment(img_origin, net_mosaic_pos, size=360, gpu_id=opt.gpu_id)
    except Exception as e:
        print(f"Segmentation error: {e}")
        return None, 0, 0, 0
    if raw is None or raw.size == 0:
        return None, 0, 0, 0
    if not isinstance(raw, np.ndarray):
        print(f"Invalid mask type: {type(raw)}")
        return None, 0, 0, 0
    try:
        raw_resized = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LANCZOS4)
    except Exception as e:
        print(f"Mask resize error: {e}")
        return None, 0, 0, 0
    auto = not bool(getattr(opt, 'no_auto_adapt', False))
    ex_mun = int(min(h, w) / 20)
    base_th = int(getattr(opt, 'mask_threshold', 48))
    base_area = int(getattr(opt, 'min_mosaic_area', 150))
    if not auto:
        mask = impro.mask_threshold(raw_resized, ex_mun=ex_mun, threshold=base_th)
        return _process_mask_to_roi(mask, opt)
    thresholds = []
    for t in [base_th, 40, 32, 24, 16, 12, 64]:
        if 8 <= t <= 96 and t not in thresholds:
            thresholds.append(t)
    thresholds = sorted(set(thresholds), reverse=True)
    area_candidates = sorted(set([base_area, 200, 150, 100, 50, 30]), reverse=True)
    best = (None, 0, 0, 0)
    best_score = -1
    for th in thresholds:
        for min_a in area_candidates:
            class _Proxy: pass
            p = _Proxy()
            p.__dict__.update(opt.__dict__)
            p.min_mosaic_area = min_a
            mask_th = impro.mask_threshold(raw_resized, ex_mun=ex_mun, threshold=th)
            m, x, y, s = _process_mask_to_roi(mask_th, p)
            if m is None or s <= 0:
                continue
            if s < 12 or s > 500:
                continue
            try:
                real_area = int(np.sum(cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)[1] > 0))
            except Exception:
                real_area = s * s
            th_penalty = abs(th - base_th) * 10
            score = real_area - th_penalty
            if score > best_score:
                best_score = score
                best = (m, x, y, s)
        if best_score > 0 and th == base_th:
            try:
                if int(raw_resized.max()) >= base_th:
                    break
            except Exception:
                pass
    if best[0] is not None:
        return best
    mask = impro.mask_threshold(raw_resized, ex_mun=ex_mun, threshold=thresholds[-1])
    return _process_mask_to_roi(mask, opt)
