import argparse
import os
import sys


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):

        #base
        self.parser.add_argument('--debug', action='store_true', help='if specified, start debug mode')
        self.parser.add_argument('--gpu_id', type=str,default='0', help='(legacy) if -1, use cpu; prefer --device')
        self.parser.add_argument('--device', type=str, default=None,
            help='device for all models -> auto | cuda | mps | directml | cpu (default auto)')
        self.parser.add_argument('--media_path', type=str, default='./imgs/ruoruo.jpg',help='your videos or images path')
        self.parser.add_argument('-ss', '--start_time', type=str, default='00:00:00',help='start position of video, default is the beginning of video')
        self.parser.add_argument('-t', '--last_time', type=str, default='00:00:00',help='duration of the video, default is the entire video')
        self.parser.add_argument('--mode', type=str, default='auto',help='Program running mode. auto | add | clean | style')
        self.parser.add_argument('--model', type=str, default='auto',
            help='restoration model for clean mode -> auto | quality | lite | traditional | '
                 '<manifest id> | legacy.pth | lite:/path.pth | portable:/path.pth '
                 '(see docs/model_manifest.md)')
        self.parser.add_argument('--manifest', type=str, nargs='+', default=None,
            help='extra model manifest yaml(s), merged over the built-in registry')
        self.parser.add_argument('--model_path', type=str, default='./pretrained_models/mosaic/add_face.pth',help='(legacy) pretrained model path')
        self.parser.add_argument('--result_dir', type=str, default='./result',help='output media will be saved here')
        self.parser.add_argument('--temp_dir', type=str, default='./tmp', help='Temporary files will go here')
        self.parser.add_argument('--tempimage_type', type=str, default='jpg',help='type of temp image, png | jpg, png is better but occupy more storage space')
        self.parser.add_argument('--netG', type=str, default='auto',
            help='(legacy) select netG architecture when using classic checkpoints -> auto | unet_128 | unet_256 | resnet_9blocks | HD | video')
        self.parser.add_argument('--fps', type=int, default=0,help='read and output fps, if 0-> origin')
        self.parser.add_argument('--no_preview', action='store_true', help='if specified,do not preview images when processing video. eg.(when run it on server)')
        self.parser.add_argument('--output_size', type=int, default=0,help='size of output media, if 0 -> origin')
        self.parser.add_argument('--mask_threshold', type=int, default=48,help='Mosaic detection threshold (0~255). The smaller is it, the more likely judged as a mosaic area. Auto-adapted when --no_auto_adapt not set.')

        #AddMosaic
        self.parser.add_argument('--mosaic_mod', type=str, default='squa_avg',help='type of mosaic -> squa_avg | squa_random | squa_avg_circle_edge | rect_avg | random')
        self.parser.add_argument('--mosaic_size', type=int, default=0,help='mosaic size,if 0 auto size')
        self.parser.add_argument('--mask_extend', type=int, default=10,help='extend mosaic area')

        #CleanMosaic
        self.parser.add_argument('--mosaic_position_model_path', type=str, default='auto',help='name of model use to find mosaic position')
        self.parser.add_argument('--traditional', action='store_true', help='(legacy) use traditional image processing methods to clean mosaic (same as --model traditional)')
        self.parser.add_argument('--tr_blur', type=int, default=10, help='ksize of blur when using traditional method, it will affect final quality')
        self.parser.add_argument('--tr_down', type=int, default=10, help='downsample when using traditional method,it will affect final quality')
        self.parser.add_argument('--no_feather', action='store_true', help='if specified, no edge feather and color correction, but run faster')
        self.parser.add_argument('--keep_frames', action='store_true', help='if specified, do not delete source frames from video2image dir during cleaning (useful for UI seeking)')
        self.parser.add_argument('--encode_crf', type=int, default=18, help='CRF quality for output video encode (0=lossless, 51=worst, default 18)')
        self.parser.add_argument('--decode_qv', type=int, default=1, help='JPEG quality for extracted frames (1=best, 31=worst, default 1)')
        self.parser.add_argument('--restore_strength', type=float, default=1.0,
            help='blend restored detail into the source mosaic region (0=source, 1=full model)')
        self.parser.add_argument('--max_restore_side', type=int, default=0,
            help='maximum source-content crop side before backend-required padding; '
                 '0 selects a device-memory-safe value')
        self.parser.add_argument('--restore_clip_len', type=int, default=0,
            help='maximum temporal frames per neural call; 0 selects a device-memory-safe value')
        self.parser.add_argument('--luma_sharpen', action='store_true',
            help='apply luma-channel unsharp-mask to cleaned region')
        self.parser.add_argument('--luma_sharpen_amount', type=float, default=1.0,
            help='luma sharpening strength (0.5=mild, 1.0=normal, 2.0=strong)')
        self.parser.add_argument('--bilateral_sharpen', action='store_true',
            help='edge-preserving bilateral sharpening on cleaned region')
        self.parser.add_argument('--bilateral_sharpen_amount', type=float, default=0.5,
            help='bilateral sharpening strength (0.2=subtle, 0.5=moderate, 1.0=strong)')
        self.parser.add_argument('--freq_inject', action='store_true',
            help='inject high-frequency edges from original mosaic into cleaned patch')
        self.parser.add_argument('--freq_inject_amount', type=float, default=0.3,
            help='frequency injection blend strength (0.1=subtle, 0.3=moderate, 0.6=strong)')
        self.parser.add_argument('--all_mosaic_area', action='store_true', help='if specified, find all mosaic area, else only find the largest area')
        self.parser.add_argument('--min_mosaic_area', type=int, default=150,
            help='minimum connected component area (pixels) to keep in mosaic mask. '
                 'Lower values detect smaller mosaic regions. Default 150 (auto-adapted if --no_auto_adapt not set).')
        self.parser.add_argument('--min_mosaic_size', type=int, default=40,
            help='minimum bounding-box half-size (pixels) to attempt cleaning. '
                 'Lower values process smaller detected regions. Default 40 (auto-adapted if --no_auto_adapt not set).')
        self.parser.add_argument('--medfilt_num', type=int, default=5,help='medfilt window of mosaic movement in the video')
        self.parser.add_argument('--ex_mult', type=str, default='auto',help='mosaic area expansion')
        self.parser.add_argument('--no_auto_adapt', action='store_true',
            help='disable automatic mosaic threshold/size adaptation; use fixed --mask_threshold/--min_mosaic_* values')

        #StyleTransfer
        self.parser.add_argument('--preprocess', type=str, default='resize', help='resize and cropping of images at load time [ resize | resize_scale_width | edges | gray] or resize,edges(use comma to split)')
        self.parser.add_argument('--edges', action='store_true', help='if specified, use edges to generate pictures,(input_nc = 1)')  
        self.parser.add_argument('--canny', type=int, default=150,help='threshold of canny')
        self.parser.add_argument('--only_edges', action='store_true', help='if specified, output media will be edges')

        self.initialized = True


    def getparse(self, test_flag = False):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        # Expand user-local checkpoint paths before validation and preserve the
        # explicit backend prefix used by new trainer outputs.
        if self.opt.model.startswith(('lite:', 'portable:')):
            checkpoint_kind, checkpoint_path = self.opt.model.split(':', 1)
            self.opt.model = checkpoint_kind + ':' + os.path.expanduser(checkpoint_path)
        elif self.opt.model.endswith(('.pth', '.pt', '.ckpt')):
            self.opt.model = os.path.expanduser(self.opt.model)
        self.opt.model_path = os.path.expanduser(self.opt.model_path)

        model_name = os.path.basename(self.opt.model_path)
        self.opt.temp_dir = os.path.join(self.opt.temp_dir, 'DeepMosaics_temp')

        # ---- device selection (unified) --------------------------------
        if self.opt.device is None:
            self.opt.device = 'cpu' if str(self.opt.gpu_id) == '-1' else 'auto'
        if self.opt.gpu_id != '-1':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.opt.gpu_id)

        # ---- mode / model resolution ------------------------------------
        modern_model = self.opt.model not in ('auto', 'legacy')

        if test_flag:
            if not 0.0 <= self.opt.restore_strength <= 1.0:
                self.parser.error('--restore_strength must be between 0 and 1')
            if self.opt.max_restore_side < 0 or self.opt.restore_clip_len < 0:
                self.parser.error(
                    '--max_restore_side and --restore_clip_len must be non-negative')
            media_is_dir = os.path.isdir(self.opt.media_path)
            if not media_is_dir and not os.path.exists(self.opt.media_path):
                self.parser.error('Media does not exist: %s' % self.opt.media_path)

            typed_checkpoint = self.opt.model.startswith(('lite:', 'portable:'))
            explicit_checkpoint = (
                self.opt.model.endswith(('.pth', '.pt', '.ckpt'))
                and not typed_checkpoint)
            checkpoint_to_check = (
                self.opt.model.split(':', 1)[1] if typed_checkpoint else
                (self.opt.model if explicit_checkpoint else self.opt.model_path))
            needs_legacy_checkpoint = self.opt.model == 'legacy' or explicit_checkpoint
            needs_checkpoint = needs_legacy_checkpoint or typed_checkpoint
            if needs_checkpoint and not os.path.exists(checkpoint_to_check):
                self.parser.error('Model does not exist: %s' % checkpoint_to_check)

            if self.opt.mode == 'auto':
                hinted_add = 'add' in model_name
                hinted_style = ('style' in model_name) or ('edges' in model_name)
                if modern_model or self.opt.traditional:
                    # Explicit restoration ids/checkpoints take precedence over
                    # hints from the unrelated legacy --model_path default.
                    self.opt.mode = 'clean'
                elif hinted_style:
                    self.opt.mode = 'style'
                elif hinted_add:
                    self.opt.mode = 'add'
                else:
                    # explicit --model ids imply clean; classic filename hints too
                    self.opt.mode = 'clean'

            if self.opt.output_size == 0 and self.opt.mode == 'style':
                self.opt.output_size = 512

            if 'edges' in model_name or 'edges' in self.opt.preprocess:
                self.opt.edges = True

            if self.opt.mode == 'clean':
                if self.opt.traditional:
                    self.opt.model = 'traditional'
                if self.opt.model == 'legacy' or explicit_checkpoint:
                    # legacy behaviour: guess generator family from filename
                    clean_model_name = os.path.basename(
                        self.opt.model if explicit_checkpoint else self.opt.model_path)
                    if 'unet_256' in clean_model_name:
                        self.opt.netG = 'unet_256'
                    elif 'unet_128' in clean_model_name:
                        self.opt.netG = 'unet_128'
                    elif 'resnet_9blocks' in clean_model_name:
                        self.opt.netG = 'resnet_9blocks'
                    elif 'HD' in clean_model_name and 'video' not in clean_model_name:
                        self.opt.netG = 'HD'
                    elif 'video' in clean_model_name:
                        self.opt.netG = 'video'
                    else:
                        self.parser.error(
                            'Could not infer legacy generator type from: %s' %
                            clean_model_name)

            if self.opt.ex_mult == 'auto':
                if 'face' in model_name:
                    self.opt.ex_mult = 1.1
                else:
                    self.opt.ex_mult = 1.5
            else:
                self.opt.ex_mult = float(self.opt.ex_mult)

            # Keep ``auto`` declarative. RestorationService resolves the
            # detector through ModelManager on every auto path, including an
            # already cached file, so SHA-256 verification cannot be bypassed
            # by an old/unverified pretrained_models/mosaic_position.pth.

        return self.opt
