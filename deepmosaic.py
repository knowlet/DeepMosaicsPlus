import os
import sys
import traceback
from cores import init

try:
    from cores import Options, add, clean, style
    from util import util
    from models import loadmodel
except Exception as e:
    print(e, file=sys.stderr)
    sys.exit(1)

opt = Options().getparse(test_flag=True)
if not os.path.isdir(opt.temp_dir):
    util.file_init(opt)

def main():
    if os.path.isdir(opt.media_path):
        files = util.Traversal(opt.media_path)
    else:
        files = [opt.media_path]

    if opt.mode == 'add':
        netS = loadmodel.bisenet(opt, 'roi')
        for file in files:
            opt.media_path = file
            if util.is_img(file):
                add.addmosaic_img(opt, netS)
            elif util.is_video(file):
                add.addmosaic_video(opt, netS)
                util.clean_tempfiles(opt, tmp_init=False)
            else:
                print('This type of file is not supported')
            util.clean_tempfiles(opt, tmp_init=False)

    elif opt.mode == 'clean':
        # unified restoration service (manifest + device + backend + detector)
        from restoration import RestorationService

        service = RestorationService.create(opt)
        netM = service.detector_net

        for file in files:
            opt.media_path = file
            if util.is_img(file):
                clean.cleanmosaic_img(opt, service)
            elif util.is_video(file):
                clean.cleanmosaic_video(opt, service, netM)
                util.clean_tempfiles(opt, tmp_init=False)
            else:
                print('This type of file is not supported')

    elif opt.mode == 'style':
        netG = loadmodel.style(opt)
        for file in files:
            opt.media_path = file
            if util.is_img(file):
                style.styletransfer_img(opt, netG)
            elif util.is_video(file):
                style.styletransfer_video(opt, netG)
                util.clean_tempfiles(opt, tmp_init=False)
            else:
                print('This type of file is not supported')

    util.clean_tempfiles(opt, tmp_init=False)

if __name__ == '__main__':
    if opt.debug:
        main()
        sys.exit(0)
    try:
        main()
        print('Finished!')
    except Exception as ex:
        print('--------------------ERROR--------------------')
        print('--------------Environment--------------')
        print('DeepMosaicsPlus: 0.6.0')
        print('Python:', sys.version)
        try:
            import torch
            print('Pytorch:', torch.__version__)
        except Exception:
            pass
        try:
            import cv2
            print('OpenCV:', cv2.__version__)
        except Exception:
            pass
        import platform
        print('Platform:', platform.platform())

        print('--------------BUG--------------')
        ex_type, ex_val, ex_stack = sys.exc_info()
        print('Error Type:', ex_type)
        print(ex_val)
        for stack in traceback.extract_tb(ex_stack):
            print(stack)
        sys.exit(1)
