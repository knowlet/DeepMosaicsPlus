import os
import sys
import base64

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cores import Options, clean
from util import image_processing as impro

# python server.py --device auto --model quality
# The detector remains "auto" so ModelManager downloads/verifies it.
opt = Options()
opt.parser.add_argument('--port', type=int, default=4000, help='http port')
opt.parser.add_argument(
    '--source_url', default=os.environ.get('DEEPMOSAICS_SOURCE_URL', ''),
    help='public corresponding-source URL for this exact deployed version '
         '(required for the AGPL network service)')
opt = opt.getparse(True)
if not opt.source_url:
    raise RuntimeError(
        'Server mode requires --source_url (or DEEPMOSAICS_SOURCE_URL) '
        'pointing to the complete source for this exact deployment.')

from flask import Flask, request
import numpy as np
import cv2

from restoration import RestorationService

# One shared service for every request: same manifest/device/backend rules
# as the CLI and GUI.
service = RestorationService.create(opt)

app = Flask(__name__)


@app.route('/', methods=['GET'])
def service_info():
    return {
        'service': 'DeepMosaicsPlus',
        'source': opt.source_url,
        'license': 'AGPL-3.0-only quality path',
    }

@app.route("/handle", methods=["POST"])
def handle():
    result = {'source': opt.source_url}
    imgRec = request.form.get('img', '')
    try:
        if not imgRec:
            raise ValueError('missing img form field')
        imgByte = base64.b64decode(imgRec, validate=True)
        img_np_arr = np.frombuffer(imgByte, np.uint8)
        img = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError('invalid encoded image')
    except Exception:
        result['img'] = imgRec
        result['info'] = 'readfailed'
        return result

    try:
        if max(img.shape) > 1080:
            img = impro.resize(img, 720, interpolation=cv2.INTER_CUBIC)
        img = clean.cleanmosaic_img_server(opt, img, service)
    except Exception:
        result['img'] = imgRec
        result['info'] = 'procfailed'
        return result

    imgbytes = cv2.imencode('.jpg', img)[1]
    result['img'] = base64.b64encode(imgbytes).decode('utf-8')
    result['info'] = 'ok'
    return result

app.run("0.0.0.0", port=opt.port, debug=opt.debug)
