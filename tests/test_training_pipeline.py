import random
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from train.mosaicvr.train import OnlineMosaicClipDataset
from train.mosaicvr.train import feature_matching
from models import BVDNet, model_util


class TestTemporalAugmentation(unittest.TestCase):
    def test_all_frames_share_one_spatial_crop(self):
        """Identical source frames must stay aligned after clip augmentation."""
        dataset = object.__new__(OnlineMosaicClipDataset)
        dataset.opt = SimpleNamespace(
            T=3, S=1, finesize=16, degradate_p=0.0,
        )
        dataset.videos = [("fixture", 3)]
        dataset.mods = ["squa_avg"]

        yy, xx = np.mgrid[:40, :44]
        frame = np.stack((xx, yy, xx + yy), axis=2).astype(np.uint8)
        mask = np.full(frame.shape[:2], 255, np.uint8)
        dataset._load = lambda _vdir, _idx: (frame.copy(), mask.copy())

        random.seed(9)
        with mock.patch(
            "train.mosaicvr.train.mosaic_util.get_random_parameter",
            return_value=(4, "squa_avg", 1.0, 0),
        ), mock.patch(
            "train.mosaicvr.train.mosaic_util.addmosaic_base",
            side_effect=lambda img, *_args, **_kwargs: img.copy(),
        ):
            mosaic, original = dataset.get_clip()

        self.assertEqual(mosaic.shape, (3, 3, 16, 16))
        np.testing.assert_array_equal(original[0], original[1])
        np.testing.assert_array_equal(original[1], original[2])
        np.testing.assert_array_equal(mosaic, original)


class TestMosaicSpecificLosses(unittest.TestCase):
    def test_discriminator_exposes_intermediate_features(self):
        net = BVDNet.define_D(
            ndf=8, n_layers_D=2, num_D=2, gpu_id="-1",
            return_features=True)
        outputs = net(__import__("torch").randn(1, 6, 32, 32))
        self.assertEqual(len(outputs), 2)
        self.assertTrue(all(len(scale) > 1 for scale in outputs))
        self.assertTrue(all(scale[-1].shape[1] == 1 for scale in outputs))

    def test_feature_matching_ignores_final_logits(self):
        torch = __import__("torch")
        intermediate = torch.zeros(1, 4, 4, 4)
        fake = [[intermediate, torch.ones(1, 1, 2, 2)]]
        real = [[intermediate.clone(), torch.zeros(1, 1, 2, 2)]]
        self.assertEqual(float(feature_matching(fake, real)), 0.0)

        fake[0][0] = torch.ones_like(intermediate)
        self.assertGreater(float(feature_matching(fake, real)), 0.0)

    def test_vgg_input_normalization(self):
        torch = __import__("torch")
        black = torch.full((1, 3, 2, 2), -1.0)
        white = torch.full((1, 3, 2, 2), 1.0)
        normalized_black = model_util.normalize_vgg_input(black)
        normalized_white = model_util.normalize_vgg_input(white)
        expected_black = torch.tensor(
            [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225])
        expected_white = torch.tensor(
            [(1 - 0.485) / 0.229, (1 - 0.456) / 0.224,
             (1 - 0.406) / 0.225])
        torch.testing.assert_close(normalized_black[0, :, 0, 0], expected_black)
        torch.testing.assert_close(normalized_white[0, :, 0, 0], expected_white)


if __name__ == "__main__":
    unittest.main()
