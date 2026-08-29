import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cores.options import Options


class TestCliModelResolution(unittest.TestCase):
    def _parse(self, *args):
        with tempfile.NamedTemporaryFile(suffix=".png") as media, \
                mock.patch.object(sys, "argv", ["deepmosaic.py", "--media_path",
                                                media.name, *args]), \
                mock.patch("builtins.input", return_value=""):
            return Options().getparse(test_flag=True)

    def test_explicit_modern_model_implies_clean_mode(self):
        opt = self._parse("--model", "quality")
        self.assertEqual(opt.mode, "clean")

    def test_clean_auto_does_not_require_legacy_model_path(self):
        opt = self._parse("--mode", "clean", "--model", "auto",
                          "--model_path", "/definitely/missing/add_face.pth")
        self.assertEqual(opt.mode, "clean")
        self.assertEqual(opt.model, "auto")

    def test_detector_auto_remains_managed(self):
        opt = self._parse("--mode", "clean", "--model", "traditional")
        self.assertEqual(opt.mosaic_position_model_path, "auto")

    def test_missing_legacy_checkpoint_exits_nonzero(self):
        with self.assertRaises(SystemExit) as caught:
            self._parse("--mode", "clean", "--model", "legacy",
                        "--model_path", "/definitely/missing/model.pth")
        self.assertEqual(caught.exception.code, 2)

    def test_typed_lite_checkpoint_routes_clean_without_legacy_inference(self):
        with tempfile.NamedTemporaryFile(suffix=".pth") as checkpoint:
            opt = self._parse("--model", "lite:" + checkpoint.name)
        self.assertEqual(opt.mode, "clean")
        self.assertEqual(opt.netG, "auto")

    def test_typed_checkpoint_expands_tilde_before_validation(self):
        home = os.path.expanduser("~")
        with tempfile.NamedTemporaryFile(dir=home, suffix=".pth") as checkpoint:
            tilde_path = "~" + checkpoint.name[len(home):]
            opt = self._parse("--model", "lite:" + tilde_path)
            self.assertEqual(opt.model, "lite:" + checkpoint.name)

    def test_restore_strength_out_of_range_is_rejected(self):
        with self.assertRaises(SystemExit) as caught:
            self._parse("--model", "traditional", "--restore_strength", "1.1")
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
