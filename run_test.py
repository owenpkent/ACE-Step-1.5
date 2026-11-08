"""Tests for the run.py double-click launcher (argv building and flag parsing only)."""

import unittest
from unittest.mock import mock_open, patch

import run


class BuildCommandTest(unittest.TestCase):
    """build_command produces the same launch the default .bat launchers use."""

    def test_ui_defaults_match_start_gradio_ui(self):
        cmd = run.build_command("uv", api=False, port=None, extra=[])
        self.assertEqual(cmd[:4], ["uv", "run", "--no-sync", "acestep"])
        self.assertIn("7860", cmd)
        self.assertEqual(cmd[cmd.index("--server-name") + 1], "127.0.0.1")
        self.assertEqual(cmd[cmd.index("--config_path") + 1], "acestep-v15-turbo")

    def test_extra_overrides_a_default_instead_of_duplicating_it(self):
        cmd = run.build_command("uv", False, None, ["--config_path", "acestep-v15-base"])
        self.assertEqual(cmd.count("--config_path"), 1)
        self.assertEqual(cmd[-2:], ["--config_path", "acestep-v15-base"])

    def test_models_come_from_env_not_the_fallback(self):
        env = {"ACESTEP_CONFIG_PATH": "acestep-v15-xl-sft",
               "ACESTEP_LM_MODEL_PATH": "acestep-5Hz-lm-4B"}
        cmd = run.build_command("uv", False, None, [], env)
        self.assertEqual(cmd[cmd.index("--config_path") + 1], "acestep-v15-xl-sft")
        self.assertEqual(cmd[cmd.index("--lm_model_path") + 1], "acestep-5Hz-lm-4B")

    def test_empty_env_value_uses_fallback(self):
        cmd = run.build_command("uv", False, None, [], {"ACESTEP_CONFIG_PATH": ""})
        self.assertEqual(cmd[cmd.index("--config_path") + 1], "acestep-v15-turbo")

    def test_api_mode_uses_api_entry_point_and_port(self):
        cmd = run.build_command("uv", api=True, port=None, extra=[])
        self.assertEqual(cmd[3:], ["acestep-api", "--host", "127.0.0.1", "--port", "8001"])

    def test_custom_port(self):
        self.assertIn("7861", run.build_command("uv", False, 7861, []))


class ReadEnvFileTest(unittest.TestCase):
    """read_env_file handles comments, quotes and a missing file."""

    def test_parses_values_and_skips_comments(self):
        text = "# ACESTEP_CONFIG_PATH=nope\nACESTEP_CONFIG_PATH=\"xl\"  # mine\nBARE\n"
        with patch("builtins.open", mock_open(read_data=text)):
            self.assertEqual(run.read_env_file(".env"), {"ACESTEP_CONFIG_PATH": "xl"})

    def test_missing_file_is_empty(self):
        self.assertEqual(run.read_env_file("no/such/.env"), {})


class ParseArgsTest(unittest.TestCase):
    """parse_args separates launcher flags from pass-through app args."""

    def test_passthrough_after_double_dash(self):
        args, extra = run.parse_args(["--api", "--", "--port", "9"])
        self.assertTrue(args.api)
        self.assertIsNone(args.port)
        self.assertEqual(extra, ["--port", "9"])

    def test_no_args(self):
        args, extra = run.parse_args([])
        self.assertFalse(args.api or args.no_browser or args.hold)
        self.assertEqual(extra, [])


class FindUvTest(unittest.TestCase):
    """find_uv prefers PATH and reports None when uv is nowhere."""

    def test_prefers_path(self):
        with patch("run.shutil.which", return_value="C:/uv.exe"):
            self.assertEqual(run.find_uv(), "C:/uv.exe")

    def test_none_when_missing(self):
        with patch("run.shutil.which", return_value=None), \
                patch("run.os.path.isfile", return_value=False):
            self.assertIsNone(run.find_uv())


class RunTest(unittest.TestCase):
    """run fails cleanly, without launching anything, when uv is missing."""

    def test_missing_uv_returns_error(self):
        args, extra = run.parse_args(["--no-browser"])
        with patch("run.find_uv", return_value=None), \
                patch("run.subprocess.Popen") as popen:
            self.assertEqual(run.run(args, extra), 1)
            popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
