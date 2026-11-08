"""Tests for the run.py double-click launcher (argv building and flag parsing only)."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, mock_open, patch

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

    def test_passthrough_port_replaces_the_default_port(self):
        for extra in (["--port", "9001"], ["--port=9001"]):
            for api in (True, False):
                cmd = run.build_command("uv", api, None, extra)
                self.assertEqual([a.split("=")[0] for a in cmd].count("--port"), 1, (extra, api))
                self.assertNotIn("8001" if api else "7860", cmd)


class PassthroughPortTest(unittest.TestCase):
    """passthrough_port finds the port the app's argparse will settle on."""

    def test_both_spellings_and_last_wins(self):
        self.assertEqual(run.passthrough_port(["--port", "9001"]), 9001)
        self.assertEqual(run.passthrough_port(["--port=9001"]), 9001)
        self.assertEqual(run.passthrough_port(["--port", "1", "--port=2"]), 2)

    def test_absent_or_unusable_port_is_none(self):
        self.assertIsNone(run.passthrough_port(["--offload_to_cpu", "true"]))
        self.assertIsNone(run.passthrough_port(["--port"]))
        self.assertIsNone(run.passthrough_port(["--port", "abc"]))


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


class EnsureVenvTest(unittest.TestCase):
    """ensure_venv trusts a completed sync, not the mere existence of .venv."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.mkdir(os.path.join(self.tmp.name, ".venv"))  # as a failed sync can leave it
        self.stamp = os.path.join(self.tmp.name, ".venv", ".run-py-synced")
        patcher = patch("run.SYNC_STAMP", self.stamp)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_failed_sync_is_retried_on_the_next_launch(self):
        with patch("run.subprocess.call", return_value=1) as call:
            self.assertFalse(run.ensure_venv("uv"))
        self.assertEqual(call.call_count, 2)  # online, then offline
        self.assertFalse(os.path.exists(self.stamp))
        with patch("run.subprocess.call", return_value=0) as call:
            self.assertTrue(run.ensure_venv("uv"))
        call.assert_called_once()
        self.assertTrue(os.path.isfile(self.stamp))

    def test_completed_sync_is_not_repeated(self):
        open(self.stamp, "w").close()
        with patch("run.subprocess.call") as call:
            self.assertTrue(run.ensure_venv("uv"))
        call.assert_not_called()


class RunTest(unittest.TestCase):
    """run fails cleanly when uv is missing and points the browser at the real port."""

    def test_missing_uv_returns_error(self):
        args, extra = run.parse_args(["--no-browser"])
        with patch("run.find_uv", return_value=None), \
                patch("run.subprocess.Popen") as popen:
            self.assertEqual(run.run(args, extra), 1)
            popen.assert_not_called()

    def test_browser_follows_a_passthrough_port(self):
        for tail in (["--port", "9001"], ["--port=9001"]):
            args, extra = run.parse_args(["--api", "--", *tail])
            with patch("run.find_uv", return_value="uv"),                     patch("run.ensure_venv", return_value=True),                     patch("run.subprocess.Popen", return_value=MagicMock()),                     patch("run.threading.Thread") as thread:
                run.run(args, extra)
            url, port, _proc = thread.call_args.kwargs["args"]
            self.assertEqual((url, port), ("http://127.0.0.1:9001/docs", 9001), tail)


if __name__ == "__main__":
    unittest.main()
