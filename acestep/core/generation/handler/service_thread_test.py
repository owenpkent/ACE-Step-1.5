"""Unit tests for the long-lived ``service_generate`` worker threads."""

import threading
import unittest
from unittest.mock import patch

from acestep.core.generation.handler import service_thread


class RunOnServiceThreadTests(unittest.TestCase):
    """``run_on_service_thread`` reuses idle threads and reports timeouts."""

    def test_sequential_calls_reuse_one_thread(self):
        """Back-to-back calls run on the same long-lived thread, never the caller's."""
        seen = []
        for _ in range(3):
            finished = service_thread.run_on_service_thread(
                lambda: seen.append(threading.get_ident()), timeout=5
            )
            self.assertTrue(finished)
        self.assertEqual(len(set(seen)), 1)
        self.assertNotEqual(seen[0], threading.get_ident())

    def test_timeout_returns_false_and_next_call_runs_on_another_thread(self):
        """A call still running at ``timeout`` returns False and doesn't block the next call."""
        release = threading.Event()
        stuck_finished = threading.Event()
        seen = []

        def stuck():
            seen.append(threading.get_ident())
            release.wait(5)
            stuck_finished.set()

        try:
            self.assertFalse(service_thread.run_on_service_thread(stuck, timeout=0.05))
            self.assertTrue(
                service_thread.run_on_service_thread(
                    lambda: seen.append(threading.get_ident()), timeout=5
                )
            )
        finally:
            release.set()
            stuck_finished.wait(5)
        self.assertEqual(len(set(seen)), 2)

    def test_exception_is_logged_and_thread_is_reused(self):
        """An exception escaping ``fn`` is logged and the same thread serves the next call."""
        seen = []

        def boom():
            seen.append(threading.get_ident())
            raise ValueError("boom")

        with patch.object(service_thread, "logger") as mock_logger:
            self.assertTrue(service_thread.run_on_service_thread(boom, timeout=5))
            self.assertTrue(
                service_thread.run_on_service_thread(
                    lambda: seen.append(threading.get_ident()), timeout=5
                )
            )
        mock_logger.exception.assert_called_once()
        self.assertEqual(len(set(seen)), 1)


if __name__ == "__main__":
    unittest.main()
