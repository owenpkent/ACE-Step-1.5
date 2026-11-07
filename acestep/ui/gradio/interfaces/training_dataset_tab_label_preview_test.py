"""Regression tests for the dataset label/preview controls under real gradio."""

import unittest

import gradio as gr

from acestep.ui.gradio.interfaces.training_dataset_tab_label_preview import (
    build_dataset_label_and_preview_controls,
)


class BuildDatasetLabelAndPreviewControlsTests(unittest.TestCase):
    """Builds the controls with real gradio to catch constructor validation errors."""

    def test_sample_selector_has_valid_range(self):
        """Construction succeeds and the slider minimum is below its maximum."""
        with gr.Blocks():
            controls = build_dataset_label_and_preview_controls()

        self.assertIn("sample_selector", controls)
        selector = controls["sample_selector"]
        self.assertLess(selector.minimum, selector.maximum)

    def test_sample_selector_hidden_initially(self):
        """The selector starts hidden because no dataset samples are loaded."""
        with gr.Blocks():
            controls = build_dataset_label_and_preview_controls()

        self.assertFalse(controls["sample_selector"].visible)


if __name__ == "__main__":
    unittest.main()
