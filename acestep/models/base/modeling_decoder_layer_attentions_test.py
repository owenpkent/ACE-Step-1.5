"""Regression test for AceStepDiTLayer with cross-attention disabled.

With ``use_cross_attention=False`` and ``output_attentions=True`` the layer
used to raise UnboundLocalError because ``cross_attn_weights`` was never set.
"""

import unittest

try:
    import torch

    from acestep.models.base.configuration_acestep_v15 import AceStepConfig
    from acestep.models.base.modeling_acestep_v15_base import AceStepDiTLayer
except ImportError:  # pragma: no cover - environment without torch/transformers
    torch = None


@unittest.skipIf(torch is None, "torch/transformers unavailable")
class TestDiTLayerOutputAttentions(unittest.TestCase):
    """Tests for the DiT layer forward with attention outputs."""

    def _run(self, use_cross_attention: bool):
        """Run a tiny CPU layer forward with output_attentions=True."""
        config = AceStepConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            layer_types=["full_attention"],
        )
        config._attn_implementation = "eager"
        layer = AceStepDiTLayer(config, 0, use_cross_attention=use_cross_attention)
        seq = 4
        pos = (torch.ones(1, seq, 16), torch.zeros(1, seq, 16))
        kwargs = {}
        if use_cross_attention:
            kwargs["encoder_hidden_states"] = torch.randn(1, 3, 32)
        return layer(
            torch.randn(1, seq, 32),
            position_embeddings=pos,
            temb=torch.randn(1, 6, 32),
            output_attentions=True,
            **kwargs,
        )

    def test_no_cross_attention_returns_none_weights(self):
        """Disabled cross-attention yields None cross weights, not an error."""
        outputs = self._run(use_cross_attention=False)
        self.assertEqual(len(outputs), 3)
        self.assertIsNone(outputs[2])

    def test_cross_attention_enabled_returns_weights(self):
        """Enabled cross-attention still returns cross-attention weights."""
        outputs = self._run(use_cross_attention=True)
        self.assertIsNotNone(outputs[2])


if __name__ == "__main__":
    unittest.main()
