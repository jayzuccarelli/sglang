"""Unit tests for LoRAManager.validate_new_adapter — no server, no model loading."""

import unittest

from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.lora_registry import LoRARef
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_manager():
    manager = LoRAManager.__new__(LoRAManager)
    manager.enable_dp_attention = False
    manager.lora_refs = {}
    manager.num_pinned_loras = 0
    manager.max_loras_per_batch = 4
    return manager


def _make_config(**overrides):
    config_dict = {
        "peft_type": "LORA",
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["q_proj", "k_proj", "v_proj"],
    }
    config_dict.update(overrides)
    return LoRAConfig.from_dict(config_dict)


class TestValidateNewAdapter(CustomTestCase):
    def test_accepts_adapter_without_bias(self):
        manager = _make_manager()
        for config in (_make_config(), _make_config(bias="none")):
            manager.validate_new_adapter(config, LoRARef(lora_name="a", lora_path="a"))

    def test_rejects_adapter_with_trained_bias(self):
        # PEFT bias="lora_only"/"all" ship trained base-layer biases, which
        # normalize_qkv_proj would otherwise stack into the lora_B buffer.
        manager = _make_manager()
        for bias in ("lora_only", "all"):
            with self.subTest(bias=bias):
                with self.assertRaisesRegex(ValueError, "bias"):
                    manager.validate_new_adapter(
                        _make_config(bias=bias),
                        LoRARef(lora_name="a", lora_path="a"),
                    )


if __name__ == "__main__":
    unittest.main()
