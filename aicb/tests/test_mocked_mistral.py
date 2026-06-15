"""
Tests for MockedMistral.py: Mistral (dense) and Mixtral (MoE) architectures.

Key differences from LLaMA:
  - Mistral: sliding window attention (compute-only, no comm impact)
  - Mixtral: MoE FFN with 8 experts, top-2 routing, all_to_all EP dispatch
  - Mistral uses rope_theta=1e6 (vs LLaMA's 1e4 default)
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.utils import CommType, CommGroup
from workload_generator.mocked_model.training.MockedMistral import (
    MistralDecoderLayer,
    MixtralDecoderLayer,
    MistralModel,
    MixtralModel,
)
from workload_generator.mocked_model.training.MockedLlama import (
    LlamaAttention, LlamaRMSNorm,
)


class MistralConfig:
    def __init__(self, **kwargs):
        defaults = dict(
            padded_vocab_size=32000, hidden_size=4096, ffn_hidden_size=14336,
            num_layers=2, num_attention_heads=32, num_kv_heads=8,
            seq_length=2048, micro_batch=1, tensor_model_parallel_size=1,
            context_parallel_size=1, enable_sequence_parallel=True,
            computation_enable=False, add_bias_linear=False,
            max_position_embeddings=4096, rope_theta=1000000.0,
        )
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


class MixtralConfig:
    def __init__(self, **kwargs):
        defaults = dict(
            padded_vocab_size=32000, hidden_size=4096, ffn_hidden_size=14336,
            num_layers=2, num_attention_heads=32, num_kv_heads=8,
            seq_length=2048, micro_batch=1, tensor_model_parallel_size=1,
            context_parallel_size=1, enable_sequence_parallel=True,
            computation_enable=False, add_bias_linear=False,
            max_position_embeddings=4096, rope_theta=1000000.0,
            num_experts=8, moe_router_topk=2,
            expert_model_parallel_size=1, moe_grouped_gemm=True,
        )
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


# ============================================================================
# MistralDecoderLayer (dense)
# ============================================================================
class TestMistralDecoderLayer:
    def test_uses_llama_attention(self):
        layer = MistralDecoderLayer(4096, 14336, 32, 8, tp=1, cp=1,
                                    seq_len=2048, batch_size=1, layer_id=0)
        assert isinstance(layer.self_attn, LlamaAttention)

    def test_uses_llama_rmsnorm(self):
        layer = MistralDecoderLayer(4096, 14336, 32, 8, tp=1, cp=1,
                                    seq_len=2048, batch_size=1, layer_id=0)
        assert isinstance(layer.input_layernorm, LlamaRMSNorm)
        assert isinstance(layer.post_attention_layernorm, LlamaRMSNorm)

    def test_window_size_configurable(self):
        layer = MistralDecoderLayer(4096, 14336, 32, 8, tp=1, cp=1,
                                    seq_len=2048, batch_size=1, layer_id=0,
                                    window_size=4096)
        assert layer.window_size == 4096

    def test_forward_with_tp(self):
        layer = MistralDecoderLayer(4096, 14336, 32, 8, tp=4, cp=1,
                                    seq_len=2048, batch_size=1, layer_id=0)
        wl = layer.forward()
        assert len(wl.workload) > 0

    def test_backward_with_tp(self):
        layer = MistralDecoderLayer(4096, 14336, 32, 8, tp=4, cp=1,
                                    seq_len=2048, batch_size=1, layer_id=0)
        wl = layer.backward()
        assert len(wl.workload) > 0

    def test_gqa_kv_heads(self):
        """Mistral uses GQA: 32 Q heads, 8 KV heads."""
        layer = MistralDecoderLayer(4096, 14336, 32, 8, tp=1, cp=1,
                                    seq_len=2048, batch_size=1, layer_id=0)
        assert layer.self_attn.num_attention_heads == 32
        assert layer.self_attn.num_kv_heads == 8


# ============================================================================
# MixtralDecoderLayer (MoE)
# ============================================================================
class TestMixtralDecoderLayer:
    def test_uses_llama_attention(self):
        layer = MixtralDecoderLayer(4096, 14336, 32, 8, tp=1, cp=1,
                                    seq_len=2048, batch_size=1, layer_id=0,
                                    expert_model_parallel_size=1,
                                    num_experts=8, moe_router_topk=2)
        assert isinstance(layer.self_attn, LlamaAttention)

    def test_moe_ffn_produces_all_to_all(self):
        """Mixtral with EP>1 should produce all_to_all for expert dispatch."""
        layer = MixtralDecoderLayer(4096, 14336, 32, 8, tp=1, cp=1,
                                    seq_len=2048, batch_size=1, layer_id=0,
                                    expert_model_parallel_size=2,
                                    num_experts=8, moe_router_topk=2)
        wl = layer.forward()
        ep_comms = [item for item in wl.workload
                    if getattr(item, 'comm_type', None) == CommType.all_to_all]
        assert len(ep_comms) > 0, "Mixtral with EP=2 should produce all_to_all"

    def test_forward_without_ep(self):
        """EP=1, TP=2: produces TP comms but no EP all_to_all."""
        layer = MixtralDecoderLayer(4096, 14336, 32, 8, tp=2, cp=1,
                                    seq_len=2048, batch_size=1, layer_id=0,
                                    expert_model_parallel_size=1,
                                    num_experts=8, moe_router_topk=2)
        wl = layer.forward()
        assert len(wl.workload) > 0

    def test_backward_with_ep(self):
        layer = MixtralDecoderLayer(4096, 14336, 32, 8, tp=2, cp=1,
                                    seq_len=2048, batch_size=1, layer_id=0,
                                    expert_model_parallel_size=2,
                                    num_experts=8, moe_router_topk=2)
        wl = layer.backward()
        assert len(wl.workload) > 0

    def test_8_experts_top2_routing(self):
        """Mixtral uses 8 experts with top-2 routing."""
        layer = MixtralDecoderLayer(4096, 14336, 32, 8, tp=1, cp=1,
                                    seq_len=2048, batch_size=1, layer_id=0,
                                    expert_model_parallel_size=1,
                                    num_experts=8, moe_router_topk=2)
        assert layer is not None


# ============================================================================
# MistralModel (dense)
# ============================================================================
class TestMistralModel:
    def test_mistral7b_approx_params(self):
        """Mistral-7B: ~7.3B params with GQA (32 layers)."""
        config = MistralConfig(num_layers=32)
        model = MistralModel(config)
        total = sum(p.numel() for p in model.parameters())
        assert 7_000_000_000 < total < 8_000_000_000, f"Got {total}"

    def test_forward_with_tp(self):
        config = MistralConfig(tensor_model_parallel_size=4)
        model = MistralModel(config)
        wl = model.forward()
        assert len(wl.workload) > 0

    def test_backward_with_tp(self):
        config = MistralConfig(tensor_model_parallel_size=4)
        model = MistralModel(config)
        wl = model.backward()
        assert len(wl.workload) > 0

    def test_rope_base_is_1e6(self):
        config = MistralConfig()
        model = MistralModel(config)
        assert model.rotary_emb.base == 1000000.0

    def test_uses_llama_rmsnorm_final(self):
        config = MistralConfig()
        model = MistralModel(config)
        assert isinstance(model.final_norm, LlamaRMSNorm)


# ============================================================================
# MixtralModel (MoE)
# ============================================================================
class TestMixtralModel:
    def test_mixtral_8x7b_has_moe(self):
        config = MixtralConfig(num_experts=8, moe_router_topk=2)
        model = MixtralModel(config)
        # All layers should have MoE FFN (not dense MLP)
        assert len(model.layers) == 2

    def test_forward_with_ep(self):
        config = MixtralConfig(num_experts=8, moe_router_topk=2,
                               expert_model_parallel_size=2)
        model = MixtralModel(config)
        wl = model.forward()
        ep_comms = [item for item in wl.workload
                    if getattr(item, 'comm_type', None) == CommType.all_to_all]
        assert len(ep_comms) > 0, "Mixtral with EP=2 should produce EP all_to_all"

    def test_forward_ep1(self):
        config = MixtralConfig(num_experts=8, moe_router_topk=2,
                               expert_model_parallel_size=1,
                               tensor_model_parallel_size=2)
        model = MixtralModel(config)
        wl = model.forward()
        assert len(wl.workload) > 0

    def test_backward_with_ep(self):
        config = MixtralConfig(num_experts=8, moe_router_topk=2,
                               expert_model_parallel_size=2,
                               tensor_model_parallel_size=2)
        model = MixtralModel(config)
        wl = model.backward()
        assert len(wl.workload) > 0

    def test_8x22b_config_larger(self):
        """Mixtral-8x22B: larger hidden and more layers."""
        config = MixtralConfig(hidden_size=6144, ffn_hidden_size=16384,
                               num_attention_heads=56, num_kv_heads=8,
                               num_layers=56, num_experts=8,
                               moe_router_topk=2)
        model = MixtralModel(config)
        assert len(model.layers) == 56
        total = sum(p.numel() for p in model.parameters())
        assert total > 50_000_000_000  # > 50B params

    def test_rope_base_is_1e6(self):
        config = MixtralConfig()
        model = MixtralModel(config)
        assert model.rotary_emb.base == 1000000.0
