"""
Tests for MockedFalconH1.py: SSM TP sharding, parallel hybrid block, layer types.

Falcon-H1 (arXiv:2507.22448) is a parallel hybrid architecture combining:
  - Mamba-2 SSM (state-space model) with tensor parallelism
  - Group Query Attention (GQA) with RoPE
  - Configurable layer allocation (parallel_hybrid, pure_mamba, pure_attention)

Key differences from standard decoder-only:
  - SSM has replicate conv1d (not TP-sharded)
  - in_proj and out_proj are TP-sharded (column/row style)
  - Parallel hybrid block runs attention and SSM concurrently
  - Configurable channel ratio (default SA_M layout: 2:1 SSM:Attention)
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from workload_generator.mocked_model.training.MockedFalconH1 import (
    Mamba2SSM,
    FalconH1MLP,
    FalconH1Attention,
    ParallelHybridBlock,
    FalconH1Layer,
    FalconH1Model,
)
from workload_generator.mocked_model.training.MockedLlama import LlamaRMSNorm


class FalconConfig:
    def __init__(self, **kwargs):
        defaults = dict(
            padded_vocab_size=32000, hidden_size=2048, ffn_hidden_size=5632,
            num_layers=2, num_attention_heads=16, num_kv_heads=4,
            ssm_state_dim=128, ssm_head_dim=64, n_ssm_heads=16,
            seq_length=2048, micro_batch=1, tensor_model_parallel_size=1,
            enable_sequence_parallel=True, computation_enable=False,
            add_bias_linear=False, layer_allocation=None,
        )
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


# ============================================================================
# Mamba2SSM
# ============================================================================
class TestMamba2SSM:
    def test_has_required_projections(self):
        ssm = Mamba2SSM(hidden_size=2048, ssm_state_dim=128, ssm_head_dim=64,
                        n_ssm_heads=16, tp=1, seq_len=2048, batch_size=1, layer_id=0)
        assert hasattr(ssm, "in_proj")
        assert hasattr(ssm, "out_proj")
        # SSM A_log, D, dt_bias parameters exist
        assert hasattr(ssm, "A_log")
        assert hasattr(ssm, "D")
        assert hasattr(ssm, "dt_bias")

    def test_parameters_exist(self):
        ssm = Mamba2SSM(hidden_size=2048, ssm_state_dim=128, ssm_head_dim=64,
                        n_ssm_heads=16, tp=1, seq_len=2048, batch_size=1, layer_id=0)
        total = sum(p.numel() for p in ssm.parameters())
        assert total > 0

    def test_conv1d_replicated_in_tp(self):
        """conv1d is replicated across TP ranks (not sharded)."""
        ssm_tp1 = Mamba2SSM(hidden_size=2048, ssm_state_dim=128, ssm_head_dim=64,
                            n_ssm_heads=16, tp=1, seq_len=2048, batch_size=1, layer_id=0)
        ssm_tp4 = Mamba2SSM(hidden_size=2048, ssm_state_dim=128, ssm_head_dim=64,
                            n_ssm_heads=16, tp=4, seq_len=2048, batch_size=1, layer_id=0)
        conv1_params = sum(p.numel() for p in ssm_tp1.parameters()
                          if "conv" in (p.name or "").lower())
        conv4_params = sum(p.numel() for p in ssm_tp4.parameters()
                          if "conv" in (p.name or "").lower())
        # conv1d params should be identical regardless of TP
        assert conv1_params == conv4_params

    def test_forward_produces_workload(self):
        ssm = Mamba2SSM(hidden_size=2048, ssm_state_dim=128, ssm_head_dim=64,
                        n_ssm_heads=16, tp=2, seq_len=2048, batch_size=1, layer_id=0)
        wl = ssm.forward()
        assert len(wl.workload) > 0

    def test_backward_produces_workload(self):
        ssm = Mamba2SSM(hidden_size=2048, ssm_state_dim=128, ssm_head_dim=64,
                        n_ssm_heads=16, tp=2, seq_len=2048, batch_size=1, layer_id=0)
        wl = ssm.backward()
        assert len(wl.workload) > 0


# ============================================================================
# FalconH1Attention (GQA)
# ============================================================================
class TestFalconH1Attention:
    def test_gqa_separate_projections(self):
        attn = FalconH1Attention(num_attention_heads=16, num_kv_heads=4,
                                 hidden_size=2048, tp=1, seq_len=2048,
                                 batch_size=1, layer_id=0)
        assert hasattr(attn, "q_proj")
        assert hasattr(attn, "k_proj")
        assert hasattr(attn, "v_proj")
        assert hasattr(attn, "o_proj")

    def test_kv_smaller_than_q(self):
        """GQA: K/V params should be fewer than Q params."""
        attn = FalconH1Attention(num_attention_heads=16, num_kv_heads=4,
                                 hidden_size=2048, tp=1, seq_len=2048,
                                 batch_size=1, layer_id=0)
        q_params = sum(p.numel() for p in attn.q_proj.parameters())
        k_params = sum(p.numel() for p in attn.k_proj.parameters())
        assert k_params < q_params  # 4x fewer KV heads

    def test_forward_produces_workload(self):
        attn = FalconH1Attention(num_attention_heads=16, num_kv_heads=4,
                                 hidden_size=2048, tp=2, seq_len=2048,
                                 batch_size=1, layer_id=0)
        wl = attn.forward()
        assert len(wl.workload) > 0


# ============================================================================
# FalconH1MLP (SwiGLU)
# ============================================================================
class TestFalconH1MLP:
    def test_three_projections(self):
        mlp = FalconH1MLP(hidden_size=2048, intermediate_size=5632, tp=1,
                          seq_len=2048, batch_size=1, layer_id=0)
        assert hasattr(mlp, "gate_proj")
        assert hasattr(mlp, "up_proj")
        assert hasattr(mlp, "down_proj")

    def test_forward_produces_workload(self):
        mlp = FalconH1MLP(hidden_size=2048, intermediate_size=5632, tp=2,
                          seq_len=2048, batch_size=1, layer_id=0)
        wl = mlp.forward()
        assert len(wl.workload) > 0


# ============================================================================
# ParallelHybridBlock (SSM + Attention in parallel)
# ============================================================================
class TestParallelHybridBlock:
    def test_has_both_ssm_and_attention(self):
        block = ParallelHybridBlock(
            hidden_size=2048, ssm_state_dim=128, ssm_head_dim=64,
            n_ssm_heads=16, num_attention_heads=16, num_kv_heads=4,
            tp=1, seq_len=2048, batch_size=1, layer_id=0,
        )
        assert isinstance(block.ssm, Mamba2SSM)
        assert isinstance(block.attention, FalconH1Attention)

    def test_forward_combines_both_paths(self):
        block = ParallelHybridBlock(
            hidden_size=2048, ssm_state_dim=128, ssm_head_dim=64,
            n_ssm_heads=16, num_attention_heads=16, num_kv_heads=4,
            tp=2, seq_len=2048, batch_size=1, layer_id=0,
        )
        wl = block.forward()
        assert len(wl.workload) > 0

    def test_backward_combines_both_paths_gradients(self):
        block = ParallelHybridBlock(
            hidden_size=2048, ssm_state_dim=128, ssm_head_dim=64,
            n_ssm_heads=16, num_attention_heads=16, num_kv_heads=4,
            tp=2, seq_len=2048, batch_size=1, layer_id=0,
        )
        wl = block.backward()
        assert len(wl.workload) > 0

    def test_channel_ratio_default(self):
        """Default 2:1 ratio SSM:Attention channels."""
        block = ParallelHybridBlock(
            hidden_size=2048, ssm_state_dim=128, ssm_head_dim=64,
            n_ssm_heads=16, num_attention_heads=16, num_kv_heads=4,
            tp=1, seq_len=2048, batch_size=1, layer_id=0,
        )
        # SSM gets 2/3 of hidden_size, attention gets 1/3
        assert block.ssm_channel_dim + block.attn_channel_dim == 2048
        assert block.ssm_channel_dim > block.attn_channel_dim


# ============================================================================
# FalconH1Layer (configurable layer types)
# ============================================================================
class TestFalconH1Layer:
    def test_parallel_hybrid_layer_has_hybrid_block(self):
        layer = FalconH1Layer(
            hidden_size=2048, intermediate_size=5632,
            ssm_state_dim=128, ssm_head_dim=64, n_ssm_heads=16,
            num_attention_heads=16, num_kv_heads=4,
            tp=1, seq_len=2048, batch_size=1, layer_id=0,
            layer_type="parallel_hybrid",
        )
        assert layer.has_hybrid is True

    def test_pure_mamba_layer_has_mamba_block(self):
        layer = FalconH1Layer(
            hidden_size=2048, intermediate_size=5632,
            ssm_state_dim=128, ssm_head_dim=64, n_ssm_heads=16,
            num_attention_heads=16, num_kv_heads=4,
            tp=1, seq_len=2048, batch_size=1, layer_id=1,
            layer_type="pure_mamba",
        )
        assert layer.has_ssm_only is True

    def test_pure_attention_layer_has_attn_block(self):
        layer = FalconH1Layer(
            hidden_size=2048, intermediate_size=5632,
            ssm_state_dim=128, ssm_head_dim=64, n_ssm_heads=16,
            num_attention_heads=16, num_kv_heads=4,
            tp=1, seq_len=2048, batch_size=1, layer_id=2,
            layer_type="pure_attention",
        )
        assert hasattr(layer, "attn_block")

    def test_forward_all_layer_types(self):
        for ltype in ["parallel_hybrid", "pure_mamba", "pure_attention"]:
            layer = FalconH1Layer(
                hidden_size=2048, intermediate_size=5632,
                ssm_state_dim=128, ssm_head_dim=64, n_ssm_heads=16,
                num_attention_heads=16, num_kv_heads=4,
                tp=2, seq_len=2048, batch_size=1, layer_id=0,
                layer_type=ltype,
            )
            wl = layer.forward()
            assert len(wl.workload) > 0, f"Empty workload for {ltype}"


# ============================================================================
# FalconH1Model
# ============================================================================
class TestFalconH1Model:
    def test_default_layer_allocation(self):
        config = FalconConfig(num_layers=8)
        model = FalconH1Model(config)
        assert len(model.layers) == 8
        assert model.layers[0].layer_type == "parallel_hybrid"
        assert model.layers[1].layer_type == "parallel_hybrid"
        assert model.layers[6].layer_type == "parallel_hybrid"
        assert model.layers[7].layer_type == "parallel_hybrid"

    def test_custom_layer_allocation(self):
        config = FalconConfig(
            num_layers=3,
            layer_allocation=["parallel_hybrid", "pure_mamba", "pure_attention"],
        )
        model = FalconH1Model(config)
        assert model.layers[0].layer_type == "parallel_hybrid"
        assert model.layers[1].layer_type == "pure_mamba"
        assert model.layers[2].layer_type == "pure_attention"

    def test_forward_produces_workload(self):
        config = FalconConfig(num_layers=2, tensor_model_parallel_size=2)
        model = FalconH1Model(config)
        wl = model.forward()
        assert len(wl.workload) > 0

    def test_backward_produces_workload(self):
        config = FalconConfig(num_layers=2, tensor_model_parallel_size=2)
        model = FalconH1Model(config)
        wl = model.backward()
        assert len(wl.workload) > 0

    def test_uses_llama_rmsnorm(self):
        config = FalconConfig()
        model = FalconH1Model(config)
        assert isinstance(model.final_norm, LlamaRMSNorm)

    def test_embedding_present(self):
        config = FalconConfig()
        model = FalconH1Model(config)
        assert hasattr(model, "embedding")

    def test_lm_head_present(self):
        config = FalconConfig()
        model = FalconH1Model(config)
        assert hasattr(model, "lm_head")
