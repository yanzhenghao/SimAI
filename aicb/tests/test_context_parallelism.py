"""
Tests for Context Parallelism (CP) communication support.

Covers:
  - CommGroup.cp_group, cp_dp_group, cp_tp_group enums
  - rank_mapper CP group mappings
  - CP all_to_all KV exchange in LlamaAttention and MegatronAttention
  - CP interaction with GQA (KV size scaling)
  - Backward compatibility: cp=1 produces no CP comms
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from workload_generator.mocked_model.MockedModel import MockedParam
from workload_generator.mocked_model.training.MockedLlama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaModel,
)
from workload_generator.mocked_model.training.MockedMegatron import (
    MegatronAttention,
    MegatronModel,
    MegatronTransformorLayer,
)
from workload_generator.mocked_model.training.MockedDeepSeek import DeepSeekV3Model
from utils.utils import CommGroup, CommType
from utils.rank_mapper import (
    _COMM_GROUP_TOKEN_MAP,
    get_rank_list_for_comm_group,
    build_rank_mapping_table,
)


# ---------------------------------------------------------------------------
# Mock config for LlamaModel and MegatronModel
# ---------------------------------------------------------------------------
class MockConfig:
    """Minimal config that supplies all attributes expected by LlamaModel."""

    def __init__(self, **kwargs):
        defaults = dict(
            padded_vocab_size=32000,
            hidden_size=4096,
            ffn_hidden_size=11008,
            num_layers=2,
            num_attention_heads=32,
            num_kv_heads=32,
            seq_length=4096,
            micro_batch=1,
            tensor_model_parallel_size=1,
            enable_sequence_parallel=True,
            computation_enable=False,
            add_bias_linear=False,
            max_position_embeddings=4096,
            rope_theta=10000.0,
            context_parallel_size=1,
        )
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


class MockMegatronConfig:
    """Minimal config for MegatronModel."""

    def __init__(self, **kwargs):
        defaults = dict(
            padded_vocab_size=32000,
            hidden_size=4096,
            ffn_hidden_size=11008,
            num_layers=2,
            num_attention_heads=32,
            seq_length=4096,
            micro_batch=1,
            tensor_model_parallel_size=1,
            pipeline_model_parallel=1,
            enable_sequence_parallel=True,
            computation_enable=False,
            add_bias_linear=False,
            context_parallel_size=1,
            expert_model_parallel_size=1,
            moe_router_topk=2,
            num_experts=1,
            moe_grouped_gemm=True,
            moe_enable=False,
        )
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# CommGroup CP enums
# ---------------------------------------------------------------------------
class TestCommGroupCP:
    def test_cp_group_exists(self):
        assert hasattr(CommGroup, "cp_group")
        assert CommGroup.cp_group.value == "cp_group"

    def test_cp_dp_group_exists(self):
        assert hasattr(CommGroup, "cp_dp_group")
        assert CommGroup.cp_dp_group.value == "cp_dp_group"

    def test_cp_tp_group_exists(self):
        assert hasattr(CommGroup, "cp_tp_group")
        assert CommGroup.cp_tp_group.value == "cp_tp_group"


# ---------------------------------------------------------------------------
# rank_mapper CP mappings
# ---------------------------------------------------------------------------
class TestRankMapperCP:
    def test_cp_group_in_token_map(self):
        assert CommGroup.cp_group in _COMM_GROUP_TOKEN_MAP

    def test_cp_dp_group_in_token_map(self):
        assert CommGroup.cp_dp_group in _COMM_GROUP_TOKEN_MAP

    def test_cp_tp_group_in_token_map(self):
        assert CommGroup.cp_tp_group in _COMM_GROUP_TOKEN_MAP

    def test_cp_group_token_is_cp(self):
        token, independent_ep = _COMM_GROUP_TOKEN_MAP[CommGroup.cp_group]
        assert token == "cp"
        assert independent_ep is False

    def test_cp_dp_token_is_cp_dp(self):
        token, _ = _COMM_GROUP_TOKEN_MAP[CommGroup.cp_dp_group]
        assert token == "cp-dp"

    def test_cp_tp_token_is_cp_tp(self):
        token, _ = _COMM_GROUP_TOKEN_MAP[CommGroup.cp_tp_group]
        assert token == "cp-tp"

    def test_build_rank_mapping_includes_cp_groups(self):
        """build_rank_mapping_table should include cp groups."""
        from utils.utils import RankGenerator
        rg = RankGenerator(tp=2, ep=1, dp=2, pp=1, cp=2, order="tp-cp-dp-pp")
        rows = build_rank_mapping_table(rg)
        group_names = [r["group"] for r in rows]
        assert "cp_group" in group_names
        assert "cp_dp_group" in group_names
        assert "cp_tp_group" in group_names

    def test_cp_group_ranks(self):
        """CP group ranks should be contiguous blocks of cp_size."""
        from utils.utils import RankGenerator
        rg = RankGenerator(tp=1, ep=1, dp=1, pp=1, cp=4, order="tp-cp-dp-pp")
        ranks = get_rank_list_for_comm_group(rg, CommGroup.cp_group)
        assert len(ranks) == 4
        assert sorted(ranks) == [0, 1, 2, 3]

    def test_cp_with_tp_rank_ordering(self):
        """With tp=2, cp=2, order=tp-cp-dp-pp, CP ranks should be interleaved."""
        from utils.utils import RankGenerator
        rg = RankGenerator(tp=2, ep=1, dp=1, pp=1, cp=2, order="tp-cp-dp-pp")
        ranks = get_rank_list_for_comm_group(rg, CommGroup.cp_group)
        assert len(ranks) == 2
        # With tp=2, cp=2: tp_rank=0 has cp ranks [0, 2]; tp_rank=1 has [1, 3]
        # The ref_rank=0 lookup returns one of the cp groups containing rank 0
        assert 0 in ranks

    def test_cp_dp_group_ranks(self):
        from utils.utils import RankGenerator
        rg = RankGenerator(tp=1, ep=1, dp=4, pp=1, cp=2, order="tp-cp-dp-pp")
        ranks = get_rank_list_for_comm_group(rg, CommGroup.cp_dp_group)
        assert len(ranks) == 8  # cp * dp = 8 (but dp=4, cp=2, world_size=8)
        # Actually: tp=1, cp=2, dp=4, pp=1 -> world=8. cp-dp group = all ranks.


# ---------------------------------------------------------------------------
# LlamaAttention CP communication
# ---------------------------------------------------------------------------
class TestLlamaAttentionCP:
    def test_cp1_no_cp_comms(self):
        """With cp=1, no CP all_to_all should be generated."""
        attn = LlamaAttention(32, 32, 4096, tp=1, cp=1, seq_len=4096,
                              batch_size=1, layer_id=0)
        fwd = attn.forward()
        comm_types = [item.comm_type for item in fwd.workload
                      if hasattr(item, 'comm_type')]
        assert CommType.all_to_all not in comm_types

    def test_cp4_generates_all_to_all(self):
        """With cp=4, CP all_to_all KV exchange should be generated."""
        attn = LlamaAttention(32, 32, 4096, tp=1, cp=4, seq_len=4096,
                              batch_size=1, layer_id=0)
        fwd = attn.forward()
        cp_comms = [item for item in fwd.workload
                    if getattr(item, 'comm_type', None) == CommType.all_to_all
                    and getattr(item, 'comm_group', None) == CommGroup.cp_group]
        assert len(cp_comms) == 1

    def test_cp_all_to_all_has_correct_group_size(self):
        attn = LlamaAttention(32, 32, 4096, tp=1, cp=4, seq_len=4096,
                              batch_size=1, layer_id=0)
        fwd = attn.forward()
        cp_items = [item for item in fwd.workload
                    if getattr(item, 'comm_group', None) == CommGroup.cp_group]
        assert len(cp_items) > 0
        for item in cp_items:
            assert item.comm_group_size == 4

    def test_cp_comm_size_matches_kv_data(self):
        """CP comm size = 2 * num_kv_heads * head_dim * seq_len * batch."""
        attn = LlamaAttention(32, 8, 4096, tp=1, cp=4, seq_len=2048,
                              batch_size=2, layer_id=0)
        fwd = attn.forward()
        cp_items = [item for item in fwd.workload
                    if getattr(item, 'comm_group', None) == CommGroup.cp_group]
        assert len(cp_items) > 0
        expected_size = 2 * 8 * 128 * 2048 * 2  # K+V: 2 * kv_heads * head_dim * seq * batch
        assert cp_items[0].msg_size == expected_size

    def test_cp_with_gqa_reduces_comm_size(self):
        """GQA (num_kv_heads < num_heads) reduces CP K/V comm."""
        # MHA: 32 KV heads
        attn_mha = LlamaAttention(32, 32, 4096, tp=1, cp=4, seq_len=2048,
                                  batch_size=1, layer_id=0)
        # GQA: 8 KV heads
        attn_gqa = LlamaAttention(32, 8, 4096, tp=1, cp=4, seq_len=2048,
                                  batch_size=1, layer_id=0)

        mha_cp = [item for item in attn_mha.forward().workload
                  if getattr(item, 'comm_group', None) == CommGroup.cp_group]
        gqa_cp = [item for item in attn_gqa.forward().workload
                   if getattr(item, 'comm_group', None) == CommGroup.cp_group]

        assert len(mha_cp) > 0 and len(gqa_cp) > 0
        # GQA CP comm should be num_kv_heads/num_heads = 1/4 of MHA
        assert mha_cp[0].msg_size == gqa_cp[0].msg_size * 4

    def test_cp_backward_generates_all_to_all(self):
        """Backward pass also generates CP all_to_all for gradient exchange."""
        attn = LlamaAttention(32, 32, 4096, tp=1, cp=2, seq_len=4096,
                              batch_size=1, layer_id=0)
        bwd = attn.backward()
        cp_items = [item for item in bwd.workload
                    if getattr(item, 'comm_group', None) == CommGroup.cp_group]
        assert len(cp_items) == 1

    def test_cp_forward_stage_name(self):
        """CP exchange should have a descriptive stage name."""
        attn = LlamaAttention(32, 32, 4096, tp=1, cp=2, seq_len=4096,
                              batch_size=1, layer_id=0)
        fwd = attn.forward()
        cp_items = [item for item in fwd.workload
                    if getattr(item, 'comm_group', None) == CommGroup.cp_group]
        assert len(cp_items) > 0
        assert "cp_kv_exchange" in cp_items[0].stage.lower()
        assert "forward" in cp_items[0].stage.lower()

    def test_cp_backward_stage_name(self):
        attn = LlamaAttention(32, 32, 4096, tp=1, cp=2, seq_len=4096,
                              batch_size=1, layer_id=0)
        bwd = attn.backward()
        cp_items = [item for item in bwd.workload
                    if getattr(item, 'comm_group', None) == CommGroup.cp_group]
        assert len(cp_items) > 0
        assert "cp_kv_exchange" in cp_items[0].stage.lower()
        assert "backward" in cp_items[0].stage.lower()


# ---------------------------------------------------------------------------
# MegatronAttention CP communication
# ---------------------------------------------------------------------------
class TestMegatronAttentionCP:
    def test_cp1_no_cp_comms_megatron(self):
        attn = MegatronAttention(32, 4096, tp=1, cp=1, seq_len=4096,
                                 batch_size=1, layer_id=0,
                                 sequence_parallel_enabled=True,
                                 computation_enable=False,
                                 add_bias_linear=False)
        fwd = attn.forward()
        comm_types = [item.comm_type for item in fwd.workload
                      if hasattr(item, 'comm_type')]
        assert CommType.all_to_all not in comm_types

    def test_cp4_generates_all_to_all_megatron(self):
        attn = MegatronAttention(32, 4096, tp=1, cp=4, seq_len=4096,
                                 batch_size=1, layer_id=0,
                                 sequence_parallel_enabled=True,
                                 computation_enable=False,
                                 add_bias_linear=False)
        fwd = attn.forward()
        cp_comms = [item for item in fwd.workload
                    if getattr(item, 'comm_type', None) == CommType.all_to_all
                    and getattr(item, 'comm_group', None) == CommGroup.cp_group]
        assert len(cp_comms) == 1


# ---------------------------------------------------------------------------
# Full model integration: LlamaModel with CP
# ---------------------------------------------------------------------------
class TestLlamaModelCP:
    def test_cp1_model_no_cp_comms(self):
        config = MockConfig(context_parallel_size=1)
        model = LlamaModel(config)
        fwd = model.forward()
        cp_items = [item for item in fwd.workload
                    if getattr(item, 'comm_group', None) == CommGroup.cp_group]
        assert len(cp_items) == 0

    def test_cp2_model_has_cp_comms(self):
        config = MockConfig(context_parallel_size=2, seq_length=4096)
        model = LlamaModel(config)
        fwd = model.forward()
        cp_items = [item for item in fwd.workload
                    if getattr(item, 'comm_group', None) == CommGroup.cp_group]
        # One CP exchange per layer x num_layers (2) = 2
        assert len(cp_items) == config.num_layers

    def test_cp4_model_has_cp_comms(self):
        config = MockConfig(context_parallel_size=4, seq_length=4096, num_layers=3)
        model = LlamaModel(config)
        fwd = model.forward()
        cp_items = [item for item in fwd.workload
                    if getattr(item, 'comm_group', None) == CommGroup.cp_group]
        assert len(cp_items) == 3  # one per layer


# ---------------------------------------------------------------------------
# MegatronModel with CP
# ---------------------------------------------------------------------------
class TestMegatronModelCP:
    def test_cp1_megatron_no_cp_comms(self):
        config = MockMegatronConfig(context_parallel_size=1)
        model = MegatronModel(config)
        fwd = model.forward()
        cp_items = [item for item in fwd.workload
                    if getattr(item, 'comm_group', None) == CommGroup.cp_group]
        assert len(cp_items) == 0

    def test_cp2_megatron_has_cp_comms(self):
        config = MockMegatronConfig(context_parallel_size=2, seq_length=4096)
        model = MegatronModel(config)
        fwd = model.forward()
        cp_items = [item for item in fwd.workload
                    if getattr(item, 'comm_group', None) == CommGroup.cp_group]
        assert len(cp_items) == config.num_layers


# ---------------------------------------------------------------------------
# Backward compatibility: existing models with default cp=1
# ---------------------------------------------------------------------------
class TestBackwardCompatibilityCP:
    def test_deepseek_model_still_works(self):
        """DeepSeekV3Model should still work with default cp=1 (no CP comms)."""
        config = MockMegatronConfig(
            moe_enable=True,
            num_experts=64,
            expert_model_parallel_size=1,
            context_parallel_size=1,
        )
        config.n_shared_expert = 2  # DeepSeekV3Model requires this attribute
        config.n_dense_layers = 1
        config.qk_rope_dim = 64
        config.qk_nope_dim = 128
        config.q_lora_rank = 1536
        config.kv_lora_rank = 512
        config.v_head_dim = 128
        # Should not raise any error
        model = DeepSeekV3Model(config)
        assert model is not None
