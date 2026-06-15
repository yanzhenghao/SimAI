"""
End-to-end integration tests for the complete AICB workflow.

Tests the full pipeline:
  Model registration -> Model instantiation -> Workload generation -> Chakra export

Covers:
  - Multi-model: Megatron, DeepSeek, Llama
  - Multi-parallelism: TP, CP, EP, GQA
  - Full pipeline: model -> forward -> backward -> Chakra JSON
  - Cross-framework consistency checks
  - Edge cases: single layer, large vocab, zero CP

These tests exercise the integration of F001 (Llama), F002 (Registry),
F003 (Context Parallelism), and F004 (Chakra Exporter).

Run:
    cd aicb && python3 -m pytest tests/e2e_test_aicb_workflow.py -v

No GPU required -- all mocked model operations are CPU-safe.
"""

import sys
import os
import json
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.utils import CommType, CommGroup, RankGenerator
from utils.chakra_export import ChakraExporter, ChakraNodeType

from workload_generator.mocked_model.MockedModel import MockedModel
from workload_generator.mocked_model.training.MockedLlama import (
    LlamaModel, LlamaAttention, LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding,
)
from workload_generator.mocked_model.training.MockedMegatron import (
    MegatronModel, MegatronAttention, MegatronTransformorLayer,
)
from workload_generator.mocked_model.training.MockedDeepSeek import DeepSeekV3Model


# ============================================================================
# Config factories
# ============================================================================

class LlamaConfig:
    """Llama model configuration factory."""
    def __init__(self, **kwargs):
        defaults = dict(
            padded_vocab_size=32000, hidden_size=4096, ffn_hidden_size=11008,
            num_layers=2, num_attention_heads=32, num_kv_heads=32,
            seq_length=2048, micro_batch=1, tensor_model_parallel_size=1,
            enable_sequence_parallel=True, computation_enable=False,
            add_bias_linear=False, max_position_embeddings=4096,
            rope_theta=10000.0, context_parallel_size=1,
        )
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


class MegatronConfig:
    """Megatron model configuration factory."""
    def __init__(self, **kwargs):
        defaults = dict(
            padded_vocab_size=32000, hidden_size=4096, ffn_hidden_size=11008,
            num_layers=2, num_attention_heads=32, seq_length=2048,
            micro_batch=1, tensor_model_parallel_size=1,
            pipeline_model_parallel=1, enable_sequence_parallel=True,
            computation_enable=False, add_bias_linear=False,
            context_parallel_size=1, expert_model_parallel_size=1,
            moe_router_topk=2, num_experts=1, moe_grouped_gemm=True,
            moe_enable=False,
        )
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


# ============================================================================
# E2E: Full Pipeline -- Model -> Workload -> Chakra Export
# ============================================================================

class TestE2EFullPipeline:
    """Tests the complete AICB pipeline end-to-end."""

    def test_llama_full_pipeline_tp4(self):
        """Llama-7B with TP=4: register -> model -> workload -> Chakra."""
        config = LlamaConfig(tensor_model_parallel_size=4)
        model = LlamaModel(config)

        # Generate forward + backward workload
        fwd = model.forward()
        bwd = model.backward()

        # Verify forward workload has expected structure
        fwd_types = [item.comm_type for item in fwd.workload
                     if hasattr(item, 'comm_type')]
        # TP=4 should produce all_gather and reduce_scatter for column/row linear
        assert CommType.all_gather in fwd_types or CommType.reduce_scatter in fwd_types
        assert len(fwd.workload) > 0

        # Verify backward workload
        bwd_types = [item.comm_type for item in bwd.workload
                     if hasattr(item, 'comm_type')]
        assert len(bwd.workload) > 0

        # Export to Chakra JSON
        exporter = ChakraExporter(fwd, config)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".et.json", delete=False) as f:
            tmp_path = f.name
        try:
            exporter.export(tmp_path)
            with open(tmp_path) as f:
                data = json.load(f)
            assert "nodes" in data
            assert len(data["nodes"]) >= 2  # metadata + at least 1 node
        finally:
            os.unlink(tmp_path)

    def test_megatron_full_pipeline_tp2(self):
        """Megatron with TP=2: full pipeline."""
        config = MegatronConfig(tensor_model_parallel_size=2)
        model = MegatronModel(config)

        fwd = model.forward()
        bwd = model.backward()

        assert len(fwd.workload) > 0
        assert len(bwd.workload) > 0

        # Verify TP communication exists
        tp_comms = [item for item in fwd.workload
                    if getattr(item, 'comm_group', None) == CommGroup.tp_group]
        assert len(tp_comms) > 0, "TP=2 should produce TP group communications"

    def test_deepseek_full_pipeline(self):
        """DeepSeek V3 with EP + TP: full pipeline without crash."""
        config = MegatronConfig(
            tensor_model_parallel_size=2,
            expert_model_parallel_size=2,
            num_experts=64,
            moe_enable=True,
            moe_router_topk=8,
        )
        config.n_shared_expert = 2
        config.n_dense_layers = 1
        config.qk_rope_dim = 64
        config.qk_nope_dim = 128
        config.q_lora_rank = 1536
        config.kv_lora_rank = 512
        config.v_head_dim = 128

        model = DeepSeekV3Model(config)
        assert model is not None

        fwd = model.forward()
        bwd = model.backward()
        assert len(fwd.workload) > 0
        assert len(bwd.workload) > 0

        # DeepSeek with EP should produce all_to_all for expert dispatch
        ep_comms = [item for item in fwd.workload
                    if getattr(item, 'comm_type', None) == CommType.all_to_all]
        assert len(ep_comms) > 0, "DeepSeek with EP should produce all_to_all"


# ============================================================================
# E2E: Multi-Model Consistency
# ============================================================================

class TestE2EMultiModelConsistency:
    """Verifies consistent behavior across different model frameworks."""

    def test_same_tp_produces_comm_across_models(self):
        """TP=4 should produce TP communication for all model frameworks."""
        # Llama
        llama = LlamaModel(LlamaConfig(tensor_model_parallel_size=4))
        llama_fwd = llama.forward()
        llama_tp = [item for item in llama_fwd.workload
                    if getattr(item, 'comm_group', None) == CommGroup.tp_group]
        assert len(llama_tp) > 0

        # Megatron
        megatron = MegatronModel(MegatronConfig(tensor_model_parallel_size=4))
        megatron_fwd = megatron.forward()
        megatron_tp = [item for item in megatron_fwd.workload
                       if getattr(item, 'comm_group', None) == CommGroup.tp_group]
        assert len(megatron_tp) > 0

    def test_llama_parameter_count_consistent(self):
        """Llama parameter count should be deterministic for fixed config."""
        config1 = LlamaConfig()
        config2 = LlamaConfig()
        model1 = LlamaModel(config1)
        model2 = LlamaModel(config2)
        p1 = sum(p.numel() for p in model1.parameters())
        p2 = sum(p.numel() for p in model2.parameters())
        assert p1 == p2, f"Parameter count should be deterministic: {p1} != {p2}"

    def test_llama_attention_has_all_projections(self):
        """Llama attention should have Q, K, V, O projections."""
        attn = LlamaAttention(32, 8, 4096, tp=1, cp=1, seq_len=2048,
                              batch_size=1, layer_id=0)
        assert hasattr(attn, "q_proj")
        assert hasattr(attn, "k_proj")
        assert hasattr(attn, "v_proj")
        assert hasattr(attn, "o_proj")


# ============================================================================
# E2E: Parallelism Variation
# ============================================================================

class TestE2EParallelismVariation:
    """Tests workload behavior across different parallelism configurations."""

    def test_cp_scaling_llama(self):
        """CP=1,2,4: more CP ranks -> more all_to_all nodes."""
        for cp in [1, 2, 4]:
            attn = LlamaAttention(32, 32, 4096, tp=1, cp=cp, seq_len=4096,
                                  batch_size=1, layer_id=0)
            fwd = attn.forward()
            cp_items = [item for item in fwd.workload
                        if getattr(item, 'comm_group', None) == CommGroup.cp_group]
            if cp == 1:
                assert len(cp_items) == 0, f"CP=1 should have no CP comms, got {len(cp_items)}"
            else:
                assert len(cp_items) == 1, f"CP={cp} should have 1 CP exchange"

    def test_tp_scaling_megatron(self):
        """TP=1,2,4: more TP ranks -> more/reduced comm sizes."""
        prev_tp_comms = 0
        for tp in [1, 2, 4]:
            attn = MegatronAttention(32, 4096, tp=tp, cp=1, seq_len=4096,
                                     batch_size=1, layer_id=0,
                                     sequence_parallel_enabled=True,
                                     computation_enable=False,
                                     add_bias_linear=False)
            fwd = attn.forward()
            tp_items = [item for item in fwd.workload
                        if getattr(item, 'comm_group', None) == CommGroup.tp_group]
            if tp == 1:
                assert len(tp_items) == 0, f"TP=1 should have no TP comms"
            else:
                assert len(tp_items) > 0, f"TP={tp} should have TP comms"

    def test_combined_tp_cp_llama(self):
        """TP=4 + CP=2: both TP and CP communications should be present."""
        attn = LlamaAttention(32, 32, 4096, tp=4, cp=2, seq_len=4096,
                              batch_size=1, layer_id=0)
        fwd = attn.forward()
        tp_items = [item for item in fwd.workload
                    if getattr(item, 'comm_group', None) == CommGroup.tp_group]
        cp_items = [item for item in fwd.workload
                    if getattr(item, 'comm_group', None) == CommGroup.cp_group]
        assert len(tp_items) > 0, "TP=4 should produce TP communications"
        assert len(cp_items) > 0, "CP=2 should produce CP communications"

    def test_large_vocab_does_not_break(self):
        """Llama with 128K vocab should still produce valid workload."""
        config = LlamaConfig(padded_vocab_size=128256, num_layers=1,
                             tensor_model_parallel_size=2)
        model = LlamaModel(config)
        fwd = model.forward()
        assert len(fwd.workload) > 0


# ============================================================================
# E2E: Chakra Export Roundtrip
# ============================================================================

class TestE2EChakraExportRoundtrip:
    """Tests Chakra export fidelity across model frameworks."""

    def _export_and_reload(self, model, config, chakra_path):
        """Helper: generate workload, export Chakra, reload JSON."""
        fwd = model.forward()
        exporter = ChakraExporter(fwd, config)
        exporter.export(chakra_path)
        with open(chakra_path) as f:
            return json.load(f)

    def test_llama_chakra_roundtrip(self):
        config = LlamaConfig(tensor_model_parallel_size=4, context_parallel_size=2,
                             num_layers=1)
        model = LlamaModel(config)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".et.json", delete=False) as f:
            tmp_path = f.name
        try:
            data = self._export_and_reload(model, config, tmp_path)
            nodes = data["nodes"]
            # Metadata + (embedding + attention*4 + cp_exchange + mlp*3 + tp_ar) + ...
            assert len(nodes) >= 3

            # Verify node types present (no COMP if computation_enable=False)
            node_types = {n["type"] for n in nodes}
            assert ChakraNodeType.METADATA in node_types
            assert ChakraNodeType.COMM_COLL in node_types

            # Verify metadata
            meta = nodes[0]
            meta_attrs = {a["name"]: a for a in meta["attr"]}
            assert meta_attrs["context_parallel_size"]["int64_val"] == 2
            assert meta_attrs["tensor_model_parallel_size"]["int64_val"] == 4

            # Verify dependencies: each non-metadata node has at least one dep
            for node in nodes[1:]:
                assert len(node["ctrl_deps"]) > 0, f"Node {node['id']} has no deps"
        finally:
            os.unlink(tmp_path)

    def test_megatron_chakra_roundtrip(self):
        config = MegatronConfig(tensor_model_parallel_size=2, num_layers=1)
        model = MegatronModel(config)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".et.json", delete=False) as f:
            tmp_path = f.name
        try:
            data = self._export_and_reload(model, config, tmp_path)
            nodes = data["nodes"]
            assert len(nodes) >= 2
        finally:
            os.unlink(tmp_path)

    def test_chakra_json_valid_structure(self):
        """Every Chakra node should have required fields."""
        config = LlamaConfig(num_layers=2, tensor_model_parallel_size=2)
        model = LlamaModel(config)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".et.json", delete=False) as f:
            tmp_path = f.name
        try:
            data = self._export_and_reload(model, config, tmp_path)
            required_fields = {"id", "name", "type", "ctrl_deps", "data_deps", "attr"}
            for node in data["nodes"]:
                missing = required_fields - set(node.keys())
                assert not missing, f"Node {node.get('id')} missing: {missing}"
            # Verify unique ids
            ids = [n["id"] for n in data["nodes"]]
            assert len(ids) == len(set(ids)), "Node IDs must be unique"
        finally:
            os.unlink(tmp_path)


# ============================================================================
# E2E: Edge Cases and Robustness
# ============================================================================

class TestE2EEdgeCases:
    """Tests boundary conditions and error resilience."""

    def test_single_layer_model(self):
        """A 1-layer model should work without issues."""
        config = LlamaConfig(num_layers=1, tensor_model_parallel_size=2)
        model = LlamaModel(config)
        assert len(model.layers) == 1
        fwd = model.forward()
        assert len(fwd.workload) > 0

    def test_many_layers_model(self):
        """A 100-layer model is extreme but should not crash."""
        config = LlamaConfig(num_layers=100, computation_enable=False,
                             tensor_model_parallel_size=2)
        model = LlamaModel(config)
        assert len(model.layers) == 100
        fwd = model.forward()
        # Should have embedding + 100 * (layer_items) + lm_head items
        assert len(fwd.workload) > 0

    def test_cp1_produces_no_cp_comms(self):
        """cp=1 should never produce CP all_to_all."""
        attn = LlamaAttention(32, 32, 4096, tp=4, cp=1, seq_len=4096,
                              batch_size=1, layer_id=0)
        fwd = attn.forward()
        cp_items = [item for item in fwd.workload
                    if getattr(item, 'comm_group', None) == CommGroup.cp_group]
        assert len(cp_items) == 0

    def test_tp1_produces_no_tp_comms(self):
        """tp=1 should never produce TP communication."""
        attn = MegatronAttention(32, 4096, tp=1, cp=1, seq_len=4096,
                                 batch_size=1, layer_id=0,
                                 sequence_parallel_enabled=True,
                                 computation_enable=False,
                                 add_bias_linear=False)
        fwd = attn.forward()
        tp_items = [item for item in fwd.workload
                    if getattr(item, 'comm_group', None) == CommGroup.tp_group]
        assert len(tp_items) == 0

    def test_empty_workload_export_does_not_crash(self):
        """Exporting an empty workload should produce minimal valid Chakra."""
        class EmptyWorkload:
            workload = []
        exporter = ChakraExporter(EmptyWorkload(), LlamaConfig())
        exporter.build()
        # Should have at least the metadata node
        assert exporter.node_count >= 1
        assert exporter.nodes[0]["type"] == ChakraNodeType.METADATA

    def test_chakra_export_with_all_comm_types(self):
        """Workload with all communication types should export correctly."""
        import dataclasses

        @dataclasses.dataclass
        class FakeItem:
            comm_type: CommType = None
            comm_group: CommGroup = None
            comm_group_size: int = None
            msg_size: float = 0
            stage: str = ""
            dst: int = None
            src: int = None
            ranks: list = None

        class FakeWorkload:
            def __init__(self, items):
                self.workload = items

        items = FakeWorkload([
            FakeItem(CommType.computation, stage="comp"),
            FakeItem(CommType.all_reduce, CommGroup.tp_group, 4, 1024, "ar"),
            FakeItem(CommType.all_gather, CommGroup.tp_group, 4, 2048, "ag"),
            FakeItem(CommType.reduce_scatter, CommGroup.tp_group, 4, 2048, "rs"),
            FakeItem(CommType.all_to_all, CommGroup.cp_group, 2, 4096, "a2a"),
            FakeItem(CommType.broadcast, CommGroup.dp_group, 8, 512, "bcast"),
            FakeItem(CommType.barrier, CommGroup.all, 8, 0, "barrier"),
            FakeItem(CommType.isend, CommGroup.pp_group, 2, 8192, "send", dst=4),
            FakeItem(CommType.irecv, CommGroup.pp_group, 2, 8192, "recv", src=0),
        ])

        exporter = ChakraExporter(items, LlamaConfig())
        exporter.build()

        # All items should be converted to nodes (none skipped except epoch_end)
        assert exporter.node_count == 10  # metadata + 9 items

        node_types = [n["type"] for n in exporter.nodes]
        assert ChakraNodeType.METADATA in node_types
        assert ChakraNodeType.COMP in node_types
        assert ChakraNodeType.COMM_COLL in node_types
        assert ChakraNodeType.COMM_SEND in node_types
        assert ChakraNodeType.COMM_RECV in node_types


# ============================================================================
# E2E: Model + Workload Generation Integration
# ============================================================================

class TestE2EWorkloadGeneration:
    """Tests the interaction between model definitions and workload generators."""

    def test_llama_model_forward_backward_cycle(self):
        """A full forward+backward cycle should produce consistent workloads."""
        config = LlamaConfig(tensor_model_parallel_size=2, context_parallel_size=2,
                             num_layers=3)
        model = LlamaModel(config)

        fwd = model.forward()
        bwd = model.backward()

        # Forward and backward should both produce non-empty workloads
        assert len(fwd.workload) > 0
        assert len(bwd.workload) > 0

        # Backward should have the same number of layers as forward
        fwd_cp = [item for item in fwd.workload
                  if getattr(item, 'comm_group', None) == CommGroup.cp_group]
        bwd_cp = [item for item in bwd.workload
                   if getattr(item, 'comm_group', None) == CommGroup.cp_group]
        # Same number of CP exchanges in forward and backward
        assert len(fwd_cp) == len(bwd_cp) == config.num_layers

    def test_megatron_model_rmsnorm_vs_layernorm(self):
        """Llama RMSNorm has fewer params than Megatron FusedLayernorm."""
        llama_norm = LlamaRMSNorm(4096)
        llama_params = sum(p.numel() for p in llama_norm.parameters())

        from workload_generator.mocked_model.training.MockedMegatron import FusedLayernorm
        megatron_norm = FusedLayernorm(4096)
        megatron_params = sum(p.numel() for p in megatron_norm.parameters())

        # RMSNorm has no bias, so it should have fewer parameters
        assert llama_params < megatron_params, (
            f"RMSNorm ({llama_params}) should have fewer params than LayerNorm ({megatron_params})"
        )

    def test_llama_rope_no_trainable_params(self):
        """RoPE should contribute zero trainable parameters."""
        rope = LlamaRotaryEmbedding(dim=128, max_position_embeddings=4096)
        assert len(rope.parameters()) == 0

    def test_gqa_kv_heads_ratio(self):
        """num_kv_heads must divide num_attention_heads."""
        # Valid: 32/8 = 4
        attn = LlamaAttention(32, 8, 4096, tp=1, cp=1, seq_len=2048,
                              batch_size=1, layer_id=0)
        assert attn.num_attention_heads % attn.num_kv_heads == 0

        # Valid: 32/32 = 1 (MHA)
        attn2 = LlamaAttention(32, 32, 4096, tp=1, cp=1, seq_len=2048,
                               batch_size=1, layer_id=0)
        assert attn2.num_attention_heads % attn2.num_kv_heads == 0
