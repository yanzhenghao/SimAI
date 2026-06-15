"""
Tests for Chakra ET exporter (utils/chakra_export.py).

Covers:
  - CommType -> Chakra collective mapping
  - Node type assignment (COMP, COMM_COLL, COMM_SEND, COMM_RECV, METADATA)
  - Dependency chain construction
  - Attribute serialization
  - JSON output structure validation
  - Backward compatibility with real LogItem data
"""

import sys
import os
import json
import tempfile
import dataclasses
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.utils import CommType, CommGroup
from utils.chakra_export import (
    ChakraExporter,
    ChakraNodeType,
    _COMMTYPE_TO_CHAKRA_COLL,
    _make_attr,
    _prod,
)


# ---------------------------------------------------------------------------
# Helpers: build minimal LogItem-like objects for testing
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class FakeLogItem:
    """Minimal LogItem stand-in for exporter tests."""
    comm_type: CommType = None
    comm_group: CommGroup = None
    comm_group_size: int = None
    msg_size: float = 0
    stage: str = ""
    dst: int = None
    src: int = None
    ranks: list = None


class FakeWorkload:
    """Minimal Workload stand-in."""
    def __init__(self, items):
        self.workload = items


class FakeArgs:
    """Minimal args stand-in."""
    world_size = 8
    tensor_model_parallel_size = 4
    pipeline_model_parallel = 1
    dp_num = 2
    context_parallel_size = 1
    num_layers = 32
    hidden_size = 4096
    seq_length = 2048
    micro_batch = 1
    global_batch = 64


# ---------------------------------------------------------------------------
# _make_attr helper
# ---------------------------------------------------------------------------
class TestMakeAttr:
    def test_int_value(self):
        attr = _make_attr("test", 42)
        assert attr["name"] == "test"
        assert attr["int64_val"] == 42

    def test_float_value(self):
        attr = _make_attr("pi", 3.14)
        assert attr["double_val"] == 3.14

    def test_string_value(self):
        attr = _make_attr("label", "hello")
        assert attr["string_val"] == "hello"

    def test_bool_value(self):
        attr = _make_attr("flag", True)
        assert attr["bool_val"] is True

    def test_int_list(self):
        attr = _make_attr("ranks", [0, 1, 2, 3])
        assert attr["int64_list"]["values"] == [0, 1, 2, 3]

    def test_str_list_falls_back_to_json(self):
        attr = _make_attr("tags", ["a", "b"])
        assert "string_val" in attr  # dumped as JSON string


# ---------------------------------------------------------------------------
# CommType -> Chakra mapping
# ---------------------------------------------------------------------------
class TestCommTypeMapping:
    def test_all_reduce_maps_to_all_reduce(self):
        assert _COMMTYPE_TO_CHAKRA_COLL[CommType.all_reduce] == "ALL_REDUCE"

    def test_all_gather_mapping(self):
        assert _COMMTYPE_TO_CHAKRA_COLL[CommType.all_gather] == "ALL_GATHER"

    def test_reduce_scatter_mapping(self):
        assert _COMMTYPE_TO_CHAKRA_COLL[CommType.reduce_scatter] == "REDUCE_SCATTER"

    def test_all_to_all_mapping(self):
        assert _COMMTYPE_TO_CHAKRA_COLL[CommType.all_to_all] == "ALL_TO_ALL"

    def test_broadcast_mapping(self):
        assert _COMMTYPE_TO_CHAKRA_COLL[CommType.broadcast] == "BROADCAST"

    def test_barrier_mapping(self):
        assert _COMMTYPE_TO_CHAKRA_COLL[CommType.barrier] == "BARRIER"

    def test_reduce_mapping(self):
        assert _COMMTYPE_TO_CHAKRA_COLL[CommType.reduce] == "REDUCE"

    def test_computation_not_in_comm_map(self):
        assert CommType.computation not in _COMMTYPE_TO_CHAKRA_COLL


# ---------------------------------------------------------------------------
# Node construction from LogItems
# ---------------------------------------------------------------------------
class TestNodeConstruction:
    def test_metadata_node_always_first(self):
        items = FakeWorkload([FakeLogItem(CommType.computation, stage="forward.emb")])
        exporter = ChakraExporter(items, FakeArgs())
        exporter.build()
        assert exporter.nodes[0]["type"] == ChakraNodeType.METADATA
        assert exporter.nodes[0]["name"] == "METADATA"

    def test_computation_becomes_comp_node(self):
        items = FakeWorkload([
            FakeLogItem(CommType.computation, stage="forward.Linear.mlp_gate",
                        msg_size=((2048, 1, 4096), (4096, 11008)))
        ])
        exporter = ChakraExporter(items, FakeArgs())
        exporter.build()
        comp_nodes = [n for n in exporter.nodes if n["type"] == ChakraNodeType.COMP]
        assert len(comp_nodes) == 1
        assert "forward.Linear.mlp_gate" in comp_nodes[0]["name"]

    def test_all_reduce_becomes_comm_coll_node(self):
        items = FakeWorkload([
            FakeLogItem(CommType.all_reduce, CommGroup.tp_group, 4,
                        msg_size=67108864, stage="forward.tp_all_reduce",
                        ranks=[0, 1, 2, 3])
        ])
        exporter = ChakraExporter(items, FakeArgs())
        exporter.build()
        coll_nodes = [n for n in exporter.nodes if n["type"] == ChakraNodeType.COMM_COLL]
        assert len(coll_nodes) == 1
        attrs = {a["name"]: a for a in coll_nodes[0]["attr"]}
        assert attrs["comm_type"]["string_val"] == "ALL_REDUCE"
        assert attrs["comm_group"]["string_val"] == "tp_group"
        assert attrs["comm_group_size"]["int64_val"] == 4

    def test_all_gather_becomes_comm_coll_node(self):
        items = FakeWorkload([
            FakeLogItem(CommType.all_gather, CommGroup.tp_group, 4,
                        msg_size=33554432, stage="forward.tp_all_gather",
                        ranks=[0, 1, 2, 3])
        ])
        exporter = ChakraExporter(items, FakeArgs())
        exporter.build()
        coll = [n for n in exporter.nodes if n["type"] == ChakraNodeType.COMM_COLL]
        assert len(coll) == 1
        attrs = {a["name"]: a for a in coll[0]["attr"]}
        assert attrs["comm_type"]["string_val"] == "ALL_GATHER"

    def test_all_to_all_becomes_comm_coll_node(self):
        items = FakeWorkload([
            FakeLogItem(CommType.all_to_all, CommGroup.cp_group, 2,
                        msg_size=134217728, stage="forward.cp_kv_exchange",
                        ranks=[0, 1])
        ])
        exporter = ChakraExporter(items, FakeArgs())
        exporter.build()
        coll = [n for n in exporter.nodes if n["type"] == ChakraNodeType.COMM_COLL]
        assert len(coll) == 1
        attrs = {a["name"]: a for a in coll[0]["attr"]}
        assert attrs["comm_type"]["string_val"] == "ALL_TO_ALL"
        assert attrs["comm_group"]["string_val"] == "cp_group"

    def test_isend_becomes_comm_send_node(self):
        items = FakeWorkload([
            FakeLogItem(CommType.isend, CommGroup.pp_group, 2,
                        msg_size=4096, stage="send_activation",
                        dst=4)
        ])
        exporter = ChakraExporter(items, FakeArgs())
        exporter.build()
        send_nodes = [n for n in exporter.nodes if n["type"] == ChakraNodeType.COMM_SEND]
        assert len(send_nodes) == 1

    def test_irecv_becomes_comm_recv_node(self):
        items = FakeWorkload([
            FakeLogItem(CommType.irecv, CommGroup.pp_group, 2,
                        msg_size=4096, stage="recv_activation",
                        src=0)
        ])
        exporter = ChakraExporter(items, FakeArgs())
        exporter.build()
        recv_nodes = [n for n in exporter.nodes if n["type"] == ChakraNodeType.COMM_RECV]
        assert len(recv_nodes) == 1

    def test_epoch_end_is_skipped(self):
        items = FakeWorkload([
            FakeLogItem(CommType.computation, stage="forward.layer0"),
            FakeLogItem(CommType.epoch_end, stage="epoch_end"),
            FakeLogItem(CommType.computation, stage="forward.layer0"),
        ])
        exporter = ChakraExporter(items, FakeArgs())
        exporter.build()
        comp_nodes = [n for n in exporter.nodes if n["type"] == ChakraNodeType.COMP]
        assert len(comp_nodes) == 2  # epoch_end skipped, not a node


# ---------------------------------------------------------------------------
# Dependency chain
# ---------------------------------------------------------------------------
class TestDependencyChain:
    def test_linear_dependency_chain(self):
        items = FakeWorkload([
            FakeLogItem(CommType.computation, stage="a"),
            FakeLogItem(CommType.computation, stage="b"),
            FakeLogItem(CommType.computation, stage="c"),
        ])
        exporter = ChakraExporter(items, FakeArgs())
        exporter.build()
        # Metadata + 3 compute nodes = 4 nodes
        assert exporter.node_count == 4
        # metadata(id=0) -> a(id=1) -> b(id=2) -> c(id=3)
        # a depends on metadata, b depends on a, c depends on b
        for i in range(1, 4):
            deps = exporter.nodes[i]["ctrl_deps"]
            assert deps == [i - 1]

    def test_mixed_comm_and_comp_chain(self):
        items = FakeWorkload([
            FakeLogItem(CommType.computation, stage="forward.qkv"),
            FakeLogItem(CommType.all_reduce, CommGroup.tp_group, 4,
                        msg_size=1024, stage="tp_all_reduce"),
            FakeLogItem(CommType.computation, stage="forward.o"),
        ])
        exporter = ChakraExporter(items, FakeArgs())
        exporter.build()
        # metadata + comp + comm + comp = 4
        assert exporter.node_count == 4
        types = [n["type"] for n in exporter.nodes]
        assert types == [ChakraNodeType.METADATA, ChakraNodeType.COMP,
                         ChakraNodeType.COMM_COLL, ChakraNodeType.COMP]


# ---------------------------------------------------------------------------
# Metadata node content
# ---------------------------------------------------------------------------
class TestMetadataContent:
    def test_metadata_contains_world_config(self):
        exporter = ChakraExporter(FakeWorkload([]), FakeArgs())
        exporter.build()
        meta = exporter.nodes[0]
        attrs = {a["name"]: a for a in meta["attr"]}
        assert attrs["world_size"]["int64_val"] == 8
        assert attrs["tensor_model_parallel_size"]["int64_val"] == 4
        assert attrs["pipeline_model_parallel"]["int64_val"] == 1
        assert attrs["dp_num"]["int64_val"] == 2
        assert attrs["context_parallel_size"]["int64_val"] == 1

    def test_metadata_contains_model_config(self):
        exporter = ChakraExporter(FakeWorkload([]), FakeArgs())
        exporter.build()
        meta = exporter.nodes[0]
        attrs = {a["name"]: a for a in meta["attr"]}
        assert attrs["num_layers"]["int64_val"] == 32
        assert attrs["hidden_size"]["int64_val"] == 4096
        assert attrs["seq_length"]["int64_val"] == 2048

    def test_metadata_contains_generator_tag(self):
        exporter = ChakraExporter(FakeWorkload([]), FakeArgs())
        exporter.build()
        meta = exporter.nodes[0]
        attrs = {a["name"]: a for a in meta["attr"]}
        assert "AICB-ChakraExporter" in attrs["generator"]["string_val"]


# ---------------------------------------------------------------------------
# JSON export round-trip
# ---------------------------------------------------------------------------
class TestJSONExport:
    def test_export_writes_valid_json(self):
        items = FakeWorkload([
            FakeLogItem(CommType.computation, stage="forward.emb"),
            FakeLogItem(CommType.all_reduce, CommGroup.tp_group, 4,
                        msg_size=1024, stage="tp_ar", ranks=[0, 1, 2, 3]),
        ])
        exporter = ChakraExporter(items, FakeArgs())
        with tempfile.NamedTemporaryFile(mode="w", suffix=".et.json", delete=False) as f:
            tmp_path = f.name
        try:
            exporter.export(tmp_path)
            with open(tmp_path) as f:
                data = json.load(f)
            assert "nodes" in data
            assert isinstance(data["nodes"], list)
            assert len(data["nodes"]) == 3  # metadata + comp + comm
        finally:
            os.unlink(tmp_path)

    def test_export_preserves_node_structure(self):
        items = FakeWorkload([
            FakeLogItem(CommType.computation, stage="forward.emb", msg_size=((1, 2048, 4096),)),
        ])
        exporter = ChakraExporter(items, FakeArgs())
        with tempfile.NamedTemporaryFile(mode="w", suffix=".et.json", delete=False) as f:
            tmp_path = f.name
        try:
            exporter.export(tmp_path)
            with open(tmp_path) as f:
                data = json.load(f)
            # Each node has required fields
            for node in data["nodes"]:
                assert "id" in node
                assert "name" in node
                assert "type" in node
                assert "ctrl_deps" in node
                assert "data_deps" in node
                assert "attr" in node
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Real-world workload simulation
# ---------------------------------------------------------------------------
class TestRealisticWorkload:
    def test_llama_microbatch_workload(self):
        """Simulate a Llama single-layer microbatch: forward + backward."""
        items = FakeWorkload([
            # Forward
            FakeLogItem(CommType.computation, stage="forward.embedding",
                        msg_size=((1, 2048, 4096), (32000, 4096))),
            # Layer 0
            FakeLogItem(CommType.computation, stage="forward.Linear.attention_q_column",
                        msg_size=((2048, 1, 4096), (4096, 4096))),
            FakeLogItem(CommType.computation, stage="forward.Linear.attention_k_column",
                        msg_size=((2048, 1, 4096), (4096, 1024))),
            FakeLogItem(CommType.computation, stage="forward.Linear.attention_v_column",
                        msg_size=((2048, 1, 4096), (4096, 1024))),
            FakeLogItem(CommType.computation, stage="forward.Linear.attention_o_row",
                        msg_size=((2048, 1, 4096), (4096, 4096))),
            FakeLogItem(CommType.all_to_all, CommGroup.cp_group, 4,
                        msg_size=2 * 8 * 128 * 2048 * 1, stage="forward.cp_kv_exchange",
                        ranks=[0, 1, 2, 3]),
            # MLP
            FakeLogItem(CommType.computation, stage="forward.Linear.mlp_gate_column",
                        msg_size=((2048, 1, 4096), (4096, 11008))),
            FakeLogItem(CommType.computation, stage="forward.Linear.mlp_up_column",
                        msg_size=((2048, 1, 4096), (4096, 11008))),
            FakeLogItem(CommType.computation, stage="forward.Linear.mlp_down_row",
                        msg_size=((2048, 1, 11008), (11008, 4096))),
            # TP all_reduce
            FakeLogItem(CommType.all_reduce, CommGroup.tp_group, 4,
                        msg_size=16777216, stage="forward.tp_ar"),
            # Backward (partial)
            FakeLogItem(CommType.all_reduce, CommGroup.tp_group, 4,
                        msg_size=16777216, stage="backward.tp_ar"),
            FakeLogItem(CommType.computation, stage="backward.Linear.mlp_down_row",
                        msg_size=((11008, 2048), (2048, 4096))),
        ])

        exporter = ChakraExporter(items, FakeArgs())
        exporter.build()

        # Count node types
        type_counts = {}
        for node in exporter.nodes:
            t = node["type"]
            type_counts[t] = type_counts.get(t, 0) + 1

        assert type_counts.get(ChakraNodeType.METADATA, 0) == 1
        assert type_counts.get(ChakraNodeType.COMP, 0) == 9  # embedding + 4 attn + 3 mlp + 1 backward
        assert type_counts.get(ChakraNodeType.COMM_COLL, 0) == 3  # cp all_to_all + 2 tp all_reduce

        # Verify chain is contiguous (no gaps in dependencies)
        for i in range(1, exporter.node_count):
            deps = exporter.nodes[i]["ctrl_deps"]
            assert len(deps) > 0, f"Node {i} has no dependencies"
