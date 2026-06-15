"""
Chakra Execution Trace (ET) exporter for AICB workloads.

Converts AICB LogItem/Workload objects to the Chakra ET JSON format
consumed by ASTRA-sim, PARAM, and other Chakra-ecosystem tools.

Format reference: MLCommons Chakra schema (et_def.proto)
  - NodeType: 1=METADATA, 2=MEM_LOAD, 3=MEM_STORE, 4=COMP, 5=COMM_SEND,
               6=COMM_RECV, 7=COMM_COLL
  - CollectiveCommType: ALL_REDUCE, ALL_GATHER, REDUCE_SCATTER, ALL_TO_ALL,
                        BROADCAST, BARRIER, REDUCE, etc.

Design decisions:
  - JSON output (no protobuf dependency -- astra-sim accepts JSON)
  - Linear dependency chain by default (each node depends on previous)
  - Metadata node with global attributes (world_size, tp, pp, dp, cp)
  - Epoch boundaries as control dependencies (no reordering across epochs)

Usage:
    from utils.chakra_export import ChakraExporter
    exporter = ChakraExporter(workload, args)
    exporter.export("output.et.json")

File: chakra_export.py
License: Apache 2.0
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from utils.utils import CommType, CommGroup


# ---------------------------------------------------------------------------
# Chakra NodeType enum (integer values matching et_def.proto)
# ---------------------------------------------------------------------------
class ChakraNodeType:
    INVALID = 0
    METADATA = 1
    MEM_LOAD = 2
    MEM_STORE = 3
    COMP = 4
    COMM_SEND = 5
    COMM_RECV = 6
    COMM_COLL = 7


# ---------------------------------------------------------------------------
# Mapping: AICB CommType -> Chakra collective name
# ---------------------------------------------------------------------------
_COMMTYPE_TO_CHAKRA_COLL: Dict[CommType, str] = {
    CommType.all_reduce: "ALL_REDUCE",
    CommType.all_gather: "ALL_GATHER",
    CommType.reduce_scatter: "REDUCE_SCATTER",
    CommType.all_to_all: "ALL_TO_ALL",
    CommType.broadcast: "BROADCAST",
    CommType.barrier: "BARRIER",
    CommType.reduce: "REDUCE",
    CommType.all_gather_into_tensor: "ALL_GATHER",
    CommType.reduce_scatter_tensor: "REDUCE_SCATTER",
}

# CommTypes that map to COMM_SEND / COMM_RECV (P2P)
_P2P_COMM_TYPES = {CommType.isend, CommType.irecv}

# CommTypes to skip (not meaningful in Chakra DAG)
_SKIP_COMM_TYPES = {CommType.epoch_end}


def _make_attr(name: str, value: Any) -> Dict[str, Any]:
    """Create a Chakra AttributeProto-compatible dict.

    Infers the correct type field suffix based on Python type:
      - int -> int64_val
      - float -> double_val
      - str -> string_val
      - bool -> bool_val
      - list[int] -> int64_list
    """
    attr: Dict[str, Any] = {"name": name}
    if isinstance(value, bool):
        attr["bool_val"] = value
    elif isinstance(value, int):
        attr["int64_val"] = value
    elif isinstance(value, float):
        attr["double_val"] = value
    elif isinstance(value, str):
        attr["string_val"] = value
    elif isinstance(value, list):
        if all(isinstance(x, int) for x in value):
            attr["int64_list"] = {"values": [int(x) for x in value]}
        else:
            attr["string_val"] = json.dumps(value)
    else:
        attr["string_val"] = str(value)
    return attr


class ChakraExporter:
    """Exports AICB Workload to Chakra ET JSON format.

    Parameters
    ----------
    workload : Workload
        A Workload object containing a list of LogItem entries.
    args : argparse.Namespace or dict
        Configuration containing world_size, tp, pp, dp, cp, and model params.
    """

    def __init__(self, workload, args):
        self._workload = workload
        self._args = args
        self._nodes: List[Dict[str, Any]] = []
        self._node_id: int = 0

    # ------- helpers -------

    def _next_id(self) -> int:
        nid = self._node_id
        self._node_id += 1
        return nid

    def _add_node(
        self,
        name: str,
        node_type: int,
        ctrl_deps: Optional[List[int]] = None,
        data_deps: Optional[List[int]] = None,
        attrs: Optional[List[Dict[str, Any]]] = None,
        comm_group: Optional[CommGroup] = None,
        comm_group_size: Optional[int] = None,
    ) -> int:
        nid = self._next_id()
        node: Dict[str, Any] = {
            "id": nid,
            "name": name,
            "type": node_type,
            "ctrl_deps": ctrl_deps or [],
            "data_deps": data_deps or [],
            "attr": attrs or [],
        }
        if comm_group is not None:
            node["attr"].append(_make_attr("comm_group", comm_group.value))
        if comm_group_size is not None:
            node["attr"].append(_make_attr("comm_group_size", comm_group_size))
        self._nodes.append(node)
        return nid

    # ------- core build logic -------

    def _build_metadata_node(self) -> int:
        """Create the global METADATA node describing the simulation config."""
        args = self._args
        attrs = []

        def _g(name, default=1):
            return int(getattr(args, name, default) or default)

        attrs.append(_make_attr("world_size", _g("world_size")))
        attrs.append(_make_attr("tensor_model_parallel_size", _g("tensor_model_parallel_size")))
        attrs.append(_make_attr("pipeline_model_parallel", _g("pipeline_model_parallel")))
        attrs.append(_make_attr("dp_num", _g("dp_num")))
        attrs.append(_make_attr("context_parallel_size", _g("context_parallel_size", 1)))
        attrs.append(_make_attr("num_layers", _g("num_layers", 1)))
        attrs.append(_make_attr("hidden_size", _g("hidden_size", 0)))
        attrs.append(_make_attr("seq_length", _g("seq_length", 0)))
        attrs.append(_make_attr("micro_batch", _g("micro_batch", 1)))
        attrs.append(_make_attr("global_batch", _g("global_batch", 1)))
        attrs.append(_make_attr("generator", "AICB-ChakraExporter-v1"))

        return self._add_node("METADATA", ChakraNodeType.METADATA, attrs=attrs)

    def _logitem_to_node(self, item, prev_id: int) -> Optional[int]:
        """Convert a single LogItem to a Chakra node.

        Returns the new node id, or None if the item is skipped.
        """
        comm_type = item.comm_type

        # Skip epoch boundaries
        if comm_type in _SKIP_COMM_TYPES:
            return None

        stage = getattr(item, "stage", "") or ""
        raw_msg_size = getattr(item, "msg_size", 0) or 0
        # msg_size may be a tuple for computation LogItems (GEMM shapes).
        # In that case, extract the numeric magnitude for the comm_size attr.
        if isinstance(raw_msg_size, tuple):
            msg_size = _msg_size_from_shapes(raw_msg_size)
        else:
            msg_size = int(raw_msg_size)
        comm_group = getattr(item, "comm_group", None)
        comm_group_size = getattr(item, "comm_group_size", None)
        ranks = getattr(item, "ranks", None)

        # Build attributes
        attrs: List[Dict[str, Any]] = []
        if msg_size:
            attrs.append(_make_attr("comm_size", msg_size))
        if ranks:
            attrs.append(_make_attr("ranks", list(ranks)))

        # P2P communication
        if comm_type in _P2P_COMM_TYPES:
            node_type = ChakraNodeType.COMM_SEND if comm_type == CommType.isend else ChakraNodeType.COMM_RECV
            dst = getattr(item, "dst", None)
            src = getattr(item, "src", None)
            if dst is not None:
                attrs.append(_make_attr("dst", int(dst)))
            if src is not None:
                attrs.append(_make_attr("src", int(src)))
            return self._add_node(
                stage or comm_type.value,
                node_type,
                ctrl_deps=[prev_id] if prev_id >= 0 else [],
                attrs=attrs,
                comm_group=comm_group,
                comm_group_size=comm_group_size,
            )

        # Collective communication
        chakra_coll = _COMMTYPE_TO_CHAKRA_COLL.get(comm_type)
        if chakra_coll is not None:
            attrs.append(_make_attr("comm_type", chakra_coll))
            return self._add_node(
                stage or chakra_coll,
                ChakraNodeType.COMM_COLL,
                ctrl_deps=[prev_id] if prev_id >= 0 else [],
                attrs=attrs,
                comm_group=comm_group,
                comm_group_size=comm_group_size,
            )

        # Computation
        if comm_type == CommType.computation:
            # Extract FLOPs info from msg_size if it's a tuple
            flops = 0
            if isinstance(msg_size, tuple):
                # (input_shape, weight_shape) -> approximate FLOPs = 2 * prod(shapes)
                shapes = [item for item in msg_size if isinstance(item, tuple)]
                flops = sum(2 * _prod(shape) for shape in shapes)
                attrs.append(_make_attr("input_shape", [list(s) for s in msg_size if isinstance(s, tuple)]))
            if flops:
                attrs.append(_make_attr("flops", flops))
            return self._add_node(
                stage or "compute",
                ChakraNodeType.COMP,
                ctrl_deps=[prev_id] if prev_id >= 0 else [],
                attrs=attrs,
            )

        # Unknown type -- treat as metadata
        attrs.append(_make_attr("raw_comm_type", comm_type.value if comm_type else "unknown"))
        return self._add_node(
            stage or str(comm_type),
            ChakraNodeType.METADATA,
            ctrl_deps=[prev_id] if prev_id >= 0 else [],
            attrs=attrs,
        )

    def build(self) -> List[Dict[str, Any]]:
        """Build the full Chakra node list from the workload.

        Returns the list of node dicts (self._nodes is also populated).
        """
        self._nodes = []
        self._node_id = 0

        # 1. METADATA node
        metadata_id = self._build_metadata_node()

        # 2. Iterate over workload items
        prev_id = metadata_id
        items = getattr(self._workload, "workload", self._workload)
        for item in items:
            result_id = self._logitem_to_node(item, prev_id)
            if result_id is not None:
                prev_id = result_id

        return self._nodes

    # ------- export -------

    def export(self, filepath: str) -> None:
        """Build and write the Chakra ET JSON file.

        Parameters
        ----------
        filepath : str
            Path to the output JSON file (typically ending in .et.json).
        """
        self.build()
        output: Dict[str, Any] = {
            "nodes": self._nodes,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)
        return None

    # ------- accessors -------

    @property
    def nodes(self) -> List[Dict[str, Any]]:
        """Return the built node list (calls build() if empty)."""
        if not self._nodes:
            self.build()
        return self._nodes

    @property
    def node_count(self) -> int:
        return len(self.nodes)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _prod(shape: Tuple[int, ...]) -> int:
    """Product of tuple elements, handling empty tuples."""
    if not shape:
        return 1
    result = 1
    for s in shape:
        result *= s
    return result


def _msg_size_from_shapes(shapes: tuple) -> int:
    """Estimate a scalar magnitude from GEMM shape tuples for metadata.

    For a computation LogItem, msg_size is like ((B, S, H), (H, O)).
    Return the product of all dimensions as an approximate byte count.
    """
    total = 0
    for item in shapes:
        if isinstance(item, tuple):
            total += _prod(item)
    return total * 2  # BF16: 2 bytes per element
