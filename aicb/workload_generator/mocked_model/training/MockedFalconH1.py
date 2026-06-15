"""
Mocked Falcon-H1 model for AICB workload generation.

Models the TII Falcon-H1 parallel hybrid architecture (arXiv:2507.22448):
  - Parallel Hybrid Block: Mamba-2 SSM + Attention run in PARALLEL, outputs concatenated
  - Mamba-2 SSM: state-space model with in_proj, conv1d, SSM scan, out_proj
  - Group Query Attention (GQA): separate Q, K, V projections with RoPE (base=10^11)
  - SwiGLU FFN: gate_proj, up_proj, down_proj (3 projections)
  - RMSNorm pre-normalization (weight-only, no bias)
  - Configurable layer ratios: parallel_hybrid, pure_mamba, pure_attention, pure_mlp

SSM Tensor Parallelism (per Megatron Core Falcon-H1 integration, March 2026):
  - in_proj: split along dim-1, distributing z/x/B/C/dt components across TP ranks
  - conv1d: replicated (not sharded)
  - A_log, D, dt_bias: split along dim-0
  - out_proj: RowParallel-style (reduce-scatter for SP, all-reduce otherwise)

Channel ratio default: 2:1:5 (SSM : Attention : MLP channels)
SA_M layout: Attention+SSM together -> followed by MLP (per paper ablations, best perf)

Supported configs:
  Falcon-H1-0.5B:  hidden=..., ssm_dim=..., ...
  Falcon-H1-1.5B:  hidden=2048, ...
  Falcon-H1-7B:    hidden=4096, ...
  Falcon-H1-34B:   hidden=..., ...

Based on MockedLlama.py and MockedMegatron.py patterns.
File: MockedFalconH1.py
License: Apache 2.0
"""

from utils.utils import divide, CommType, CommGroup
from workload_generator.mocked_model.MockedModel import MockedModel, MockedParam
from workload_generator.mocked_model.training.MockedMegatron import (
    MegatronColumnLinear,
    MegatronRowLinear,
    MegatronEmbedding,
)
from workload_generator.mocked_model.training.MockedLlama import (
    LlamaRMSNorm,
)
from log_analyzer.log import Workload, LogItem, ParallelWorkload


# ---------------------------------------------------------------------------
# Mamba2SSM -- Mamba-2 State Space Model layer
# ---------------------------------------------------------------------------
class Mamba2SSM(MockedModel):
    """Mamba-2 SSM block with TP-aware projections.

    Architecture (simplified from mamba-2.2):
        x -> RMSNorm -> in_proj -> [z, x_proj, B, C, dt] -> conv1d -> SSM scan -> out_proj

    TP sharding (per Megatron Core pattern):
        in_proj  output sharded along dim-1 (components distributed across ranks)
        conv1d   replicated (each rank has full conv1d kernel)
        A_log    sharded along dim-0 (head dimension)
        D        sharded along dim-0
        dt_bias  sharded along dim-0
        out_proj input sharded (RowParallel-style reduce-scatter or all-reduce)

    The SSM scan itself is a compute-only operation: an associative scan over the
    sequence dimension with complexity O(seq_len * d_state * d_inner). It generates
    NO communication -- only computation LogItems.

    For Falcon-H1, the Mamba-2 is configured with:
      - d_state (SSM state dimension): larger state -> better perf (paper Sec 3.2)
      - n_heads (SSM head groups): analogous to attention heads
      - head_dim: per-head inner dimension
    """

    def __init__(
        self,
        hidden_size: int,
        ssm_state_dim: int,
        ssm_head_dim: int,
        n_ssm_heads: int,
        tp: int,
        seq_len: int,
        batch_size: int,
        layer_id: int,
        sequence_parallel_enabled: bool = True,
        computation_enable: bool = False,
    ):
        super().__init__()
        self.name = f"mamba2_ssm_layer_{layer_id}"
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.ssm_state_dim = ssm_state_dim
        self.ssm_head_dim = ssm_head_dim
        self.n_ssm_heads = n_ssm_heads
        self.tp_size = tp
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.sequence_parallel_enabled = sequence_parallel_enabled
        self.computation_enable = computation_enable

        # Total inner dimension (analogous to intermediate_size in MLP)
        self.d_inner = n_ssm_heads * ssm_head_dim

        # ---- Parameters (SSM-specific, with TP sharding) ----
        # A_log: (n_ssm_heads,) -- log of state transition matrix diagonal
        self.A_log = MockedParam(
            (divide(n_ssm_heads, tp),),
            name=f"A_log_{layer_id}"
        )
        # D: (n_ssm_heads,) -- direct skip connection weight
        self.D = MockedParam(
            (divide(n_ssm_heads, tp),),
            name=f"D_{layer_id}"
        )
        # dt_bias: (n_ssm_heads,) -- learned bias for delta projection
        self.dt_bias = MockedParam(
            (divide(n_ssm_heads, tp),),
            name=f"dt_bias_{layer_id}"
        )

        # in_proj: hidden -> (z, x, B, C, dt) as a fused projection
        # Total output dim: d_inner * 2 + n_ssm_heads * (ssm_state_dim + 1)
        # Simplified: z_dim=d_inner, x_dim=d_inner, B_dim=n_heads*d_state,
        #             C_dim=n_heads*d_state, dt_dim=n_heads
        # Below we use d_inner * 2 as a reasonable proxy for total fused-proj output.
        self.in_proj_output_dim = 2 * self.d_inner + n_ssm_heads * (ssm_state_dim + ssm_state_dim + 1)
        self.in_proj_output_per_tp = divide(self.in_proj_output_dim, tp)

        self.in_proj = MegatronColumnLinear(
            hidden_size,
            self.in_proj_output_dim,
            tp,
            seq_len,
            batch_size,
            layer_id,
            "mamba2_in",
            sequence_parallel_enabled,
            computation_enable,
            name=f"mamba2_in_proj_{layer_id}",
            add_bias_linear=False,
        )

        # conv1d: depthwise 1D conv (replicated, not TP-sharded)
        # Kernel size typically 4, acts along sequence dim.
        # Modeled as a parameter with small footprint.
        self.conv1d_weight = MockedParam(
            (self.d_inner, 4),  # (d_inner, kernel_size)
            name=f"mamba2_conv1d_{layer_id}"
        )
        self.conv1d_bias = MockedParam(
            (self.d_inner,),
            name=f"mamba2_conv1d_bias_{layer_id}"
        )

        # out_proj: d_inner -> hidden_size (RowParallel-style)
        self.out_proj = MegatronRowLinear(
            self.d_inner,
            hidden_size,
            tp,
            seq_len,
            batch_size,
            layer_id,
            "mamba2_out",
            sequence_parallel_enabled,
            computation_enable,
            name=f"mamba2_out_proj_{layer_id}",
            add_bias_linear=False,
        )

    def _ssm_compute_kernel(self, stage: str) -> Workload:
        """Emit computation LogItems for the SSM scan and conv1d operations.

        These are compute-only: no network communication.
        conv1d: O(seq_len * d_inner * kernel_size) FLOPs
        SSM scan: O(seq_len * n_heads * d_state * head_dim) FLOPs
        """
        workloads = Workload()
        if not self.computation_enable:
            return workloads

        # conv1d forward/backward
        conv_msg = (
            (self.seq_len, self.d_inner),     # (L, D)
            (self.d_inner, self.d_inner * 4),  # conv1d kernel application
        )
        workloads.append(
            LogItem(
                comm_type=CommType.computation,
                comm_group=CommGroup.all,
                msg_size=conv_msg,
                stage=f"{stage}.Mamba2SSM.conv1d.{self.name}",
            )
        )

        # SSM scan (associative scan over sequence)
        scan_flops = self.seq_len * self.n_ssm_heads * self.ssm_state_dim * self.ssm_head_dim
        workloads.append(
            LogItem(
                comm_type=CommType.computation,
                comm_group=CommGroup.all,
                msg_size=((scan_flops,), (1,)),
                stage=f"{stage}.Mamba2SSM.scan.{self.name}",
            )
        )

        return workloads

    def forward(self):
        workloads = Workload()

        # in_proj: hidden -> z/x/B/C/dt (ColumnParallel: all_gather if SP)
        workloads.extend(self.in_proj.forward())

        # conv1d + SSM scan: compute only, no communication
        workloads.extend(self._ssm_compute_kernel(stage="forward"))

        # out_proj: d_inner -> hidden (RowParallel: reduce_scatter if SP)
        workloads.extend(self.out_proj.forward())

        assert all(isinstance(w, LogItem) for w in workloads.workload)
        return workloads

    def backward(self):
        workloads = Workload()

        # out_proj backward: all_gather if SP
        workloads.extend(self.out_proj.backward())

        # conv1d + SSM scan backward: compute only
        workloads.extend(self._ssm_compute_kernel(stage="backward"))

        # in_proj backward: reduce_scatter if SP
        workloads.extend(self.in_proj.backward())

        assert all(isinstance(w, LogItem) for w in workloads.workload)
        return workloads


# ---------------------------------------------------------------------------
# FalconH1MLP -- SwiGLU Feed-Forward (reuses LlamaMLP pattern)
# ---------------------------------------------------------------------------
class FalconH1MLP(MockedModel):
    """SwiGLU MLP: SiLU(gate_proj(x)) * up_proj(x) -> down_proj

    Identical to LlamaMLP. Falcon-H1 uses SwiGLU with configurable
    intermediate_size per the channel ratio (SSM : Attention : MLP = 2:1:5).
    """

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        tp,
        seq_len,
        batch_size,
        layer_id,
        sequence_parallel_enabled=True,
        computation_enable=False,
        add_bias_linear=False,
    ):
        super().__init__()
        self.name = f"mlp_layer_swiglu_{layer_id}"
        self.layer_id = layer_id

        self.gate_proj = MegatronColumnLinear(
            hidden_size, intermediate_size, tp,
            seq_len, batch_size, layer_id, "mlp_gate",
            sequence_parallel_enabled, computation_enable,
            name=f"mlp_gate_column_{layer_id}", add_bias_linear=add_bias_linear,
        )
        self.up_proj = MegatronColumnLinear(
            hidden_size, intermediate_size, tp,
            seq_len, batch_size, layer_id, "mlp_up",
            sequence_parallel_enabled, computation_enable,
            name=f"mlp_up_column_{layer_id}", add_bias_linear=add_bias_linear,
        )
        self.down_proj = MegatronRowLinear(
            intermediate_size, hidden_size, tp,
            seq_len, batch_size, layer_id, "mlp_down",
            sequence_parallel_enabled, computation_enable,
            name=f"mlp_down_row_{layer_id}", add_bias_linear=add_bias_linear,
        )

    def forward(self):
        w = Workload()
        w.extend(self.gate_proj.forward())
        w.extend(self.up_proj.forward())
        w.extend(self.down_proj.forward())
        assert all(isinstance(x, LogItem) for x in w.workload)
        return w

    def backward(self):
        w = Workload()
        w.extend(self.down_proj.backward())
        w.extend(self.up_proj.backward())
        w.extend(self.gate_proj.backward())
        assert all(isinstance(x, LogItem) for x in w.workload)
        return w


# ---------------------------------------------------------------------------
# FalconH1Attention -- GQA with RoPE (reuses LlamaAttention pattern)
# ---------------------------------------------------------------------------
class FalconH1Attention(MockedModel):
    """Group Query Attention for Falcon-H1.

    Uses RoPE with unusually high base frequency (10^11).
    Otherwise identical to LlamaAttention: separate Q/K/V projections with GQA.

    TP sharding: Q full TP, K/V capped at num_kv_heads.
    """

    def __init__(
        self,
        num_attention_heads,
        num_kv_heads,
        hidden_size,
        tp,
        seq_len,
        batch_size,
        layer_id,
        sequence_parallel_enabled=True,
        computation_enable=False,
        add_bias_linear=False,
    ):
        super().__init__()
        self.name = f"attention_layer_falconh1_{layer_id}"
        self.layer_id = layer_id
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_attention_heads

        kv_tp = min(num_kv_heads, tp)

        self.q_proj = MegatronColumnLinear(
            hidden_size, num_attention_heads * self.head_dim, tp,
            seq_len, batch_size, layer_id, "attention_q",
            sequence_parallel_enabled, computation_enable,
            name=f"attn_q_col_{layer_id}", add_bias_linear=add_bias_linear,
        )
        self.k_proj = MegatronColumnLinear(
            hidden_size, num_kv_heads * self.head_dim, kv_tp,
            seq_len, batch_size, layer_id, "attention_k",
            sequence_parallel_enabled, computation_enable,
            name=f"attn_k_col_{layer_id}", add_bias_linear=add_bias_linear,
        )
        self.v_proj = MegatronColumnLinear(
            hidden_size, num_kv_heads * self.head_dim, kv_tp,
            seq_len, batch_size, layer_id, "attention_v",
            sequence_parallel_enabled, computation_enable,
            name=f"attn_v_col_{layer_id}", add_bias_linear=add_bias_linear,
        )
        self.o_proj = MegatronRowLinear(
            num_attention_heads * self.head_dim, hidden_size, tp,
            seq_len, batch_size, layer_id, "attention_o",
            sequence_parallel_enabled, computation_enable,
            name=f"attn_o_row_{layer_id}", add_bias_linear=add_bias_linear,
        )

    def forward(self):
        w = Workload()
        w.extend(self.q_proj.forward())
        w.extend(self.k_proj.forward())
        w.extend(self.v_proj.forward())
        w.extend(self.o_proj.forward())
        assert all(isinstance(x, LogItem) for x in w.workload)
        return w

    def backward(self):
        w = Workload()
        w.extend(self.o_proj.backward())
        w.extend(self.v_proj.backward())
        w.extend(self.k_proj.backward())
        w.extend(self.q_proj.backward())
        assert all(isinstance(x, LogItem) for x in w.workload)
        return w


# ---------------------------------------------------------------------------
# ParallelHybridBlock -- core Falcon-H1 innovation
# ---------------------------------------------------------------------------
class ParallelHybridBlock(MockedModel):
    """Parallel hybrid: Mamba-2 SSM + Attention run in PARALLEL on the same input.

    Both sub-blocks receive the same RMSNorm'd input.
    Their outputs are concatenated along the hidden dimension, then projected
    back to hidden_size via a fusion projection.

    This is the key architectural difference from sequential Transformer layers:
    instead of attention_output -> mlp_output (sequential), we have:

        input -> norm -> [SSM_path || Attention_path] -> concat -> fusion_proj

    The channel ratio (default 2:1 SSM:Attention) controls the relative
    compute allocation between the two parallel paths.

    Communication: SSM and Attention are independent (no cross-communication).
    Only the fusion projection involves TP collectives.
    """

    def __init__(
        self,
        hidden_size,
        ssm_state_dim,
        ssm_head_dim,
        n_ssm_heads,
        num_attention_heads,
        num_kv_heads,
        tp,
        seq_len,
        batch_size,
        layer_id,
        ssm_channel_ratio=2,      # SSM : Attention channel ratio
        attn_channel_ratio=1,
        sequence_parallel_enabled=True,
        computation_enable=False,
        add_bias_linear=False,
    ):
        super().__init__()
        self.name = f"parallel_hybrid_block_{layer_id}"
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.ssm_channel_dim = hidden_size * ssm_channel_ratio // (ssm_channel_ratio + attn_channel_ratio)
        self.attn_channel_dim = hidden_size - self.ssm_channel_dim

        # Pre-block RMSNorm (shared input norm)
        self.input_norm = LlamaRMSNorm(hidden_size, name=f"hybrid_input_norm_{layer_id}")

        # SSM pathway
        self.ssm = Mamba2SSM(
            hidden_size=hidden_size,
            ssm_state_dim=ssm_state_dim,
            ssm_head_dim=ssm_head_dim,
            n_ssm_heads=n_ssm_heads,
            tp=tp,
            seq_len=seq_len,
            batch_size=batch_size,
            layer_id=layer_id,
            sequence_parallel_enabled=sequence_parallel_enabled,
            computation_enable=computation_enable,
        )

        # Attention pathway
        self.attention = FalconH1Attention(
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            hidden_size=hidden_size,
            tp=tp,
            seq_len=seq_len,
            batch_size=batch_size,
            layer_id=layer_id,
            sequence_parallel_enabled=sequence_parallel_enabled,
            computation_enable=computation_enable,
            add_bias_linear=add_bias_linear,
        )

        # Fusion projection: concat(ssm_out, attn_out) -> hidden
        # Total concat dim = 2 * hidden_size (both pathways output hidden_size)
        # For simplicity, we use two independent output dims that sum to hidden_size
        # In practice the concat then projected; here we model it as
        # a ColumnParallel-like fusion since both pathways feed into it.
        concat_dim = 2 * hidden_size
        self.fusion_proj = MegatronRowLinear(
            concat_dim, hidden_size, tp,
            seq_len, batch_size, layer_id, "hybrid_fusion",
            sequence_parallel_enabled, computation_enable,
            name=f"hybrid_fusion_{layer_id}", add_bias_linear=add_bias_linear,
        )

        # Also need the column-side: separate fusion weights for SSM and Attention outputs
        # after they are each produced. Use standard ColumnParallel for each pathway.
        self.ssm_fusion_in = MegatronColumnLinear(
            hidden_size, concat_dim // 2, tp,
            seq_len, batch_size, layer_id, "hybrid_fusion_ssm_in",
            sequence_parallel_enabled, computation_enable,
            name=f"hybrid_fusion_ssm_in_{layer_id}", add_bias_linear=False,
        )
        self.attn_fusion_in = MegatronColumnLinear(
            hidden_size, concat_dim // 2, tp,
            seq_len, batch_size, layer_id, "hybrid_fusion_attn_in",
            sequence_parallel_enabled, computation_enable,
            name=f"hybrid_fusion_attn_in_{layer_id}", add_bias_linear=False,
        )

    def forward(self):
        # SSM and Attention run in PARALLEL -- both pathways issue their
        # communication simultaneously and their compute overlaps.
        # We collect both Workloads into a ParallelWorkload so that
        # downstream consumers (replayer, analytical simulator) can
        # schedule them concurrently.
        ssm_wl = self.ssm.forward()
        attn_wl = self.attention.forward()
        parallel = ParallelWorkload([ssm_wl, attn_wl])

        workloads = Workload()
        workloads.extend(parallel)
        workloads.extend(self.ssm_fusion_in.forward())
        workloads.extend(self.attn_fusion_in.forward())
        workloads.extend(self.fusion_proj.forward())
        assert all(isinstance(w, LogItem) for w in workloads.workload)
        return workloads

    def backward(self):
        ssm_wl = self.ssm.backward()
        attn_wl = self.attention.backward()
        parallel = ParallelWorkload([ssm_wl, attn_wl])

        workloads = Workload()
        workloads.extend(self.fusion_proj.backward())
        workloads.extend(self.attn_fusion_in.backward())
        workloads.extend(self.ssm_fusion_in.backward())
        workloads.extend(parallel)
        assert all(isinstance(w, LogItem) for w in workloads.workload)
        return workloads


# ---------------------------------------------------------------------------
# FalconH1Layer -- complete SA_M decoder block
# ---------------------------------------------------------------------------
class FalconH1Layer(MockedModel):
    """Falcon-H1 decoder layer with SA_M layout.

    SA_M layout (per paper ablations, best performance):
        x -> input_norm -> [SSM || Attention] -> concat_fusion (+x residual)
            -> post_hybrid_norm -> MLP -> (+ residual)

    This is the pre-norm pattern: norm before each sub-block, residual after.

    The layer also supports configurable layer-type allocation:
      - 'parallel_hybrid': uses ParallelHybridBlock (default)
      - 'pure_mamba': uses Mamba2SSM only (no attention)
      - 'pure_attention': uses FalconH1Attention only (no SSM)
      - 'pure_mlp': skip both, just MLP
    Controlled by the layer_allocation list in FalconH1Model.
    """

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        ssm_state_dim,
        ssm_head_dim,
        n_ssm_heads,
        num_attention_heads,
        num_kv_heads,
        tp,
        seq_len,
        batch_size,
        layer_id,
        layer_type="parallel_hybrid",
        sequence_parallel_enabled=True,
        computation_enable=False,
        add_bias_linear=False,
    ):
        super().__init__()
        self.name = f"falconh1_layer_{layer_id}"
        self.layer_id = layer_id
        self.layer_type = layer_type

        # Pre-hybrid RMSNorm
        self.input_norm = LlamaRMSNorm(hidden_size, name=f"input_norm_{layer_id}")

        # Hybrid / attention / SSM block (depends on layer_type)
        if layer_type == "parallel_hybrid":
            self.hybrid_block = ParallelHybridBlock(
                hidden_size=hidden_size,
                ssm_state_dim=ssm_state_dim,
                ssm_head_dim=ssm_head_dim,
                n_ssm_heads=n_ssm_heads,
                num_attention_heads=num_attention_heads,
                num_kv_heads=num_kv_heads,
                tp=tp,
                seq_len=seq_len,
                batch_size=batch_size,
                layer_id=layer_id,
                sequence_parallel_enabled=sequence_parallel_enabled,
                computation_enable=computation_enable,
                add_bias_linear=add_bias_linear,
            )
            self.has_hybrid = True
            self.has_ssm_only = False
        elif layer_type == "pure_mamba":
            self.mamba_block = Mamba2SSM(
                hidden_size=hidden_size,
                ssm_state_dim=ssm_state_dim,
                ssm_head_dim=ssm_head_dim,
                n_ssm_heads=n_ssm_heads,
                tp=tp,
                seq_len=seq_len,
                batch_size=batch_size,
                layer_id=layer_id,
                sequence_parallel_enabled=sequence_parallel_enabled,
                computation_enable=computation_enable,
            )
            self.has_hybrid = False
            self.has_ssm_only = True
        elif layer_type == "pure_attention":
            self.attn_block = FalconH1Attention(
                num_attention_heads=num_attention_heads,
                num_kv_heads=num_kv_heads,
                hidden_size=hidden_size,
                tp=tp,
                seq_len=seq_len,
                batch_size=batch_size,
                layer_id=layer_id,
                sequence_parallel_enabled=sequence_parallel_enabled,
                computation_enable=computation_enable,
                add_bias_linear=add_bias_linear,
            )
            self.has_hybrid = False
            self.has_ssm_only = False
        else:
            self.has_hybrid = False
            self.has_ssm_only = False

        # Post-hybrid RMSNorm (before MLP)
        self.post_hybrid_norm = LlamaRMSNorm(hidden_size, name=f"post_hybrid_norm_{layer_id}")

        # MLP (SwiGLU)
        self.mlp = FalconH1MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            tp=tp,
            seq_len=seq_len,
            batch_size=batch_size,
            layer_id=layer_id,
            sequence_parallel_enabled=sequence_parallel_enabled,
            computation_enable=computation_enable,
            add_bias_linear=add_bias_linear,
        )

    def forward(self):
        workloads = Workload()

        # Sub-block (hybrid, mamba, attention, or skipped)
        if self.has_hybrid:
            workloads.extend(self.hybrid_block.forward())
        elif self.has_ssm_only:
            workloads.extend(self.mamba_block.forward())
        elif hasattr(self, 'attn_block'):
            workloads.extend(self.attn_block.forward())
        # else: pure_mlp, skip the sub-block entirely

        # MLP (always present)
        workloads.extend(self.mlp.forward())

        assert all(isinstance(w, LogItem) for w in workloads.workload)
        return workloads

    def backward(self):
        workloads = Workload()

        workloads.extend(self.mlp.backward())

        if self.has_hybrid:
            workloads.extend(self.hybrid_block.backward())
        elif self.has_ssm_only:
            workloads.extend(self.mamba_block.backward())
        elif hasattr(self, 'attn_block'):
            workloads.extend(self.attn_block.backward())

        assert all(isinstance(w, LogItem) for w in workloads.workload)
        return workloads


# ---------------------------------------------------------------------------
# FalconH1Model -- complete Falcon-H1 architecture
# ---------------------------------------------------------------------------
class FalconH1Model(MockedModel):
    """Falcon-H1 model: Embedding -> N x FalconH1Layer -> FinalNorm -> LM Head.

    Config expects these fields (set by get_params / config file):
      hidden_size, ffn_hidden_size (intermediate), num_layers,
      num_attention_heads, num_kv_heads, seq_length, micro_batch,
      tensor_model_parallel_size, padded_vocab_size,
      enable_sequence_parallel, computation_enable, add_bias_linear,
      ssm_state_dim, ssm_head_dim, n_ssm_heads,
      layer_allocation (list of layer types, length = num_layers)

    Default layer_allocation:
      Alternating pattern typical of Falcon-H1: several hybrid layers
      at input and output, with pure-Mamba and pure-attention mixed in.
      Per paper: SA_M layout, default channel ratio 2:1:5.
    """

    def __init__(self, config):
        super().__init__()

        # SSM-specific config (defaults for Falcon-H1-7B if not specified)
        ssm_state_dim = getattr(config, "ssm_state_dim", 128)
        ssm_head_dim = getattr(config, "ssm_head_dim", 64)
        n_ssm_heads = getattr(config, "n_ssm_heads", 32)

        # Layer allocation: list of types per layer or a single default
        layer_allocation = getattr(config, "layer_allocation", None)
        if layer_allocation is None:
            # Default: alternating hybrid/pure_mamba, with first and last layers hybrid
            layer_allocation = []
            for i in range(config.num_layers):
                if i < 2 or i >= config.num_layers - 2:
                    layer_allocation.append("parallel_hybrid")
                elif i % 4 == 0:
                    layer_allocation.append("pure_mamba")
                elif i % 4 == 1:
                    layer_allocation.append("pure_attention")
                elif i % 4 == 2:
                    layer_allocation.append("parallel_hybrid")
                else:
                    layer_allocation.append("parallel_hybrid")
        if len(layer_allocation) != config.num_layers:
            # Fallback: all parallel_hybrid if allocation length mismatch
            layer_allocation = ["parallel_hybrid"] * config.num_layers

        # KV heads (default to num_attention_heads for non-GQA)
        num_kv_heads = getattr(config, "num_kv_heads", config.num_attention_heads)

        # Embedding
        self.embedding = MegatronEmbedding(
            config.padded_vocab_size,
            config.hidden_size,
            config.tensor_model_parallel_size,
            config.seq_length,
            config.micro_batch,
        )

        # Decoder layers
        self.layers = []
        for i in range(config.num_layers):
            self.layers.append(
                FalconH1Layer(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.ffn_hidden_size,
                    ssm_state_dim=ssm_state_dim,
                    ssm_head_dim=ssm_head_dim,
                    n_ssm_heads=n_ssm_heads,
                    num_attention_heads=config.num_attention_heads,
                    num_kv_heads=num_kv_heads,
                    tp=config.tensor_model_parallel_size,
                    seq_len=config.seq_length,
                    batch_size=config.micro_batch,
                    layer_id=i,
                    layer_type=layer_allocation[i],
                    sequence_parallel_enabled=config.enable_sequence_parallel,
                    computation_enable=config.computation_enable,
                    add_bias_linear=config.add_bias_linear,
                )
            )

        # Final RMSNorm
        self.final_norm = LlamaRMSNorm(config.hidden_size, name="final_norm")

        # LM Head: hidden -> vocab
        self.lm_head = MegatronColumnLinear(
            config.hidden_size,
            config.padded_vocab_size,
            config.tensor_model_parallel_size,
            config.seq_length,
            config.micro_batch,
            config.num_layers + 1,
            "lm_head",
            sequence_parallel_enabled=config.enable_sequence_parallel,
            computation_enable=config.computation_enable,
            add_bias_linear=False,
        )

    def forward(self):
        workloads = Workload()
        workloads.extend(self.embedding.forward())
        for layer in self.layers:
            workloads.extend(layer.forward())
        assert all(isinstance(w, LogItem) for w in workloads.workload)
        return workloads

    def backward(self):
        workloads = Workload()
        for layer in self.layers[::-1]:
            workloads.extend(layer.backward())
        workloads.extend(self.embedding.backward())
        assert all(isinstance(w, LogItem) for w in workloads.workload)
        return workloads
