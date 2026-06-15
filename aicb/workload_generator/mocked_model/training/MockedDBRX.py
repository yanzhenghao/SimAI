"""
Mocked DBRX model for AICB workload generation.

DBRX (Databricks, 2024): Fine-grained MoE with 16 experts, top-4 routing.
Total 132B params, 36B active per token.

Architecture:
  - RMSNorm pre-normalization
  - RoPE position embeddings
  - GQA (if num_kv_heads < num_heads)
  - FFN: fine-grained MoE with 16 experts and top-4 gating
  - Each expert is a smaller FFN block for fine-grained load balancing

Communication-wise, DBRX is identical to Mixtral with different expert
count and routing top-k. Reuses MixtralDecoderLayer and MixtralModel
from MockedMistral.py.

Supported config:
  DBRX-132B: hidden=6144, ffn=24576, layers=40, num_heads=48,
             num_kv_heads=8, num_experts=16, moe_router_topk=4,

File: MockedDBRX.py
License: Apache 2.0
"""

from workload_generator.mocked_model.MockedModel import MockedModel
from workload_generator.mocked_model.training.MockedMistral import MixtralModel


class DBRXModel(MixtralModel):
    """DBRX model: 16 experts, top-4 routing, fine-grained MoE.

    Reuses MixtralModel with DBRX-specific defaults:
      - num_experts=16 (vs Mixtral's 8)
      - moe_router_topk=4 (vs Mixtral's 2)
      - Fine-grained experts: smaller FFN per expert for load balance
    """

    pass
