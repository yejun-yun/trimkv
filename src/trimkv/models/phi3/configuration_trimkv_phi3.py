from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging
from transformers.models.phi3 import Phi3Config


logger = logging.get_logger(__name__)


class TrimKVPhi3Config(Phi3Config):
    def __init__(
        self,
        retention_gate_bias_init=10.0,
        retention_weight=1.0,
        memory_size=1024,
        retention_gate='rg',
        base_loss='fwkl',
        attn_impl='rg_attn_flex',
        compress_memory=False,
        compress_strategy='alpha',
        floor_budget_ratio=0.,
        buffer_size=1,
        trainable_params=None,
        max_seq_len=20480,
        retention_gate_intermediate_size=1024,
        **kwargs,
    ):
        self.retention_gate_bias_init = retention_gate_bias_init
        self.retention_weight = retention_weight
        self.retention_gate_intermediate_size = retention_gate_intermediate_size
        self.memory_size = memory_size
        self.retention_gate = retention_gate
        self.attn_impl = attn_impl
        self.base_loss = base_loss
        self.trainable_params = trainable_params
        self.compress_memory = compress_memory
        self.compress_strategy = compress_strategy
        self.floor_budget_ratio = floor_budget_ratio
        self.buffer_size = buffer_size # run compression every `buffer_size` tokens
        self.max_seq_len = max_seq_len
        super().__init__(
            **kwargs,
        )


__all__ = ["TrimKVPhi3Config"]
