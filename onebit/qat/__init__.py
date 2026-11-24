"""Phase 1 KD-QAT package: ternary/binary QAT modules, streaming distillation, and training loop."""
from onebit.qat.convert import (
    QATConfig,
    build_bit_report,
    convert_model_to_qat,
    copy_qat_weights_to_fp_model,
    parameter_groups,
    unwrap_model,
)
from onebit.qat.data import (
    PackedStreamingBatchIterator,
    SyntheticBatchIterator,
    TokenBufferBatchIterator,
    build_train_dataloader,
)
from onebit.qat.distill import DistillationLoss
from onebit.qat.modules import (
    FP_META_BITS,
    TERNARY_BITS,
    BinaryLinear,
    QuantizedEmbedding,
    QuantizedLinearInt,
    QuantizedModule,
    TernaryLinear,
    TiedQuantizedLMHead,
)
from onebit.qat.ste import clamp_round_ste, lsq_grad_scale, round_ste, sign_ste
from onebit.qat.trainer import (
    TrainConfig,
    build_cosine_scheduler,
    load_checkpoint,
    run_kd_qat_training,
    save_checkpoint,
    train_from_pretrained,
)

__all__ = [
    "FP_META_BITS",
    "TERNARY_BITS",
    "QuantizedModule",
    "TernaryLinear",
    "BinaryLinear",
    "QuantizedEmbedding",
    "TiedQuantizedLMHead",
    "QuantizedLinearInt",
    "sign_ste",
    "round_ste",
    "clamp_round_ste",
    "lsq_grad_scale",
    "QATConfig",
    "convert_model_to_qat",
    "build_bit_report",
    "parameter_groups",
    "unwrap_model",
    "copy_qat_weights_to_fp_model",
    "DistillationLoss",
    "build_train_dataloader",
    "PackedStreamingBatchIterator",
    "TokenBufferBatchIterator",
    "SyntheticBatchIterator",
    "TrainConfig",
    "build_cosine_scheduler",
    "save_checkpoint",
    "load_checkpoint",
    "run_kd_qat_training",
    "train_from_pretrained",
]
