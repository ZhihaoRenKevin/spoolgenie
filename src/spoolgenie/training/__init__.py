"""Local RTX GPU LoRA fine-tuning (character 2D styles) & MakerWorld CAD Coder SFT dataset/trainer tools."""

from spoolgenie.training.trainer import (
    export_cad_sft_dataset,
    prepare_character_lora_dataset,
)

__all__ = ["prepare_character_lora_dataset", "export_cad_sft_dataset"]
