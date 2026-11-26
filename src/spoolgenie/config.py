"""Cross-platform configuration, printer presets, and Bambu AMS filament palettes."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True)
class PrinterProfile:
    id: str
    name: str
    bed_x_mm: float
    bed_y_mm: float
    bed_z_mm: float
    ams_slots: int
    nozzle_mm: float = 0.4
    default_layer_height_mm: float = 0.2


PRINTER_PROFILES: dict[str, PrinterProfile] = {
    "bambu_p1s": PrinterProfile("bambu_p1s", "Bambu Lab P1S / P1P", 256.0, 256.0, 256.0, 4),
    "bambu_x1c": PrinterProfile("bambu_x1c", "Bambu Lab X1-Carbon", 256.0, 256.0, 256.0, 4),
    "bambu_a1": PrinterProfile("bambu_a1", "Bambu Lab A1 (AMS lite)", 256.0, 256.0, 256.0, 4),
    "bambu_a1_mini": PrinterProfile("bambu_a1_mini", "Bambu Lab A1 mini (AMS lite)", 180.0, 180.0, 180.0, 4),
    "bambu_h2d": PrinterProfile("bambu_h2d", "Bambu Lab H2D", 325.0, 320.0, 325.0, 8),
    "prusa_mk4s": PrinterProfile("prusa_mk4s", "Prusa MK4S (MMU3)", 250.0, 210.0, 220.0, 5),
    "generic_fdm": PrinterProfile("generic_fdm", "Generic FDM (220x220x250)", 220.0, 220.0, 250.0, 1),
}


@dataclass(frozen=True)
class BambuOfficialFilament:
    name: str
    name_zh: str
    hex_color: str
    series: str  # "PLA Matte", "PLA Basic", "PLA Silk"
    store_url_cn: str
    store_url_global: str


_URL_MATTE_CN = "https://store.bambulab.cn/products/pla-matte"
_URL_MATTE_GL = "https://store.bambulab.com/products/pla-matte"
_URL_BASIC_CN = "https://store.bambulab.cn/products/pla-basic-filament"
_URL_BASIC_GL = "https://store.bambulab.com/products/pla-basic-filament"
_URL_SILK_CN = "https://store.bambulab.cn/products/pla-silk"
_URL_SILK_GL = "https://store.bambulab.com/products/pla-silk"

BAMBU_OFFICIAL_FILAMENTS: tuple[BambuOfficialFilament, ...] = (
    BambuOfficialFilament("Matte Charcoal", "哑光炭黑", "#2B2D30", "PLA Matte", _URL_MATTE_CN, _URL_MATTE_GL),
    BambuOfficialFilament("Matte Ivory White", "哑光象牙白", "#F2EFE9", "PLA Matte", _URL_MATTE_CN, _URL_MATTE_GL),
    BambuOfficialFilament("Matte Mandarin Orange", "哑光橘红", "#F36C21", "PLA Matte", _URL_MATTE_CN, _URL_MATTE_GL),
    BambuOfficialFilament("Matte Sakura Pink", "哑光樱花粉", "#F2B6C1", "PLA Matte", _URL_MATTE_CN, _URL_MATTE_GL),
    BambuOfficialFilament("Matte Ice Blue", "哑光冰蓝", "#8BBEE8", "PLA Matte", _URL_MATTE_CN, _URL_MATTE_GL),
    BambuOfficialFilament("Matte Marine Blue", "哑光深海蓝", "#1E4C8A", "PLA Matte", _URL_MATTE_CN, _URL_MATTE_GL),
    BambuOfficialFilament("Matte Grass Green", "哑光草绿", "#4D9A4E", "PLA Matte", _URL_MATTE_CN, _URL_MATTE_GL),
    BambuOfficialFilament("Matte Lemon Yellow", "哑光柠檬黄", "#F4D03F", "PLA Matte", _URL_MATTE_CN, _URL_MATTE_GL),
    BambuOfficialFilament("Basic Sunflower Yellow", "基础向日葵黄", "#F0B418", "PLA Basic", _URL_BASIC_CN, _URL_BASIC_GL),
    BambuOfficialFilament("Matte Scarlet Red", "哑光猩红", "#C9302C", "PLA Matte", _URL_MATTE_CN, _URL_MATTE_GL),
    BambuOfficialFilament("Matte Ash Gray", "哑光浅灰", "#8D9297", "PLA Matte", _URL_MATTE_CN, _URL_MATTE_GL),
    BambuOfficialFilament("Matte Latte Brown", "哑光拿铁棕", "#9C7A54", "PLA Matte", _URL_MATTE_CN, _URL_MATTE_GL),
    BambuOfficialFilament("Matte Desert Tan", "哑光沙漠米", "#DFC096", "PLA Matte", _URL_MATTE_CN, _URL_MATTE_GL),
    BambuOfficialFilament("Matte Lilac Purple", "哑光丁香紫", "#A28BC7", "PLA Matte", _URL_MATTE_CN, _URL_MATTE_GL),
    BambuOfficialFilament("Silk Gold", "丝绸金", "#D4AF37", "PLA Silk", _URL_SILK_CN, _URL_SILK_GL),
    BambuOfficialFilament("Silk Silver", "丝绸银", "#C0C5CE", "PLA Silk", _URL_SILK_CN, _URL_SILK_GL),
    BambuOfficialFilament("Basic Jade White", "基础玉石白", "#FFFFFF", "PLA Basic", _URL_BASIC_CN, _URL_BASIC_GL),
    BambuOfficialFilament("Basic Black", "基础纯黑", "#18181A", "PLA Basic", _URL_BASIC_CN, _URL_BASIC_GL),
)

BAMBU_PLA_SWATCHES: dict[str, str] = {f.name: f.hex_color for f in BAMBU_OFFICIAL_FILAMENTS}


class SpoolGenieSettings(BaseSettings):
    """Runtime settings read from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_prefix="SPOOLGENIE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Workspace & server
    workspace_dir: Path = Path("outputs")
    host: str = "127.0.0.1"
    port: int = 8765
    default_printer: str = "bambu_p1s"

    # LLM / VLM backend ("auto", "ollama", "openai", "fallback")
    llm_provider: str = "auto"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "auto"
    ollama_vlm_model: str = "auto"

    # Optional OpenAI-compatible API (works with OpenAI, Gemini OpenAI-compat, DeepSeek, OpenRouter)
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_vlm_model: str = "gpt-4o-mini"

    # 2D concept generation ("auto", "diffusers", "openai_image", "studio")
    image_backend: str = "auto"
    diffusers_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    image_steps: int = 22

    # Organic 3D generation ("auto", "hunyuan3d", "http_3d", "tripo", "meshy", "relief")
    mesh_backend: str = "auto"
    hunyuan3d_repo: str = "tencent/Hunyuan3D-2"
    hunyuan3d_subfolder: str = "hunyuan3d-dit-v2-0"
    local_3d_server_url: str = ""
    tripo_api_key: str = ""
    meshy_api_key: str = ""

    # Slicer path override (auto-discovered if empty)
    slicer_path: str = ""


def get_settings() -> SpoolGenieSettings:
    settings = SpoolGenieSettings()
    # Respect standard OPENAI_API_KEY if SPOOLGENIE_OPENAI_API_KEY is unset
    if not settings.openai_api_key and os.getenv("OPENAI_API_KEY"):
        settings.openai_api_key = os.environ["OPENAI_API_KEY"]
    settings.workspace_dir.mkdir(parents=True, exist_ok=True)
    return settings
