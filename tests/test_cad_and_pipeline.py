"""Tests for build123d AST sandbox, CAD engine, KB/training tools, and full SpoolGenieSession pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from spoolgenie.agent.cad_coder import UnsafeCADCodeError, validate_cad_code_ast
from spoolgenie.config import SpoolGenieSettings
from spoolgenie.knowledge.character_catalog import CHARACTER_CATALOG, match_character_catalog
from spoolgenie.knowledge.kb import PrintKnowledgeBase
from spoolgenie.llm.client import UnifiedLLMClient
from spoolgenie.pipeline import SpoolGenieSession
from spoolgenie.training.trainer import export_cad_sft_dataset


def test_cad_ast_sandbox_blocks_unsafe_code() -> None:
    with pytest.raises(UnsafeCADCodeError):
        validate_cad_code_ast("import os\nos.system('echo hi')")
    with pytest.raises(UnsafeCADCodeError):
        validate_cad_code_ast("from build123d import *\nopen('/etc/passwd')")
    validate_cad_code_ast("from build123d import *\nPARTS = [{'name':'a','slot':1,'solid':Box(10,10,10)}]")


def test_full_e2e_no_ams_headphone_stand_pipeline(tmp_path: Path) -> None:
    settings = SpoolGenieSettings(workspace_dir=tmp_path, llm_provider="fallback", image_backend="studio")
    session = SpoolGenieSession(project_id="test_stand", settings=settings, llm=UnifiedLLMClient(settings))

    res = session.chat("做一个桌面耳机支架，带底部理线槽，高210mm，哑光黑和橘红双色，我没有AMS，用燕尾槽免胶水拆件打印")
    assert res["concept"] is not None
    assert Path(res["concept"].image_path).exists()
    assert session.state.spec.ams_units == 0

    state = session.confirm_and_build(run_slicer=True)
    assert state.dfm is not None
    assert state.dfm.watertight is True
    assert state.dfm.effective_multicolor_mode == "split_no_ams"
    assert state.dfm.joint_type_used == "dovetail"
    assert Path(state.exports.threemf_path).exists()
    assert Path(state.exports.split_plates_3mf_path).exists()
    assert Path(state.exports.stl_path).exists()
    assert Path(state.exports.step_path).exists()
    assert Path(state.exports.preview_glb_path).exists()
    assert len(state.exports.per_color_stls) >= 2


def test_full_e2e_organic_character_ams_pipeline(tmp_path: Path) -> None:
    settings = SpoolGenieSettings(workspace_dir=tmp_path, llm_provider="fallback", image_backend="studio", mesh_backend="relief")
    session = SpoolGenieSession(project_id="test_chibi", settings=settings, llm=UnifiedLLMClient(settings))

    session.chat("做一个可爱Q版动漫猫咪手办，高100mm，1台AMS，黑、白、橙三色")
    state = session.confirm_and_build(run_slicer=False)
    assert state.dfm is not None
    assert state.dfm.watertight is True
    assert state.dfm.ams_slots_used == 3
    assert Path(state.exports.threemf_path).exists()
    assert Path(state.exports.preview_glb_path).exists()


def test_character_catalog_covers_movies_anime_animals_and_plants() -> None:
    franchises = {p.franchise for p in CHARACTER_CATALOG.values()}
    assert {
        "Marvel / DC",
        "Disney / Pixar",
        "Universal / Sci-Fi",
        "Anime / Game",
        "Animals",
        "Plants & Decor",
        "Space & Classics",
        "Vehicles & Machines",
        "Food & Cafe",
        "Music & Gaming",
        "Myth & Culture",
        "Seasonal & Gifts",
    } <= franchises
    assert len(CHARACTER_CATALOG) >= 170
    assert match_character_catalog("做一个漫威钢铁侠Q版摆件").id == "marvel_ironman"
    assert match_character_catalog("迪士尼史迪奇手办").id == "disney_stitch"
    assert match_character_catalog("环球小黄人摆件").id == "universal_minion"
    assert match_character_catalog("皮卡丘Q版摆件").id == "anime_pikachu"
    assert match_character_catalog("做一个头顶橘子的卡皮巴拉水豚").id == "animal_capybara"
    assert match_character_catalog("做一个小熊猫摆件").id == "animal_red_panda"
    assert match_character_catalog("做一个多肉盆栽花盆").id == "plant_succulent"
    assert match_character_catalog("做一个龟背竹杯垫").id == "plant_monstera"
    assert match_character_catalog("做一个宇航员摆件").id == "decor_astronaut"
    assert match_character_catalog("做一个黑神话悟空手办").id == "game_wukong"
    assert match_character_catalog("做一个赛博皮卡摆件").id == "veh_cybertruck"
    assert match_character_catalog("做一个珍珠奶茶杯").id == "food_boba"
    assert match_character_catalog("做一个国潮醒狮摆件").id == "myth_lion_dance"
    assert match_character_catalog("做一个圣诞树").id == "season_xmas_tree"
    # Every prompt_2d stays compact (<=55 words) so SDXL CLIP never truncates "no text, no letters"
    for p in CHARACTER_CATALOG.values():
        assert len(p.prompt_2d.split()) <= 55


def test_knowledge_base_and_cad_sft_export(tmp_path: Path) -> None:
    kb = PrintKnowledgeBase(tmp_path / "kb.jsonl")
    hits = kb.retrieve("gridfinity 收纳盒 organizer", domain="functional_makerworld", top_k=2)
    assert hits and "gridfinity" in hits[0].id

    sft = export_cad_sft_dataset(tmp_path / "sft_out", workspace_dir=tmp_path, kb=kb)
    assert sft["sample_count"] >= 8
    assert Path(sft["dataset_jsonl"]).exists()
    assert Path(sft["qlora_script"]).exists()
    assert Path(sft["ollama_modelfile"]).exists()
