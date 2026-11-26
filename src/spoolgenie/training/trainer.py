"""Domain-adaptation & local fine-tuning toolkit for SpoolGenie.

Supports two open-source-clean training pipelines runnable locally on RTX 5090 / 4090 / 3090 (or cloud GPU):
  1. Character / Figurine 2D Style LoRA:
     - Auto-crops user-provided reference images + runs local Ollama VLM (`qwen2.5vl:7b`) to generate
       FDM-printable 3D-toy captions (`metadata.jsonl`).
     - Emits a ready-to-run Diffusers + PEFT `bf16` LoRA training script (`train_character_lora.py`)
       whose `.safetensors` output hot-loads straight into `spoolgenie.engines.concept_2d`.
  2. MakerWorld Functional CAD Coder SFT (QLoRA -> Ollama):
     - Exports the MakerWorld Functional Knowledge Bank + all user-verified `build123d` builds into
       instruction-tuning JSONL (`cad_sft_dataset.jsonl`).
     - Emits a TRL + PEFT QLoRA script (`train_cad_coder_qlora.py`) + Ollama `Modelfile` so the
       fine-tuned CAD specialist runs locally at zero API cost!
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

from PIL import Image

from spoolgenie.agent.cad_coder import generate_template_cad_code
from spoolgenie.knowledge.kb import PrintKnowledgeBase
from spoolgenie.llm.client import UnifiedLLMClient
from spoolgenie.schemas import DesignSpec, DimensionsMM, FilamentSlot


def prepare_character_lora_dataset(
    images_dir: str | Path,
    output_dir: str | Path,
    trigger_token: str = "pf_chibi_print",
    style_hint: str = "stylized 3D-printable matte PLA collectible figurine on pure white background",
    resolution: int = 768,
    llm: UnifiedLLMClient | None = None,
) -> dict[str, str | int]:
    """Prepare square-cropped images + VLM-captioned `metadata.jsonl` + RTX 5090 LoRA training script."""
    src = Path(images_dir)
    out = Path(output_dir)
    img_out = out / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    valid_exts = {".png", ".jpg", ".jpeg", ".webp"}
    files = [p for p in sorted(src.iterdir()) if p.is_file() and p.suffix.lower() in valid_exts] if src.exists() else []

    records: list[dict[str, str]] = []
    for idx, img_path in enumerate(files, start=1):
        img = Image.open(img_path).convert("RGB")
        # Pad cleanly onto pure white square canvas (preserves full character silhouette for 3D printing!)
        side = max(img.width, img.height)
        square = Image.new("RGB", (side, side), (255, 255, 255))
        square.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
        square = square.resize((resolution, resolution), Image.Resampling.LANCZOS)

        out_name = f"char_{idx:04d}.png"
        saved_path = img_out / out_name
        square.save(saved_path, format="PNG")

        caption = f"{trigger_token}, {style_hint}, {img_path.stem.replace('_', ' ').replace('-', ' ')}"
        if llm and llm.available:
            vlm_cap = llm.analyze_image(
                saved_path,
                (
                    "Describe this character / figurine in one concise English sentence for 3D-print concept training: "
                    "mention silhouette, pose, clothing/armor color blocks, and expression. No camera jargon."
                ),
            )
            if vlm_cap:
                caption = f"{trigger_token}, {vlm_cap.strip()}, {style_hint}"

        records.append({"file_name": f"images/{out_name}", "text": caption})

    meta_path = out / "metadata.jsonl"
    with open(meta_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write a clean, self-contained RTX 5090 (bf16) / CUDA / MPS Diffusers PEFT LoRA training script
    train_script = out / "train_character_lora.py"
    train_script.write_text(
        dedent(
            f"""
            \"\"\"Local 2D Character Style LoRA Trainer (Diffusers + PEFT, bf16 RTX 5090 / CUDA ready).

            Install training extras first:
              uv pip install diffusers transformers accelerate peft datasets bitsandbytes

            Run:
              python {train_script.name} --steps 600 --rank 16
            \"\"\"
            import argparse
            import json
            from pathlib import Path

            import torch
            import torch.nn.functional as F
            from PIL import Image
            from torch.utils.data import DataLoader, Dataset
            from torchvision import transforms


            class CaptionImageDataset(Dataset):
                def __init__(self, root: Path, size: int = {resolution}):
                    self.root = root
                    self.items = [json.loads(l) for l in (root / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
                    self.tfm = transforms.Compose([
                        transforms.Resize((size, size)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),
                    ])

                def __len__(self):
                    return len(self.items)

                def __getitem__(self, idx):
                    item = self.items[idx]
                    img = Image.open(self.root / item["file_name"]).convert("RGB")
                    return self.tfm(img), item["text"]


            def main():
                parser = argparse.ArgumentParser()
                parser.add_argument("--base-model", default="stabilityai/sdxl-turbo")
                parser.add_argument("--steps", type=int, default=500)
                parser.add_argument("--rank", type=int, default=16)
                parser.add_argument("--lr", type=float, default=1e-4)
                parser.add_argument("--output-dir", default="lora_weights")
                args = parser.parse_args()

                from diffusers import AutoPipelineForText2Image, DDPMScheduler
                from peft import LoraConfig

                device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
                dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
                print(f"Training Character LoRA on {{device}} ({{dtype}}) from {{args.base_model}} ...")

                pipe = AutoPipelineForText2Image.from_pretrained(args.base_model, torch_dtype=dtype)
                unet = pipe.unet.to(device)
                vae = pipe.vae.to(device).eval()
                noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

                lora_cfg = LoraConfig(
                    r=args.rank,
                    lora_alpha=args.rank,
                    init_lora_weights="gaussian",
                    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                )
                unet.add_adapter(lora_cfg)
                trainable = [p for p in unet.parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW(trainable, lr=args.lr)

                ds = CaptionImageDataset(Path(__file__).parent)
                if len(ds) == 0:
                    raise RuntimeError("No images found in metadata.jsonl! Add reference images first.")
                loader = DataLoader(ds, batch_size=1, shuffle=True)

                step = 0
                while step < args.steps:
                    for pixels, captions in loader:
                        if step >= args.steps:
                            break
                        pixels = pixels.to(device=device, dtype=dtype)
                        with torch.no_grad():
                            latents = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
                            prompt_embeds, _, pooled, _ = pipe.encode_prompt(
                                prompt=list(captions),
                                device=device,
                                num_images_per_prompt=1,
                                do_classifier_free_guidance=False,
                            )
                            add_time_ids = torch.tensor([[{resolution}, {resolution}, 0, 0, {resolution}, {resolution}]], device=device, dtype=dtype)

                        noise = torch.randn_like(latents)
                        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
                        noisy = noise_scheduler.add_noise(latents, noise, timesteps)

                        pred = unet(
                            noisy,
                            timesteps,
                            encoder_hidden_states=prompt_embeds,
                            added_cond_kwargs={{"text_embeds": pooled, "time_ids": add_time_ids}},
                        ).sample
                        loss = F.mse_loss(pred.float(), noise.float(), reduction="mean")
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                        step += 1
                        if step % 25 == 0 or step == args.steps:
                            print(f"Step {{step}}/{{args.steps}} | loss={{loss.item():.4f}}")

                out_dir = Path(__file__).parent / args.output_dir
                out_dir.mkdir(parents=True, exist_ok=True)
                unet.save_attn_procs(out_dir)
                print(f"Saved Character LoRA weights to {{out_dir}} — load via SpoolGenie --lora {{out_dir}}")


            if __name__ == "__main__":
                main()
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    return {
        "dataset_dir": str(out.resolve()),
        "metadata_jsonl": str(meta_path.resolve()),
        "train_script": str(train_script.resolve()),
        "image_count": len(records),
        "trigger_token": trigger_token,
    }


def export_cad_sft_dataset(
    output_dir: str | Path,
    workspace_dir: str | Path = "outputs",
    kb: PrintKnowledgeBase | None = None,
) -> dict[str, str | int]:
    """Export MakerWorld functional priors + parametric templates + verified user builds into QLoRA SFT JSONL + Ollama Modelfile."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    kb = kb or PrintKnowledgeBase()

    samples: list[dict[str, object]] = []

    # 1. Synthesize clean multi-size, multi-color parametric build123d training pairs across functional categories
    seed_scenarios = [
        ("headphone_stand", "Desktop Headphone Stand with Rear Cable Slot", (115, 105, 220), 2),
        ("headphone_stand", "Compact A1-mini Headphone Stand", (95, 90, 170), 3),
        ("phone_stand", "Angled Desk Phone Dock with USB-C Pass-Through", (82, 95, 115), 2),
        ("desk_organizer", "Tiered 3-Compartment Desk Organizer", (150, 100, 85), 3),
        ("cable_clip", "5-Slot Matte Black & Orange Desk Cable Comb", (78, 28, 20), 2),
        ("enclosure", "Ventilated ESP32 / Sensor Enclosure with M3 Standoffs", (110, 75, 42), 2),
        ("coaster_sign", "Hexagonal 3-Color Inlay Geometric Coaster", (96, 96, 6), 3),
        ("planter_vase", "12-Facet Geometric Desk Planter with Saucer", (105, 105, 130), 3),
    ]
    palette_pool = [
        FilamentSlot(slot=1, name="Matte Charcoal", hex_color="#2B2D30", role="main body"),
        FilamentSlot(slot=2, name="Matte Mandarin Orange", hex_color="#F36C21", role="accent"),
        FilamentSlot(slot=3, name="Matte Ivory White", hex_color="#F2EFE9", role="detail"),
    ]

    for cat, title, (w, d, h), n_cols in seed_scenarios:
        spec = DesignSpec(
            title=title,
            category=cat,
            dimensions=DimensionsMM(width_mm=w, depth_mm=d, height_mm=h),
            filaments=palette_pool[:n_cols],
        )
        code = generate_template_cad_code(spec)
        kb_hits = kb.retrieve(f"{cat} {title}", domain="functional_makerworld", top_k=2)
        rules_str = "; ".join(r for e in kb_hits for r in e.design_rules[:2])
        samples.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are SpoolGenie-CAD, a build123d functional 3D printing engineer trained on MakerWorld design tolerances and Bambu AMS multi-part rules.",
                    },
                    {
                        "role": "user",
                        "content": f"Design a {title} ({w}x{d}x{h} mm, {n_cols} filament slots). Engineering rules: {rules_str}",
                    },
                    {"role": "assistant", "content": f"```python\n{code}\n```"},
                ]
            }
        )

    # 2. Harvest any verified user CAD projects from workspace_dir
    ws = Path(workspace_dir)
    if ws.exists():
        for state_file in ws.glob("*/project.json"):
            try:
                data = json.loads(state_file.read_text(encoding="utf-8"))
                cad_code = data.get("cad_code", "").strip()
                dfm = data.get("dfm") or {}
                if cad_code and dfm.get("watertight"):
                    sp = data.get("spec") or {}
                    samples.append(
                        {
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "You are SpoolGenie-CAD, a build123d functional 3D printing engineer.",
                                },
                                {
                                    "role": "user",
                                    "content": f"Generate verified build123d script for: {sp.get('title', 'Custom Part')} — {sp.get('summary', '')}",
                                },
                                {"role": "assistant", "content": f"```python\n{cad_code}\n```"},
                            ]
                        }
                    )
            except Exception:
                continue

    dataset_path = out / "cad_sft_dataset.jsonl"
    with open(dataset_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Emit ready-to-run QLoRA trainer (TRL + PEFT, RTX 5090 24GB ready) + Ollama Modelfile
    qlora_script = out / "train_cad_coder_qlora.py"
    qlora_script.write_text(
        dedent(
            """
            \"\"\"Fine-tune Qwen2.5-Coder-7B-Instruct on MakerWorld + SpoolGenie build123d SFT pairs (RTX 5090 24GB bf16).

            Install optional fine-tuning deps:
              uv pip install transformers peft trl datasets bitsandbytes accelerate

            Run:
              python train_cad_coder_qlora.py
            \"\"\"
            import argparse
            from pathlib import Path
            import torch


            def main():
                parser = argparse.ArgumentParser()
                parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
                parser.add_argument("--epochs", type=int, default=3)
                parser.add_argument("--output-dir", default="spoolgenie_cad_qlora")
                args = parser.parse_args()

                from datasets import load_dataset
                from peft import LoraConfig
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
                from trl import SFTConfig, SFTTrainer

                ds = load_dataset("json", data_files=str(Path(__file__).parent / "cad_sft_dataset.jsonl"), split="train")
                bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
                tokenizer = AutoTokenizer.from_pretrained(args.base_model)
                model = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=bnb, device_map="auto")

                peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
                cfg = SFTConfig(
                    output_dir=str(Path(__file__).parent / args.output_dir),
                    num_train_epochs=args.epochs,
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=4,
                    learning_rate=2e-4,
                    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                    logging_steps=5,
                )
                trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tokenizer, peft_config=peft_cfg)
                trainer.train()
                trainer.save_model()
                print("Finished MakerWorld CAD Coder QLoRA fine-tuning!")


            if __name__ == "__main__":
                main()
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    modelfile = out / "Modelfile.spoolgenie-cad"
    modelfile.write_text(
        dedent(
            """
            # Instant zero-training Ollama specialist (or point FROM to your merged GGUF after QLoRA!)
            FROM qwen2.5:14b
            PARAMETER temperature 0.15
            SYSTEM \"\"\"You are SpoolGenie-CAD, an expert 3D printing mechanical engineer specializing in build123d Python CAD, MakerWorld functional standards (Gridfinity 42mm, Skadis 40mm, M3 clearances), Bambu Lab AMS multicolor parts, and glue-free mechanical joints (dovetails, compliant snap-pins, press tenons with 0.22mm clearance). Always output valid, watertight build123d scripts defining PARTS = [{"name": ..., "slot": 1..N, "solid": ...}].\"\"\"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    return {
        "dataset_jsonl": str(dataset_path.resolve()),
        "qlora_script": str(qlora_script.resolve()),
        "ollama_modelfile": str(modelfile.resolve()),
        "sample_count": len(samples),
    }
