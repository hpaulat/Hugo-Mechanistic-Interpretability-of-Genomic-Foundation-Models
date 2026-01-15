from __future__ import annotations

import argparse
from pathlib import Path
import sys
import os
import time
from typing import Dict, List

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM

seq_len_bp = 3000
layers = [1,9,18,22, 28]
batch_size = 2
max_samples = 20
head_mode = "mean"
model_name = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"

# --- Paths / Imports ---
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
GENOMIC_FM_DIR = REPO_ROOT / "external" / "genomic-FM"
DATA_DIR = GENOMIC_FM_DIR / "root" / "data"

print("\n[1/6] Setting up import paths...")
print(f"  - REPO_ROOT:      {REPO_ROOT}")
print(f"  - GENOMIC_FM_DIR: {GENOMIC_FM_DIR}")
print(f"  - DATA_DIR:       {DATA_DIR}")

sys.path.insert(0, str(GENOMIC_FM_DIR))
os.chdir(GENOMIC_FM_DIR)

from src.dataloader.data_wrapper import RealClinVar

# --- Load data ---
print("\n[2/6] Loading ClinVar data via RealClinVar...")
loader = RealClinVar(num_records=max_samples, all_records=False)
data = loader.get_data(Seq_length=seq_len_bp)

print("  - Loaded successfully.")
print(f"  - number of records: {len(data)} (variants)")

# --- Extract sequences + labels ---
print("\n[3/6] Extracting reference/alternative sequences and labels...")
ref_seqs = [r[0][0] for r in data]
alt_seqs = [r[0][1] for r in data]
labels = [r[1] for r in data]

all_sequences = ref_seqs + alt_seqs
n = len(ref_seqs)

unique_labels = sorted(set(labels))
print(f"  - unique labels: {unique_labels}")
print(f"  - sequences: {len(all_sequences)} (REF+ALT = {n}+{n})")

# --- Load tokenizer/model ---
print("\n[4/6] Loading tokenizer + model...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

print(f"  - model: {model_name}")
print(f"  - device: {device}")
print(f"  - tokenizer.model_max_length: {tokenizer.model_max_length}")

# --- Tokenize ---
print("\n[5/6] Tokenizing sequences...")
t0 = time.time()

tok_max_len = min(int(seq_len_bp / 6 + 5), int(tokenizer.model_max_length))

enc = tokenizer(
    all_sequences,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=tok_max_len,
)

input_ids = enc["input_ids"]
attention_mask = enc["attention_mask"]

print(f"  - token_max_length: {input_ids.shape[1]}")
print(f"  - input_ids shape: {tuple(input_ids.shape)}")
print(f"  - attention_mask shape: {tuple(attention_mask.shape)}")
print(f"  - tokenization done in {time.time() - t0:.2f}s")

# --- Attention extraction ---
print("\n[6/6] Running model inference and extracting attentions...")
t0 = time.time()

# outputs.attentions is a tuple length = num_layers
# Each element: (B, num_heads, L, L)
layers_str = ",".join(map(str, layers))
print(f"  - layers requested: {layers_str}")
print(f"  - head_mode: {head_mode}")
print(f"  - batch_size: {batch_size}")

# Store as lists, concatenate at end.
attn_store: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}

use_amp = (device.type == "cuda")

with torch.inference_mode():
    for i in range(0, input_ids.size(0), batch_size):
        batch_input_ids = input_ids[i:i + batch_size].to(device)
        batch_mask = attention_mask[i:i + batch_size].to(device)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_mask,
                    output_attentions=True,
                    output_hidden_states=False,
                    return_dict=True,
                )
        else:
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_mask,
                output_attentions=True,
                output_hidden_states=False,
                return_dict=True,
            )

        # outputs.attentions: tuple[num_layers], each (B, H, L, L)
        for l in layers:
            a = outputs.attentions[l]  # (B, H, L, L)

            if head_mode == "mean":
                a = a.mean(dim=1)  # (B, L, L)

            # move to CPU immediately
            attn_store[l].append(a.float().cpu())

        if (i // batch_size) % 5 == 0 and device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            print(f"  - batch {i//batch_size:04d} | processed {min(i+batch_size, input_ids.size(0))}/{input_ids.size(0)} "
                    f"| GPU allocated {allocated:.1f} GiB, reserved {reserved:.1f} GiB")

        # help fragmentation
        if device.type == "cuda":
            torch.cuda.empty_cache()

# Concatenate
attn_by_layer = {l: torch.cat(attn_store[l], dim=0) for l in layers}

# Shapes:
# mean mode: (2N, L, L)
# full mode: (2N, H, L, L)
example_layer = layers[0]
print(f"  - example saved attention shape (layer {example_layer}): {tuple(attn_by_layer[example_layer].shape)}")
print(f"  - done in {time.time() - t0:.2f}s")

# --- Save ---
print("\nSaving attentions to disk...")
DATA_DIR.mkdir(parents=True, exist_ok=True)

out_path = DATA_DIR / (f"clinvar_attentions__n{n}__bp{seq_len_bp}__tok{input_ids.shape[1]}__layers{len(layers)}__heads_{head_mode}.pt")

payload = {
    "model_name": model_name,
    "bp_window_len": seq_len_bp,
    "token_max_length": int(input_ids.shape[1]),
    "layers": layers,
    "head_mode": head_mode,
    "labels": labels,
    "ref_sequences": ref_seqs,
    "alt_sequences": alt_seqs,
    "attentions_by_layer": attn_by_layer,
    # helpful metadata
    "note": "Attentions are HF attentions[layer] with shape (B,H,L,L). If head_mode=='mean', heads are averaged -> (B,L,L). Order is REF then ALT.",
}

torch.save(payload, out_path)
print(f"  - saved to: {out_path}")
print("Done.\n")
