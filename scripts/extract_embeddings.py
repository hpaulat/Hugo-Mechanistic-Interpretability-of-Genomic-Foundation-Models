from pathlib import Path
import sys
import os
import time

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# --- Paths / Imports ---
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
GENOMIC_FM_DIR = REPO_ROOT / "external" / "genomic-FM"
DATA_DIR = GENOMIC_FM_DIR / "root" / "data"  # where verified_real_clinvar.csv likely lives

print("\n[1/6] Setting up import paths...")
print(f"  - REPO_ROOT:     {REPO_ROOT}")
print(f"  - GENOMIC_FM_DIR:{GENOMIC_FM_DIR}")
print(f"  - DATA_DIR:      {DATA_DIR}")

sys.path.insert(0, str(GENOMIC_FM_DIR))
os.chdir(GENOMIC_FM_DIR)


from src.dataloader.data_wrapper import RealClinVar

# load data
Seq_length = 3000   # length of sequences to extract from ClinVar records

print("\n[2/6] Loading ClinVar data via RealClinVar...")
loader = RealClinVar(num_records=151, all_records=False)    # change number of records based on needs
data = loader.get_data(Seq_length=Seq_length)

print("  - Loaded successfully.")
print(f"  - data type: {type(data)}")
print(f"  - number of records: {len(data)}")

# retrieve sequences and print information
print("\n[3/6] Extracting Reference/Alternative sequences and labels...")
lc_reference_sequences = [record[0][0] for record in data]
lc_alternative_sequences = [record[0][1] for record in data]
lc_labels = [record[1] for record in data]

unique_labels = sorted(set(lc_labels))
print(f"  - unique labels: {unique_labels}")
print(f"  - number of reference sequences: {len(lc_reference_sequences)}")
print(f"  - number of alternative sequences: {len(lc_alternative_sequences)}")

# import tokenizer and model
print("\n[4/6] Loading tokenizer + model...")
MODEL_NAME = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

print(f"  - model: {MODEL_NAME}")
print(f"  - device: {device}")
print(f"  - tokenizer.model_max_length: {tokenizer.model_max_length}")

# tokenize
print("\n[5/6] Tokenizing sequences...")
t0 = time.time()

all_sequences = lc_reference_sequences + lc_alternative_sequences
n = len(lc_reference_sequences)
max_length = Seq_length/6 + 5
max_length = min(int(max_length), tokenizer.model_max_length)

enc = tokenizer(
    all_sequences,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=max_length,
)

input_ids = enc["input_ids"]           # (2N, L)
attention_mask = enc["attention_mask"]   # (2N, L)

print(f"  - total sequences tokenized (REF+ALT): {len(all_sequences)} = {2*n}")
print(f"  - input_ids shape: {tuple(input_ids.shape)}")
print(f"  - attention_mask shape: {tuple(attention_mask.shape)}")
print(f"  - tokenization done in {time.time() - t0:.2f}s")


# Inference and Pooling
print("\n[6/6] Running model inference and computing pooled embeddings...")
t0 = time.time()

batch_size = 2

layers = [1, 3, 5, 9, 12, 15, 18, 22, 25, 28]  # 0..28 indexing
layer_to_embeds = {l: [] for l in layers}

use_amp = (device.type == "cuda"
           )
with torch.inference_mode():
    for i in range(0, input_ids.size(0), batch_size):
        batch_input_ids = input_ids[i:i+batch_size].to(device)
        batch_mask = attention_mask[i:i+batch_size].to(device)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_mask,
                    output_hidden_states=True,
                    output_attentions=False,
                    return_dict=True,
                )
        else:
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_mask,
                output_hidden_states=True,
                output_attentions=False,
                return_dict=True,
            )

        mask = batch_mask.unsqueeze(-1).to(dtype = outputs.hidden_states[0].dtype)   # (B, L, 1)
        denom = mask.sum(dim=1).clamp(min=1e-9)                             # (B, 1)

        for l in layers:
            h = outputs.hidden_states[l]                 # (B, L, D)
            pooled = (h * mask).sum(dim=1) / denom       # (B, D)
            layer_to_embeds[l].append(pooled.float().cpu())

        if (i // batch_size) % 10 == 0:
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            print(f"  - batch {i//batch_size:04d} | processed {i+len(batch_input_ids)}/{input_ids.size(0)} "
                  f"| GPU allocated {allocated:.1f} GiB, reserved {reserved:.1f} GiB")
            
layer_to_seq = {l: torch.cat(v, dim=0) for l, v in layer_to_embeds.items()}  # each (2N, 1024)
seq_embeddings_by_layer = {l: layer_to_seq[l] for l in layers}  # each (2N, 1024)

print("\nSaving embeddings to disk...")
DATA_DIR.mkdir(parents=True, exist_ok=True)

payload = {
    "model_name": MODEL_NAME,
    "bp_window_len": Seq_length,
    "token_max_length": int(input_ids.shape[1]),
    "pooling": "masked_mean",
    "layers": layers,
    "labels": lc_labels,
    "embeddings_by_layer": seq_embeddings_by_layer,  # (2N, 1024) per layer
}
save_path = DATA_DIR / f"clinvar_pooled_embeddings__n{n}__bp{Seq_length}__tok{input_ids.shape[1]}__layers10.pt"
torch.save(payload, save_path)


print(f"  - saved to: {save_path}")
print(f"  - done in {time.time() - t0:.2f}s")