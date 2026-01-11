from pathlib import Path
import sys
import os
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# --- Resolve repo paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
GENOMIC_FM_DIR = REPO_ROOT / "external" / "genomic-FM"

# --- Make genomic-FM importable ---
sys.path.insert(0, str(GENOMIC_FM_DIR))

# --- Some loaders rely on relative paths ---
os.chdir(GENOMIC_FM_DIR)

# --- Import loader ---
from src.dataloader.data_wrapper import RealClinVar

# --- Load data ---
loader = RealClinVar(
    num_records=151,     # change number of records based on needs
    all_records=False
)

data = loader.get_data(Seq_length=1024)

print("Loaded data successfully")
print("Data Type:", type(data))
print("Number of Records:", len(data))

# --- Retrieve Sequences and Print Information ---
lc_sequences = [record[0][0] for record in data]
lc_labels = [record[1] for record in data]

print("Unique labels in dataset:", set(lc_labels))
average_length = sum(len(seq) for seq in lc_sequences) / len(lc_sequences)
print("Average sequence length:", average_length)
maximum_length = max(len(seq) for seq in lc_sequences)
print("Maximum sequence length:", maximum_length)

# --- Import Tokenizer and Model ---
tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", trust_remote_code=True)

# length of tokens which the input sequences are padded
max_length = (tokenizer.model_max_length // maximum_length) + 1

# tokenize
tokens_ids = tokenizer.batch_encode_plus(lc_sequences, return_tensors="pt", padding="max_length", max_length = max_length)["input_ids"]
print("Tokens shape:", tokens_ids.shape, "\n")

# attention masks
attention_mask = tokens_ids != tokenizer.pad_token_id
print("Attention Mask", attention_mask)

torch_outs = model(
    tokens_ids,
    attention_mask=attention_mask,  # prevents attention to padding tokens
    output_attentions = True,
    encoder_attention_mask=attention_mask,
    output_hidden_states=True       # to get all layer embeddings
)



