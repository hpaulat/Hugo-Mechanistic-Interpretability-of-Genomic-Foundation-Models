# Tokenizes DNA sequences for further processing.

# import sequences and preprocess them into token IDs
from bowanglab.models.ntv2 import embed


sequences = [
    "ATTCCGAAATCGCTGACCGATCGTACGAAA",
    "ATTTCTCTCTCTCTCTGAGATCGATCGATCGATATCTCTCGAGCTAGC",
]

result = embed(sequences, model_name="250M_multi_species_v2", layer=20, max_positions=512, pooling="mean")