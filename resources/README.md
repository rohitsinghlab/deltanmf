# Foundation Model Resources

This folder contains gene similarity matrices derived from foundation model embeddings.

## Setup

### Transcriptformer

1. Download embeddings from https://github.com/czi-ai/transcriptformer
   - Get `homo_sapiens_gene.h5` from their releases

2. Run the conversion script:
   ```bash
   python create_transformer_similarity_matrix_transcriptformer.py
   ```

3. Outputs (saved to `transcriptformer/`):
   - `S_E_relu.npy` - gene similarity matrix
   - `genes_order.json` - gene name list

### scGPT

1. Download embeddings from https://github.com/bowang-lab/scGPT
   - Get gene embeddings from their model weights

2. Run the conversion script:
   ```bash
   python create_transformer_similarity_matrix_scgpt.py
   ```

3. Outputs (saved to `scgpt/`):
   - `S_E_relu.npy` - gene similarity matrix  
   - `genes_order.json` - gene name list

## Directory Structure

```
resources/
├── README.md
├── scgpt/
│   ├── S_E_relu.npy
│   └── genes_order.json
└── transcriptformer/
    ├── S_E_relu.npy
    └── genes_order.json
```

## Usage

Point DeltaNMF to either foundation model:

```python
S_E_PATH = "resources/scgpt/S_E_relu.npy"
S_E_GENES_PATH = "resources/scgpt/genes_order.json"
```