# DeltaNMF: A Two-Stage Neural NMF for Differential Gene Program Discovery
A lightweight Python implementation of (1) single-stage neural NMF and (2) a two-stage case-control NMF that separates baseline (control) programs from case-specific programs, with optional foundation model–informed gene-similarity regularization.

---

## Overview
DeltaNMF is a neural-network reformulation of non-negative matrix factorization (NMF) designed for single-cell transcriptomics. It supports:

- **One-stage (control-only) factorization** to learn a baseline program dictionary
- **Two-stage (case-control) factorization** that learns baseline programs on controls, then learns additional case-specific programs on case cells while keeping baseline programs active
- **Optional gene-similarity regularization** via a precomputed gene–gene similarity matrix derived from transformer gene embeddings (e.g., scGPT or TranscriptFormer)

## Key Features
- **GPU acceleration:** consensus initialization + PyTorch optimization (when CUDA is available)
- **Interpretable gene programs:** returns `W` (genes × programs) and `H` (programs × cells)
- **Foundation-model regularization:** add a Laplacian penalty using precomputed gene–gene similarity (`S_E`) from scGPT or TranscriptFormer embeddings
- **Two-stage case-control modeling:** explicitly separates *shifted usage of shared programs* from *emergence of novel case-specific programs*


## Input
DeltaNMF expects expression matrices in **genes × cells** orientation:

- `X_control` / `X_ntc`: shape `(n_genes, n_control_cells)`
- `X_case` / `X_spec`: shape `(n_genes, n_case_cells)` (two-stage only)
- `gene_names`: list/array of length `n_genes` matching the rows of `X_*`

If you start from an `.h5ad`, `deltanmf/io.py::h5ad_to_npy(...)` will:
- split cells by `adata.obs[condition_key]`
- extract `adata.X` or `adata.layers[layer]`
- convert to dense (if sparse)
- transpose to **genes × cells**

---
## Quickstart: One-Stage (Control-Only)
Edit `scripts/run_onestage.py` and set:
- `h5ad_path`
- `out`
- `ntc_key`, `condition_key`
- resource choice (`scgpt` vs `transcriptformer`)
- `K`

Then run:
```bash
python scripts/run_onestage.py
```

If you're calling the API directly, stage-1 NTC fitting can run in either mode:
- full-batch (default): `use_minibatch_ntc=False`
- minibatch over cells: `use_minibatch_ntc=True` and `minibatch_size_ntc=40960` (or your preferred size)

## Quickstart: Two-Stage (Case-Control)
Edit `scripts/run_twostage.py` and set:
- `h5ad_path`
- `out`
- `ntc_key`, `condition_key`, and optionally `case_key`
- resource choice (`scgpt` vs `transcriptformer`)
- k values

Then run:
```bash
python scripts/run_twostage.py
```

Two-stage has the same stage-1 option:
- full-batch baseline fitting: `stage1_use_minibatch_ntc=False`
- minibatch baseline fitting: `stage1_use_minibatch_ntc=True` and `stage1_minibatch_size_ntc=40960`

**Running without foundation-model regularization** (`use_fm`, default `True`):

Both `run_onestage_deltanmf` and `run_twostage_deltanmf` accept a `use_fm` flag. When `use_fm=False`, the FM Laplacian term is disabled entirely: `S_E` is never loaded, the gene set is not intersected with `S_E`'s gene list, and the solver runs plain (regularizer-free, aside from non-negativity) NMF / two-stage case-control NMF. Use this when you want a pure-data baseline, lack a precomputed `S_E`, or want to compare against the FM-regularized variant.

- One-stage: `run_onestage_deltanmf(..., use_fm=False)` — `S_E_PATH`/`S_E_GENES_PATH` may be omitted.
- Two-stage: `run_twostage_deltanmf(..., use_fm=False)` — `S_E_PATH`/`S_E_GENES_PATH` may be omitted; `stage1_rel_alpha`, `stage2_rel_alpha`, `stage2_rel_gamma` are ignored.

Default (`use_fm=True`) preserves the existing behavior: `S_E` is loaded, genes are intersected with `S_E`'s gene list, and FM regularization is applied according to the `rel_alpha`/`rel_gamma` parameters.

**Stage-2 memory mode** (`stage2_use_hybrid_memory`, default `False`):

By default, stage 2 uses the standard GPU solver (`solve_specific_with_fixed_ntc`), which loads the entire `X_spec` matrix into VRAM upfront. This is simple and fast but will OOM if `X_spec` is too large to fit alongside the model parameters.

For large case matrices, set `stage2_use_hybrid_memory=True` to use a memory-aware GPU solver (`solve_specific_with_fixed_ntc_hybrid_fast`) that adapts to available VRAM:
1. **Tier 1** — keeps `X_spec` on GPU (like the default solver) but auto-tunes the batch size based on remaining VRAM after model allocation
2. **Tier 2** — if VRAM is insufficient (< 40 % free or OOM), keeps `X_spec` on CPU and streams batches to GPU via pinned memory

The solver prints VRAM diagnostics at startup so you can see which tier was selected. Use this when your case matrix is large enough that the default solver OOMs.


## Resources: scGPT and TranscriptFormer Gene Similarity

DeltaNMF’s optional regularization uses a gene–gene similarity matrix `S_E`, aligned to your dataset’s genes via a companion gene-order JSON.

The runner scripts expect:
- `resources/<model>/S_E_relu.npy`
- `resources/<model>/genes_order.json`

### Generate scGPT and TranscriptFormer resources
You can generate scGPT or TranscriptFormer-derived files by running:
- `resources/transcriptformer/create_transformer_similarity_matrix_transcriptformer.py`
- `resources/scgpt/create_transformer_similarity_matrix_scgpt.py`

After generation, place outputs at:

- `resources/{}/S_E_relu.npy`
- `resources/{}/genes_order.json`


## Notes

- **Gene alignment matters:** `S_E` must correspond to the same gene naming convention as `gene_names` (the code aligns and filters genes; mismatches reduce coverage). Use ENSEMBL IDs
- **Memory:** `S_E` is dense and can be large. Use `float32` and store on disk; load only when needed.


## Citation

If you use this code, please cite:

Karpurapu, A., Gersbach, C. A., & Singh, R. (2026). **DeltaNMF: A Two-Stage Neural NMF for Differential Gene Program Discovery**. *bioRxiv*. https://doi.org/10.64898/2026.01.22.701049

```bibtex
@article{karpurapu2026deltanmf,
  title = {DeltaNMF: A Two-Stage Neural NMF for Differential Gene Program Discovery},
  author = {Karpurapu, Anish and Gersbach, Charles A. and Singh, Rohit},
  journal = {bioRxiv},
  year = {2026},
  doi = {10.64898/2026.01.22.701049},
  url = {https://www.biorxiv.org/content/10.64898/2026.01.22.701049v1}
}
```

## License

CC BY NC SA 4.0

This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.


## Contact
For questions and issues, please open a GitHub issue or contact the maintainers.

