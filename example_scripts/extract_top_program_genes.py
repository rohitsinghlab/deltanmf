#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load_gene_names(run_dir: Path) -> np.ndarray:
    candidates = [
        run_dir / "gene_names_aligned.npy",
        run_dir / "gene_names_aligned.tsv",
        run_dir / "genes_aligned.txt",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".npy":
            genes = np.load(path, allow_pickle=True)
        else:
            genes = np.loadtxt(path, dtype=str, ndmin=1)
        genes = np.asarray(genes, dtype=object).reshape(-1)
        if genes.size == 0:
            raise ValueError(f"Gene name file is empty: {path}")
        return genes
    raise FileNotFoundError(
        "Could not find gene names. Expected one of: "
        "gene_names_aligned.npy, gene_names_aligned.tsv, genes_aligned.txt"
    )


def _load_program_matrices(run_dir: Path) -> tuple[np.ndarray, list[str]]:
    current_twostage = (run_dir / "W_stage1.npy", run_dir / "W_stage2.npy")
    legacy_twostage = (run_dir / "W_ntc_final.npy", run_dir / "W_spec_final.npy")
    onestage = run_dir / "W.npy"

    if current_twostage[0].exists() and current_twostage[1].exists():
        w_ntc = np.load(current_twostage[0])
        w_spec = np.load(current_twostage[1])
        matrix = np.hstack([w_ntc, w_spec])
        names = [f"NTC_{i+1}" for i in range(w_ntc.shape[1])]
        names += [f"Specific_{i+1}" for i in range(w_spec.shape[1])]
        return matrix, names

    if legacy_twostage[0].exists() and legacy_twostage[1].exists():
        w_ntc = np.load(legacy_twostage[0])
        w_spec = np.load(legacy_twostage[1])
        matrix = np.hstack([w_ntc, w_spec])
        names = [f"NTC_{i+1}" for i in range(w_ntc.shape[1])]
        names += [f"Specific_{i+1}" for i in range(w_spec.shape[1])]
        return matrix, names

    if onestage.exists():
        w = np.load(onestage)
        names = [f"Program_{i+1}" for i in range(w.shape[1])]
        return w, names

    raise FileNotFoundError(
        f"Could not find W matrices in {run_dir}. Expected one of: "
        "W.npy, (W_stage1.npy and W_stage2.npy), or "
        "(W_ntc_final.npy and W_spec_final.npy)."
    )


def _build_outputs(
    w_matrix: np.ndarray,
    gene_names: np.ndarray,
    program_names: list[str],
    n_top: int,
) -> pd.DataFrame:
    if w_matrix.ndim != 2:
        raise ValueError(f"W matrix must be 2D, got shape {w_matrix.shape}")
    if w_matrix.shape[0] != gene_names.shape[0]:
        raise ValueError(
            f"Gene count mismatch: W has {w_matrix.shape[0]} rows but "
            f"gene list has {gene_names.shape[0]} entries"
        )

    n_top = min(n_top, w_matrix.shape[0])
    long_rows: list[dict[str, object]] = []

    for j, program_name in enumerate(program_names):
        weights = w_matrix[:, j]
        order = np.argsort(weights)[::-1][:n_top]
        top_genes = gene_names[order]
        top_weights = weights[order]

        for rank, (gene, weight) in enumerate(zip(top_genes, top_weights), start=1):
            long_rows.append(
                {
                    "program": program_name,
                    "rank": rank,
                    "gene": str(gene),
                    "weight": float(weight),
                }
            )

    return pd.DataFrame(long_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract top genes per DeltaNMF program from saved W matrices."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Directory containing DeltaNMF outputs.",
    )
    parser.add_argument(
        "--n-top",
        type=int,
        default=300,
        help="Number of top genes to keep per program.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <run_dir>/top_program_genes.",
    )
    args = parser.parse_args()

    if args.n_top <= 0:
        raise ValueError("--n-top must be positive")

    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir is not None else run_dir / "top_program_genes"
    out_dir.mkdir(parents=True, exist_ok=True)

    gene_names = _load_gene_names(run_dir)
    w_matrix, program_names = _load_program_matrices(run_dir)

    long_df = _build_outputs(
        w_matrix=w_matrix,
        gene_names=gene_names,
        program_names=program_names,
        n_top=args.n_top,
    )

    long_path = out_dir / f"top_{args.n_top}_genes.tsv"

    long_df.to_csv(long_path, sep="\t", index=False)

    print(f"Loaded {len(program_names)} programs and {len(gene_names)} genes.")
    print(f"Saved top genes table to: {long_path}")


if __name__ == "__main__":
    main()
