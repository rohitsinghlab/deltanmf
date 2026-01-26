from pathlib import Path
import numpy as np

from deltanmf.api import run_onestage_deltanmf
from deltanmf.io import h5ad_to_npy


def main():
    h5ad_path = Path("/path/to/data.h5ad")
    out = Path("/path/to/outdir")
    out.mkdir(parents=True, exist_ok=True)

    X_control, _, gene_names = h5ad_to_npy(
        h5ad_path,
        ntc_key="NTC",
        condition_key="condition",
        case_key=None,
        layer=None,
    )

    pkg_dir = Path(__file__).parent.parent
    resources = pkg_dir / "resources"
    S_E_PATH = resources / "scgpt" / "S_E_relu.npy" # can replace scgpt with transcriptformer
    S_E_GENES_PATH = resources / "scgpt" / "genes_order.json" # can replace scgpt with transcriptformer

    if not S_E_PATH.exists():
        raise FileNotFoundError(f"Missing: {S_E_PATH}")
    if not S_E_GENES_PATH.exists():
        raise FileNotFoundError(f"Missing: {S_E_GENES_PATH}")

    res = run_onestage_deltanmf(
        X_control, gene_names, S_E_PATH, S_E_GENES_PATH,
        K=30,
        MIN_CELLS=215,
        REMOVE_GENES=["ENSG00000164796"],
        rel_alpha=0.0,
        max_iter=10000,
        FM_NONNEG="softplus",
        FM_SOFTPLUS_BETA=5.0,
        lr=0.01,
    )

    np.save(out / "W.npy", res["W"])
    np.save(out / "H.npy", res["H"])


if __name__ == "__main__":
    main()
