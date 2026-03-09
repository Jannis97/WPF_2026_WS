"""
04_dimensionality_reduction.py
Dimensionsreduktion: PCA, UMAP, t-SNE für beide Datensätze.
Je Methode 2 Plots: gefärbt nach Probenklasse und nach Hesperidin-Gehalt.
"""

import json
import logging
import numpy as np
import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

INTERACTIVE = False
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap
from pathlib import Path

BASE_DIR = Path(__file__).parent
PLOT_DIR = BASE_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "04_dimensionality_reduction.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def load_preprocessed(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def plot_embedding(embedding, sample_ids, hesperidin, method_name, dataset_name, color_by="sample"):
    """Plottet 2D-Embedding."""
    fig, ax = plt.subplots(figsize=(9, 7))

    if color_by == "sample":
        unique_ids = sorted(set(sample_ids), key=lambda x: (not x.isdigit(), x))
        cmap = plt.get_cmap("tab20", len(unique_ids))
        colors = {sid: cmap(i) for i, sid in enumerate(unique_ids)}

        for sid in unique_ids:
            mask = [s == sid for s in sample_ids]
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       color=colors[sid], label=f"Probe {sid}", s=80,
                       edgecolors="k", linewidth=0.5)

        ax.legend(fontsize=8, loc="best", title="Proben")
        title_suffix = "nach Probe"
    else:
        hesp_arr = np.array(hesperidin)
        norm = plt.Normalize(hesp_arr.min(), hesp_arr.max())
        cmap_c = plt.get_cmap("viridis")
        sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                        c=hesperidin, cmap=cmap_c, s=80,
                        edgecolors="k", linewidth=0.5)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label("Hesperidin-Gehalt (%)")
        title_suffix = "nach Gehalt"

    ax.set_xlabel(f"{method_name} Dimension 1")
    ax.set_ylabel(f"{method_name} Dimension 2")
    ax.set_title(f"{dataset_name} – {method_name} ({title_suffix})", fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    suffix = "by_sample" if color_by == "sample" else "by_concentration"
    fname = f"{dataset_name.lower().replace(' ', '_')}_{method_name.lower()}_{suffix}.png"
    fig.savefig(PLOT_DIR / fname, dpi=150, bbox_inches="tight")
    logger.info(f"Plot gespeichert: {fname}")
    if INTERACTIVE:
        plt.show()
    else:
        plt.close(fig)


def run_pca(X, sample_ids, hesperidin, dataset_name):
    """PCA Dimensionsreduktion."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    logger.info(f"{dataset_name} PCA: Erklärte Varianz = "
                f"{pca.explained_variance_ratio_[0]:.3f}, {pca.explained_variance_ratio_[1]:.3f} "
                f"(Total: {sum(pca.explained_variance_ratio_):.3f})")

    plot_embedding(X_pca, sample_ids, hesperidin, "PCA", dataset_name, "sample")
    plot_embedding(X_pca, sample_ids, hesperidin, "PCA", dataset_name, "concentration")
    return X_pca


def run_tsne(X, sample_ids, hesperidin, dataset_name):
    """t-SNE Dimensionsreduktion."""
    perplexity = min(30, X.shape[0] - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    X_tsne = tsne.fit_transform(X)
    logger.info(f"{dataset_name} t-SNE: KL-Divergenz = {tsne.kl_divergence_:.4f}")

    plot_embedding(X_tsne, sample_ids, hesperidin, "tSNE", dataset_name, "sample")
    plot_embedding(X_tsne, sample_ids, hesperidin, "tSNE", dataset_name, "concentration")
    return X_tsne


def run_umap(X, sample_ids, hesperidin, dataset_name):
    """UMAP Dimensionsreduktion."""
    n_neighbors = min(15, X.shape[0] - 1)
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X)
    logger.info(f"{dataset_name} UMAP: Embedding shape = {X_umap.shape}")

    plot_embedding(X_umap, sample_ids, hesperidin, "UMAP", dataset_name, "sample")
    plot_embedding(X_umap, sample_ids, hesperidin, "UMAP", dataset_name, "concentration")
    return X_umap


def process_dataset(data, dataset_name):
    """Führt alle Dimensionsreduktionen durch."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Dimensionsreduktion: {dataset_name}")
    logger.info(f"{'='*60}")

    X = np.array(data["snv_spectra"])
    sample_ids = data["sample_ids"]
    hesperidin = data["hesperidin_content"]

    # Standardisierung
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info(f"X shape: {X_scaled.shape}")

    run_pca(X_scaled, sample_ids, hesperidin, dataset_name)
    run_umap(X_scaled, sample_ids, hesperidin, dataset_name)
    run_tsne(X_scaled, sample_ids, hesperidin, dataset_name)


def main():
    tango_prep = load_preprocessed(BASE_DIR / "tango_preprocessed.json")
    neo_prep = load_preprocessed(BASE_DIR / "neospectra_preprocessed.json")

    process_dataset(tango_prep, "TANGO")
    process_dataset(neo_prep, "NeoSpectra")


if __name__ == "__main__":
    main()
