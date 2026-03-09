"""
02_preprocessing.py
Vorverarbeitung der NIR-Spektren: Mitteln, SNV.
Erstellt Plots mit Subplots für die Verarbeitungsschritte,
gefärbt nach Probenklasse und nach Hesperidin-Gehalt.
"""

import json
import logging
import numpy as np
import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

INTERACTIVE = False

BASE_DIR = Path(__file__).parent
PLOT_DIR = BASE_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "02_preprocessing.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def load_data(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def mean_spectra(spectra, sample_ids):
    """Mittelt Wiederholungsmessungen pro Probe."""
    unique_ids = sorted(set(sample_ids), key=lambda x: (not x.isdigit(), x))
    mean_specs = []
    for sid in unique_ids:
        indices = [i for i, s in enumerate(sample_ids) if s == sid]
        mean_spec = np.mean([spectra[i] for i in indices], axis=0)
        mean_specs.append(mean_spec)
    return np.array(mean_specs), unique_ids


def snv(spectra):
    """Standard Normal Variate Transformation."""
    result = np.zeros_like(spectra, dtype=float)
    for i in range(spectra.shape[0]):
        mean = np.mean(spectra[i])
        std = np.std(spectra[i])
        if std > 0:
            result[i] = (spectra[i] - mean) / std
        else:
            result[i] = spectra[i] - mean
    return result


def plot_preprocessing_steps(wavelengths, raw, averaged, snv_data,
                             sample_ids, hesperidin, dataset_name, color_by="sample"):
    """Erstellt Subplot-Figur mit 2 Verarbeitungsschritten."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"{dataset_name} – Vorverarbeitung"
                 f" (gefärbt nach {'Probe' if color_by == 'sample' else 'Hesperidin-Gehalt'})",
                 fontsize=14, fontweight="bold")

    steps = [
        ("Rohspektren (gemittelt)", averaged),
        ("SNV", snv_data),
    ]

    # Color setup
    if color_by == "sample":
        unique_ids = sorted(set(sample_ids), key=lambda x: (not x.isdigit(), x))
        cmap = plt.get_cmap("tab20", len(unique_ids))
        colors = {sid: cmap(i) for i, sid in enumerate(unique_ids)}
        spec_colors = [colors[sid] for sid in sample_ids]
    else:
        hesp_arr = np.array(hesperidin)
        norm = plt.Normalize(hesp_arr.min(), hesp_arr.max())
        cmap = plt.get_cmap("viridis")
        spec_colors = [cmap(norm(h)) for h in hesperidin]

    x_label = "Wellenzahl (cm⁻¹)" if "TANGO" in dataset_name else "Wellenlänge (nm)"

    for ax, (title, data) in zip(axes, steps):
        for i in range(data.shape[0]):
            ax.plot(wavelengths, data[i], color=spec_colors[i], alpha=0.7, linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Intensität")
        ax.grid(True, alpha=0.3)

    # Legend / Colorbar
    if color_by == "sample":
        handles = [plt.Line2D([0], [0], color=colors[sid], label=f"Probe {sid}")
                   for sid in sorted(set(sample_ids), key=lambda x: (not x.isdigit(), x))]
        fig.legend(handles=handles, loc="center right", bbox_to_anchor=(1.12, 0.5),
                   fontsize=8, title="Proben")
    else:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.02])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("Hesperidin-Gehalt (%)")

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    suffix = "by_sample" if color_by == "sample" else "by_concentration"
    fname = f"{dataset_name.lower().replace(' ', '_')}_preprocessing_{suffix}.png"
    fig.savefig(PLOT_DIR / fname, dpi=150, bbox_inches="tight")
    logger.info(f"Plot gespeichert: {fname}")
    if INTERACTIVE:
        plt.show()
    else:
        plt.close(fig)


def process_dataset(data, dataset_name):
    """Verarbeitet einen Datensatz komplett."""
    wavelengths = np.array(data["wavelengths"])
    spectra = np.array(data["spectra"])
    sample_ids = data["sample_ids"]
    hesperidin = data["hesperidin_content"]

    logger.info(f"\n{'='*60}")
    logger.info(f"Verarbeite: {dataset_name}")
    logger.info(f"{'='*60}")

    # Mitteln
    averaged, unique_ids = mean_spectra(spectra, sample_ids)
    logger.info(f"Gemittelt: {averaged.shape[0]} Proben x {averaged.shape[1]} Wellenlängen")

    # Zugehörige Hesperidin-Werte (gemittelt pro Probe)
    hesp_map = {}
    for sid, h in zip(sample_ids, hesperidin):
        hesp_map[sid] = h
    avg_hesperidin = [hesp_map[sid] for sid in unique_ids]

    # SNV
    snv_data = snv(averaged)
    logger.info(f"SNV angewendet: shape={snv_data.shape}")

    # Plots
    plot_preprocessing_steps(wavelengths, spectra, averaged, snv_data,
                             unique_ids, avg_hesperidin, dataset_name, color_by="sample")
    plot_preprocessing_steps(wavelengths, spectra, averaged, snv_data,
                             unique_ids, avg_hesperidin, dataset_name, color_by="concentration")

    # Speichere vorverarbeitete Daten
    preprocessed = {
        "wavelengths": wavelengths.tolist(),
        "averaged_spectra": averaged.tolist(),
        "snv_spectra": snv_data.tolist(),
        "sample_ids": unique_ids,
        "hesperidin_content": avg_hesperidin,
    }
    out_path = BASE_DIR / f"{dataset_name.lower().replace(' ', '_')}_preprocessed.json"
    with open(out_path, "w") as f:
        json.dump(preprocessed, f, indent=2)
    logger.info(f"Vorverarbeitete Daten gespeichert: {out_path}")

    return preprocessed


def main():
    tango_data = load_data(BASE_DIR / "tango_data.json")
    neo_data = load_data(BASE_DIR / "neospectra_data.json")

    tango_prep = process_dataset(tango_data, "TANGO")
    neo_prep = process_dataset(neo_data, "NeoSpectra")

    return tango_prep, neo_prep


if __name__ == "__main__":
    main()
