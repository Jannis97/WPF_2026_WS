#!/usr/bin/env python3
"""
1_data_preprocessing.py – Preprocessing der Benchtop-Spektren.

Schritte (konfigurierbar via params dict):
  1. SNV (Standard Normal Variate)
  2. Savitzky-Golay Glaettung / Ableitung

Keine Mittelung der Replikate – alle Einzelmessungen bleiben erhalten
fuer gruppenbasierte Kreuzvalidierung.

Erzeugt:
  - plots/preprocessing_<variant>.png          (Dual-Panel pro Variante)
  - plots/preprocessing_pipeline_overview.png  (Alle Schritte gestapelt)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import savgol_filter
from importlib.util import spec_from_file_location, module_from_spec

BENCHTOP_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(BENCHTOP_DIR, "plots")
DPI = 300


def _load_dataloading():
    spec = spec_from_file_location("dataloading", os.path.join(BENCHTOP_DIR, "0_dataloading.py"))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─── Preprocessing-Funktionen ────────────────────────────────

def snv(spectra):
    """Standard Normal Variate: zeilenweise Zentrierung und Skalierung."""
    means = spectra.mean(axis=1, keepdims=True)
    stds = spectra.std(axis=1, keepdims=True, ddof=0)
    stds[stds == 0] = 1.0
    return (spectra - means) / stds


def apply_savgol(spectra, window_length=15, polyorder=2, deriv=1):
    """Savitzky-Golay Filter auf jedes Spektrum."""
    wl = window_length
    if wl > spectra.shape[1]:
        wl = spectra.shape[1] if spectra.shape[1] % 2 == 1 else spectra.shape[1] - 1
    if wl % 2 == 0:
        wl += 1
    return savgol_filter(spectra, window_length=wl, polyorder=polyorder,
                         deriv=deriv, axis=1)


def preprocess(spectra, params=None):
    """Fuehrt die Preprocessing-Pipeline aus.

    Args:
        spectra: np.array (n_samples, n_features)
        params: dict mit keys:
            - snv: bool (default True)
            - savgol: bool (default False)
            - savgol_window: int (default 15)
            - savgol_poly: int (default 2)
            - savgol_deriv: int (default 1)

    Returns:
        preprocessed: np.array gleiche Shape
        steps: list of str (Namen der angewandten Schritte)
    """
    if params is None:
        params = {}

    result = spectra.copy()
    steps = ["raw"]

    if params.get("snv", True):
        result = snv(result)
        steps.append("SNV")

    if params.get("savgol", False):
        window = params.get("savgol_window", 15)
        poly = params.get("savgol_poly", 2)
        deriv = params.get("savgol_deriv", 1)
        result = apply_savgol(result, window_length=window, polyorder=poly, deriv=deriv)
        steps.append(f"SG(w={window},p={poly},d={deriv})")

    return result, steps


# ─── Plot-Funktionen ─────────────────────────────────────────

def plot_spectra_dual(wavenumbers, sample_ids, spectra, conc_map, type_map,
                      title_suffix="", save_path=None):
    """Dual-Panel Spektren: links Konzentration, rechts Probenklasse."""
    dl = _load_dataloading()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Links: nach Konzentration
    colors_conc, norm, cmap = dl._get_colors_by_concentration(sample_ids, conc_map)
    for i in range(spectra.shape[0]):
        ax1.plot(wavenumbers, spectra[i], color=colors_conc[i], alpha=0.4, lw=0.5)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax1, label="Hesperidin [%]")
    ax1.set_xlabel("Wellenzahl [cm$^{-1}$]")
    ax1.set_ylabel("Intensitaet")
    ax1.set_title(f"Nach Konzentration{title_suffix}")
    ax1.invert_xaxis()

    # Rechts: nach Probenklasse
    colors_class, legend_handles = dl._get_colors_by_probe_class(sample_ids, type_map)
    for i in range(spectra.shape[0]):
        ax2.plot(wavenumbers, spectra[i], color=colors_class[i], alpha=0.4, lw=0.5)
    ax2.legend(handles=legend_handles, fontsize=6, loc="upper right", ncol=2)
    ax2.set_xlabel("Wellenzahl [cm$^{-1}$]")
    ax2.set_ylabel("Intensitaet")
    ax2.set_title(f"Nach Probenklasse{title_suffix}")
    ax2.invert_xaxis()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"  -> {save_path}")
    plt.close(fig)


def plot_pipeline_overview(wavenumbers, sample_ids, spectra, conc_map, type_map,
                           params=None, save_path=None):
    """Uebersichtsplot: alle Preprocessing-Schritte untereinander, je dual-panel."""
    if params is None:
        params = {}
    dl = _load_dataloading()

    # Collect stages
    stages = [("Rohspektren", spectra.copy())]
    current = spectra.copy()

    if params.get("snv", True):
        current = snv(current)
        stages.append(("SNV", current.copy()))

    if params.get("savgol", False):
        window = params.get("savgol_window", 15)
        poly = params.get("savgol_poly", 2)
        deriv = params.get("savgol_deriv", 1)
        current = apply_savgol(current, window_length=window, polyorder=poly, deriv=deriv)
        stages.append((f"SG(w={window},d={deriv})", current.copy()))

    n_stages = len(stages)
    fig, axes = plt.subplots(n_stages, 2, figsize=(16, 5 * n_stages))
    if n_stages == 1:
        axes = axes.reshape(1, 2)

    for row_idx, (name, data) in enumerate(stages):
        ax1, ax2 = axes[row_idx]

        colors_conc, norm, cmap = dl._get_colors_by_concentration(sample_ids, conc_map)
        for i in range(data.shape[0]):
            ax1.plot(wavenumbers, data[i], color=colors_conc[i], alpha=0.4, lw=0.5)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax1, label="Hesperidin [%]")
        ax1.set_title(f"{name} \u2013 Konzentration")
        ax1.set_xlabel("Wellenzahl [cm$^{-1}$]")
        ax1.invert_xaxis()

        colors_class, legend_handles = dl._get_colors_by_probe_class(sample_ids, type_map)
        for i in range(data.shape[0]):
            ax2.plot(wavenumbers, data[i], color=colors_class[i], alpha=0.4, lw=0.5)
        ax2.legend(handles=legend_handles, fontsize=5, loc="upper right", ncol=2)
        ax2.set_title(f"{name} \u2013 Probenklasse")
        ax2.set_xlabel("Wellenzahl [cm$^{-1}$]")
        ax2.invert_xaxis()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"  -> {save_path}")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────

def main():
    print("=== 1_data_preprocessing ===")
    os.makedirs(PLOT_DIR, exist_ok=True)
    dl = _load_dataloading()

    wavenumbers, sample_ids, spectra = dl.load_spectra()
    hplc_df = dl.load_hplc()
    conc_map = dl.get_concentration_map(hplc_df)
    type_map = dl.get_sample_type_map(hplc_df)

    # Preprocessing-Varianten
    variants = {
        "SNV": {"snv": True, "savgol": False},
        "SNV_SG1d": {"snv": True, "savgol": True, "savgol_window": 15,
                     "savgol_poly": 2, "savgol_deriv": 1},
        "SG1d": {"snv": False, "savgol": True, "savgol_window": 15,
                 "savgol_poly": 2, "savgol_deriv": 1},
    }

    for vname, params in variants.items():
        print(f"\nPreprocessing: {vname}")
        processed, steps = preprocess(spectra, params)
        print(f"  Schritte: {' -> '.join(steps)}")
        print(f"  Shape: {processed.shape}")

        plot_spectra_dual(wavenumbers, sample_ids, processed, conc_map, type_map,
                          title_suffix=f" \u2013 {vname}",
                          save_path=os.path.join(PLOT_DIR, f"preprocessing_{vname}.png"))

    # Pipeline-Uebersicht fuer SNV+SG1d
    print("\nPipeline-Uebersicht (SNV+SG1d)...")
    plot_pipeline_overview(wavenumbers, sample_ids, spectra, conc_map, type_map,
                           params=variants["SNV_SG1d"],
                           save_path=os.path.join(PLOT_DIR, "preprocessing_pipeline_overview.png"))

    print("Done.")


if __name__ == "__main__":
    main()
