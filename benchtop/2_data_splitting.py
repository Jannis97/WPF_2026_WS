#!/usr/bin/env python3
"""
2_data_splitting.py – Gruppenbasierte Datenaufteilung fuer Kreuzvalidierung.

Alle Replikate einer Probe bilden eine Gruppe.
Keine Replikate duerfen zwischen Train und Test aufgeteilt werden.

Klassen:
  - LeaveOneGroupOut: alle Replikate einer Probe als Test
  - GroupKFold: K-Fold mit Gruppen-Respekt

Erzeugt:
  - plots/splitting_schemes.png
  - plots/splitting_cv_folds_*.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from abc import ABC, abstractmethod
from importlib.util import spec_from_file_location, module_from_spec

BENCHTOP_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(BENCHTOP_DIR, "plots")
DPI = 300


def _load_dataloading():
    spec = spec_from_file_location("dataloading", os.path.join(BENCHTOP_DIR, "0_dataloading.py"))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─── Splitter-Klassen ────────────────────────────────────────

class DataSplitter(ABC):
    """Basisklasse fuer gruppenbasierte Datenaufteilung."""

    @abstractmethod
    def split(self, X, y, groups):
        """Yield (train_idx, test_idx) Tupel.

        Args:
            X: Feature-Matrix (n_samples, n_features)
            y: Zielwerte (n_samples,)
            groups: Gruppen-Labels (n_samples,) – z.B. Proben-IDs
        """
        pass

    @abstractmethod
    def name(self):
        """Menschenlesbarer Name."""
        pass

    def n_splits(self, X, y, groups):
        """Anzahl der Splits."""
        return len(list(self.split(X, y, groups)))


class LeaveOneGroupOut(DataSplitter):
    """Leave-One-Group-Out: alle Replikate einer Probe als Test."""

    def split(self, X, y, groups):
        unique_groups = sorted(set(groups),
                               key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 0, x))
        groups_arr = np.array(groups)
        for g in unique_groups:
            test_mask = groups_arr == g
            train_mask = ~test_mask
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

    def name(self):
        return "GroupLeaveOneOut"


class GroupKFold(DataSplitter):
    """K-Fold CV mit Gruppen-Respekt (keine Replicate-Leakage)."""

    def __init__(self, n_splits=4):
        self._n_splits = n_splits

    def split(self, X, y, groups):
        unique_groups = sorted(set(groups),
                               key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 0, x))
        n_groups = len(unique_groups)
        k = min(self._n_splits, n_groups)

        # Gruppen den Folds zuordnen
        fold_assignment = {}
        for i, g in enumerate(unique_groups):
            fold_assignment[g] = i % k

        groups_arr = np.array(groups)
        for fold in range(k):
            test_groups = [g for g, f in fold_assignment.items() if f == fold]
            test_mask = np.isin(groups_arr, test_groups)
            train_idx = np.where(~test_mask)[0]
            test_idx = np.where(test_mask)[0]
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

    def name(self):
        return f"GroupKFold(k={self._n_splits})"


# ─── Plot-Funktionen ─────────────────────────────────────────

def plot_splitting_scheme(sample_ids, splitters, save_path=None):
    """Uebersicht der Splitting-Schemata als Heatmap."""
    groups = np.array(sample_ids)
    unique_groups = sorted(set(sample_ids),
                           key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 0, x))
    n_groups = len(unique_groups)

    n_splitters = len(splitters)
    fig, axes = plt.subplots(1, n_splitters, figsize=(8 * n_splitters, max(6, n_groups * 0.4)))
    if n_splitters == 1:
        axes = [axes]

    dummy_X = np.zeros((len(sample_ids), 1))
    dummy_y = np.zeros(len(sample_ids))

    for ax, splitter in zip(axes, splitters):
        splits = list(splitter.split(dummy_X, dummy_y, sample_ids))
        n_folds = len(splits)

        matrix = np.zeros((n_groups, n_folds))
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            test_groups_in_fold = set(groups[test_idx])
            for g_idx, g in enumerate(unique_groups):
                if g in test_groups_in_fold:
                    matrix[g_idx, fold_idx] = 1

        cmap_split = plt.cm.RdYlGn_r
        ax.imshow(matrix, cmap=cmap_split, aspect='auto', interpolation='nearest',
                  vmin=0, vmax=1)
        ax.set_yticks(range(n_groups))
        ax.set_yticklabels(unique_groups, fontsize=8)

        if n_folds <= 20:
            ax.set_xticks(range(n_folds))
            ax.set_xticklabels([f"F{i+1}" for i in range(n_folds)],
                               fontsize=7, rotation=45, ha="right")
        else:
            ax.set_xlabel("Fold")

        ax.set_title(splitter.name(), fontsize=11)
        ax.set_ylabel("Probe")

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=cmap_split(0.0), label='Train'),
                           Patch(facecolor=cmap_split(1.0), label='Test')]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"  -> {save_path}")
    plt.close(fig)


def plot_cv_folds_spectra(wavenumbers, sample_ids, spectra, splitter,
                          conc_map, type_map, max_folds=6, save_path=None):
    """Spektren fuer CV-Folds: Train (blau/farbig) und Test (rot)."""
    dl = _load_dataloading()
    dummy_y = np.zeros(len(sample_ids))
    splits = list(splitter.split(spectra, dummy_y, sample_ids))

    n_folds = min(len(splits), max_folds)
    fig, axes = plt.subplots(n_folds, 2, figsize=(16, 4 * n_folds))
    if n_folds == 1:
        axes = axes.reshape(1, 2)

    for fold_idx in range(n_folds):
        train_idx, test_idx = splits[fold_idx]
        ax1, ax2 = axes[fold_idx]

        test_groups = sorted(set(np.array(sample_ids)[test_idx]))
        fold_label = f"Fold {fold_idx+1} (Test: {', '.join(test_groups)})"

        # Links: Konzentration
        colors_conc, norm, cmap = dl._get_colors_by_concentration(sample_ids, conc_map)
        for i in train_idx:
            ax1.plot(wavenumbers, spectra[i], color=colors_conc[i], alpha=0.2, lw=0.3)
        for i in test_idx:
            ax1.plot(wavenumbers, spectra[i], color='red', alpha=0.7, lw=1.0)
        ax1.set_title(f"{fold_label} \u2013 Konzentration")
        ax1.set_xlabel("Wellenzahl [cm$^{-1}$]")
        ax1.invert_xaxis()

        # Rechts: Klasse
        colors_class, _ = dl._get_colors_by_probe_class(sample_ids, type_map)
        for i in train_idx:
            ax2.plot(wavenumbers, spectra[i], color=colors_class[i], alpha=0.2, lw=0.3)
        for i in test_idx:
            ax2.plot(wavenumbers, spectra[i], color='red', alpha=0.7, lw=1.0)

        test_h = plt.Line2D([0], [0], color='red', lw=2, label='Test')
        train_h = plt.Line2D([0], [0], color='gray', lw=2, alpha=0.3, label='Train')
        ax2.legend(handles=[train_h, test_h], fontsize=8, loc='upper right')
        ax2.set_title(f"{fold_label} \u2013 Probenklasse")
        ax2.set_xlabel("Wellenzahl [cm$^{-1}$]")
        ax2.invert_xaxis()

    plt.suptitle(f"CV-Folds: {splitter.name()}", fontsize=14, y=1.01)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"  -> {save_path}")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────

def main():
    print("=== 2_data_splitting ===")
    os.makedirs(PLOT_DIR, exist_ok=True)
    dl = _load_dataloading()

    wavenumbers, sample_ids, spectra = dl.load_spectra()
    hplc_df = dl.load_hplc()
    conc_map = dl.get_concentration_map(hplc_df)
    type_map = dl.get_sample_type_map(hplc_df)

    # Nur Proben mit HPLC-Referenz
    known_ids = set(conc_map.keys())
    mask = [sid in known_ids for sid in sample_ids]
    sample_ids_f = [sid for sid, m in zip(sample_ids, mask) if m]
    spectra_f = spectra[mask]

    print(f"Proben mit Referenz: {len(set(sample_ids_f))} "
          f"unique ({len(sample_ids_f)} Spektren)")

    splitters = [LeaveOneGroupOut(), GroupKFold(n_splits=4)]

    print("Plotting Splitting-Schemata...")
    plot_splitting_scheme(sample_ids_f, splitters,
                          save_path=os.path.join(PLOT_DIR, "splitting_schemes.png"))

    for splitter in splitters:
        sname = splitter.name().replace("(", "_").replace(")", "").replace("=", "")
        print(f"Plotting CV-Folds fuer {splitter.name()}...")
        plot_cv_folds_spectra(wavenumbers, sample_ids_f, spectra_f,
                              splitter, conc_map, type_map, max_folds=4,
                              save_path=os.path.join(PLOT_DIR, f"splitting_cv_folds_{sname}.png"))

    print("Done.")


if __name__ == "__main__":
    main()
