#!/usr/bin/env python3
"""
0_dataloading.py – Benchtop (TANGO FT-NIR) Spektren und HPLC-Referenzdaten laden.

Erzeugt:
  - plots/raw_spectra_dual.png       (Rohspektren: Konzentration + Probenklasse)
  - plots/replicate_count.png        (Wiederholungsmessungen pro Probe)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
BENCHTOP_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(BENCHTOP_DIR, "plots")
DPI = 300


# ─── Daten laden ──────────────────────────────────────────────

def load_spectra(path=None):
    """Lädt TANGO Spektren aus Spektren.txt.

    Returns:
        wavenumbers: np.array (n_wn,)
        sample_ids: list[str] – eine ID pro Spektrum (Replikate haben gleiche ID)
        spectra: np.array (n_spectra, n_wn)
    """
    if path is None:
        path = os.path.join(DATA_DIR, "Spektren.txt")

    with open(path, "r") as f:
        lines = f.readlines()

    # Zeile 5 (Index 4) = Header mit Wellenzahlen
    header = lines[4].strip().split("\t")
    wavenumbers = np.array([float(x) for x in header[1:] if x.strip()])

    sample_ids = []
    spectra = []
    for line in lines[5:]:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        sample_ids.append(parts[0])
        vals = [float(x) for x in parts[1:] if x.strip()]
        spectra.append(vals[:len(wavenumbers)])

    spectra = np.array(spectra)
    return wavenumbers, sample_ids, spectra


def load_hplc(path=None):
    """Lädt HPLC-Referenzwerte aus .ods Datei.

    Returns:
        DataFrame mit Spalten: probe, probenummer, einwaage_mg, peakflaeche,
                                c_mg_ml, gehalt_pct
    """
    if path is None:
        path = os.path.join(DATA_DIR,
                            "WPF Mandarinen Hesperidin-Qauntifizierung mittels HPLC.ods")
    df_raw = pd.read_excel(path, engine="odf", sheet_name="Tabelle1", header=None)
    rows = []
    for i in range(34, len(df_raw)):
        row = df_raw.iloc[i]
        probe = row[1]
        if pd.isna(probe):
            continue
        try:
            probenummer = str(int(row[2]))
        except (ValueError, TypeError):
            continue
        rows.append({
            "probe": str(probe).strip(),
            "probenummer": probenummer,
            "einwaage_mg": float(row[3]) if not pd.isna(row[3]) else np.nan,
            "peakflaeche": float(row[4]) if not pd.isna(row[4]) else np.nan,
            "c_mg_ml": float(row[5]) if not pd.isna(row[5]) else np.nan,
            "gehalt_pct": float(row[6]) if not pd.isna(row[6]) else np.nan,
        })
    return pd.DataFrame(rows)


def get_concentration_map(hplc_df=None):
    """Erstellt Mapping probenummer -> mittlerer Gehalt (%).

    Returns:
        dict: {probenummer_str: gehalt_pct_mean}
    """
    if hplc_df is None:
        hplc_df = load_hplc()
    return hplc_df.groupby("probenummer")["gehalt_pct"].mean().to_dict()


def get_sample_type_map(hplc_df=None):
    """Erstellt Mapping probenummer -> Probentyp (M/G/Z/L).

    Returns:
        dict: {probenummer_str: 'M'/'G'/'Z'/'L'}
    """
    if hplc_df is None:
        hplc_df = load_hplc()
    type_map = {}
    for _, row in hplc_df.iterrows():
        probe = row["probe"]
        pnr = row["probenummer"]
        if probe.startswith("G"):
            type_map[pnr] = "G"
        elif probe.startswith("Z"):
            type_map[pnr] = "Z"
        elif probe.startswith("L"):
            type_map[pnr] = "L"
        else:
            type_map[pnr] = "M"
    return type_map


# ─── Farb-Hilfsfunktionen ────────────────────────────────────

def _get_colors_by_concentration(sample_ids, conc_map):
    """Farben nach Hesperidin-Konzentration (viridis), unbekannte grau.

    Returns:
        colors: list of RGBA tuples
        norm: Normalize instance
        cmap: colormap
    """
    cmap = cm.viridis
    concentrations = [conc_map.get(sid, np.nan) for sid in sample_ids]
    valid_conc = [c for c in concentrations if not np.isnan(c)]
    if valid_conc:
        vmin, vmax = min(valid_conc), max(valid_conc)
    else:
        vmin, vmax = 0, 1
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colors = []
    for c in concentrations:
        if np.isnan(c):
            colors.append((0.7, 0.7, 0.7, 0.5))
        else:
            colors.append(cmap(norm(c)))
    return colors, norm, cmap


def _get_colors_by_probe_class(sample_ids, type_map):
    """Farben nach Proben-ID mit Typanzeige in Legende.

    Returns:
        colors: list of RGBA tuples
        legend_handles: list of Line2D handles
    """
    unique_ids = sorted(set(sample_ids),
                        key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 0, x))
    base_cmap = cm.tab20
    id_color_map = {}
    for i, uid in enumerate(unique_ids):
        id_color_map[uid] = base_cmap(i % 20)

    colors = [id_color_map[sid] for sid in sample_ids]

    legend_handles = []
    for uid in unique_ids:
        stype = type_map.get(uid, "?")
        label = f"{uid} ({stype})"
        legend_handles.append(plt.Line2D([0], [0], color=id_color_map[uid], lw=2, label=label))

    return colors, legend_handles


# ─── Plot-Funktionen ─────────────────────────────────────────

def plot_raw_spectra_dual(wavenumbers, sample_ids, spectra, conc_map, type_map,
                          save_path=None):
    """Dual-Panel Rohspektren: links Konzentration, rechts Probenklasse."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Links: nach Konzentration
    colors_conc, norm, cmap = _get_colors_by_concentration(sample_ids, conc_map)
    for i in range(spectra.shape[0]):
        ax1.plot(wavenumbers, spectra[i], color=colors_conc[i], alpha=0.4, lw=0.5)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax1, label="Hesperidin [%]")
    ax1.set_xlabel("Wellenzahl [cm$^{-1}$]")
    ax1.set_ylabel("Absorbanz")
    ax1.set_title("Rohspektren \u2013 nach Konzentration")
    ax1.invert_xaxis()

    # Rechts: nach Probenklasse
    colors_class, legend_handles = _get_colors_by_probe_class(sample_ids, type_map)
    for i in range(spectra.shape[0]):
        ax2.plot(wavenumbers, spectra[i], color=colors_class[i], alpha=0.4, lw=0.5)
    ax2.legend(handles=legend_handles, fontsize=6, loc="upper right", ncol=2)
    ax2.set_xlabel("Wellenzahl [cm$^{-1}$]")
    ax2.set_ylabel("Absorbanz")
    ax2.set_title("Rohspektren \u2013 nach Probenklasse")
    ax2.invert_xaxis()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"  -> {save_path}")
    plt.close(fig)


def plot_replicate_count(sample_ids, save_path=None):
    """Balkendiagramm: Anzahl Wiederholungsmessungen pro Probe."""
    from collections import Counter
    counts = Counter(sample_ids)
    unique_ids = sorted(counts.keys(),
                        key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 0, x))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(unique_ids)), [counts[uid] for uid in unique_ids],
                  color="#4ECDC4", edgecolor="k")
    ax.set_xticks(range(len(unique_ids)))
    ax.set_xticklabels(unique_ids, rotation=45, ha="right")
    ax.set_xlabel("Proben-ID")
    ax.set_ylabel("Anzahl Wiederholungen")
    ax.set_title("Wiederholungsmessungen pro Probe")

    for bar, uid in zip(bars, unique_ids):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.1,
                str(counts[uid]), ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"  -> {save_path}")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────

def main():
    print("=== 0_dataloading ===")
    os.makedirs(PLOT_DIR, exist_ok=True)

    wavenumbers, sample_ids, spectra = load_spectra()
    print(f"Spektren: {spectra.shape[0]} Messungen, {spectra.shape[1]} Wellenzahlen")
    print(f"Wellenzahlen: {wavenumbers[0]:.0f} - {wavenumbers[-1]:.0f} cm^-1")
    print(f"Proben: {sorted(set(sample_ids))}")

    hplc_df = load_hplc()
    print(f"HPLC-Referenz: {len(hplc_df)} Eintraege")

    conc_map = get_concentration_map(hplc_df)
    type_map = get_sample_type_map(hplc_df)
    print(f"Konzentrationen: {conc_map}")
    print(f"Probentypen: {type_map}")

    plot_raw_spectra_dual(wavenumbers, sample_ids, spectra, conc_map, type_map,
                          save_path=os.path.join(PLOT_DIR, "raw_spectra_dual.png"))

    plot_replicate_count(sample_ids,
                         save_path=os.path.join(PLOT_DIR, "replicate_count.png"))

    print("Done.")


if __name__ == "__main__":
    main()
