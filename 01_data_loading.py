"""
01_data_loading.py
Lädt NIR-Spektren (TANGO + NeoSpectra) und HPLC-Referenzwerte,
strukturiert sie als Dicts und speichert als JSON.
Parameter load_json=True lädt vorhandene JSONs statt neu zu parsen.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "01_data_loading.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def load_hplc_reference():
    """Lädt HPLC-Referenzwerte aus der .ods Datei."""
    ods_path = DATA_DIR / "WPF Mandarinen Hesperidin-Qauntifizierung mittels HPLC.ods"
    df = pd.read_excel(ods_path, engine="odf", header=None)

    # Finde die Probentabelle (Zeile mit "Probe" in Spalte 1)
    probe_start = None
    for i, row in df.iterrows():
        if str(row.iloc[1]).strip() == "Probe":
            probe_start = i + 1
            break

    if probe_start is None:
        raise ValueError("Konnte Probentabelle in HPLC-Datei nicht finden")

    # Parse Proben: Probenummer -> mittlerer Gehalt
    sample_gehalt = {}
    sample_info = {}
    for i in range(probe_start, len(df)):
        row = df.iloc[i]
        probe_name = str(row.iloc[1]).strip()
        try:
            probe_nr = str(int(float(row.iloc[2])))
        except (ValueError, TypeError):
            continue
        try:
            gehalt = float(row.iloc[6])
        except (ValueError, TypeError):
            continue

        if probe_nr not in sample_gehalt:
            sample_gehalt[probe_nr] = []
            sample_info[probe_nr] = probe_name
        sample_gehalt[probe_nr].append(gehalt)

    # Mitteln der Duplikate
    hesperidin = {k: float(np.mean(v)) for k, v in sample_gehalt.items()}
    logger.info(f"HPLC-Referenzwerte geladen: {len(hesperidin)} Proben")
    for k, v in sorted(hesperidin.items(), key=lambda x: int(x[0])):
        logger.info(f"  Probe {k} ({sample_info[k]}): {v:.4f}%")
    return hesperidin, sample_info


def load_tango_spectra():
    """Lädt TANGO FT-NIR Spektren aus Spektren.txt."""
    txt_path = DATA_DIR / "Spektren.txt"
    logger.info(f"Lade TANGO-Spektren aus {txt_path}")

    with open(txt_path, "r") as f:
        lines = f.readlines()

    # Zeile 5 (Index 4): Header mit Wellenzahlen
    header_line = lines[4].strip().split("\t")
    wavelengths = [float(x) for x in header_line[1:] if x.strip()]

    # Zeilen 6+ (Index 5+): Spektren
    sample_ids = []
    spectra = []
    for line in lines[5:]:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        sample_id = parts[0].strip()
        values = [float(x) for x in parts[1:] if x.strip()]
        if len(values) == len(wavelengths):
            sample_ids.append(sample_id)
            spectra.append(values)

    spectra = np.array(spectra)
    logger.info(f"TANGO: {spectra.shape[0]} Spektren, {spectra.shape[1]} Wellenzahlen")
    logger.info(f"TANGO: Wellenzahlbereich {wavelengths[0]:.0f} - {wavelengths[-1]:.0f} cm⁻¹")
    unique_ids = sorted(set(sample_ids), key=lambda x: (not x.isdigit(), x))
    logger.info(f"TANGO: {len(unique_ids)} einzigartige Proben: {unique_ids}")

    return {
        "wavelengths": wavelengths,
        "spectra": spectra.tolist(),
        "sample_ids": sample_ids,
    }


def load_neospectra_spectra():
    """Lädt NeoSpectra Handheld NIR Spektren aus .Spectrum Dateien."""
    neo_dir = DATA_DIR / "NeoSpectra"
    logger.info(f"Lade NeoSpectra-Spektren aus {neo_dir}")

    files = sorted(neo_dir.glob("*.Spectrum"))
    sample_ids = []
    spectra = []
    wavelengths = None

    for fpath in files:
        # Parse filename: "M 25_Perc_1.Spectrum" -> sample_id="25", replicate=1
        fname = fpath.stem  # e.g. "M 25_Perc_1"
        # Extract sample number
        parts = fname.split("_Perc_")
        if len(parts) != 2:
            logger.warning(f"Unerwarteter Dateiname: {fname}")
            continue

        sample_part = parts[0].strip()
        # Remove prefix like "M ", "M", "R1 " etc.
        # Samples: "M 25", "M3.", "M 21.", "R1 Mandarinenschalen", "R2 Orangenblüten"
        if sample_part.startswith("R"):
            sample_id = sample_part.split()[0]  # "R1" or "R2"
        else:
            # Remove "M" prefix and dots/spaces
            sample_id = sample_part.replace("M", "").replace(".", "").strip()

        # Read spectrum file
        wl = []
        refl = []
        with open(fpath, "r") as f:
            for i, line in enumerate(f):
                if i == 0:  # Header
                    continue
                vals = line.strip().split("\t")
                if len(vals) == 2:
                    try:
                        wl.append(float(vals[0]))
                        refl.append(float(vals[1]))
                    except ValueError:
                        continue

        if wavelengths is None:
            wavelengths = wl
        spectra.append(refl)
        sample_ids.append(sample_id)

    spectra = np.array(spectra)
    logger.info(f"NeoSpectra: {spectra.shape[0]} Spektren, {spectra.shape[1]} Wellenlängen")
    logger.info(f"NeoSpectra: Wellenlängenbereich {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm")
    unique_ids = sorted(set(sample_ids), key=lambda x: (not x.isdigit(), x))
    logger.info(f"NeoSpectra: {len(unique_ids)} einzigartige Proben: {unique_ids}")

    return {
        "wavelengths": wavelengths,
        "spectra": spectra.tolist(),
        "sample_ids": sample_ids,
    }


def build_dataset(spectra_dict, hesperidin, dataset_name, exclude_ids=None):
    """Verknüpft Spektren mit HPLC-Referenzwerten."""
    if exclude_ids is None:
        exclude_ids = set()
    else:
        exclude_ids = set(exclude_ids)

    sample_ids = spectra_dict["sample_ids"]
    spectra = spectra_dict["spectra"]
    wavelengths = spectra_dict["wavelengths"]

    # Nur Proben mit HPLC-Werten behalten, exkludierte rausfiltern
    filtered_ids = []
    filtered_spectra = []
    filtered_hesperidin = []

    for sid, spec in zip(sample_ids, spectra):
        if sid in hesperidin and sid not in exclude_ids:
            filtered_ids.append(sid)
            filtered_spectra.append(spec)
            filtered_hesperidin.append(hesperidin[sid])

    logger.info(f"{dataset_name}: {len(filtered_spectra)} Spektren mit HPLC-Referenz "
                f"(von {len(spectra)} total)")
    unique_filtered = sorted(set(filtered_ids), key=lambda x: (not x.isdigit(), x))
    logger.info(f"{dataset_name}: Proben mit HPLC: {unique_filtered}")

    return {
        "wavelengths": wavelengths,
        "spectra": filtered_spectra,
        "sample_ids": filtered_ids,
        "hesperidin_content": filtered_hesperidin,
        "all_spectra": spectra,
        "all_sample_ids": sample_ids,
    }


def save_to_json(data, filepath):
    """Speichert Dict als JSON."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Gespeichert: {filepath}")


def load_from_json(filepath):
    """Lädt Dict aus JSON."""
    with open(filepath, "r") as f:
        data = json.load(f)
    logger.info(f"Geladen aus JSON: {filepath}")
    return data


def print_data_summary(data, name):
    """Gibt Keys und Shapes der Daten aus."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Datensatz: {name}")
    logger.info(f"{'='*60}")
    for key, val in data.items():
        if isinstance(val, list):
            arr = np.array(val) if val and isinstance(val[0], (list, float, int)) else val
            if isinstance(arr, np.ndarray):
                logger.info(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
            else:
                logger.info(f"  {key}: len={len(val)}, unique={len(set(val))}")
        else:
            logger.info(f"  {key}: {type(val).__name__}")


def main(load_json=False):
    tango_json = BASE_DIR / "tango_data.json"
    neo_json = BASE_DIR / "neospectra_data.json"

    if load_json and tango_json.exists() and neo_json.exists():
        logger.info("Lade Daten aus JSON-Dateien...")
        tango_data = load_from_json(tango_json)
        neo_data = load_from_json(neo_json)
    else:
        logger.info("Parse Rohdaten und erstelle JSON-Dateien...")
        hesperidin, sample_info = load_hplc_reference()

        tango_raw = load_tango_spectra()
        neo_raw = load_neospectra_spectra()

        # Proben 3 und 5 ausschließen (keine Mandarinen)
        exclude = ["3", "5"]
        tango_data = build_dataset(tango_raw, hesperidin, "TANGO", exclude_ids=exclude)
        neo_data = build_dataset(neo_raw, hesperidin, "NeoSpectra", exclude_ids=exclude)

        save_to_json(tango_data, tango_json)
        save_to_json(neo_data, neo_json)

    print_data_summary(tango_data, "TANGO")
    print_data_summary(neo_data, "NeoSpectra")

    return tango_data, neo_data


if __name__ == "__main__":
    import sys
    load_json = "--load-json" in sys.argv
    main(load_json=load_json)
