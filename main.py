"""
main.py
Führt alle Auswertungsschritte aus mit interaktiven Plots.
Schließe jeweils das Plot-Fenster, um zum nächsten Schritt zu gelangen.

Usage:
    .venv/bin/python main.py              # Interaktive Plots
    .venv/bin/python main.py --no-show    # Nur PNGs speichern, keine Fenster
    .venv/bin/python main.py --load-json  # JSON-Cache nutzen statt neu parsen
"""

import sys
import importlib

interactive = "--no-show" not in sys.argv
load_json = "--load-json" in sys.argv

import matplotlib
if interactive:
    matplotlib.use("Qt5Agg")
else:
    matplotlib.use("Agg")

print("=" * 60)
print("NIR Hesperidin-Quantifizierung – Gesamtauswertung")
print("=" * 60)
if interactive:
    print("Plots öffnen sich interaktiv. Fenster schließen → nächster Plot.\n")

# --- Schritt 1: Daten laden ---
print("━" * 60)
print("SCHRITT 1: Daten laden")
print("━" * 60)
data_loading = importlib.import_module("01_data_loading")
tango_data, neo_data = data_loading.main(load_json=load_json)

# --- Schritt 2: Vorverarbeitung ---
print("\n" + "━" * 60)
print("SCHRITT 2: Vorverarbeitung")
print("━" * 60)
preprocessing = importlib.import_module("02_preprocessing")
preprocessing.INTERACTIVE = interactive
preprocessing.main()

# --- Schritt 3: PLS-Regression ---
print("\n" + "━" * 60)
print("SCHRITT 3: PLS-Regression")
print("━" * 60)
pls = importlib.import_module("03_pls_regression")
pls.INTERACTIVE = interactive
pls.main()

# --- Schritt 4: Dimensionsreduktion ---
print("\n" + "━" * 60)
print("SCHRITT 4: Dimensionsreduktion (PCA, UMAP, t-SNE)")
print("━" * 60)
dimred = importlib.import_module("04_dimensionality_reduction")
dimred.INTERACTIVE = interactive
dimred.main()

# --- Schritt 5: Report ---
print("\n" + "━" * 60)
print("SCHRITT 5: Report generieren")
print("━" * 60)
report = importlib.import_module("05_generate_report")
report.main()

print("\n" + "=" * 60)
print("FERTIG! Alle Ergebnisse in plots/, logs/, report.pdf")
print("=" * 60)