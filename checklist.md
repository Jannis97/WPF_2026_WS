# Checkliste – NIR Hesperidin-Quantifizierung

## 1. Projektstruktur
- [x] `.venv` erstellen
- [x] `goal.md` erstellen
- [x] `checklist.md` erstellen
- [x] `CLAUDE.md` erstellen

## 2. Daten laden (`01_data_loading.py`)
- [x] TANGO Spektren.txt einlesen (tab-separiert, Wellenzahlen als Header)
- [x] NeoSpectra .Spectrum Dateien einlesen (Wellenlänge + Reflektanz)
- [x] HPLC .ods einlesen (Hesperidin-Gehalt pro Probe)
- [x] Daten als Dicts strukturieren (keys: wavelengths, spectra, sample_ids, hesperidin_content)
- [x] Als JSON speichern (tango_data.json, neospectra_data.json)
- [x] `load_json=True` Parameter: JSON laden statt neu parsen
- [x] Keys und Shapes printen

## 3. Vorverarbeitung (`02_preprocessing.py`)
- [x] Wiederholungsmessungen mitteln
- [x] SNV (Standard Normal Variate)
- [x] Savitzky-Golay Glättung
- [x] Je Datensatz 2 PNGs mit Subplots der Verarbeitungsschritte:
  - [x] `tango_preprocessing_by_sample.png` – gefärbt nach Probenklasse
  - [x] `tango_preprocessing_by_concentration.png` – gefärbt nach Hesperidin-Gehalt
  - [x] `neospectra_preprocessing_by_sample.png` – gefärbt nach Probenklasse
  - [x] `neospectra_preprocessing_by_concentration.png` – gefärbt nach Hesperidin-Gehalt

## 4. PLS-Regression (`03_pls_regression.py`)
- [x] PLS mit Kreuzvalidierung (Leave-One-Out oder k-fold)
- [x] Optimale Komponentenzahl bestimmen (1–11 Komponenten getestet)
- [x] Je Datensatz 3 PNGs:
  - [x] `tango_pls_components.png` – RMSECV über Komponentenzahl
  - [x] `tango_pls_scatter.png` – Predicted vs. Measured mit RMSE und R²
  - [x] `tango_pls_residuals.png` – Residuenplot
  - [x] `neospectra_pls_components.png`
  - [x] `neospectra_pls_scatter.png`
  - [x] `neospectra_pls_residuals.png`

## 5. Dimensionsreduktion (`04_dimensionality_reduction.py`)
- [x] PCA, UMAP, t-SNE je Datensatz
- [x] Je Methode 2 Plots (nach Probenklasse + nach Gehalt)
- [x] Ergibt je Datensatz 6 PNGs:
  - [x] `tango_pca_by_sample.png`, `tango_pca_by_concentration.png`
  - [x] `tango_umap_by_sample.png`, `tango_umap_by_concentration.png`
  - [x] `tango_tsne_by_sample.png`, `tango_tsne_by_concentration.png`
  - [x] (analog für neospectra_*)

## 6. Logging
- [x] Logger in allen Skripten, der Metriken und Ergebnisse in `logs/` speichert

## 7. Report
- [x] `report.md` mit allen Ergebnissen und eingebetteten Grafiken
- [x] `report.pdf` aus der .md generieren

## Zusammenfassung der erwarteten Grafiken
~30 PNGs total (je ~15 pro Datensatz)
