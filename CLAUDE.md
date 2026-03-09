# CLAUDE.md – Projektkontext

## Projekt
NIR-Spektroskopie zur Hesperidin-Quantifizierung in Mandarinenschalen.

## Verzeichnisstruktur
```
WPF_2026_WS/
├── data/
│   ├── Spektren.txt                    # TANGO FT-NIR Spektren (tab-sep)
│   ├── NeoSpectra/                     # NeoSpectra Handheld NIR (.Spectrum)
│   └── WPF Mandarinen Hesperidin-...   # HPLC Referenzwerte (.ods)
├── pipeline.py                         # Parametrisierte Analyse-Pipeline
├── main.py                             # Führt 8 Konfigurationen aus
├── 01_data_loading.py                  # Daten laden (wird von pipeline importiert)
├── 02_preprocessing.py                 # Preprocessing (wird von pipeline importiert)
├── 03_pls_regression.py                # Standalone Modellvergleich (alt)
├── 04_dimensionality_reduction.py      # PCA, UMAP, t-SNE
├── 05_generate_report.py               # MD→PDF Report
├── runs/                               # Ergebnisse pro Konfiguration
│   ├── tango_SNV_all/                  # TANGO, SNV, alle Proben
│   ├── tango_SNV_excl3+5/             # TANGO, SNV, ohne Probe 3+5
│   ├── tango_SNV_SG1d_all/            # TANGO, SNV+SavGol1d, alle
│   ├── tango_SNV_SG1d_excl3+5/        # TANGO, SNV+SavGol1d, ohne 3+5
│   ├── neospectra_SNV_all/             # analog für NeoSpectra
│   ├── neospectra_SNV_excl3+5/
│   ├── neospectra_SNV_SG1d_all/
│   ├── neospectra_SNV_SG1d_excl3+5/
│   └── comparison.json                 # Gesamtvergleich
├── logs/
├── goal.md
├── checklist.md
└── report.md / report.pdf
```

## Pipeline-Konfiguration
`main.py` führt 8 Runs aus: 2 Spektrometer × 2 Preprocessing × 2 Probenfilter.
Jeder Run erzeugt in `runs/<name>/`:
- `preprocessing.png` – Rohspektren, SNV, ggf. SavGol 1. Ableitung
- `varsel_on_spectrum.png` – Gewählte Wellenlängen auf mittlerem Spektrum
- `varsel_r2_gain.png` – R²CV pro Variablenanzahl
- `varsel_univariate.png` – RMSECV pro Einzelwellenlänge
- `scatter_top_models.png` – LOO-Vorhersagen der Top-6-Modelle
- `model_comparison.png` – Balkendiagramm aller Modelle
- `results.json` – Alle Metriken

## Konventionen
- Python-Skripte mit `.venv/bin/python` ausführen
- Pipeline-Parameter als Dict übergeben
- `exclude_ids` steuert Probenfilterung (z.B. `["3", "5"]` für nur Mandarinen)
- Modelle: VarSel(1-8) LR, PLS(1-4), PCR(1-4) × {LR, Ridge, Lasso}, ICA(1-4) × {LR, Ridge, Lasso}

## Datenformat
- **TANGO:** Zeile 5 = Wellenzahlen-Header (4000–10000 cm⁻¹), Zeilen 6+ = Spektren, Spalte 1 = Proben-ID
- **NeoSpectra:** Pro Datei: Zeile 1 = Header, dann Wellenlänge(nm) \t Reflektanz(%)
- **HPLC:** .ods mit Kalibriergerade und Proben-Gehaltstabelle (Gehalt in %)
- Proben-IDs: Nummern (1–28), R1, R2; Prefix M=Mandarine, G=Grapefruit, Z=Zitrone, L=Limette
