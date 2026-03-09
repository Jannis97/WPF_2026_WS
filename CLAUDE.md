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
├── 01_data_loading.py
├── 02_preprocessing.py
├── 03_pls_regression.py
├── 04_dimensionality_reduction.py
├── plots/                              # Alle erzeugten PNGs
├── logs/                               # Log-Dateien
├── goal.md
├── checklist.md
└── report.md / report.pdf
```

## Konventionen
- Python-Skripte mit `.venv/bin/python` ausführen
- Daten als Dicts verarbeiten, JSON-Caching mit `load_json` Parameter
- Logger nutzen für alle Metriken
- Plots in `plots/` speichern

## Datenformat
- **TANGO:** Zeile 5 = Wellenzahlen-Header (4000–10000 cm⁻¹), Zeilen 6+ = Spektren, Spalte 1 = Proben-ID
- **NeoSpectra:** Pro Datei: Zeile 1 = Header, dann Wellenlänge(nm) \t Reflektanz(%)
- **HPLC:** .ods mit Kalibriergerade und Proben-Gehaltstabelle (Gehalt in %)
- Proben-IDs: Nummern (1–28), R1, R2; Prefix M=Mandarine, G=Grapefruit, Z=Zitrone, L=Limette
