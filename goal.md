# Projektziel

## Quantifizierung von Hesperidin in Mandarinenschalen mittels NIR-Spektroskopie

### Hintergrund
Hesperidin ist ein Flavonoid, das in Zitrusfrüchten vorkommt. Die quantitative Bestimmung erfolgt
üblicherweise mittels HPLC, was zeit- und kostenintensiv ist. NIR-Spektroskopie bietet eine
schnelle, zerstörungsfreie Alternative zur Vorhersage des Hesperidin-Gehalts.

### Daten
- **TANGO (Bruker FT-NIR):** Reflexionsspektren im Bereich 4000–10000 cm⁻¹ (1501 Wellenzahlen), 21 Proben à 10 Wiederholungsmessungen
- **NeoSpectra (Handheld NIR):** Reflexionsspektren im Bereich ~1350–2558 nm (141 Wellenlängen), 20 Proben à 10 Wiederholungsmessungen
- **HPLC-Referenzwerte:** Hesperidin-Gehalt (%) für 12 Proben, bestimmt mittels HPLC

### Pipeline-Auswertung
Die Auswertung erfolgt als parametrisierte Pipeline (`pipeline.py`) die über `main.py` in 8 Konfigurationen ausgeführt wird:

| Faktor | Varianten |
|--------|-----------|
| Spektrometer | TANGO, NeoSpectra |
| Preprocessing | SNV, SNV + SavGol 1. Ableitung |
| Probenfilter | Alle (n=12), Nur Mandarinen ohne Probe 3+5 (n=10) |

### Modelle
- **Variablenselektion (Forward Greedy):** 1–8 Variablen + Lineare Regression
- **PLS:** 1–4 Komponenten
- **PCR (PCA + Regression):** 1–4 Komponenten × {LinReg, Ridge, Lasso}
- **ICA + Regression:** 1–4 Komponenten × {LinReg, Ridge, Lasso}

### Evaluation
- **LOO-CV:** Leave-One-Out Kreuzvalidierung (Train auf n-1, Test auf 1)
- **Metriken:** R²CV, RMSECV (Test), R²train, RMSEtrain (Training)
- **Plots:** Scatter (Predicted vs Measured), Modellvergleich, Variablenselektion

### Ergebnisse
Pro Run werden alle Plots und Metriken in `runs/<config_name>/` gespeichert.
Am Ende druckt `main.py` eine Gesamtvergleichstabelle über alle 8 Konfigurationen.
