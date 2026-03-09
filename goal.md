# Projektziel

## Quantifizierung von Hesperidin in Mandarinenschalen mittels NIR-Spektroskopie

### Hintergrund
Hesperidin ist ein Flavonoid, das in Zitrusfrüchten vorkommt. Die quantitative Bestimmung erfolgt
üblicherweise mittels HPLC, was zeit- und kostenintensiv ist. NIR-Spektroskopie bietet eine
schnelle, zerstörungsfreie Alternative zur Vorhersage des Hesperidin-Gehalts.

### Daten
- **TANGO (Bruker FT-NIR):** Reflexionsspektren im Bereich 4000–10000 cm⁻¹ (1502 Wellenzahlen), 21 Proben à 10 Wiederholungsmessungen
- **NeoSpectra (Handheld NIR):** Reflexionsspektren im Bereich ~1350–2558 nm (141 Wellenlängen), 20 Proben à 10 Wiederholungsmessungen
- **HPLC-Referenzwerte:** Hesperidin-Gehalt (%) für die Proben, bestimmt mittels HPLC

### Ziel der Auswertung
1. **Daten laden und strukturieren** – Beide Spektrometerdatensätze einlesen, mit HPLC-Referenzwerten verknüpfen, als JSON speichern
2. **Vorverarbeitung** – Mitteln der Wiederholungsmessungen, SNV, Savitzky-Golay-Glättung; Visualisierung der Verarbeitungsschritte
3. **PLS-Regression** – Partial Least Squares Regression mit Kreuzvalidierung zur Bestimmung der optimalen Komponentenzahl für beide Spektrometer
4. **Dimensionsreduktion** – PCA, UMAP, t-SNE zur explorativen Datenanalyse, gefärbt nach Probenklasse und Hesperidin-Gehalt
5. **Ergebnisbericht** – Alle Metriken, Plots und Erkenntnisse in einem Report zusammenfassen
