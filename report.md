# NIR-Spektroskopie zur Hesperidin-Quantifizierung in Mandarinenschalen

## 1. Projektübersicht

Ziel dieses Projekts ist die Quantifizierung des Hesperidin-Gehalts in Zitrusschalen mittels NIR-Spektroskopie. Zwei NIR-Spektrometer wurden verglichen:

- **TANGO (Bruker FT-NIR):** Laborspektrometer, 4000–10000 cm⁻¹, 1501 Wellenzahlen
- **NeoSpectra (Handheld NIR):** Tragbares Spektrometer, 1351–2558 nm, 141 Wellenlängen

Als Referenz dienen HPLC-bestimmte Hesperidin-Gehalte (%) für 12 Proben.

## 2. Daten

### 2.1 Probenmaterial

| Probe | Beschreibung | Hesperidin (%) |
|-------|-------------|----------------|
| 2 | G, EDEKA (Grapefruit) | 3.33 |
| 3 | Z, Rewe Spanien (Zitrone) | 2.47 |
| 5 | L, Rewe Vietnam (Limette) | 2.12 |
| 15 | M, Bio Markt lose Pa | 4.27 |
| 16 | M, Bio Markt lose Pb | 3.30 |
| 18 | M, EDEKA Spanien Pb | 2.96 |
| 19 | M, Rewe Spanien Pa | 3.69 |
| 21 | M, Penny Spanien Pa | 2.84 |
| 24 | M, EDEKA Spanien Nadorcott Pb | 3.63 |
| 25 | M, Rewe Bio Spanien Pb | 4.50 |
| 26 | M, Rewe Bio Spanien Pa | 4.22 |
| 28 | M, EDEKA Bio Spanien Pb | 3.82 |

### 2.2 Spektren

- **TANGO:** 219 Spektren total (21 Proben × ~10 Wiederholungen), davon 120 mit HPLC-Referenz (12 Proben)
- **NeoSpectra:** 200 Spektren total (20 Proben × 10 Wiederholungen), davon 120 mit HPLC-Referenz (12 Proben)
- Nach Mittelung der Wiederholungsmessungen: je 12 gemittelte Spektren pro Datensatz

## 3. Vorverarbeitung

Folgende Vorverarbeitungsschritte wurden durchgeführt:
1. **Mittelung** der Wiederholungsmessungen pro Probe
2. **SNV (Standard Normal Variate)** – Korrektur von Streueffekten
3. **Savitzky-Golay Glättung** (Fenster=15, Polynom=2)
4. **SG 1. Ableitung** – zur Verstärkung spektraler Merkmale

### TANGO – Vorverarbeitung (nach Probe)
![TANGO Preprocessing by Sample](plots/tango_preprocessing_by_sample.png)

### TANGO – Vorverarbeitung (nach Gehalt)
![TANGO Preprocessing by Concentration](plots/tango_preprocessing_by_concentration.png)

### NeoSpectra – Vorverarbeitung (nach Probe)
![NeoSpectra Preprocessing by Sample](plots/neospectra_preprocessing_by_sample.png)

### NeoSpectra – Vorverarbeitung (nach Gehalt)
![NeoSpectra Preprocessing by Concentration](plots/neospectra_preprocessing_by_concentration.png)

## 4. PLS-Regression

PLS-Regression (Partial Least Squares) wurde mit Leave-One-Out Kreuzvalidierung (LOO-CV) auf den SNV-vorverarbeiteten, gemittelten Spektren durchgeführt. Die Komponentenzahl wurde von 1–11 optimiert.

### 4.1 TANGO

- **Optimale Komponentenzahl:** 1
- **RMSECV:** 1.1895%
- **R²CV:** -1.8555

Das negative R²CV zeigt, dass das Modell schlechter als der Mittelwert vorhersagt. Bei nur 12 Proben und LOO-CV ist die Validierung instabil.

![TANGO PLS Components](plots/tango_pls_components.png)

![TANGO PLS Scatter](plots/tango_pls_scatter.png)

![TANGO PLS Residuals](plots/tango_pls_residuals.png)

### 4.2 NeoSpectra

- **Optimale Komponentenzahl:** 7
- **RMSECV:** 0.7472%
- **R²CV:** -0.1268

Das NeoSpectra-Modell performt besser als das TANGO-Modell, ist aber ebenfalls nicht prädiktiv (R²CV < 0).

![NeoSpectra PLS Components](plots/neospectra_pls_components.png)

![NeoSpectra PLS Scatter](plots/neospectra_pls_scatter.png)

![NeoSpectra PLS Residuals](plots/neospectra_pls_residuals.png)

### 4.3 Bewertung

Die schlechte PLS-Performance ist auf die **sehr geringe Probenanzahl (n=12)** zurückzuführen. Für robuste PLS-Modelle werden typischerweise >30 Kalibierproben benötigt. Zudem ist die Varianz der Hesperidin-Gehalte gering (2.1–4.5%). Empfehlungen:
- Mehr Kalibierproben sammeln
- Breiteren Konzentrationsbereich abdecken
- Verschiedene Vorverarbeitungen systematisch testen

## 5. Dimensionsreduktion

PCA, UMAP und t-SNE wurden auf den SNV-vorverarbeiteten, gemittelten Spektren durchgeführt.

### 5.1 TANGO

#### PCA
- Erklärte Varianz: PC1=57.3%, PC2=21.7% (Total: 79.0%)

![TANGO PCA by Sample](plots/tango_pca_by_sample.png)

![TANGO PCA by Concentration](plots/tango_pca_by_concentration.png)

#### UMAP

![TANGO UMAP by Sample](plots/tango_umap_by_sample.png)

![TANGO UMAP by Concentration](plots/tango_umap_by_concentration.png)

#### t-SNE
- KL-Divergenz: 0.2869

![TANGO tSNE by Sample](plots/tango_tsne_by_sample.png)

![TANGO tSNE by Concentration](plots/tango_tsne_by_concentration.png)

### 5.2 NeoSpectra

#### PCA
- Erklärte Varianz: PC1=54.4%, PC2=25.8% (Total: 80.2%)

![NeoSpectra PCA by Sample](plots/neospectra_pca_by_sample.png)

![NeoSpectra PCA by Concentration](plots/neospectra_pca_by_concentration.png)

#### UMAP

![NeoSpectra UMAP by Sample](plots/neospectra_umap_by_sample.png)

![NeoSpectra UMAP by Concentration](plots/neospectra_umap_by_concentration.png)

#### t-SNE
- KL-Divergenz: 0.2904

![NeoSpectra tSNE by Sample](plots/neospectra_tsne_by_sample.png)

![NeoSpectra tSNE by Concentration](plots/neospectra_tsne_by_concentration.png)

## 6. Zusammenfassung

| Metrik | TANGO | NeoSpectra |
|--------|-------|------------|
| Wellenlängenbereich | 4000–10000 cm⁻¹ | 1351–2558 nm |
| Spektrale Auflösung | 1501 Punkte | 141 Punkte |
| Proben mit HPLC | 12 | 12 |
| Wiederholungsmessungen | 10/Probe | 10/Probe |
| Optimale PLS-Komponenten | 1 | 7 |
| RMSECV | 1.19% | 0.75% |
| R²CV | -1.86 | -0.13 |
| PCA Varianz (PC1+PC2) | 79.0% | 80.2% |

### Kernerkenntnisse

1. **Datenqualität:** Beide Spektrometer liefern reproduzierbare Spektren (geringe Streuung innerhalb der Wiederholungsmessungen).
2. **PLS-Modelle:** Nicht prädiktiv aufgrund der geringen Probenzahl (n=12). Das NeoSpectra performt tendenziell besser.
3. **Dimensionsreduktion:** PCA zeigt eine gute Varianzaufklärung (≈80% in 2 Komponenten). Die Proben zeigen keine klare Separierung nach Hesperidin-Gehalt, was die schwache PLS-Performance erklärt.
4. **Empfehlung:** Erhöhung der Probenzahl auf mindestens 30–50 für robuste Kalibriermodelle.
