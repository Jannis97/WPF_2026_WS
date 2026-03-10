#!/usr/bin/env python3
"""
3_regression_models.py – Regressionsmodelle fuer NIR-Kalibrierung.

Modelle:
  - PLSModel (Partial Least Squares) mit VIP-Score-Berechnung
  - PCRModel (Principal Component Regression)
  - LinearModel (OLS auf PC-Scores)

Variable Selection:
  - VIP scores aus PLS
  - Korrelationsbasiert

Erzeugt:
  - plots/model_<name>_coefficients.png
  - plots/model_<name>_predicted_vs_actual.png
  - plots/model_<name>_vip_scores.png  (nur PLS)
  - plots/model_<name>_selected_wavelengths.png  (nur PLS)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from abc import ABC, abstractmethod
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from importlib.util import spec_from_file_location, module_from_spec

BENCHTOP_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(BENCHTOP_DIR, "plots")
DPI = 300


def _load_dataloading():
    spec = spec_from_file_location("dataloading", os.path.join(BENCHTOP_DIR, "0_dataloading.py"))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─── Basisklasse ─────────────────────────────────────────────

class RegressionModel(ABC):
    """Basisklasse fuer Regressionsmodelle."""

    def __init__(self, params=None):
        self.params = params or {}
        self.is_fitted = False

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def fit_predict(self, X_train, y_train, X_test):
        self.fit(X_train, y_train)
        return self.predict(X_test)

    @abstractmethod
    def get_coefficients(self):
        """Regressionskoeffizienten im Originalraum."""
        pass

    def plot_coefficients(self, wavenumbers, save_path=None):
        """Dual-Panel: Koeffizienten ueber Wellenzahlen."""
        coefs = self.get_coefficients()
        if coefs is None:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

        # Links: Koeffizienten-Linie
        ax1.plot(wavenumbers, coefs, linewidth=0.8, color="#2196F3")
        ax1.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax1.fill_between(wavenumbers, 0, coefs, alpha=0.3, color="#2196F3")
        ax1.set_xlabel("Wellenzahl [cm$^{-1}$]")
        ax1.set_ylabel("Koeffizient")
        ax1.set_title(f"{self.name()} \u2013 Regressionskoeffizienten")
        ax1.invert_xaxis()

        # Rechts: Absolute Koeffizienten (Wichtigkeit)
        abs_coefs = np.abs(coefs)
        ax2.plot(wavenumbers, abs_coefs, linewidth=0.8, color="#FF5722")
        ax2.fill_between(wavenumbers, 0, abs_coefs, alpha=0.3, color="#FF5722")
        ax2.set_xlabel("Wellenzahl [cm$^{-1}$]")
        ax2.set_ylabel("|Koeffizient|")
        ax2.set_title(f"{self.name()} \u2013 Absolute Wichtigkeit")
        ax2.invert_xaxis()

        plt.tight_layout()
        if save_path is None:
            safe = self.name().replace(" ", "_").replace("(", "").replace(")", "").lower()
            save_path = os.path.join(PLOT_DIR, f"model_{safe}_coefficients.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"  -> {save_path}")
        plt.close(fig)

    def plot_predicted_vs_actual(self, y_true, y_pred, sample_ids=None,
                                 conc_map=None, type_map=None, label="",
                                 save_path=None):
        """Dual-Panel: Predicted vs Actual, farbig nach Konzentration und Klasse."""
        dl = _load_dataloading()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        mn = min(y_true.min(), y_pred.min()) - 0.2
        mx = max(y_true.max(), y_pred.max()) + 0.2

        for ax in [ax1, ax2]:
            ax.plot([mn, mx], [mn, mx], "k--", linewidth=0.8)
            ax.set_xlim(mn, mx)
            ax.set_ylim(mn, mx)
            ax.set_xlabel("Tatsaechlich [%]")
            ax.set_ylabel("Vorhergesagt [%]")
            ax.set_aspect("equal")

        # Links: nach Konzentration
        if sample_ids and conc_map:
            colors_conc, norm, cmap = dl._get_colors_by_concentration(sample_ids, conc_map)
            for i in range(len(y_true)):
                ax1.scatter(y_true[i], y_pred[i], color=colors_conc[i],
                            edgecolors="k", s=60, zorder=3)
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax1, label="Hesperidin [%]")
        else:
            ax1.scatter(y_true, y_pred, edgecolors="k", facecolors="#4ECDC4", s=60, zorder=3)

        from sklearn.metrics import r2_score, mean_squared_error
        r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float('nan')
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        ax1.set_title(f"{self.name()} {label} \u2013 Konzentration\nR\u00b2={r2:.3f}, RMSE={rmse:.3f}")

        # Rechts: nach Probenklasse
        if sample_ids and type_map:
            colors_class, legend_handles = dl._get_colors_by_probe_class(sample_ids, type_map)
            for i in range(len(y_true)):
                ax2.scatter(y_true[i], y_pred[i], color=colors_class[i],
                            edgecolors="k", s=60, zorder=3)
            ax2.legend(handles=legend_handles, fontsize=7, loc="upper left")
        else:
            ax2.scatter(y_true, y_pred, edgecolors="k", facecolors="#4ECDC4", s=60, zorder=3)

        ax2.set_title(f"{self.name()} {label} \u2013 Probenklasse")

        plt.tight_layout()
        if save_path is None:
            safe = self.name().replace(" ", "_").replace("(", "").replace(")", "").lower()
            safe_l = label.replace(" ", "_").lower()
            save_path = os.path.join(PLOT_DIR, f"model_{safe}_pred_vs_actual_{safe_l}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"  -> {save_path}")
        plt.close(fig)


# ─── PLS Modell ──────────────────────────────────────────────

class PLSModel(RegressionModel):
    """Partial Least Squares Regression mit VIP-Score-Berechnung."""

    def __init__(self, params=None):
        super().__init__(params)
        self.n_components = self.params.get("n_components", 3)
        self.model = None

    def name(self):
        return f"PLS({self.n_components})"

    def fit(self, X_train, y_train):
        n_comp = min(self.n_components, X_train.shape[0] - 1, X_train.shape[1])
        self.model = PLSRegression(n_components=n_comp)
        self.model.fit(X_train, y_train)
        self._n_comp_actual = n_comp
        self.is_fitted = True

    def predict(self, X):
        return self.model.predict(X).ravel()

    def get_coefficients(self):
        if not self.is_fitted:
            return None
        return self.model.coef_.ravel()

    def compute_vip(self):
        """VIP-Scores berechnen.

        VIP_j = sqrt(p * sum_k(SS_k * w_jk^2) / sum_k(SS_k))

        wobei:
          p = Anzahl der Praediktoren
          w_jk = Gewicht des j-ten Praediktors in der k-ten Komponente
          SS_k = erklaerte Summe der Quadrate der k-ten Komponente
        """
        if not self.is_fitted:
            return None

        T = self.model.x_scores_      # (n, n_comp)
        W = self.model.x_weights_      # (p, n_comp)
        Q = self.model.y_loadings_     # (1, n_comp)

        p = W.shape[0]
        n_comp = T.shape[1]

        # SS_k = sum of squared y-loadings * sum of squared scores
        SS = np.zeros(n_comp)
        for k in range(n_comp):
            SS[k] = (Q[0, k] ** 2) * np.sum(T[:, k] ** 2)

        SS_total = np.sum(SS)
        if SS_total == 0:
            return np.ones(p)

        vip = np.zeros(p)
        for j in range(p):
            ss_weighted = np.sum(SS * (W[j, :] ** 2))
            vip[j] = np.sqrt(p * ss_weighted / SS_total)

        return vip

    def plot_vip_scores(self, wavenumbers, save_path=None):
        """Plot VIP-Scores ueber Wellenzahlen."""
        vip = self.compute_vip()
        if vip is None:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

        # Links: VIP Scores
        ax1.plot(wavenumbers, vip, linewidth=0.8, color="#9C27B0")
        ax1.axhline(1.0, color="red", linestyle="--", linewidth=1, label="VIP=1.0 Schwelle")
        ax1.fill_between(wavenumbers, 0, vip, where=vip >= 1.0,
                         alpha=0.3, color="#9C27B0", label="VIP >= 1.0")
        ax1.set_xlabel("Wellenzahl [cm$^{-1}$]")
        ax1.set_ylabel("VIP Score")
        ax1.set_title(f"{self.name()} \u2013 VIP Scores")
        ax1.legend()
        ax1.invert_xaxis()

        # Rechts: Top-N VIP markiert
        n_selected = np.sum(vip >= 1.0)
        top_idx = np.argsort(vip)[::-1][:max(n_selected, 20)]
        ax2.plot(wavenumbers, vip, linewidth=0.5, color="gray", alpha=0.5)
        ax2.scatter(wavenumbers[top_idx], vip[top_idx], color="#9C27B0",
                    s=20, zorder=3, label=f"Top {len(top_idx)} Wellenzahlen")
        ax2.axhline(1.0, color="red", linestyle="--", linewidth=1)
        ax2.set_xlabel("Wellenzahl [cm$^{-1}$]")
        ax2.set_ylabel("VIP Score")
        ax2.set_title(f"Top VIP Wellenzahlen (n={len(top_idx)})")
        ax2.legend()
        ax2.invert_xaxis()

        plt.tight_layout()
        if save_path is None:
            safe = self.name().replace(" ", "_").replace("(", "").replace(")", "").lower()
            save_path = os.path.join(PLOT_DIR, f"model_{safe}_vip_scores.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"  -> {save_path}")
        plt.close(fig)

    def plot_selected_wavelengths(self, wavenumbers, spectra, vip_threshold=1.0,
                                  save_path=None):
        """Plot: Mittelspektrum mit markierten VIP-Wellenzahlen."""
        vip = self.compute_vip()
        if vip is None:
            return

        mean_spectrum = spectra.mean(axis=0)
        selected = vip >= vip_threshold

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

        # Links: Spektrum mit markierten Bereichen
        ax1.plot(wavenumbers, mean_spectrum, color="gray", lw=1, label="Mittelspektrum")
        ax1.fill_between(wavenumbers, mean_spectrum.min(), mean_spectrum.max(),
                         where=selected, alpha=0.3, color="#9C27B0",
                         label=f"VIP >= {vip_threshold}")
        ax1.set_xlabel("Wellenzahl [cm$^{-1}$]")
        ax1.set_ylabel("Intensitaet")
        ax1.set_title(f"Ausgewaehlte Wellenzahlen (n={np.sum(selected)})")
        ax1.legend()
        ax1.invert_xaxis()

        # Rechts: VIP mit Schwelle
        ax2.bar(wavenumbers, vip, width=4, color=np.where(selected, "#9C27B0", "lightgray"))
        ax2.axhline(vip_threshold, color="red", linestyle="--", linewidth=1)
        ax2.set_xlabel("Wellenzahl [cm$^{-1}$]")
        ax2.set_ylabel("VIP Score")
        ax2.set_title("VIP-basierte Variablenselektion")
        ax2.invert_xaxis()

        plt.tight_layout()
        if save_path is None:
            safe = self.name().replace(" ", "_").replace("(", "").replace(")", "").lower()
            save_path = os.path.join(PLOT_DIR, f"model_{safe}_selected_wavelengths.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"  -> {save_path}")
        plt.close(fig)


# ─── PCR Modell ──────────────────────────────────────────────

class PCRModel(RegressionModel):
    """Principal Component Regression."""

    def __init__(self, params=None):
        super().__init__(params)
        self.n_components = self.params.get("n_components", 3)
        self.pca = None
        self.regressor = None

    def name(self):
        return f"PCR({self.n_components})"

    def fit(self, X_train, y_train):
        n_comp = min(self.n_components, X_train.shape[0] - 1, X_train.shape[1])
        self.pca = PCA(n_components=n_comp)
        scores = self.pca.fit_transform(X_train)
        self.regressor = LinearRegression()
        self.regressor.fit(scores, y_train)
        self.is_fitted = True

    def predict(self, X):
        scores = self.pca.transform(X)
        return self.regressor.predict(scores).ravel()

    def get_coefficients(self):
        if not self.is_fitted:
            return None
        return self.pca.components_.T @ self.regressor.coef_


# ─── Linear Model (OLS auf PC-Scores) ───────────────────────

class LinearModel(RegressionModel):
    """OLS auf PCA-reduzierten Daten."""

    def __init__(self, params=None):
        super().__init__(params)
        self.n_components = self.params.get("n_components", 3)
        self.pca = None
        self.regressor = None

    def name(self):
        return f"OLS_PC({self.n_components})"

    def fit(self, X_train, y_train):
        n_comp = min(self.n_components, X_train.shape[0] - 1, X_train.shape[1])
        self.pca = PCA(n_components=n_comp)
        scores = self.pca.fit_transform(X_train)
        self.regressor = LinearRegression()
        self.regressor.fit(scores, y_train)
        self.is_fitted = True

    def predict(self, X):
        scores = self.pca.transform(X)
        return self.regressor.predict(scores).ravel()

    def get_coefficients(self):
        if not self.is_fitted:
            return None
        return self.pca.components_.T @ self.regressor.coef_


# ─── Variable Selection ─────────────────────────────────────

def select_variables_vip(X, y, n_components=3, top_n=None, threshold=1.0):
    """Variablenselektion ueber VIP-Scores aus PLS.

    Args:
        X: Feature-Matrix
        y: Zielwerte
        n_components: PLS-Komponenten fuer VIP-Berechnung
        top_n: Anzahl zu waehlen (hat Vorrang ueber threshold)
        threshold: VIP-Schwelle (default 1.0)

    Returns:
        selected_idx: np.array mit Indizes der gewaehlten Variablen
        vip_scores: np.array aller VIP-Scores
    """
    pls = PLSModel({"n_components": n_components})
    pls.fit(X, y)
    vip = pls.compute_vip()

    if top_n is not None:
        selected_idx = np.argsort(vip)[::-1][:top_n]
        selected_idx = np.sort(selected_idx)
    else:
        selected_idx = np.where(vip >= threshold)[0]

    return selected_idx, vip


def select_variables_correlation(X, y, top_n=50):
    """Variablenselektion ueber Pearson-Korrelation mit y.

    Returns:
        selected_idx: np.array
        correlations: np.array
    """
    correlations = np.array([np.corrcoef(X[:, j], y)[0, 1] for j in range(X.shape[1])])
    abs_corr = np.abs(correlations)
    selected_idx = np.argsort(abs_corr)[::-1][:top_n]
    selected_idx = np.sort(selected_idx)
    return selected_idx, correlations


# ─── Factory ─────────────────────────────────────────────────

def get_model_by_name(name, params=None):
    """Factory: Modell nach Name instanziieren."""
    models = {
        "pls": PLSModel,
        "pcr": PCRModel,
        "ols": LinearModel,
    }
    key = name.lower()
    if key not in models:
        raise ValueError(f"Unbekanntes Modell: {name}. Verfuegbar: {list(models.keys())}")
    return models[key](params)


# ─── Main ─────────────────────────────────────────────────────

def main():
    print("=== 3_regression_models ===")
    os.makedirs(PLOT_DIR, exist_ok=True)

    dl = _load_dataloading()
    pp_spec = spec_from_file_location("preprocessing",
                                       os.path.join(BENCHTOP_DIR, "1_data_preprocessing.py"))
    pp = module_from_spec(pp_spec)
    pp_spec.loader.exec_module(pp)

    wavenumbers, sample_ids, spectra = dl.load_spectra()
    hplc_df = dl.load_hplc()
    conc_map = dl.get_concentration_map(hplc_df)
    type_map = dl.get_sample_type_map(hplc_df)

    # Preprocessing (SNV)
    processed, steps = pp.preprocess(spectra, {"snv": True, "savgol": False})
    print(f"Preprocessing: {' -> '.join(steps)}")

    # Nur Proben mit Konzentration
    known_ids = set(conc_map.keys())
    mask = [sid in known_ids for sid in sample_ids]
    X = processed[mask]
    y = np.array([conc_map[sid] for sid, m in zip(sample_ids, mask) if m])
    ids_f = [sid for sid, m in zip(sample_ids, mask) if m]

    print(f"Daten: {X.shape}, y: {y.shape}")

    # Demo: Modelle trainieren
    models_configs = [
        PLSModel({"n_components": 1}),
        PLSModel({"n_components": 2}),
        PLSModel({"n_components": 3}),
        PCRModel({"n_components": 1}),
        PCRModel({"n_components": 2}),
        PCRModel({"n_components": 3}),
    ]

    for model in models_configs:
        print(f"\n--- {model.name()} ---")
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
        print(f"  Train R2: {r2:.4f}")

        model.plot_coefficients(wavenumbers)
        model.plot_predicted_vs_actual(y, y_pred, sample_ids=ids_f,
                                       conc_map=conc_map, type_map=type_map,
                                       label="Train")

    # VIP-Scores fuer PLS(3)
    pls3 = PLSModel({"n_components": 3})
    pls3.fit(X, y)
    pls3.plot_vip_scores(wavenumbers)
    pls3.plot_selected_wavelengths(wavenumbers, X)

    # Variable selection demo
    sel_idx, vip = select_variables_vip(X, y, n_components=3, top_n=50)
    print(f"\nVIP Variable Selection: {len(sel_idx)} Wellenzahlen ausgewaehlt")

    sel_idx_corr, corrs = select_variables_correlation(X, y, top_n=50)
    print(f"Korrelationsbasiert: {len(sel_idx_corr)} Wellenzahlen ausgewaehlt")

    print("\nDone.")


if __name__ == "__main__":
    main()
