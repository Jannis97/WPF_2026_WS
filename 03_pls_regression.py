"""
03_pls_regression.py
Umfassender Modellvergleich mit LOO-CV:
- Variablenselektion (1-4 Variablen, Forward Selection)
- PLS (1-4 Komponenten)
- PCR = PCA + Regression (1-4 Komponenten)
- ICA + Regression (1-4 Komponenten)
Auf jeder Dimensionsreduktion: LinReg, Ridge, Lasso.
"""

import json
import logging
import warnings
import numpy as np
import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, FastICA
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

warnings.filterwarnings("ignore")

INTERACTIVE = False
BASE_DIR = Path(__file__).parent
PLOT_DIR = BASE_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "03_pls_regression.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def load_preprocessed(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def _save_plot(fig, fname):
    fig.savefig(PLOT_DIR / fname, dpi=150, bbox_inches="tight")
    logger.info(f"Plot gespeichert: {fname}")
    if INTERACTIVE:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Variablenselektion (Forward Greedy)
# ---------------------------------------------------------------------------

def forward_select_variables(X, y, max_vars=4):
    """Greedy Forward Selection: wählt schrittweise die besten Variablen."""
    loo = LeaveOneOut()
    n_features = X.shape[1]
    selected = []
    remaining = list(range(n_features))

    for step in range(max_vars):
        best_rmse = np.inf
        best_idx = None
        for j in remaining:
            cols = selected + [j]
            Xsub = X[:, cols]
            y_pred = cross_val_predict(LinearRegression(), Xsub, y, cv=loo)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            if rmse < best_rmse:
                best_rmse = rmse
                best_idx = j
        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


def variable_selection_models(X, y, wavelengths, dataset_name):
    """Variablenselektion 1-4 Variablen × {LinReg, Ridge, Lasso}."""
    logger.info(f"{dataset_name} | Forward Variable Selection ...")
    selected_indices = forward_select_variables(X, y, max_vars=8)
    selected_wls = [wavelengths[i] for i in selected_indices]
    logger.info(f"{dataset_name} | Gewählte Variablen: {[f'{w:.1f}' for w in selected_wls]}")

    loo = LeaveOneOut()
    results = []

    regressors = {
        "LinReg": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1, max_iter=10000),
    }

    for n_vars in range(1, len(selected_indices) + 1):
        cols = selected_indices[:n_vars]
        wl_str = "+".join([f"{wavelengths[c]:.0f}" for c in cols])
        Xsub = X[:, cols]

        for reg_name, reg in regressors.items():
            name = f"VarSel({n_vars}) {reg_name}"
            try:
                y_pred = cross_val_predict(reg, Xsub, y, cv=loo)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                r2 = r2_score(y, y_pred)
                results.append({
                    "name": name,
                    "rmse": rmse,
                    "r2": r2,
                    "y_pred": y_pred.flatten(),
                    "detail": wl_str,
                })
                logger.info(f"{dataset_name} | {name:30s} | RMSECV={rmse:.4f} | R²CV={r2:.4f}")
            except Exception as e:
                logger.warning(f"{dataset_name} | {name}: {e}")

    # Variablenselektions-Plot
    _plot_variable_selection(X, y, wavelengths, dataset_name)

    return results, selected_indices


def _plot_variable_selection(X, y, wavelengths, dataset_name):
    """Plot: RMSECV pro Einzelvariable."""
    loo = LeaveOneOut()
    rmses = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        y_pred = cross_val_predict(LinearRegression(), X[:, j:j+1], y, cv=loo)
        rmses[j] = np.sqrt(mean_squared_error(y, y_pred))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(wavelengths, rmses, color="steelblue", linewidth=0.8)
    best_idx = np.argmin(rmses)
    ax.axvline(wavelengths[best_idx], color="red", linestyle="--", alpha=0.7,
               label=f"Beste: {wavelengths[best_idx]:.1f} (RMSECV={rmses[best_idx]:.4f})")
    ax.set_xlabel("Wellenzahl (cm⁻¹)" if "TANGO" in dataset_name else "Wellenlänge (nm)")
    ax.set_ylabel("RMSECV (%)")
    ax.set_title(f"{dataset_name} – Univariate Variablenselektion (LOO-CV)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_plot(fig, f"{dataset_name.lower().replace(' ', '_')}_variable_selection.png")


# ---------------------------------------------------------------------------
# Dimensionsreduktions-Modelle
# ---------------------------------------------------------------------------

def dimred_models():
    """PLS 1-4, PCR 1-4 × {LinReg, Ridge, Lasso}, ICA 1-4 × {LinReg, Ridge, Lasso}."""
    models = {}

    # PLS 1-4 (eigene Regression, kein zusätzliches Ridge/Lasso nötig)
    for n in [1, 2, 3, 4]:
        models[f"PLS ({n} Komp.)"] = PLSRegression(n_components=n, scale=True)

    # PCR 1-4 × Regressoren
    for n in [1, 2, 3, 4]:
        models[f"PCR({n}) LinReg"] = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n)),
            ("reg", LinearRegression()),
        ])
        models[f"PCR({n}) Ridge"] = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n)),
            ("reg", Ridge(alpha=1.0)),
        ])
        models[f"PCR({n}) Lasso"] = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n)),
            ("reg", Lasso(alpha=0.1, max_iter=10000)),
        ])

    # ICA 1-4 × Regressoren
    for n in [1, 2, 3, 4]:
        models[f"ICA({n}) LinReg"] = Pipeline([
            ("scaler", StandardScaler()),
            ("ica", FastICA(n_components=n, max_iter=1000, random_state=42)),
            ("reg", LinearRegression()),
        ])
        models[f"ICA({n}) Ridge"] = Pipeline([
            ("scaler", StandardScaler()),
            ("ica", FastICA(n_components=n, max_iter=1000, random_state=42)),
            ("reg", Ridge(alpha=1.0)),
        ])
        models[f"ICA({n}) Lasso"] = Pipeline([
            ("scaler", StandardScaler()),
            ("ica", FastICA(n_components=n, max_iter=1000, random_state=42)),
            ("reg", Lasso(alpha=0.1, max_iter=10000)),
        ])

    return models


def evaluate_models(X, y, models, dataset_name):
    """LOO-CV für alle Modelle."""
    loo = LeaveOneOut()
    results = []

    for name, model in models.items():
        try:
            y_pred = cross_val_predict(model, X, y, cv=loo)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)
            results.append({
                "name": name,
                "rmse": rmse,
                "r2": r2,
                "y_pred": y_pred.flatten(),
            })
            logger.info(f"{dataset_name} | {name:30s} | RMSECV={rmse:.4f} | R²CV={r2:.4f}")
        except Exception as e:
            logger.warning(f"{dataset_name} | {name}: {e}")

    return results


# ---------------------------------------------------------------------------
# Ausgabe und Plots
# ---------------------------------------------------------------------------

def print_results_table(results, dataset_name):
    """Ergebnistabelle sortiert nach RMSECV."""
    results.sort(key=lambda x: x["rmse"])

    header = f"\n{'='*70}\n  {dataset_name} – Modellvergleich (LOO-CV, n=10)\n{'='*70}"
    print(header)

    fmt = "  {rank:>2}. {name:30s}  RMSECV={rmse:.4f}%  R²CV={r2:+.4f}"
    divider = "  " + "-" * 65
    print(f"  {'#':>2}  {'Modell':30s}  {'RMSECV':>10s}  {'R²CV':>10s}")
    print(divider)

    for i, r in enumerate(results):
        line = fmt.format(rank=i + 1, name=r["name"], rmse=r["rmse"], r2=r["r2"])
        marker = " << BEST" if i == 0 else ""
        print(line + marker)

    print(divider)
    best = results[0]
    print(f"  Bestes Modell: {best['name']} (RMSECV={best['rmse']:.4f}%, R²CV={best['r2']:.4f})")
    print()


def plot_scatter(y_true, y_pred, rmse, r2, model_name, dataset_name, sample_ids):
    fig, ax = plt.subplots(figsize=(8, 7))
    unique_ids = sorted(set(sample_ids), key=lambda x: (not x.isdigit(), x))
    cmap = plt.get_cmap("tab20", len(unique_ids))
    colors = {sid: cmap(i) for i, sid in enumerate(unique_ids)}

    for sid in unique_ids:
        mask = [s == sid for s in sample_ids]
        ax.scatter(np.array(y_true)[mask], np.array(y_pred)[mask],
                   color=colors[sid], label=f"Probe {sid}", s=80,
                   edgecolors="k", linewidth=0.5, zorder=3)

    lims = [min(min(y_true), min(y_pred)) - 0.2, max(max(y_true), max(y_pred)) + 0.2]
    ax.plot(lims, lims, "k--", alpha=0.5, label="1:1 Linie")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Gemessen (HPLC) – Hesperidin (%)", fontsize=12)
    ax.set_ylabel("Vorhergesagt – Hesperidin (%)", fontsize=12)
    ax.set_title(f"{dataset_name} – {model_name}\nRMSECV={rmse:.4f}%, R²CV={r2:.4f}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_plot(fig, f"{dataset_name.lower().replace(' ', '_')}_best_scatter.png")


def plot_model_comparison(results, dataset_name):
    names = [r["name"] for r in results]
    rmses = [r["rmse"] for r in results]
    r2s = [r["r2"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, max(8, len(names) * 0.35)))
    fig.suptitle(f"{dataset_name} – Modellvergleich ({len(names)} Modelle)",
                 fontsize=14, fontweight="bold")

    colors = ["forestgreen" if i == 0 else "steelblue" for i in range(len(names))]

    ax1.barh(range(len(names)), rmses, color=colors, edgecolor="k", linewidth=0.5)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel("RMSECV (%)")
    ax1.set_title("RMSECV (kleiner = besser)")
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis="x")

    ax2.barh(range(len(names)), r2s, color=colors, edgecolor="k", linewidth=0.5)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlabel("R²CV")
    ax2.set_title("R²CV (größer = besser)")
    ax2.invert_yaxis()
    ax2.axvline(0, color="red", linestyle="--", alpha=0.5)
    ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    _save_plot(fig, f"{dataset_name.lower().replace(' ', '_')}_model_comparison.png")


# ---------------------------------------------------------------------------
# Hauptprozess
# ---------------------------------------------------------------------------

def process_dataset(data, dataset_name):
    logger.info(f"\n{'='*60}")
    logger.info(f"Modellvergleich: {dataset_name}")
    logger.info(f"{'='*60}")

    X = np.array(data["snv_spectra"])
    y = np.array(data["hesperidin_content"])
    sample_ids = data["sample_ids"]
    wavelengths = np.array(data["wavelengths"])

    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
    logger.info(f"Hesperidin-Bereich: {y.min():.4f}% - {y.max():.4f}%")

    # 1) Variablenselektion
    varsel_results, _ = variable_selection_models(X, y, wavelengths, dataset_name)

    # 2) PLS, PCR, ICA Modelle
    models = dimred_models()
    dimred_results = evaluate_models(X, y, models, dataset_name)

    # Zusammenführen
    all_results = varsel_results + dimred_results
    all_results.sort(key=lambda x: x["rmse"])

    print_results_table(all_results, dataset_name)
    plot_model_comparison(all_results, dataset_name)

    best = all_results[0]
    plot_scatter(y, best["y_pred"], best["rmse"], best["r2"],
                 best["name"], dataset_name, sample_ids)

    return [{"name": r["name"], "rmse": float(r["rmse"]), "r2": float(r["r2"])}
            for r in all_results]


def main():
    tango_prep = load_preprocessed(BASE_DIR / "tango_preprocessed.json")
    neo_prep = load_preprocessed(BASE_DIR / "neospectra_preprocessed.json")

    tango_results = process_dataset(tango_prep, "TANGO")
    neo_results = process_dataset(neo_prep, "NeoSpectra")

    results = {"TANGO": tango_results, "NeoSpectra": neo_results}
    with open(BASE_DIR / "pls_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Ergebnisse gespeichert: pls_results.json")

    return results


if __name__ == "__main__":
    main()
