"""
pipeline.py
Parametrisierte NIR-Analyse-Pipeline.
Nimmt ein Config-Dict und führt die komplette Auswertung durch:
Daten laden → Preprocessing → Variablenselektion → Modellierung → Evaluation → Plots.
"""

import json
import logging
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, FastICA
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent


# ===========================================================================
# Daten laden (importiert Funktionen aus 01/02)
# ===========================================================================

def load_data(spectrometer, exclude_ids=None):
    """Lädt Rohdaten und HPLC, filtert Proben, gibt Dict zurück."""
    import importlib
    loader = importlib.import_module("01_data_loading")

    hesperidin, sample_info = loader.load_hplc_reference()

    if spectrometer == "tango":
        raw = loader.load_tango_spectra()
    else:
        raw = loader.load_neospectra_spectra()

    data = loader.build_dataset(raw, hesperidin, spectrometer.upper(),
                                exclude_ids=exclude_ids or [])
    return data, sample_info


# ===========================================================================
# Preprocessing
# ===========================================================================

def mean_spectra(spectra, sample_ids):
    """Mittelt Wiederholungsmessungen pro Probe."""
    unique_ids = sorted(set(sample_ids), key=lambda x: (not x.isdigit(), x))
    mean_specs = []
    for sid in unique_ids:
        indices = [i for i, s in enumerate(sample_ids) if s == sid]
        mean_specs.append(np.mean([spectra[i] for i in indices], axis=0))
    return np.array(mean_specs), unique_ids


def snv(spectra):
    """Standard Normal Variate."""
    result = np.zeros_like(spectra, dtype=float)
    for i in range(spectra.shape[0]):
        m, s = np.mean(spectra[i]), np.std(spectra[i])
        result[i] = (spectra[i] - m) / s if s > 0 else spectra[i] - m
    return result


def savgol_derivative(spectra, window_length=15, polyorder=2):
    """Savitzky-Golay 1. Ableitung."""
    wl = min(window_length, spectra.shape[1])
    if wl % 2 == 0:
        wl -= 1
    return savgol_filter(spectra, window_length=wl, polyorder=polyorder,
                         deriv=1, axis=1)


def preprocess(data, use_savgol=False):
    """Mitteln → SNV → optional SavGol 1. Ableitung."""
    spectra = np.array(data["spectra"])
    sample_ids = data["sample_ids"]
    wavelengths = np.array(data["wavelengths"])
    hesperidin = data["hesperidin_content"]

    averaged, unique_ids = mean_spectra(spectra, sample_ids)

    # Hesperidin pro Probe
    hesp_map = {}
    for sid, h in zip(sample_ids, hesperidin):
        hesp_map[sid] = h
    y = np.array([hesp_map[sid] for sid in unique_ids])

    # SNV
    X = snv(averaged)

    # Optional SavGol 1. Ableitung
    if use_savgol:
        X = savgol_derivative(X)

    return wavelengths, X, y, unique_ids


# ===========================================================================
# Forward Variable Selection
# ===========================================================================

def forward_select_variables(X, y, max_vars=8):
    """Greedy Forward Selection."""
    loo = LeaveOneOut()
    selected = []
    remaining = list(range(X.shape[1]))
    step_metrics = []

    for step in range(min(max_vars, X.shape[0] - 2)):
        best_rmse = np.inf
        best_idx = None
        for j in remaining:
            cols = selected + [j]
            y_pred = cross_val_predict(LinearRegression(), X[:, cols], y, cv=loo)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            if rmse < best_rmse:
                best_rmse = rmse
                best_idx = j

        selected.append(best_idx)
        remaining.remove(best_idx)

        # Metriken für diesen Schritt
        y_pred = cross_val_predict(LinearRegression(), X[:, selected], y, cv=loo)
        r2 = r2_score(y, y_pred)
        step_metrics.append({"n_vars": step + 1, "rmse": best_rmse, "r2": r2})

    return selected, step_metrics


# ===========================================================================
# Modelle bauen
# ===========================================================================

def build_all_models(max_comp=4):
    """Erstellt Dict {name: model} für PLS, PCR, ICA × Regressoren."""
    models = {}

    for n in range(1, max_comp + 1):
        models[f"PLS({n})"] = PLSRegression(n_components=n, scale=True)

        for reg_name, reg in [("LR", LinearRegression()),
                               ("Ridge", Ridge(alpha=1.0)),
                               ("Lasso", Lasso(alpha=0.1, max_iter=10000))]:
            models[f"PCR({n}) {reg_name}"] = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n)),
                ("reg", reg),
            ])
            models[f"ICA({n}) {reg_name}"] = Pipeline([
                ("scaler", StandardScaler()),
                ("ica", FastICA(n_components=n, max_iter=1000, random_state=42)),
                ("reg", reg),
            ])

    return models


# ===========================================================================
# Evaluation
# ===========================================================================

def evaluate_all(X, y, selected_indices, max_comp=4):
    """LOO-CV + Training-Metriken für alle Modelle."""
    loo = LeaveOneOut()
    results = []

    # --- VarSel Modelle ---
    for k in range(1, len(selected_indices) + 1):
        cols = selected_indices[:k]
        Xsub = X[:, cols]
        model = LinearRegression()
        name = f"VarSel({k}) LR"
        try:
            y_cv = cross_val_predict(model, Xsub, y, cv=loo).flatten()
            model.fit(Xsub, y)
            y_train = model.predict(Xsub).flatten()
            results.append(_make_result(name, y, y_cv, y_train))
        except Exception:
            pass

    # --- PLS, PCR, ICA Modelle ---
    models = build_all_models(max_comp)
    for name, model in models.items():
        try:
            y_cv = cross_val_predict(model, X, y, cv=loo).flatten()
            model.fit(X, y)
            y_train = model.predict(X).flatten()
            results.append(_make_result(name, y, y_cv, y_train))
        except Exception:
            pass

    results.sort(key=lambda r: r["RMSECV"])
    return results


def nested_loo_varsel(X, y, max_vars=8):
    """Nested LOO: Variablenselektion + Modellfit INNERHALB jedes Folds.
    Gibt ehrliche Vorhersageperformance für neue Proben."""
    loo = LeaveOneOut()
    n = X.shape[0]
    max_k = min(max_vars, n - 3)  # mind. 2 Freiheitsgrade im Training

    # Pro Variablenanzahl: LOO-Vorhersagen sammeln
    results = {}
    for k in range(1, max_k + 1):
        results[k] = np.zeros(n)

    for fold_idx, (train_idx, test_idx) in enumerate(loo.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        # Forward Selection NUR auf Trainingsdaten
        selected = []
        remaining = list(range(X_train.shape[1]))
        inner_loo = LeaveOneOut()

        for step in range(max_k):
            best_rmse = np.inf
            best_j = None
            for j in remaining:
                cols = selected + [j]
                y_inner = cross_val_predict(LinearRegression(),
                                            X_train[:, cols], y_train,
                                            cv=inner_loo)
                rmse = np.sqrt(mean_squared_error(y_train, y_inner))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_j = j
            selected.append(best_j)
            remaining.remove(best_j)

            # Modell mit k=step+1 Variablen fitten und Testprobe vorhersagen
            k = step + 1
            model = LinearRegression()
            model.fit(X_train[:, selected], y_train)
            results[k][fold_idx] = model.predict(X_test[:, selected])[0]

    # Metriken berechnen
    out = []
    for k in range(1, max_k + 1):
        y_cv = results[k]
        # Auch Trainingsmetriken (Resubstitution mit globaler VarSel)
        model = LinearRegression()
        # Globale VarSel für Trainingsmetriken (forward_select_variables)
        global_sel, _ = forward_select_variables(X, y, max_vars=k)
        model.fit(X[:, global_sel[:k]], y)
        y_train_pred = model.predict(X[:, global_sel[:k]]).flatten()

        out.append({
            "name": f"NestedLOO VarSel({k})",
            "R2_train": float(r2_score(y, y_train_pred)),
            "RMSE_train": float(np.sqrt(mean_squared_error(y, y_train_pred))),
            "R2_CV": float(r2_score(y, y_cv)),
            "RMSECV": float(np.sqrt(mean_squared_error(y, y_cv))),
            "y_pred_cv": y_cv.tolist(),
            "y_pred_train": y_train_pred.tolist(),
        })

    return out


def _make_result(name, y, y_cv, y_train):
    return {
        "name": name,
        "R2_train": float(r2_score(y, y_train)),
        "RMSE_train": float(np.sqrt(mean_squared_error(y, y_train))),
        "R2_CV": float(r2_score(y, y_cv)),
        "RMSECV": float(np.sqrt(mean_squared_error(y, y_cv))),
        "y_pred_cv": y_cv.tolist(),
        "y_pred_train": y_train.tolist(),
    }


# ===========================================================================
# Plots
# ===========================================================================

def _save_plot(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_preprocessing(wavelengths, averaged, X, use_savgol, dataset_name, out_dir):
    """Preprocessing-Schritte plotten."""
    n_steps = 3 if use_savgol else 2
    fig, axes = plt.subplots(1, n_steps, figsize=(7 * n_steps, 5))
    if n_steps == 2:
        axes = list(axes)

    x_label = "Wellenzahl (cm⁻¹)" if "TANGO" in dataset_name else "Wellenlänge (nm)"

    for i in range(averaged.shape[0]):
        axes[0].plot(wavelengths, averaged[i], alpha=0.7, linewidth=0.8)
    axes[0].set_title("Gemittelte Rohspektren")
    axes[0].set_xlabel(x_label)
    axes[0].grid(True, alpha=0.3)

    snv_data = snv(averaged)
    for i in range(snv_data.shape[0]):
        axes[1].plot(wavelengths, snv_data[i], alpha=0.7, linewidth=0.8)
    axes[1].set_title("SNV")
    axes[1].set_xlabel(x_label)
    axes[1].grid(True, alpha=0.3)

    if use_savgol:
        for i in range(X.shape[0]):
            axes[2].plot(wavelengths, X[i], alpha=0.7, linewidth=0.8)
        axes[2].set_title("SNV + SavGol 1. Ableitung")
        axes[2].set_xlabel(x_label)
        axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"{dataset_name} – Preprocessing", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_plot(fig, out_dir / "preprocessing.png")


def plot_varsel_on_spectrum(wavelengths, X, selected_indices, dataset_name, out_dir):
    """Zeigt gewählte Wellenlängen auf dem mittleren Spektrum."""
    mean_spec = np.mean(X, axis=0)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(wavelengths, mean_spec, color="gray", linewidth=0.8, label="Mittleres Spektrum")

    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_indices)))
    for rank, (idx, color) in enumerate(zip(selected_indices, colors)):
        wl = wavelengths[idx]
        ax.axvline(wl, color=color, linestyle="--", alpha=0.8, linewidth=1.5)
        ax.scatter([wl], [mean_spec[idx]], color=color, s=100, zorder=5,
                   edgecolors="k", linewidth=0.5)
        ax.annotate(f"#{rank+1}: {wl:.0f}", xy=(wl, mean_spec[idx]),
                    xytext=(5, 10 + rank * 8), textcoords="offset points",
                    fontsize=8, color=color, fontweight="bold")

    x_label = "Wellenzahl (cm⁻¹)" if "TANGO" in dataset_name else "Wellenlänge (nm)"
    ax.set_xlabel(x_label)
    ax.set_ylabel("Intensität (SNV)")
    ax.set_title(f"{dataset_name} – Gewählte Variablen (Forward Selection)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_plot(fig, out_dir / "varsel_on_spectrum.png")


def plot_varsel_r2_gain(step_metrics, dataset_name, out_dir):
    """R²CV-Gewinn pro hinzugefügter Variable."""
    ns = [m["n_vars"] for m in step_metrics]
    r2s = [m["r2"] for m in step_metrics]
    rmses = [m["rmse"] for m in step_metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{dataset_name} – Variablenselektion: Schrittweise Verbesserung",
                 fontsize=13, fontweight="bold")

    ax1.plot(ns, r2s, "o-", color="forestgreen", linewidth=2, markersize=8)
    ax1.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Anzahl Variablen")
    ax1.set_ylabel("R²CV")
    ax1.set_title("R²CV vs. Variablenanzahl")
    ax1.grid(True, alpha=0.3)

    ax2.plot(ns, rmses, "o-", color="steelblue", linewidth=2, markersize=8)
    ax2.set_xlabel("Anzahl Variablen")
    ax2.set_ylabel("RMSECV (%)")
    ax2.set_title("RMSECV vs. Variablenanzahl")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_plot(fig, out_dir / "varsel_r2_gain.png")


def plot_varsel_univariate(X, y, wavelengths, dataset_name, out_dir):
    """RMSECV pro Einzelvariable über gesamtes Spektrum."""
    loo = LeaveOneOut()
    rmses = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        y_pred = cross_val_predict(LinearRegression(), X[:, j:j+1], y, cv=loo)
        rmses[j] = np.sqrt(mean_squared_error(y, y_pred))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(wavelengths, rmses, color="steelblue", linewidth=0.8)
    best_idx = np.argmin(rmses)
    ax.axvline(wavelengths[best_idx], color="red", linestyle="--", alpha=0.7,
               label=f"Beste: {wavelengths[best_idx]:.1f} (RMSECV={rmses[best_idx]:.4f})")
    ax.set_xlabel("Wellenzahl (cm⁻¹)" if "TANGO" in dataset_name else "Wellenlänge (nm)")
    ax.set_ylabel("RMSECV (%)")
    ax.set_title(f"{dataset_name} – Univariate RMSECV pro Wellenlänge")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_plot(fig, out_dir / "varsel_univariate.png")


def plot_scatter_models(y, results, sample_ids, dataset_name, out_dir):
    """Scatter-Plots für Top-Modelle (LOO-Vorhersagen)."""
    # Top 6 Modelle plotten
    top = results[:6]
    n = len(top)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5.5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for i, r in enumerate(top):
        ax = axes[i]
        y_pred = np.array(r["y_pred_cv"])

        unique_ids = sorted(set(sample_ids), key=lambda x: (not x.isdigit(), x))
        cmap = plt.get_cmap("tab20", len(unique_ids))
        colors = {sid: cmap(j) for j, sid in enumerate(unique_ids)}

        for sid in unique_ids:
            mask = [s == sid for s in sample_ids]
            ax.scatter(np.array(y)[mask], y_pred[mask], color=colors[sid],
                       label=f"{sid}", s=60, edgecolors="k", linewidth=0.5, zorder=3)

        lims = [min(min(y), min(y_pred)) - 0.2, max(max(y), max(y_pred)) + 0.2]
        ax.plot(lims, lims, "k--", alpha=0.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Gemessen (%)")
        ax.set_ylabel("Vorhergesagt (%)")
        ax.set_title(f"{r['name']}\nR²CV={r['R2_CV']:.4f}  RMSECV={r['RMSECV']:.4f}%",
                     fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, ncol=2)

    # Leere Subplots ausblenden
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{dataset_name} – LOO-Vorhersagen (Top {n} Modelle)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_plot(fig, out_dir / "scatter_top_models.png")


def plot_model_comparison(results, dataset_name, out_dir):
    """Balkendiagramm: Alle Modelle."""
    names = [r["name"] for r in results]
    rmses = [r["RMSECV"] for r in results]
    r2s = [r["R2_CV"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, max(8, len(names) * 0.32)))
    fig.suptitle(f"{dataset_name} – Modellvergleich ({len(names)} Modelle)",
                 fontsize=14, fontweight="bold")

    colors = ["forestgreen" if i == 0 else "steelblue" for i in range(len(names))]

    ax1.barh(range(len(names)), rmses, color=colors, edgecolor="k", linewidth=0.5)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=7)
    ax1.set_xlabel("RMSECV (%)")
    ax1.set_title("RMSECV (kleiner = besser)")
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis="x")

    ax2.barh(range(len(names)), r2s, color=colors, edgecolor="k", linewidth=0.5)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=7)
    ax2.set_xlabel("R²CV")
    ax2.set_title("R²CV (größer = besser)")
    ax2.invert_yaxis()
    ax2.axvline(0, color="red", linestyle="--", alpha=0.5)
    ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    _save_plot(fig, out_dir / "model_comparison.png")


# ===========================================================================
# Tabellen-Ausgabe
# ===========================================================================

def print_results_table(results, dataset_name, run_name):
    """Ergebnistabelle mit Training- und CV-Metriken."""
    header = (f"\n{'='*90}\n"
              f"  {run_name} | {dataset_name} – Modellvergleich (LOO-CV, n={len(results[0]['y_pred_cv'])})\n"
              f"{'='*90}")
    print(header)

    print(f"  {'#':>2}  {'Modell':30s}  {'R²train':>8s}  {'RMSEtr':>8s}  │  {'R²CV':>8s}  {'RMSECV':>8s}")
    print("  " + "-" * 85)

    for i, r in enumerate(results):
        marker = " << BEST" if i == 0 else ""
        print(f"  {i+1:>2}. {r['name']:30s}  {r['R2_train']:>+8.4f}  {r['RMSE_train']:>8.4f}  │"
              f"  {r['R2_CV']:>+8.4f}  {r['RMSECV']:>8.4f}{marker}")

    print("  " + "-" * 85)
    best = results[0]
    print(f"  Bestes: {best['name']} → R²CV={best['R2_CV']:.4f}, RMSECV={best['RMSECV']:.4f}%")
    print()


def print_loo_table(y, y_pred, sample_ids, model_name, dataset_name):
    """LOO pro Probe."""
    print(f"\n  {dataset_name} – LOO-Vorhersagen ({model_name})")
    print(f"  {'Probe':>8s}  {'Gemessen%':>10s}  {'Vorherges%':>10s}  {'Fehler%':>10s}  {'Rel.%':>8s}")
    print("  " + "-" * 55)

    errors = []
    for sid, yt, yp in zip(sample_ids, y, y_pred):
        err = yp - yt
        errors.append(abs(err))
        print(f"  {sid:>8s}  {yt:>10.4f}  {yp:>10.4f}  {err:>+10.4f}  {abs(err)/yt*100:>8.1f}")

    mae = np.mean(errors)
    print("  " + "-" * 55)
    print(f"  {'MAE':>8s}  {'':>10s}  {'':>10s}  {mae:>10.4f}  {mae/np.mean(y)*100:>8.1f}")
    print()


# ===========================================================================
# Hauptpipeline
# ===========================================================================

def run_pipeline(params):
    """
    Führt komplette Pipeline aus.

    params = {
        "name": "TANGO_SNV_all",
        "spectrometer": "tango" | "neospectra",
        "use_savgol": False,
        "exclude_ids": [],
        "max_varsel": 8,
        "max_components": 4,
        "output_dir": "runs/TANGO_SNV_all",
    }
    """
    name = params["name"]
    spectrometer = params["spectrometer"]
    use_savgol = params.get("use_savgol", False)
    exclude_ids = params.get("exclude_ids", [])
    max_varsel = params.get("max_varsel", 8)
    max_comp = params.get("max_components", 4)
    out_dir = Path(params["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = spectrometer.upper()
    preproc_label = "SNV+SG1d" if use_savgol else "SNV"
    filter_label = f"excl{'+'.join(exclude_ids)}" if exclude_ids else "alle"

    print(f"\n{'#'*90}")
    print(f"  Pipeline: {name}")
    print(f"  Spektrometer: {dataset_name} | Preprocessing: {preproc_label} | "
          f"Filter: {filter_label}")
    print(f"  Output: {out_dir}")
    print(f"{'#'*90}")

    # 1) Daten laden
    data, sample_info = load_data(spectrometer, exclude_ids)

    # 2) Preprocessing
    wavelengths, X, y, sample_ids = preprocess(data, use_savgol)
    print(f"  Daten: {X.shape[0]} Proben × {X.shape[1]} Features")
    print(f"  Hesperidin: {y.min():.2f}% - {y.max():.2f}%")
    print(f"  Proben: {sample_ids}")

    # Preprocessing-Plot
    averaged, _ = mean_spectra(np.array(data["spectra"]), data["sample_ids"])
    plot_preprocessing(wavelengths, averaged, X, use_savgol, f"{name}", out_dir)

    # 3) Variablenselektion
    max_vs = min(max_varsel, X.shape[0] - 2)
    selected, step_metrics = forward_select_variables(X, y, max_vars=max_vs)
    sel_wls = [float(wavelengths[i]) for i in selected]
    print(f"  Gewählte Variablen: {[f'{w:.0f}' for w in sel_wls]}")

    # VarSel-Plots
    plot_varsel_on_spectrum(wavelengths, X, selected, f"{name}", out_dir)
    plot_varsel_r2_gain(step_metrics, f"{name}", out_dir)
    plot_varsel_univariate(X, y, wavelengths, f"{name}", out_dir)

    # 4) Alle Modelle evaluieren (Standard LOO-CV)
    results = evaluate_all(X, y, selected, max_comp)

    # 5) Nested LOO (ehrliche Performance für neue Proben)
    print(f"  Nested LOO-CV läuft (VarSel innerhalb jedes Folds)...")
    nested_results = nested_loo_varsel(X, y, max_vars=max_vs)
    all_results = nested_results + results
    all_results.sort(key=lambda r: r["RMSECV"])

    # 6) Tabellen ausgeben
    print_results_table(all_results, dataset_name, name)

    # LOO-Tabelle für bestes Nested-Modell und bestes Standard-Modell
    best_nested = min(nested_results, key=lambda r: r["RMSECV"])
    best_standard = results[0]
    print_loo_table(y, np.array(best_nested["y_pred_cv"]), sample_ids,
                    best_nested["name"], f"{name} (ehrliche Schätzung)")
    if best_standard["name"] != best_nested["name"]:
        print_loo_table(y, np.array(best_standard["y_pred_cv"]), sample_ids,
                        best_standard["name"], f"{name} (optimistisch)")

    # 7) Plots
    plot_scatter_models(y, all_results, sample_ids, f"{name}", out_dir)
    plot_model_comparison(all_results, f"{name}", out_dir)

    results = all_results

    # 8) Ergebnisse speichern
    best = results[0]
    run_result = {
        "name": name,
        "spectrometer": spectrometer,
        "preprocessing": preproc_label,
        "excluded": exclude_ids,
        "n_samples": int(X.shape[0]),
        "selected_wavelengths": sel_wls,
        "varsel_steps": step_metrics,
        "models": [{k: v for k, v in r.items() if k not in ("y_pred_cv", "y_pred_train")}
                   for r in results],
        "best_model": best["name"],
        "best_R2CV": best["R2_CV"],
        "best_RMSECV": best["RMSECV"],
        "best_nested_model": best_nested["name"],
        "best_nested_R2CV": best_nested["R2_CV"],
        "best_nested_RMSECV": best_nested["RMSECV"],
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(run_result, f, indent=2)

    return run_result
