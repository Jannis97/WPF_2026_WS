#!/usr/bin/env python3
"""
evaluation.py – Evaluierung und Visualisierung der Modellperformance.

Metriken: R2, RMSE, MAE, Bias, RPD
Alle Plots als Dual-Panel mit Konzentration und Probenklasse.

Erzeugt:
  - plots/eval_scatter_*.png
  - plots/eval_residuals_*.png
  - plots/eval_metrics_comparison.png
  - plots/eval_cv_boxplot.png
  - results_table.md
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from importlib.util import spec_from_file_location, module_from_spec

BENCHTOP_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(BENCHTOP_DIR, "plots")
DPI = 300


def _load_dataloading():
    spec = spec_from_file_location("dataloading", os.path.join(BENCHTOP_DIR, "0_dataloading.py"))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_splitting():
    spec = spec_from_file_location("splitting", os.path.join(BENCHTOP_DIR, "2_data_splitting.py"))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_models():
    spec = spec_from_file_location("models", os.path.join(BENCHTOP_DIR, "3_regression_models.py"))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─── Metriken ────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    """Berechnet Regressionsmetriken.

    Returns:
        dict mit R2, RMSE, MAE, Bias, RPD
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    bias = np.mean(y_pred - y_true)
    rpd = np.std(y_true) / rmse if rmse > 0 else float("inf")

    return {"R2": r2, "RMSE": rmse, "MAE": mae, "Bias": bias, "RPD": rpd}


# ─── Plot-Funktionen ─────────────────────────────────────────

def plot_train_test_scatter(y_train, y_pred_train, y_test, y_pred_test,
                            sample_ids_train=None, sample_ids_test=None,
                            conc_map=None, type_map=None,
                            model_name="", save_path=None):
    """2x2 Scatter: (Train/Test) x (Konzentration/Probenklasse)."""
    dl = _load_dataloading()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    datasets = [
        (axes[0, 0], axes[0, 1], y_train, y_pred_train, sample_ids_train, "Train"),
        (axes[1, 0], axes[1, 1], y_test, y_pred_test, sample_ids_test, "Test"),
    ]

    for ax_conc, ax_class, yt, yp, sids, label in datasets:
        if len(yt) == 0:
            continue

        mn = min(yt.min(), yp.min()) - 0.2
        mx = max(yt.max(), yp.max()) + 0.2

        for ax in [ax_conc, ax_class]:
            ax.plot([mn, mx], [mn, mx], "k--", linewidth=0.8)
            ax.set_xlim(mn, mx)
            ax.set_ylim(mn, mx)
            ax.set_xlabel("Tatsaechlich [%]")
            ax.set_ylabel("Vorhergesagt [%]")
            ax.set_aspect("equal")

        metrics = compute_metrics(yt, yp)

        # Konzentration
        if sids and conc_map:
            colors_conc, norm, cmap = dl._get_colors_by_concentration(sids, conc_map)
            for i in range(len(yt)):
                ax_conc.scatter(yt[i], yp[i], color=colors_conc[i],
                                edgecolors="k", s=60, zorder=3)
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax_conc, label="Hesperidin [%]")
        else:
            ax_conc.scatter(yt, yp, edgecolors="k", facecolors="#4ECDC4", s=60, zorder=3)

        ax_conc.set_title(f"{label} Konzentration \u2013 R\u00b2={metrics['R2']:.3f}, "
                          f"RMSE={metrics['RMSE']:.3f}")

        # Probenklasse
        if sids and type_map:
            colors_class, legend_handles = dl._get_colors_by_probe_class(sids, type_map)
            for i in range(len(yt)):
                ax_class.scatter(yt[i], yp[i], color=colors_class[i],
                                 edgecolors="k", s=60, zorder=3)
            ax_class.legend(handles=legend_handles, fontsize=6, loc="upper left", ncol=2)
        else:
            ax_class.scatter(yt, yp, edgecolors="k", facecolors="#FF6B6B", s=60, zorder=3)

        ax_class.set_title(f"{label} Probenklasse \u2013 R\u00b2={metrics['R2']:.3f}")

    fig.suptitle(f"{model_name} \u2013 Train vs. Test", fontsize=14)
    plt.tight_layout()

    if save_path is None:
        safe = model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        save_path = os.path.join(PLOT_DIR, f"eval_scatter_{safe}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    print(f"  -> {save_path}")
    plt.close(fig)


def plot_residuals(y_true, y_pred, sample_ids=None, conc_map=None, type_map=None,
                   model_name="", label="", save_path=None):
    """Dual-Panel Residuen: nach Konzentration und Histogramm."""
    dl = _load_dataloading()
    residuals = y_true - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Links: Residuen vs Vorhergesagt (farbig nach Konzentration)
    if sample_ids and conc_map:
        colors_conc, norm, cmap = dl._get_colors_by_concentration(sample_ids, conc_map)
        for i in range(len(y_pred)):
            ax1.scatter(y_pred[i], residuals[i], color=colors_conc[i],
                        edgecolors="k", s=60, zorder=3)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax1, label="Hesperidin [%]")
    else:
        ax1.scatter(y_pred, residuals, edgecolors="k", facecolors="#FFE66D", s=50)

    ax1.axhline(0, color="k", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("Vorhergesagt [%]")
    ax1.set_ylabel("Residuum [%]")
    ax1.set_title("Residuen vs. Vorhersage")

    # Rechts: Histogramm
    ax2.hist(residuals, bins=max(5, len(residuals) // 3),
             color="#FFE66D", edgecolor="k")
    ax2.set_xlabel("Residuum [%]")
    ax2.set_ylabel("Haeufigkeit")
    ax2.set_title("Residuen-Verteilung")

    fig.suptitle(f"{model_name} \u2013 Residuen ({label})", fontsize=13)
    plt.tight_layout()

    if save_path is None:
        safe = model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        safe_l = label.replace(" ", "_").lower()
        save_path = os.path.join(PLOT_DIR, f"eval_residuals_{safe}_{safe_l}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    print(f"  -> {save_path}")
    plt.close(fig)


def plot_metrics_comparison(results_list, save_path=None):
    """Balkendiagramm: Metriken aller Modelle/Strategien vergleichen."""
    if not results_list:
        return

    model_names = [r["model_name"] for r in results_list]
    metrics_keys = ["R2", "RMSE", "MAE", "RPD"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.ravel()
    x = np.arange(len(model_names))
    width = 0.35

    for i, key in enumerate(metrics_keys):
        ax = axes[i]
        train_vals = [r.get("train_metrics", {}).get(key, 0) for r in results_list]
        test_vals = [r.get("test_metrics", {}).get(key, 0) for r in results_list]

        ax.bar(x - width / 2, train_vals, width, label="Train",
               color="#4ECDC4", edgecolor="k")
        ax.bar(x + width / 2, test_vals, width, label="Test/CV",
               color="#FF6B6B", edgecolor="k")

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=35, ha="right", fontsize=7)
        ax.set_title(key)
        ax.legend(fontsize=8)

    fig.suptitle("Modellvergleich \u2013 Train vs. Test/CV", fontsize=14)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(PLOT_DIR, "eval_metrics_comparison.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    print(f"  -> {save_path}")
    plt.close(fig)


def plot_cv_boxplot(cv_results, metric="RMSE", save_path=None):
    """Boxplot: CV-Ergebnisse pro Modell.

    Args:
        cv_results: dict {model_name: list of fold metric values}
        metric: Name der Metrik
    """
    if not cv_results:
        return

    names = list(cv_results.keys())
    data = [cv_results[n] for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, patch_artist=True, tick_labels=names)

    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_ylabel(f"{metric} (CV)")
    ax.set_title(f"Kreuzvalidierung \u2013 {metric} pro Fold")
    if metric == "R2":
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(PLOT_DIR, "eval_cv_boxplot.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    print(f"  -> {save_path}")
    plt.close(fig)


def create_results_table(results_list, outpath=None):
    """Erstellt Markdown-Tabelle mit Modell-Metriken.

    Args:
        results_list: list of dicts mit 'model_name', 'train_metrics', 'test_metrics'

    Returns:
        str: Markdown-Tabelle
    """
    if outpath is None:
        outpath = os.path.join(BENCHTOP_DIR, "results_table.md")

    lines = [
        "# Modellvergleich \u2013 Ergebnisse",
        "",
        "| Modell | R2(Train) | RMSE(Train) | MAE(Train) | R2(CV) | RMSE(CV) | MAE(CV) | RPD(CV) | Bias(CV) |",
        "|--------|-----------|-------------|------------|--------|----------|---------|---------|----------|",
    ]

    for r in results_list:
        tr = r.get("train_metrics", {})
        te = r.get("test_metrics", {})
        lines.append(
            f"| {r['model_name']} "
            f"| {tr.get('R2', float('nan')):.4f} | {tr.get('RMSE', float('nan')):.4f} "
            f"| {tr.get('MAE', float('nan')):.4f} "
            f"| {te.get('R2', float('nan')):.4f} | {te.get('RMSE', float('nan')):.4f} "
            f"| {te.get('MAE', float('nan')):.4f} "
            f"| {te.get('RPD', float('nan')):.2f} "
            f"| {te.get('Bias', float('nan')):.4f} |"
        )

    lines.append("")
    table_str = "\n".join(lines)

    with open(outpath, "w") as f:
        f.write(table_str)
    print(f"  -> {outpath}")

    return table_str


# ─── Main ─────────────────────────────────────────────────────

def main():
    print("=== evaluation (Demo) ===")
    os.makedirs(PLOT_DIR, exist_ok=True)

    dl = _load_dataloading()
    sp = _load_splitting()
    mm = _load_models()
    pp_spec = spec_from_file_location("preprocessing",
                                       os.path.join(BENCHTOP_DIR, "1_data_preprocessing.py"))
    pp = module_from_spec(pp_spec)
    pp_spec.loader.exec_module(pp)

    wavenumbers, sample_ids, spectra = dl.load_spectra()
    hplc_df = dl.load_hplc()
    conc_map = dl.get_concentration_map(hplc_df)
    type_map = dl.get_sample_type_map(hplc_df)

    # Preprocessing
    processed, steps = pp.preprocess(spectra, {"snv": True, "savgol": False})

    # Filter: nur Proben mit Referenz
    known_ids = set(conc_map.keys())
    mask = [sid in known_ids for sid in sample_ids]
    X = processed[mask]
    y = np.array([conc_map[sid] for sid, m in zip(sample_ids, mask) if m])
    ids_f = [sid for sid, m in zip(sample_ids, mask) if m]

    print(f"Daten: X={X.shape}, y={y.shape}")

    # GroupLeaveOneOut-CV
    splitter = sp.LeaveOneGroupOut()

    models_configs = [
        ("pls", {"n_components": 1}),
        ("pls", {"n_components": 2}),
        ("pls", {"n_components": 3}),
        ("pcr", {"n_components": 2}),
        ("pcr", {"n_components": 3}),
    ]

    all_results = []
    cv_fold_metrics = {}

    for model_name, model_params in models_configs:
        y_pred_all = np.full_like(y, np.nan)
        fold_rmses = []

        for train_idx, test_idx in splitter.split(X, y, ids_f):
            model = mm.get_model_by_name(model_name, model_params)
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[test_idx])
            y_pred_all[test_idx] = preds
            if len(preds) > 0:
                fold_rmse = np.sqrt(np.mean((y[test_idx] - preds) ** 2))
                fold_rmses.append(fold_rmse)

        cv_metrics = compute_metrics(y, y_pred_all)

        # Train metrics (full data)
        full_model = mm.get_model_by_name(model_name, model_params)
        full_model.fit(X, y)
        y_pred_train = full_model.predict(X)
        train_metrics = compute_metrics(y, y_pred_train)

        model_label = mm.get_model_by_name(model_name, model_params).name()
        print(f"  {model_label}: Train R2={train_metrics['R2']:.4f}, "
              f"CV R2={cv_metrics['R2']:.4f}, CV RMSE={cv_metrics['RMSE']:.4f}")

        all_results.append({
            "model_name": model_label,
            "train_metrics": train_metrics,
            "test_metrics": cv_metrics,
        })

        cv_fold_metrics[model_label] = fold_rmses

    plot_metrics_comparison(all_results)
    plot_cv_boxplot(cv_fold_metrics, metric="RMSE")
    create_results_table(all_results)

    # Residuen-Plot fuer bestes Modell
    best_idx = np.argmin([r["test_metrics"]["RMSE"] for r in all_results])
    best_name = all_results[best_idx]["model_name"]
    print(f"\nBestes Modell (CV RMSE): {best_name}")

    # Rerun best model for residual plot
    for mname, mparams in models_configs:
        m = mm.get_model_by_name(mname, mparams)
        if m.name() == best_name:
            y_pred_cv = np.full_like(y, np.nan)
            for train_idx, test_idx in splitter.split(X, y, ids_f):
                m2 = mm.get_model_by_name(mname, mparams)
                m2.fit(X[train_idx], y[train_idx])
                y_pred_cv[test_idx] = m2.predict(X[test_idx])

            plot_residuals(y, y_pred_cv, sample_ids=ids_f, conc_map=conc_map,
                           type_map=type_map, model_name=best_name, label="GroupLOO_CV")
            break

    print("Done.")


if __name__ == "__main__":
    main()
