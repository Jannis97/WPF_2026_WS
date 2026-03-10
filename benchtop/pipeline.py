#!/usr/bin/env python3
"""
pipeline.py – Vollstaendige Pipeline fuer Benchtop NIR Hesperidin-Kalibrierung.

Strategien:
  1. Alle Proben, alle Replikate, GroupLOO-CV
  2. Nur Mandarinen (exclude 2,3,5), alle Replikate, GroupLOO-CV
  3. Variable Selection (VIP top-N), dann Modell

Preprocessing-Varianten: SNV, SNV+SG1d, SG1d

Nested CV:
  - Aeussere Schleife: GroupLOO (Evaluation)
  - Innere Schleife: GroupLOO (Modellauswahl)

Erzeugt:
  - plots/pipeline_*.png
  - results_table.md
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from importlib.util import spec_from_file_location, module_from_spec

BENCHTOP_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(BENCHTOP_DIR, "plots")
DPI = 300


def _load_module(filename):
    path = os.path.join(BENCHTOP_DIR, filename)
    spec = spec_from_file_location(filename.replace(".py", ""), path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─── Konfiguration ───────────────────────────────────────────

PREPROCESSING_VARIANTS = {
    "SNV": {"snv": True, "savgol": False},
    "SNV_SG1d": {"snv": True, "savgol": True, "savgol_window": 15,
                 "savgol_poly": 2, "savgol_deriv": 1},
    "SG1d": {"snv": False, "savgol": True, "savgol_window": 15,
             "savgol_poly": 2, "savgol_deriv": 1},
}

MODEL_CONFIGS = [
    {"name": "pls", "param_grid": [{"n_components": c} for c in [1, 2, 3]]},
    {"name": "pcr", "param_grid": [{"n_components": c} for c in [1, 2, 3]]},
]

EXCLUDE_NON_MANDARIN = ["2", "3", "5"]  # G=Grapefruit, Z=Zitrone, L=Limette


# ─── Hilfsfunktionen ─────────────────────────────────────────

def prepare_data(dl, pp, pp_params, exclude_ids=None):
    """Lade Daten, preprocesse, filtere.

    Returns:
        wavenumbers, X, y, sample_ids, conc_map, type_map
    """
    wavenumbers, sample_ids, spectra = dl.load_spectra()
    hplc_df = dl.load_hplc()
    conc_map = dl.get_concentration_map(hplc_df)
    type_map = dl.get_sample_type_map(hplc_df)

    # Preprocessing
    processed, steps = pp.preprocess(spectra, pp_params)

    # Nur Proben mit HPLC-Referenz
    known_ids = set(conc_map.keys())
    if exclude_ids:
        known_ids -= set(exclude_ids)

    mask = [sid in known_ids for sid in sample_ids]
    X = processed[mask]
    y = np.array([conc_map[sid] for sid, m in zip(sample_ids, mask) if m])
    ids_f = [sid for sid, m in zip(sample_ids, mask) if m]

    return wavenumbers, X, y, ids_f, conc_map, type_map


def run_group_loo_cv(X, y, groups, model_factory):
    """Fuehre GroupLeaveOneOut-CV aus.

    Returns:
        y_pred: np.array (n,) – CV-Vorhersagen
        fold_metrics: list of dicts – Metriken pro Fold
    """
    sp = _load_module("2_data_splitting.py")
    ev = _load_module("evaluation.py")

    splitter = sp.LeaveOneGroupOut()
    y_pred = np.full_like(y, np.nan, dtype=float)
    fold_metrics = []

    for train_idx, test_idx in splitter.split(X, y, groups):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        y_pred[test_idx] = preds

        if len(preds) > 0:
            fm = ev.compute_metrics(y[test_idx], preds)
            fm["test_group"] = list(set(np.array(groups)[test_idx]))
            fold_metrics.append(fm)

    return y_pred, fold_metrics


def select_best_model_inner_cv(X_train, y_train, groups_train, model_configs):
    """Innere GroupLOO-CV: waehlt bestes Modell.

    Returns:
        best_name, best_params, best_rmse, selection_results
    """
    rm = _load_module("3_regression_models.py")
    ev = _load_module("evaluation.py")
    sp = _load_module("2_data_splitting.py")

    splitter = sp.LeaveOneGroupOut()
    best_rmse = float("inf")
    best_name = None
    best_params = None
    selection_results = []

    for config in model_configs:
        name = config["name"]
        for params in config["param_grid"]:
            factory = lambda n=name, p=params: rm.get_model_by_name(n, p.copy())

            try:
                y_pred_inner = np.full_like(y_train, np.nan, dtype=float)
                for tr_idx, te_idx in splitter.split(X_train, y_train, groups_train):
                    m = factory()
                    m.fit(X_train[tr_idx], y_train[tr_idx])
                    y_pred_inner[te_idx] = m.predict(X_train[te_idx])

                valid = ~np.isnan(y_pred_inner)
                metrics = ev.compute_metrics(y_train[valid], y_pred_inner[valid])
                rmse = metrics["RMSE"]
                r2 = metrics["R2"]
            except Exception:
                rmse = float("inf")
                r2 = float("nan")

            model_label = rm.get_model_by_name(name, params).name()
            selection_results.append({
                "model": model_label,
                "name": name,
                "params": params,
                "rmse_cv": rmse,
                "r2_cv": r2,
            })

            if rmse < best_rmse:
                best_rmse = rmse
                best_name = name
                best_params = params.copy()

    return best_name, best_params, best_rmse, selection_results


def run_nested_cv(X, y, groups, model_configs, label=""):
    """Nested CV: aeussere GroupLOO, innere GroupLOO.

    Returns:
        results dict
    """
    rm = _load_module("3_regression_models.py")
    ev = _load_module("evaluation.py")
    sp = _load_module("2_data_splitting.py")

    splitter = sp.LeaveOneGroupOut()
    y_pred_outer = np.full_like(y, np.nan, dtype=float)
    fold_results = []
    selected_models = []

    splits = list(splitter.split(X, y, groups))
    n_outer = len(splits)

    print(f"\n{'='*60}")
    print(f"Nested CV ({label}): {n_outer} aeussere Folds")
    print(f"{'='*60}")

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = [groups[i] for i in train_idx]
        test_groups = sorted(set(np.array(groups)[test_idx]))

        # Innere CV: Modellauswahl
        best_name, best_params, best_rmse, sel_res = select_best_model_inner_cv(
            X_train, y_train, groups_train, model_configs
        )

        best_label = rm.get_model_by_name(best_name, best_params).name()
        selected_models.append(best_label)

        # Aeusseres Modell trainieren und evaluieren
        model = rm.get_model_by_name(best_name, best_params)
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        y_pred_outer[test_idx] = y_pred_test

        y_pred_train = model.predict(X_train)
        train_m = ev.compute_metrics(y_train, y_pred_train)
        test_m = ev.compute_metrics(y_test, y_pred_test)

        print(f"  Fold {fold_idx+1}/{n_outer} (Test: {test_groups}): "
              f"{best_label}, RMSE_test={test_m['RMSE']:.4f}")

        fold_results.append({
            "fold": fold_idx + 1,
            "test_groups": test_groups,
            "best_model": best_label,
            "train_metrics": train_m,
            "test_metrics": test_m,
            "selection_results": sel_res,
        })

    valid = ~np.isnan(y_pred_outer)
    overall = ev.compute_metrics(y[valid], y_pred_outer[valid])

    print(f"\nGesamt: R2={overall['R2']:.4f}, RMSE={overall['RMSE']:.4f}, "
          f"RPD={overall['RPD']:.2f}")
    print(f"Gewaehlte Modelle: {Counter(selected_models).most_common()}")

    return {
        "label": label,
        "overall_metrics": overall,
        "fold_results": fold_results,
        "y_true": y,
        "y_pred": y_pred_outer,
        "groups": groups,
        "selected_models": selected_models,
    }


def run_strategy_simple_cv(X, y, groups, model_configs, label=""):
    """Einfache GroupLOO-CV ohne Nested (feste Modelle).

    Returns:
        list of result dicts (one per model config)
    """
    rm = _load_module("3_regression_models.py")
    ev = _load_module("evaluation.py")
    sp = _load_module("2_data_splitting.py")

    splitter = sp.LeaveOneGroupOut()
    results = []

    for config in model_configs:
        name = config["name"]
        for params in config["param_grid"]:
            model_label = rm.get_model_by_name(name, params).name()
            factory = lambda n=name, p=params: rm.get_model_by_name(n, p.copy())

            y_pred = np.full_like(y, np.nan, dtype=float)
            fold_rmses = []

            for train_idx, test_idx in splitter.split(X, y, groups):
                m = factory()
                m.fit(X[train_idx], y[train_idx])
                preds = m.predict(X[test_idx])
                y_pred[test_idx] = preds
                if len(preds) > 0:
                    fold_rmses.append(np.sqrt(np.mean((y[test_idx] - preds) ** 2)))

            valid = ~np.isnan(y_pred)
            cv_metrics = ev.compute_metrics(y[valid], y_pred[valid])

            # Train auf allen Daten
            full_model = factory()
            full_model.fit(X, y)
            y_pred_train = full_model.predict(X)
            train_metrics = ev.compute_metrics(y, y_pred_train)

            results.append({
                "model_name": f"{label}_{model_label}" if label else model_label,
                "train_metrics": train_metrics,
                "test_metrics": cv_metrics,
                "fold_rmses": fold_rmses,
                "y_pred_cv": y_pred,
            })

    return results


# ─── Visualisierungen ────────────────────────────────────────

def plot_nested_cv_scatter(results, conc_map, type_map, save_path=None):
    """Scatter: Nested CV Vorhersagen vs Tatsaechlich, dual panel."""
    dl = _load_module("0_dataloading.py")

    y = results["y_true"]
    y_pred = results["y_pred"]
    groups = results["groups"]
    overall = results["overall_metrics"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    valid = ~np.isnan(y_pred)
    mn = min(y[valid].min(), y_pred[valid].min()) - 0.3
    mx = max(y[valid].max(), y_pred[valid].max()) + 0.3

    for ax in [ax1, ax2]:
        ax.plot([mn, mx], [mn, mx], "k--", linewidth=0.8)
        ax.set_xlim(mn, mx)
        ax.set_ylim(mn, mx)
        ax.set_xlabel("Tatsaechlich [%]")
        ax.set_ylabel("Vorhergesagt [%]")
        ax.set_aspect("equal")

    # Links: Konzentration
    colors_conc, norm, cmap = dl._get_colors_by_concentration(groups, conc_map)
    for i in range(len(y)):
        if valid[i]:
            ax1.scatter(y[i], y_pred[i], color=colors_conc[i],
                        edgecolors="k", s=60, zorder=3)
    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax1, label="Hesperidin [%]")
    ax1.set_title(f"Konzentration \u2013 R\u00b2={overall['R2']:.3f}, "
                  f"RMSE={overall['RMSE']:.3f}")

    # Rechts: Probenklasse mit Annotationen
    colors_class, legend_handles = dl._get_colors_by_probe_class(groups, type_map)
    for i in range(len(y)):
        if valid[i]:
            ax2.scatter(y[i], y_pred[i], color=colors_class[i],
                        edgecolors="k", s=60, zorder=3)
    # Annotate with group ID (only unique positions)
    seen = set()
    for i in range(len(y)):
        if valid[i] and groups[i] not in seen:
            ax2.annotate(groups[i], (y[i], y_pred[i]), fontsize=6,
                         xytext=(4, 4), textcoords="offset points")
            seen.add(groups[i])
    ax2.legend(handles=legend_handles, fontsize=6, loc="upper left", ncol=2)
    ax2.set_title(f"Probenklasse \u2013 RPD={overall['RPD']:.2f}")

    label = results.get("label", "Nested CV")
    fig.suptitle(f"{label}: Nested CV Vorhersagen", fontsize=14)
    plt.tight_layout()

    if save_path is None:
        safe = label.replace(" ", "_").replace("/", "_").lower()
        save_path = os.path.join(PLOT_DIR, f"pipeline_nested_cv_{safe}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    print(f"  -> {save_path}")
    plt.close(fig)


def plot_model_selection_overview(results, save_path=None):
    """Boxplot der inneren CV-RMSEs + Modellwahl-Haeufigkeit."""
    fold_results = results["fold_results"]
    selected = results["selected_models"]

    # Sammle innere RMSEs
    model_rmses = {}
    for fr in fold_results:
        if "selection_results" not in fr:
            continue
        for sr in fr["selection_results"]:
            name = sr["model"]
            if name not in model_rmses:
                model_rmses[name] = []
            if sr["rmse_cv"] < float("inf"):
                model_rmses[name].append(sr["rmse_cv"])

    if not model_rmses:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Links: Boxplot
    names = list(model_rmses.keys())
    data = [model_rmses[n] for n in names]
    bp = ax1.boxplot(data, patch_artist=True, tick_labels=names)
    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax1.set_ylabel("RMSE (innere CV)")
    ax1.set_title("Modell-Selektion: RMSE ueber aeussere Folds")
    plt.setp(ax1.get_xticklabels(), rotation=35, ha="right", fontsize=8)

    # Rechts: Haeufigkeit
    counts = Counter(selected)
    names_freq = list(counts.keys())
    ax2.barh(names_freq, [counts[n] for n in names_freq],
             color="#4ECDC4", edgecolor="k")
    ax2.set_xlabel("Anzahl Folds")
    ax2.set_title("Haeufigkeit der Modellauswahl")

    label = results.get("label", "")
    fig.suptitle(f"{label}: Modell-Selektion", fontsize=14)
    plt.tight_layout()

    if save_path is None:
        safe = label.replace(" ", "_").replace("/", "_").lower()
        save_path = os.path.join(PLOT_DIR, f"pipeline_model_selection_{safe}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    print(f"  -> {save_path}")
    plt.close(fig)


def plot_strategy_comparison(all_strategy_results, save_path=None):
    """Balkendiagramm: Vergleich aller Strategien."""
    ev = _load_module("evaluation.py")

    flat_results = []
    for sr in all_strategy_results:
        if "overall_metrics" in sr:
            # Nested CV result
            flat_results.append({
                "model_name": sr["label"],
                "train_metrics": {},  # not available for nested
                "test_metrics": sr["overall_metrics"],
            })
        elif isinstance(sr, list):
            flat_results.extend(sr)
        else:
            flat_results.append(sr)

    if flat_results:
        ev.plot_metrics_comparison(flat_results, save_path=save_path)


def create_comprehensive_results_table(all_results, outpath=None):
    """Erstellt umfassende Ergebnistabelle fuer alle Strategien."""
    if outpath is None:
        outpath = os.path.join(BENCHTOP_DIR, "results_table.md")

    lines = [
        "# Benchtop NIR Hesperidin-Kalibrierung \u2013 Gesamtergebnisse",
        "",
    ]

    for result in all_results:
        if "overall_metrics" in result:
            # Nested CV
            label = result.get("label", "Nested CV")
            overall = result["overall_metrics"]
            lines.extend([
                f"## {label}",
                "",
                f"| Metrik | Wert |",
                f"|--------|------|",
                f"| R\u00b2     | {overall['R2']:.4f} |",
                f"| RMSE   | {overall['RMSE']:.4f} |",
                f"| MAE    | {overall['MAE']:.4f} |",
                f"| Bias   | {overall['Bias']:.4f} |",
                f"| RPD    | {overall['RPD']:.2f} |",
                "",
                "### Fold-Details",
                "",
                "| Fold | Test | Modell | RMSE |",
                "|------|------|--------|------|",
            ])
            for fr in result["fold_results"]:
                te = fr["test_metrics"]
                tg = ", ".join(fr["test_groups"])
                lines.append(f"| {fr['fold']} | {tg} | {fr['best_model']} | {te['RMSE']:.4f} |")

            lines.extend(["", "### Modellwahl-Haeufigkeit", ""])
            counts = Counter(result["selected_models"])
            for name, count in counts.most_common():
                lines.append(f"- **{name}**: {count}x")
            lines.append("")

        elif isinstance(result, list):
            # Simple CV results list
            if result:
                lines.extend([
                    f"## Einfache CV-Ergebnisse",
                    "",
                    "| Modell | R\u00b2(Train) | RMSE(Train) | R\u00b2(CV) | RMSE(CV) | RPD(CV) |",
                    "|--------|-----------|-------------|--------|----------|---------|",
                ])
                for r in result:
                    tr = r.get("train_metrics", {})
                    te = r.get("test_metrics", {})
                    lines.append(
                        f"| {r['model_name']} "
                        f"| {tr.get('R2', float('nan')):.4f} "
                        f"| {tr.get('RMSE', float('nan')):.4f} "
                        f"| {te.get('R2', float('nan')):.4f} "
                        f"| {te.get('RMSE', float('nan')):.4f} "
                        f"| {te.get('RPD', float('nan')):.2f} |"
                    )
                lines.append("")

    table_str = "\n".join(lines)
    with open(outpath, "w") as f:
        f.write(table_str)
    print(f"  -> {outpath}")
    return table_str


# ─── Main Pipeline ───────────────────────────────────────────

def main():
    print("=== Pipeline: Benchtop NIR Hesperidin-Kalibrierung ===\n")
    os.makedirs(PLOT_DIR, exist_ok=True)

    dl = _load_module("0_dataloading.py")
    pp = _load_module("1_data_preprocessing.py")
    rm = _load_module("3_regression_models.py")
    ev = _load_module("evaluation.py")

    hplc_df = dl.load_hplc()
    conc_map = dl.get_concentration_map(hplc_df)
    type_map = dl.get_sample_type_map(hplc_df)

    all_results = []
    all_simple_results = []

    # ─── Strategie 1 & 2: Alle Preprocessing-Varianten ───────

    for pp_name, pp_params in PREPROCESSING_VARIANTS.items():
        for filter_label, exclude in [("alle", None), ("nurM", EXCLUDE_NON_MANDARIN)]:
            strategy_label = f"{pp_name}_{filter_label}"
            print(f"\n{'#'*60}")
            print(f"# Strategie: {strategy_label}")
            print(f"{'#'*60}")

            wavenumbers, X, y, groups, _, _ = prepare_data(dl, pp, pp_params, exclude)
            n_unique = len(set(groups))
            print(f"  Proben: {n_unique} unique, {len(groups)} Spektren")
            print(f"  y: {y.min():.2f} - {y.max():.2f} %")

            if n_unique < 3:
                print(f"  SKIP: zu wenige Proben ({n_unique})")
                continue

            # Einfache CV (alle Modelle vergleichen)
            simple_res = run_strategy_simple_cv(X, y, groups, MODEL_CONFIGS,
                                                 label=strategy_label)
            all_simple_results.extend(simple_res)

            # Nested CV
            nested_res = run_nested_cv(X, y, groups, MODEL_CONFIGS,
                                        label=strategy_label)
            all_results.append(nested_res)

            # Plots
            plot_nested_cv_scatter(nested_res, conc_map, type_map)
            plot_model_selection_overview(nested_res)

    # ─── Strategie 3: Variable Selection (VIP) ───────────────

    print(f"\n{'#'*60}")
    print("# Strategie: Variable Selection (VIP)")
    print(f"{'#'*60}")

    # Verwende SNV, nur Mandarinen als Basis
    pp_params = PREPROCESSING_VARIANTS["SNV"]
    wavenumbers, X, y, groups, _, _ = prepare_data(dl, pp, pp_params, EXCLUDE_NON_MANDARIN)

    for top_n in [50, 100, 200]:
        vip_label = f"VIP_top{top_n}_SNV_nurM"
        print(f"\n  --- {vip_label} ---")

        sel_idx, vip_scores = rm.select_variables_vip(X, y, n_components=3, top_n=top_n)
        X_sel = X[:, sel_idx]
        print(f"  Ausgewaehlte Variablen: {len(sel_idx)}")

        simple_res = run_strategy_simple_cv(X_sel, y, groups, MODEL_CONFIGS,
                                             label=vip_label)
        all_simple_results.extend(simple_res)

        # Best VIP model: nested CV
        if top_n == 100:  # only for one representative
            nested_vip = run_nested_cv(X_sel, y, groups, MODEL_CONFIGS,
                                        label=vip_label)
            all_results.append(nested_vip)
            plot_nested_cv_scatter(nested_vip, conc_map, type_map)

    # ─── Gesamtvergleich ─────────────────────────────────────

    print(f"\n{'#'*60}")
    print("# Gesamtvergleich")
    print(f"{'#'*60}")

    # Vergleichs-Balkendiagramm (alle einfachen CV-Ergebnisse)
    ev.plot_metrics_comparison(all_simple_results,
                               save_path=os.path.join(PLOT_DIR, "pipeline_comparison_all.png"))

    # Nur Nested-CV Ergebnisse vergleichen
    nested_comparison = []
    for r in all_results:
        nested_comparison.append({
            "model_name": r["label"],
            "train_metrics": {},
            "test_metrics": r["overall_metrics"],
        })
    ev.plot_metrics_comparison(nested_comparison,
                               save_path=os.path.join(PLOT_DIR, "pipeline_comparison_nested.png"))

    # CV Boxplot: RMSE pro Fold fuer einfache Ergebnisse
    cv_boxplot_data = {}
    for r in all_simple_results:
        if "fold_rmses" in r and r["fold_rmses"]:
            cv_boxplot_data[r["model_name"]] = r["fold_rmses"]
    if cv_boxplot_data:
        ev.plot_cv_boxplot(cv_boxplot_data, metric="RMSE",
                           save_path=os.path.join(PLOT_DIR, "pipeline_cv_boxplot.png"))

    # Ergebnistabelle
    create_comprehensive_results_table(all_results + [all_simple_results])

    # ─── Bestes Modell zusammenfassen ────────────────────────

    if all_results:
        best_nested = min(all_results, key=lambda r: r["overall_metrics"]["RMSE"])
        print(f"\n{'='*60}")
        print(f"BESTES Nested-CV Ergebnis: {best_nested['label']}")
        print(f"  R2={best_nested['overall_metrics']['R2']:.4f}")
        print(f"  RMSE={best_nested['overall_metrics']['RMSE']:.4f}")
        print(f"  RPD={best_nested['overall_metrics']['RPD']:.2f}")
        print(f"  Modelle: {Counter(best_nested['selected_models']).most_common()}")
        print(f"{'='*60}")

    if all_simple_results:
        best_simple = min(all_simple_results, key=lambda r: r["test_metrics"]["RMSE"])
        print(f"\nBESTE einfache CV: {best_simple['model_name']}")
        print(f"  CV R2={best_simple['test_metrics']['R2']:.4f}")
        print(f"  CV RMSE={best_simple['test_metrics']['RMSE']:.4f}")

    print("\n=== Pipeline fertig ===")
    return all_results, all_simple_results


if __name__ == "__main__":
    main()
