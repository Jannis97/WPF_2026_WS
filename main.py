"""
main.py
Führt 8 Pipeline-Konfigurationen aus:
  2 Spektrometer × 2 Preprocessing × 2 Probenfilter = 8 Runs.
Druckt am Ende eine Vergleichstabelle über alle Runs.

Usage:
    .venv/bin/python main.py
"""

from pipeline import run_pipeline
from pathlib import Path
from tqdm import tqdm

RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)


def build_configs():
    """Erstellt 8 Konfigurationen."""
    configs = []
    for spectrometer in ["tango", "neospectra"]:
        for use_savgol in [False, True]:
            for exclude_ids in [[], ["3", "5", "21"]]:
                preproc = "SNV_SG1d" if use_savgol else "SNV"
                filt = "excl3+5+21" if exclude_ids else "all"
                name = f"{spectrometer}_{preproc}_{filt}"
                configs.append({
                    "name": name,
                    "spectrometer": spectrometer,
                    "use_savgol": use_savgol,
                    "exclude_ids": list(exclude_ids),
                    "max_varsel": 8,
                    "max_components": 4,
                    "output_dir": str(RUNS_DIR / name),
                })
    return configs


def print_comparison_table(all_results):
    """Vergleichstabelle über alle Runs: Standard LOO vs Nested LOO."""
    print(f"\n{'='*120}")
    print(f"  GESAMTVERGLEICH – Standard LOO (optimistisch) vs. Nested LOO (ehrlich)")
    print(f"{'='*120}")
    print(f"  {'Run':35s}  {'n':>3s}  │ {'Std LOO Modell':25s} {'R²CV':>7s} {'RMSECV':>7s}"
          f"  │ {'Nested LOO Modell':25s} {'R²CV':>7s} {'RMSECV':>7s}")
    print("  " + "-" * 115)

    for r in all_results:
        print(f"  {r['name']:35s}  {r['n_samples']:>3d}"
              f"  │ {r['best_model']:25s} {r['best_R2CV']:>+7.3f} {r['best_RMSECV']:>7.4f}"
              f"  │ {r.get('best_nested_model','?'):25s} {r.get('best_nested_R2CV',0):>+7.3f}"
              f" {r.get('best_nested_RMSECV',0):>7.4f}")

    print("  " + "-" * 115)

    best_nested = min(all_results, key=lambda x: x.get("best_nested_RMSECV", 999))
    print(f"\n  Bester Run (Nested LOO = ehrliche Schätzung): {best_nested['name']}")
    print(f"  Modell: {best_nested.get('best_nested_model','')} → "
          f"R²CV={best_nested.get('best_nested_R2CV',0):.4f}, "
          f"RMSECV={best_nested.get('best_nested_RMSECV',0):.4f}%")
    print()


def print_detailed_comparison(all_results):
    """Detaillierte Tabelle: Top 3 Modelle pro Run."""
    print(f"\n{'='*100}")
    print(f"  DETAILVERGLEICH – Top 3 Modelle pro Run")
    print(f"{'='*100}")

    for r in all_results:
        print(f"\n  {r['name']} (n={r['n_samples']})")
        print(f"  {'#':>2}  {'Modell':30s}  {'R²train':>8s}  {'RMSEtr':>8s}  │  {'R²CV':>8s}  {'RMSECV':>8s}")
        print("  " + "-" * 80)
        for i, m in enumerate(r["models"][:3]):
            print(f"  {i+1:>2}. {m['name']:30s}  {m['R2_train']:>+8.4f}  {m['RMSE_train']:>8.4f}  │"
                  f"  {m['R2_CV']:>+8.4f}  {m['RMSECV']:>8.4f}")

    print()


def main():
    print("=" * 100)
    print("  NIR Hesperidin-Quantifizierung – 8-fache Pipeline-Auswertung")
    print("=" * 100)

    configs = build_configs()
    all_results = []

    for cfg in tqdm(configs, desc="Pipeline Runs", unit="run"):
        tqdm.write(f"\n  >>> Run: {cfg['name']}")
        result = run_pipeline(cfg)
        all_results.append(result)

    # Vergleichstabellen
    print_comparison_table(all_results)
    print_detailed_comparison(all_results)

    # Gesamtergebnis speichern
    import json
    with open(RUNS_DIR / "comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Gesamtergebnis gespeichert: {RUNS_DIR / 'comparison.json'}")


if __name__ == "__main__":
    main()
