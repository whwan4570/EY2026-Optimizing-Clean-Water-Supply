"""
DRP Structural Experiment Runner
Tests each structural toggle independently and reports CV R2/RMSE.
Usage: python drp_structure_experiments.py
"""
import subprocess
import sys
import re
import os
import csv
from datetime import datetime

SCRIPT = os.path.join(os.path.dirname(__file__), "run_benchmark_notebook.py")
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "drp_structure_results.csv")

EXPERIMENTS = [
    {
        "name": "baseline",
        "desc": "Current best: residual + TA/EC + single XGB",
        "flags": {
            "DRP_RESIDUAL_MODE": True,
            "DRP_LOG_RESIDUAL": True,
            "USE_TA_EC_FOR_DRP": True,
            "USE_DRP_STACKING": False,
            "DRP_RATIO_TARGET": False,
            "USE_LIGHTGBM_DRP": False,
        },
    },
    {
        "name": "no_residual",
        "desc": "Predict log1p(DRP) directly, no GEMS residual",
        "flags": {
            "DRP_RESIDUAL_MODE": False,
            "DRP_LOG_RESIDUAL": True,
            "USE_TA_EC_FOR_DRP": True,
            "USE_DRP_STACKING": False,
            "DRP_RATIO_TARGET": False,
            "USE_LIGHTGBM_DRP": False,
        },
    },
    {
        "name": "no_ta_ec",
        "desc": "Drop pred_TA/EC from DRP features (1-stage)",
        "flags": {
            "DRP_RESIDUAL_MODE": True,
            "DRP_LOG_RESIDUAL": True,
            "USE_TA_EC_FOR_DRP": False,
            "USE_DRP_STACKING": False,
            "DRP_RATIO_TARGET": False,
            "USE_LIGHTGBM_DRP": False,
        },
    },
    {
        "name": "ratio_target",
        "desc": "Ratio target: y / max(gems, eps)",
        "flags": {
            "DRP_RESIDUAL_MODE": True,
            "DRP_LOG_RESIDUAL": True,
            "USE_TA_EC_FOR_DRP": True,
            "USE_DRP_STACKING": False,
            "DRP_RATIO_TARGET": True,
            "USE_LIGHTGBM_DRP": False,
        },
    },
    {
        "name": "stacking",
        "desc": "XGB(shallow)+XGB(deep)+RF → Ridge stacking",
        "flags": {
            "DRP_RESIDUAL_MODE": True,
            "DRP_LOG_RESIDUAL": True,
            "USE_TA_EC_FOR_DRP": True,
            "USE_DRP_STACKING": True,
            "DRP_RATIO_TARGET": False,
            "USE_LIGHTGBM_DRP": False,
        },
    },
    {
        "name": "stacking_lgb",
        "desc": "Stacking + LightGBM as 4th base model",
        "flags": {
            "DRP_RESIDUAL_MODE": True,
            "DRP_LOG_RESIDUAL": True,
            "USE_TA_EC_FOR_DRP": True,
            "USE_DRP_STACKING": True,
            "DRP_RATIO_TARGET": False,
            "USE_LIGHTGBM_DRP": True,
        },
    },
    {
        "name": "no_residual_no_ta_ec",
        "desc": "Direct prediction without TA/EC (fully independent DRP)",
        "flags": {
            "DRP_RESIDUAL_MODE": False,
            "DRP_LOG_RESIDUAL": True,
            "USE_TA_EC_FOR_DRP": False,
            "USE_DRP_STACKING": False,
            "DRP_RATIO_TARGET": False,
            "USE_LIGHTGBM_DRP": False,
        },
    },
]


def patch_flags(script_path, flags):
    """Temporarily patch flags in the script and return original lines for rollback."""
    with open(script_path, "r", encoding="utf-8") as f:
        original = f.read()

    patched = original
    for flag_name, flag_val in flags.items():
        val_str = repr(flag_val)
        pattern = re.compile(rf'^({flag_name}\s*=\s*).*$', re.MULTILINE)
        if pattern.search(patched):
            patched = pattern.sub(rf'\g<1>{val_str}', patched, count=1)

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(patched)

    return original


def restore_script(script_path, original):
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(original)


def extract_drp_metrics(output):
    """Extract DRP R2_Train, RMSE_Train, R2_Test, RMSE_Test from script output."""
    r2_train = r2_test = rmse_train = rmse_test = None
    for line in output.split("\n"):
        if "Dissolved Reactive Phosphorus" in line and "R2_Train" not in line:
            m = re.search(r'R2_Test=([\d.]+)', line)
            if m:
                r2_test = float(m.group(1))
        if "Dissolved Reactive Phosphorus" in line:
            parts = line.strip().split()
            for i, p in enumerate(parts):
                if p == "R2_Train" and i + 1 < len(parts):
                    try:
                        r2_train = float(parts[i + 1])
                    except ValueError:
                        pass

    for line in output.split("\n"):
        if "Dissolved Reactive Phosphorus" in line and "|" not in line:
            continue
        if "Dissolved Reactive Phosphorus" in line:
            nums = re.findall(r'[\d.]+', line)
            if len(nums) >= 4:
                try:
                    r2_train = float(nums[0])
                    rmse_train = float(nums[1])
                    r2_test = float(nums[2])
                    rmse_test = float(nums[3])
                except (ValueError, IndexError):
                    pass

    return r2_train, rmse_train, r2_test, rmse_test


def run_experiment(exp):
    print(f"\n{'='*60}")
    print(f"  Experiment: {exp['name']}")
    print(f"  {exp['desc']}")
    print(f"  Flags: {exp['flags']}")
    print(f"{'='*60}\n")

    original = patch_flags(SCRIPT, exp["flags"])
    try:
        result = subprocess.run(
            [sys.executable, SCRIPT],
            capture_output=True, text=True, timeout=600,
            cwd=os.path.dirname(SCRIPT),
        )
        output = result.stdout + "\n" + result.stderr
        print(output[-3000:] if len(output) > 3000 else output)

        r2_train, rmse_train, r2_test, rmse_test = extract_drp_metrics(output)
        return {
            "name": exp["name"],
            "desc": exp["desc"],
            "r2_train": r2_train,
            "rmse_train": rmse_train,
            "r2_test": r2_test,
            "rmse_test": rmse_test,
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 600s")
        return {"name": exp["name"], "desc": exp["desc"], "r2_test": None, "exit_code": -1}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"name": exp["name"], "desc": exp["desc"], "r2_test": None, "exit_code": -2}
    finally:
        restore_script(SCRIPT, original)


def main():
    results = []
    for exp in EXPERIMENTS:
        res = run_experiment(exp)
        results.append(res)
        print(f"\n  >> {res['name']}: R2_Test={res.get('r2_test', 'N/A')}, R2_Train={res.get('r2_train', 'N/A')}")

    print(f"\n\n{'='*80}")
    print("  DRP STRUCTURAL EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Name':<25} {'R2_Test':>10} {'R2_Train':>10} {'RMSE_Test':>12}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*12}")
    for r in results:
        r2t = f"{r['r2_test']:.4f}" if r.get('r2_test') is not None else "N/A"
        r2tr = f"{r['r2_train']:.4f}" if r.get('r2_train') is not None else "N/A"
        rmse = f"{r['rmse_test']:.2f}" if r.get('rmse_test') is not None else "N/A"
        print(f"  {r['name']:<25} {r2t:>10} {r2tr:>10} {rmse:>12}")

    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "desc", "r2_train", "rmse_train", "r2_test", "rmse_test", "exit_code"])
        w.writeheader()
        w.writerows(results)
    print(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
