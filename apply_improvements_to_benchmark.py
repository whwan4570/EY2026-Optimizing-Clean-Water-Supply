"""
benchmark_model.py에 통합 모델·마코위츠·Huber 개선을 적용하는 스크립트.
실행 전 benchmark_model.py를 닫아 두세요.
"""
from pathlib import Path

BENCH = Path(__file__).parent / "benchmark_model.py"

def main():
    path = BENCH
    if not path.exists():
        print("benchmark_model.py를 찾을 수 없습니다.")
        return
    text = path.read_text(encoding="utf-8")

    # 1) import 다음에 multi_output 임포트 + 개선 상수 추가
    old1 = """from pathlib import Path
from typing import Optional


# Base directory (스크립트 위치 기준)
BASE_DIR = Path(__file__).parent

# ============ 실험 모드 (Step 1/2/3) ============"""

    new1 = """from pathlib import Path
from typing import Optional

# 통합 모델 (Multi-Output TabNet/MLP)
try:
    from multi_output_tabnet import MultiOutputTabNetWrapper
    MULTI_OUTPUT_TABNET_AVAILABLE = True
except ImportError:
    MultiOutputTabNetWrapper = None
    MULTI_OUTPUT_TABNET_AVAILABLE = False
try:
    from multi_output_mlp import MultiOutputMLPWrapper
    MULTI_OUTPUT_MLP_AVAILABLE = True
except ImportError:
    MultiOutputMLPWrapper = None
    MULTI_OUTPUT_MLP_AVAILABLE = False

# Base directory (스크립트 위치 기준)
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# ============ 개선 옵션 (통합 모델, 마코위츠, Huber) ============
USE_MARKOWITZ_BLEND = True
USE_MULTI_OUTPUT = True
USE_MULTI_OUTPUT_TABNET = True
USE_MULTI_OUTPUT_HUBER = True
MULTI_OUTPUT_BLEND_W = 0.2
MULTI_OUTPUT_HUBER_BETA = 0.1

# ============ 실험 모드 (Step 1/2/3) ============"""

    if old1 not in text:
        # 대체: 상수/임포트만 단계적으로 추가
        if "USE_MARKOWITZ_BLEND" in text:
            print("(1) 이미 적용된 상태로 스킵.")
        else:
            text = text.replace(
                "from typing import Optional\n\n\n# Base directory",
                "from typing import Optional\n\n# 통합 모델 (Multi-Output TabNet/MLP)\ntry:\n    from multi_output_tabnet import MultiOutputTabNetWrapper\n    MULTI_OUTPUT_TABNET_AVAILABLE = True\nexcept ImportError:\n    MultiOutputTabNetWrapper = None\n    MULTI_OUTPUT_TABNET_AVAILABLE = False\ntry:\n    from multi_output_mlp import MultiOutputMLPWrapper\n    MULTI_OUTPUT_MLP_AVAILABLE = True\nexcept ImportError:\n    MultiOutputMLPWrapper = None\n    MULTI_OUTPUT_MLP_AVAILABLE = False\n\n# Base directory",
                1,
            )
            text = text.replace(
                "BASE_DIR = Path(__file__).parent\n\n# ============ 실험 모드",
                "BASE_DIR = Path(__file__).parent\nDATA_DIR = BASE_DIR / \"data\"\n\n# ============ 개선 옵션 (통합 모델, 마코위츠, Huber) ============\nUSE_MARKOWITZ_BLEND = True\nUSE_MULTI_OUTPUT = True\nUSE_MULTI_OUTPUT_TABNET = True\nUSE_MULTI_OUTPUT_HUBER = True\nMULTI_OUTPUT_BLEND_W = 0.2\nMULTI_OUTPUT_HUBER_BETA = 0.1\n\n# ============ 실험 모드",
                1,
            )
            print("(1) 임포트 및 개선 상수 적용됨 (대체 패턴).")
    else:
        text = text.replace(old1, new1, 1)
        print("(1) 임포트 및 개선 상수 적용됨.")

    # 2) DRP_TOP_K 다음에 Markowitz 함수 추가
    old2 = "DRP_TOP_K = 50\n\n\ndef add_seasonality_features(df: pd.DataFrame, date_col: str = \"Sample Date\") -> pd.DataFrame:\n    \"\"\"Add month, dayofyear, sin_doy, cos_doy from Sample Date.\"\"\""

    new2 = (
        "DRP_TOP_K = 50\n\n\n"
        "def compute_markowitz_blend_weight_two(oof_a: np.ndarray, oof_b: np.ndarray, y_true: np.ndarray) -> float:\n"
        '    """OOF 예측 두 개에 대해 R² 최대화 블렌드 가중치 w (blend = (1-w)*a + w*b) 반환."""\n'
        "    oof_a = np.asarray(oof_a, dtype=np.float64).ravel()\n"
        "    oof_b = np.asarray(oof_b, dtype=np.float64).ravel()\n"
        "    y_true = np.asarray(y_true, dtype=np.float64).ravel()\n"
        "    n = min(len(oof_a), len(oof_b), len(y_true))\n"
        "    if n == 0:\n        return 0.0\n"
        "    oof_a, oof_b, y_true = oof_a[:n], oof_b[:n], y_true[:n]\n"
        "    best_r2, best_w = -np.inf, 0.0\n"
        "    for w in np.linspace(0.0, 1.0, 21):\n"
        "        blend = (1.0 - w) * oof_a + w * oof_b\n"
        "        ss_res = np.sum((y_true - blend) ** 2)\n"
        "        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)\n"
        "        r2 = float(1.0 - ss_res / ss_tot) if ss_tot >= 1e-12 else 0.0\n"
        "        if r2 > best_r2:\n            best_r2, best_w = r2, w\n"
        "    return float(best_w)\n\n\n"
        'def add_seasonality_features(df: pd.DataFrame, date_col: str = "Sample Date") -> pd.DataFrame:\n'
        '    """Add month, dayofyear, sin_doy, cos_doy from Sample Date."""'
    )
    if old2 not in text:
        print("(2) DRP_TOP_K 블록을 찾을 수 없습니다.")
    else:
        text = text.replace(old2, new2, 1)
        print("(2) compute_markowitz_blend_weight_two 추가됨.")

    # 3) EC 학습 직후 Markowitz 가중치 계산
    old3 = """    progress("EC 모델 학습 (Group K-Fold CV)")
    model_EC, scaler_EC, results_EC, _ = run_pipeline(
        X_EC, y_EC, groups_EC, "Electrical Conductance", model_config=config_EC, aggregation_cols=_agg
    )

    # DRP 튜닝 (TA, EC 모델 필요)"""

    new3 = """    progress("EC 모델 학습 (Group K-Fold CV)")
    model_EC, scaler_EC, results_EC, _ = run_pipeline(
        X_EC, y_EC, groups_EC, "Electrical Conductance", model_config=config_EC, aggregation_cols=_agg
    )

    # Markowitz 블렌드 가중치 (TA/EC vs GEMS)
    w_ta_gems, w_ec_gems = 0.15, 0.15
    if use_weak_gems_blend and USE_MARKOWITZ_BLEND:
        oof_TA = oof_predictions_with_groups(X_TA, y_TA, groups_TA, n_splits=5, model_config=config_TA)
        oof_EC = oof_predictions_with_groups(X_EC, y_EC, groups_EC, n_splits=5, model_config=config_EC)
        if "gems_Alk_Tot" in X_TA.columns:
            g_ta = np.maximum(np.nan_to_num(X_TA["gems_Alk_Tot"].values, nan=0.0), 0)
            w_ta_gems = compute_markowitz_blend_weight_two(oof_TA, g_ta, y_TA.values)
        if "gems_EC" in X_EC.columns:
            g_ec = np.maximum(np.nan_to_num(X_EC["gems_EC"].values, nan=0.0), 0)
            w_ec_gems = compute_markowitz_blend_weight_two(oof_EC, g_ec, y_EC.values)
        print(f"  Markowitz blend: w_ta_gems={w_ta_gems:.3f}, w_ec_gems={w_ec_gems:.3f}")

    # DRP 튜닝 (TA, EC 모델 필요)"""

    if old3 not in text:
        print("(3) EC 학습 직후 블록을 찾을 수 없습니다.")
    else:
        text = text.replace(old3, new3, 1)
        print("(3) Markowitz 가중치 계산 추가됨.")

    # 4) DRP 학습 후 통합 모델 학습
    old4 = """        drp_pred_TA_std_mean = drp_pred_EC_std_mean = None

    progress("Validation 예측 및 submission 생성")"""

    new4 = """        drp_pred_TA_std_mean = drp_pred_EC_std_mean = None

    # 통합 모델 (Multi-Output) 학습
    model_multi = None
    if USE_MULTI_OUTPUT and (USE_MULTI_OUTPUT_TABNET and MULTI_OUTPUT_TABNET_AVAILABLE or not USE_MULTI_OUTPUT_TABNET and MULTI_OUTPUT_MLP_AVAILABLE):
        progress("통합 모델 (Multi-Output) 학습")
        X_shared = X_DRP.copy().fillna(X_DRP.median())
        try:
            if USE_MULTI_OUTPUT_TABNET and MultiOutputTabNetWrapper is not None:
                model_multi = MultiOutputTabNetWrapper(
                    n_d=8, n_a=8, n_steps=3, max_epochs=150, patience=25,
                    batch_size=1024, loss_weights=(0.1, 0.2, 0.7), seed=42, verbose=0,
                )
            else:
                model_multi = MultiOutputMLPWrapper(
                    loss_weights=(0.1, 0.2, 0.7),
                    loss_type="huber" if USE_MULTI_OUTPUT_HUBER else "mse",
                    huber_beta=MULTI_OUTPUT_HUBER_BETA,
                    epochs=150, patience=25, seed=42,
                )
            model_multi.fit(X_shared, y_TA, y_EC, y_DRP)
            print("  Multi-Output 모델 학습 완료.")
        except Exception as e:
            print(f"  Multi-Output 스킵: {e}")
            model_multi = None

    progress("Validation 예측 및 submission 생성")"""

    if old4 not in text:
        print("(4) DRP 후 progress 블록을 찾을 수 없습니다.")
    else:
        text = text.replace(old4, new4, 1)
        print("(4) 통합 모델 학습 블록 추가됨.")

    # 5) submission: weak gems blend을 Markowitz 가중치로
    old5 = """    # Step 2: weak gems blend (final = 0.85*model + 0.15*gems)
    if use_weak_gems_blend:
        w_model, w_gems = 0.85, 0.15
        if "gems_Alk_Tot" in val_data.columns:
            g_ta = np.maximum(np.nan_to_num(val_data["gems_Alk_Tot"].values, nan=0.0), 0)
            pred_TA_submission = w_model * pred_TA_submission + w_gems * g_ta
        if "gems_EC" in val_data.columns:
            g_ec = np.maximum(np.nan_to_num(val_data["gems_EC"].values, nan=0.0), 0)
            pred_EC_submission = w_model * pred_EC_submission + w_gems * g_ec"""

    new5 = """    # Step 2: weak gems blend (Markowitz 가중치 또는 고정 0.85/0.15)
    if use_weak_gems_blend:
        w_model_ta, w_gems_ta = (1.0 - w_ta_gems), w_ta_gems
        w_model_ec, w_gems_ec = (1.0 - w_ec_gems), w_ec_gems
        if "gems_Alk_Tot" in val_data.columns:
            g_ta = np.maximum(np.nan_to_num(val_data["gems_Alk_Tot"].values, nan=0.0), 0)
            pred_TA_submission = w_model_ta * pred_TA_submission + w_gems_ta * g_ta
        if "gems_EC" in val_data.columns:
            g_ec = np.maximum(np.nan_to_num(val_data["gems_EC"].values, nan=0.0), 0)
            pred_EC_submission = w_model_ec * pred_EC_submission + w_gems_ec * g_ec"""

    if old5 not in text:
        print("(5) weak gems blend 블록을 찾을 수 없습니다.")
    else:
        text = text.replace(old5, new5, 1)
        print("(5) Markowitz 가중치 적용됨.")

    # 6) DRP 제출 직전에 통합 모델 블렌드
    old6 = """    else:
        pred_DRP_submission = np.expm1(pred_residual) if drp_log_target else pred_residual
        pred_DRP_submission = np.maximum(pred_DRP_submission, 0)

    # --- Create submission ---"""

    new6 = """    else:
        pred_DRP_submission = np.expm1(pred_residual) if drp_log_target else pred_residual
        pred_DRP_submission = np.maximum(pred_DRP_submission, 0)

    # 통합 모델 DRP 블렌드
    if model_multi is not None and MULTI_OUTPUT_BLEND_W > 0:
        shared_cols = [c for c in X_DRP.columns if c in val_data.columns]
        if shared_cols:
            submission_val_shared = val_data.reindex(columns=shared_cols).fillna(0)
            if hasattr(model_multi, "feature_names_in_") and model_multi.feature_names_in_ is not None:
                submission_val_shared = submission_val_shared.reindex(columns=model_multi.feature_names_in_).fillna(0)
            _, _, pred_DRP_multi = model_multi.predict(submission_val_shared)
            pred_DRP_multi = np.maximum(np.asarray(pred_DRP_multi, dtype=np.float64).ravel(), 0)
            if len(pred_DRP_multi) == len(pred_DRP_submission):
                w_multi = min(max(MULTI_OUTPUT_BLEND_W, 0.0), 1.0)
                pred_DRP_submission = (1.0 - w_multi) * pred_DRP_submission + w_multi * pred_DRP_multi
                pred_DRP_submission = np.maximum(pred_DRP_submission, 0)

    # --- Create submission ---"""

    if old6 not in text:
        print("(6) DRP submission 직전 블록을 찾을 수 없습니다.")
    else:
        text = text.replace(old6, new6, 1)
        print("(6) 통합 모델 DRP 블렌드 추가됨.")

    out_path = path
    try:
        path.write_text(text, encoding="utf-8")
        print("\nbenchmark_model.py 저장 완료.")
    except PermissionError:
        out_path = path.parent / "benchmark_model_updated.py"
        out_path.write_text(text, encoding="utf-8")
        print(f"\nbenchmark_model.py가 잠겨 있어 수정본을 다음 파일로 저장했습니다: {out_path.name}")
        print("  → benchmark_model.py를 닫은 뒤, 이 파일 내용을 benchmark_model.py로 복사하세요.")

if __name__ == "__main__":
    main()
