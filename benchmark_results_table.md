# EY 2026 Benchmark — Results Summary (with precip anomaly features)

**Run:** Ensemble (XGBoost + LightGBM + CatBoost), Group K-Fold CV (5 folds), GEMS 포함, 강수 이상치·수문 피처 병합

---

## 1. 최종 CV 결과 (Results Summary)

| Target | R² (CV mean) | R² (CV std) | RMSE (CV mean) | RMSE (CV std) |
|--------|--------------|-------------|----------------|---------------|
| **Total Alkalinity (TA)** | **0.805** | 0.036 | 32.57 | 3.65 |
| **Electrical Conductance (EC)** | **0.774** | 0.039 | 160.63 | 29.12 |
| **Dissolved Reactive Phosphorus (DRP)** | **0.486** | 0.085 | 35.74 | 4.05 |

---

## 2. TA / EC / DRP 상세 (Fold별)

### Total Alkalinity (TA)
| Fold | R² | RMSE |
|------|-----|------|
| 1 | 0.765 | 34.18 |
| 2 | 0.839 | 29.86 |
| 3 | 0.803 | 28.73 |
| 4 | 0.852 | 31.18 |
| 5 | 0.765 | 38.89 |
| **Mean** | **0.805** | **32.57** |

### Electrical Conductance (EC)
| Fold | R² | RMSE |
|------|-----|------|
| 1 | 0.746 | 186.29 |
| 2 | 0.796 | 170.22 |
| 3 | 0.840 | 107.86 |
| 4 | 0.750 | 152.79 |
| 5 | 0.736 | 185.97 |
| **Mean** | **0.774** | **160.63** |

### Dissolved Reactive Phosphorus (DRP)
| Fold | R² | RMSE |
|------|-----|------|
| 1 | 0.532 | 35.48 |
| 2 | 0.452 | 29.16 |
| 3 | 0.528 | 38.69 |
| 4 | 0.582 | 34.34 |
| 5 | 0.339 | 41.04 |
| **Mean** | **0.486** | **35.74** |

---

## 3. 하이퍼파라미터 튜닝 Best

| Target | Best R² (tune) | Config |
|--------|----------------|--------|
| TA | 0.800 | n_est=1200, max_d=6, lr=0.01, subsample=0.7, colsample=0.9, reg_alpha=0.05, reg_lambda=1.0 |
| EC | 0.777 | n_est=500, max_d=5, lr=0.01, subsample=0.85, colsample=0.7, reg_alpha=0.01, reg_lambda=1.0 |
| DRP | 0.486 | n_est=700, max_d=8, lr=0.04, subsample=0.7, colsample=0.7, min_child=2, reg_alpha=0.1, reg_lambda=2.5 |

---

## 4. 강수 이상치 피처 사용 현황

- **TA:** precip_anom, cumulative_anom_3m  
  (Feature importance: cumulative_anom_3m ≈ 0.0067, precip_anom ≈ 0.0054)
- **EC:** precip_anom, precip_anom_lag1, cumulative_anom_3m, wetness_index  
  (cumulative_anom_3m ≈ 0.0077, precip_anom ≈ 0.0059, wetness_index ≈ 0.0056, precip_anom_lag1 ≈ 0.0042)
- **DRP:** precip_anom, precip_anom_lag1, precip_anom_lag2, storm_pet_ratio, wetness_index, cumulative_wetness_3m, anom_pos, anom_neg  
  (wetness_index ≈ 0.0049, cumulative_wetness_3m ≈ 0.0041, precip_anom ≈ 0.0038, anom_neg ≈ 0.0037, precip_anom_lag2 ≈ 0.0037, precip_anom_lag1 ≈ 0.0035, anom_pos ≈ 0.0032, storm_pet_ratio ≈ 0.0030)

---

*Generated from benchmark_results_latest.csv and terminal output.*
