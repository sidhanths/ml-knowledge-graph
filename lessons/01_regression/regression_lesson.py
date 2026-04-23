"""
Lesson 01 — Regression family, end-to-end.
Covers graph nodes: ordinary least squares, ridge, lasso, elastic net,
huber regression, quantile regression, logistic regression, + SHAP, + conformal intervals.

Context: synthetic "fee" data shaped like Purefacts tabular problems.
Produces PNG plots in lessons/01_regression/plots/
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    HuberRegressor, QuantileRegressor, LogisticRegression,
)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import shap

PLOTS = Path(__file__).parent / "plots"
PLOTS.mkdir(exist_ok=True)
rng = np.random.default_rng(7)

# ----- synthetic "fee" dataset -----------------------------------------------
# 8 features: account_size, years_held, tier, advisor_flag, region_a, region_b,
# churn_risk, market_vol. True fee is driven by account_size (log) + years + tier.
# We sprinkle in outliers so Huber/Quantile show their value.
n = 2000
X = pd.DataFrame({
    "account_size":   rng.lognormal(mean=11, sigma=1.0, size=n),      # $
    "years_held":     rng.integers(0, 30, size=n),
    "tier":           rng.integers(1, 5, size=n),
    "advisor_flag":   rng.integers(0, 2, size=n),
    "region_a":       rng.integers(0, 2, size=n),
    "region_b":       rng.integers(0, 2, size=n),
    "churn_risk":     rng.beta(2, 5, size=n),
    "market_vol":     rng.normal(0, 1, size=n),
})
true_fee = (
    0.0008 * X["account_size"]
    + 15 * X["years_held"]
    + 40 * X["tier"]
    + 10 * X["advisor_flag"]
    + rng.normal(0, 50, size=n)
)
# outliers (operations errors / re-bills)
outlier_mask = rng.random(n) < 0.04
true_fee[outlier_mask] += rng.normal(2000, 500, size=outlier_mask.sum())
y = true_fee
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)

# ----- 1. Coefficient shrinkage: OLS vs Ridge vs Lasso vs ElasticNet ----------
models = {
    "OLS":          LinearRegression(),
    "Ridge(α=10)":  Ridge(alpha=10),
    "Lasso(α=10)":  Lasso(alpha=10, max_iter=20000),
    "ElasticNet":   ElasticNet(alpha=10, l1_ratio=0.5, max_iter=20000),
}
fig, ax = plt.subplots(figsize=(10, 5))
width = 0.2
feats = X.columns.tolist()
for i, (name, m) in enumerate(models.items()):
    m.fit(Xtr, ytr)
    coefs = m.coef_
    ax.bar(np.arange(len(feats)) + i * width, coefs, width, label=name)
ax.set_xticks(np.arange(len(feats)) + 1.5 * width)
ax.set_xticklabels(feats, rotation=30, ha="right")
ax.set_ylabel("coefficient")
ax.set_title("1. Linear family — coefficients (Lasso zeroes out weak features)")
ax.axhline(0, color="k", lw=0.5)
ax.legend()
fig.tight_layout()
fig.savefig(PLOTS / "01_coefficients.png", dpi=110)
plt.close(fig)

# ----- 2. Robustness: OLS vs Huber vs Quantile when outliers are present ------
rob = {
    "OLS":                LinearRegression(),
    "Huber":              HuberRegressor(max_iter=500),
    "Quantile(q=0.5)":    QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs"),
    "Quantile(q=0.90)":   QuantileRegressor(quantile=0.90, alpha=0.0, solver="highs"),
}
results = {}
for name, m in rob.items():
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    results[name] = (mean_absolute_error(yte, pred), r2_score(yte, pred))

fig, ax = plt.subplots(figsize=(8, 5))
names = list(results)
mae = [results[k][0] for k in names]
ax.barh(names, mae, color=["#888", "#2a9", "#27c", "#e73"])
ax.set_xlabel("MAE on held-out set (lower is better)")
ax.set_title("2. OLS vs Huber vs Quantile under 4% outliers\n"
             "Huber/Quantile ignore outlier pull → lower MAE")
for i, v in enumerate(mae):
    ax.text(v, i, f" {v:,.0f}", va="center")
fig.tight_layout()
fig.savefig(PLOTS / "02_robustness.png", dpi=110)
plt.close(fig)

# ----- 3. Quantile regression = prediction intervals --------------------------
q_lo = QuantileRegressor(quantile=0.05, alpha=0.0, solver="highs").fit(Xtr, ytr)
q_md = QuantileRegressor(quantile=0.50, alpha=0.0, solver="highs").fit(Xtr, ytr)
q_hi = QuantileRegressor(quantile=0.95, alpha=0.0, solver="highs").fit(Xtr, ytr)
lo, md, hi = q_lo.predict(Xte), q_md.predict(Xte), q_hi.predict(Xte)

order = np.argsort(md)
idx = np.arange(len(order))
fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(idx, lo[order], hi[order], alpha=0.25, label="P5–P95 band")
ax.plot(idx, md[order], lw=1.5, label="median prediction")
ax.scatter(idx, yte.values[order], s=6, alpha=0.35, color="k", label="actual")
ax.set_title("3. Quantile regression → native prediction intervals\n"
             "(a FAANG-favorite alternative to conformal)")
ax.set_xlabel("test rows, sorted by predicted median")
ax.set_ylabel("fee")
ax.legend()
fig.tight_layout()
fig.savefig(PLOTS / "03_quantile_bands.png", dpi=110)
plt.close(fig)

# ----- 4. Conformal prediction (split conformal, distribution-free) ----------
# Graph doesn't have this node yet; adding it since you called it out.
# Idea: split data into train/calibration/test. Fit any model on train.
# Compute absolute residuals on calibration. The (1-α)-quantile of those
# residuals defines a symmetric band around any new prediction — with a
# finite-sample coverage guarantee ≥ 1-α (no distributional assumptions).
Xtr2, Xcal, ytr2, ycal = train_test_split(Xtr, ytr, test_size=0.3, random_state=1)
base = GradientBoostingRegressor(random_state=0).fit(Xtr2, ytr2)
cal_resid = np.abs(ycal - base.predict(Xcal))
alpha = 0.1  # 90% coverage target
q = np.quantile(cal_resid, 1 - alpha)
pred_te = base.predict(Xte)
lo_c, hi_c = pred_te - q, pred_te + q
coverage = ((yte >= lo_c) & (yte <= hi_c)).mean()
width = (hi_c - lo_c).mean()

order = np.argsort(pred_te)
fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(np.arange(len(pred_te)), lo_c[order], hi_c[order],
                alpha=0.25, label=f"conformal band (target 90%)")
ax.plot(np.arange(len(pred_te)), pred_te[order], lw=1.5, label="GBM prediction")
ax.scatter(np.arange(len(pred_te)), yte.values[order], s=6, alpha=0.35, color="k", label="actual")
ax.set_title(f"4. Split-conformal intervals — empirical coverage {coverage:.1%}, "
             f"avg width {width:,.0f}\n"
             "Symmetric, distribution-free, finite-sample guarantee.")
ax.set_xlabel("test rows, sorted by prediction")
ax.set_ylabel("fee")
ax.legend()
fig.tight_layout()
fig.savefig(PLOTS / "04_conformal.png", dpi=110)
plt.close(fig)

# ----- 5. Logistic regression (churn) + probability calibration intuition ---
churn = (X["churn_risk"] + 0.3 * (X["tier"] == 1) + rng.normal(0, 0.1, n)) > 0.5
Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(X, churn, test_size=0.25, random_state=0)
lr = LogisticRegression(max_iter=1000).fit(Xtr_c, ytr_c)
p = lr.predict_proba(Xte_c)[:, 1]

# reliability curve
bins = np.linspace(0, 1, 11)
idx = np.digitize(p, bins) - 1
frac_pos = [yte_c.values[idx == b].mean() if (idx == b).any() else np.nan for b in range(10)]
mean_pred = [p[idx == b].mean() if (idx == b).any() else np.nan for b in range(10)]
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
ax.plot(mean_pred, frac_pos, "o-", label="logistic regression")
ax.set_xlabel("mean predicted probability")
ax.set_ylabel("empirical fraction positive")
ax.set_title("5. Logistic regression calibration (reliability diagram)\n"
             "FAANG-ads loves this question")
ax.legend()
fig.tight_layout()
fig.savefig(PLOTS / "05_calibration.png", dpi=110)
plt.close(fig)

# ----- 6. SHAP: explain the GBM fee model on a few rows ----------------------
expl = shap.TreeExplainer(base)
sv = expl.shap_values(Xte.iloc[:200])
shap.summary_plot(sv, Xte.iloc[:200], show=False)
plt.gcf().suptitle("6. SHAP summary — per-feature impact on fee prediction",
                   y=1.02, fontsize=12)
plt.savefig(PLOTS / "06_shap_summary.png", dpi=110, bbox_inches="tight")
plt.close("all")

# ----- console recap ----------------------------------------------------------
print("Saved plots:")
for p in sorted(PLOTS.iterdir()):
    print(" ", p.name)

print("\nRobustness MAE/R²:")
for name, (mae_, r2_) in results.items():
    print(f"  {name:20}  MAE={mae_:8,.0f}  R²={r2_:+.3f}")

print(f"\nConformal: target 90% coverage, got {coverage:.1%} on held-out; "
      f"avg interval width = {width:,.0f}")
