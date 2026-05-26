import json
import pandas as pd
import numpy as np
import math
import time
import warnings
from pathlib import Path
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH      = Path("0-Data")
PROCESSED_PATH = Path("1-Processed")
FACTS_DIR      = DATA_PATH / "companyfacts_extracted"

FILING_START = 20120615
FILING_END   = 20241231
PERIOD_MIN   = 20120615


# =============================================================================
# STEP 1 — XBRL EXTRACTION
# =============================================================================

def iter_quarters(start="2012q3", end="2024q4"):
    y, q = int(start[:4]), int(start[5])
    ey, eq = int(end[:4]), int(end[5])
    while (y, q) <= (ey, eq):
        yield f"{y}q{q}"
        q += 1
        if q > 4:
            q = 1
            y += 1


json_files = list(FACTS_DIR.glob("CIK*.json"))
print(f"JSON files found : {len(json_files):,}")

records = []
errors  = []

for json_file in tqdm(json_files, desc="Extracting XBRL"):
    cik = json_file.stem.replace("CIK", "")
    try:
        with open(json_file, "r") as f:
            data = json.load(f)

        us_gaap = data.get("facts", {}).get("us-gaap", {})

        for tag, tag_data in us_gaap.items():
            units = tag_data.get("units", {})

            if "USD" in units:
                unit_key, entries = "USD", units["USD"]
            elif "shares" in units:
                unit_key, entries = "shares", units["shares"]
            elif "pure" in units:
                unit_key, entries = "pure", units["pure"]
            else:
                continue

            for entry in entries:
                if (entry.get("form") in ["10-K", "10-K/A"] and
                        entry.get("fp") == "FY"):
                    end_year = int(entry["end"][:4])
                    if 2008 <= end_year <= 2024:
                        records.append((
                            cik, tag, end_year,
                            entry["end"], entry["val"],
                            entry.get("filed", "")
                        ))
    except Exception as e:
        errors.append((cik, str(e)))

print(f"Records extracted : {len(records):,}")
print(f"Errors            : {len(errors)}")

cols = ["cik", "tag", "fiscal_year_end", "end_date", "value", "filed"]
df   = pd.DataFrame(records, columns=cols)
df["value"] = pd.to_numeric(df["value"], errors="coerce")
df = df.dropna(subset=["value"])

print(f"Unique CIKs  : {df['cik'].nunique():,}")
print(f"Unique tags  : {df['tag'].nunique():,}")

df.to_parquet(PROCESSED_PATH / "xbrl_long_v2.parquet", index=False)
print("Saved xbrl_long_v2.parquet")


# =============================================================================
# STEP 2 — FILING INDEX CONSTRUCTION
# =============================================================================

QUARTERS = list(iter_quarters("2012q3", "2024q4"))
SUB_COLS  = ["adsh", "cik", "name", "sic", "form", "period",
             "fy", "fp", "filed", "fye", "wksi", "afs"]

sub_list = []
for qtr in tqdm(QUARTERS, desc="Loading submission files"):
    fp = DATA_PATH / "extracted" / qtr / "sub.txt"
    if not fp.exists():
        continue
    chunk = pd.read_csv(fp, sep="	",
                        usecols=lambda c: c in SUB_COLS,
                        dtype=str, low_memory=False)
    chunk["_qtr"] = qtr
    sub_list.append(chunk)

sub_raw = pd.concat(sub_list, ignore_index=True)

for col in ("filed", "period"):
    sub_raw[col] = pd.to_numeric(
        sub_raw[col].str.replace("-", "", regex=False), errors="coerce"
    ).astype("Int64")

sub = sub_raw[
    sub_raw["form"].isin(["10-K", "10-K/A"]) &
    sub_raw["filed"].between(FILING_START, FILING_END) &
    (sub_raw["period"] >= PERIOD_MIN)
].copy()

sub = sub[sub["fp"].isin(["FY"]) | sub["fp"].isna()].copy()

sub["period_dt"]     = pd.to_datetime(sub["period"].astype(str), format="%Y%m%d", errors="coerce")
sub["filed_dt"]      = pd.to_datetime(sub["filed"].astype(str),  format="%Y%m%d", errors="coerce")
sub["fiscal_year"]   = sub["period_dt"].dt.year.astype("Int64")
sub["calendar_year"] = sub["filed_dt"].dt.year.astype("Int64")
sub["cutoff_dt"]     = sub["period_dt"] + pd.DateOffset(months=3)
sub = sub[sub["filed_dt"] <= sub["cutoff_dt"]].copy()
sub = (sub.sort_values(["cik", "period", "filed_dt"])
          .groupby(["cik", "period"], as_index=False)
          .last())

print(f"Total filings          : {len(sub):,}")
sub.to_parquet(PROCESSED_PATH / "sub_sample_extended.parquet", index=False)
print("Saved sub_sample_extended.parquet")


# =============================================================================
# STEP 3 — FEATURE MATRIX CONSTRUCTION
# =============================================================================

df  = pd.read_parquet(PROCESSED_PATH / "xbrl_long_v2.parquet")
sub = pd.read_parquet(PROCESSED_PATH / "sub_sample_extended.parquet")

df["cik"]  = df["cik"].astype(str).str.zfill(10)
sub["cik"] = sub["cik"].astype(str).str.zfill(10)

sub["filed_dt"]      = pd.to_datetime(sub["filed"].astype(str), format="%Y%m%d", errors="coerce")
sub["calendar_year"] = sub["filed_dt"].dt.year.astype("Int64")

sample_ciks  = set(sub["cik"].unique())
sample_years = sorted(sub["fiscal_year"].dropna().astype(int).unique())
needed_years = set(range(min(sample_years) - 1, max(sample_years) + 2))

df = df[df["cik"].isin(sample_ciks) & df["fiscal_year_end"].isin(needed_years)].copy()
print(f"After CIK + year filter : {len(df):,}")

df = (df.sort_values("filed")
        .groupby(["cik", "tag", "fiscal_year_end"], as_index=False)
        .last()[["cik", "tag", "fiscal_year_end", "value"]])
print(f"After dedup             : {len(df):,}")

tag_year = (df[df["fiscal_year_end"].between(2012, 2024)]
              .groupby("tag")["fiscal_year_end"]
              .nunique()
              .reset_index(name="n_years"))

n_total = df[df["fiscal_year_end"].between(2012, 2024)]["fiscal_year_end"].nunique()
COMMON_TAGS = set(tag_year[tag_year["n_years"] == n_total]["tag"])
print(f"Common tags : {len(COMMON_TAGS):,}")

df = df[df["tag"].isin(COMMON_TAGS)].copy()

assets = (df[df["tag"] == "Assets"][["cik", "fiscal_year_end", "value"]]
            .rename(columns={"value": "total_assets"}))
assets = assets[assets["total_assets"] > 0]

PER_SHARE = ["pershare", "perdiluted", "perbasic", "per_share"]

def is_per_share(tag):
    return any(k in tag.lower() for k in PER_SHARE) or tag == "Assets"

df = df.merge(assets, on=["cik", "fiscal_year_end"], how="inner")
df["scaled"] = np.where(
    df["tag"].map(is_per_share),
    df["value"],
    df["value"] / df["total_assets"]
)

curr = df[df["fiscal_year_end"].isin(sample_years)].copy()
curr["fiscal_year"] = curr["fiscal_year_end"]
lag  = df[df["fiscal_year_end"].isin([y - 1 for y in sample_years])].copy()
lag["fiscal_year"] = lag["fiscal_year_end"] + 1

IDX = ["cik", "fiscal_year", "tag"]
curr = curr.groupby(IDX)["scaled"].mean().reset_index()
lag  = lag.groupby(IDX)["scaled"].mean().reset_index()

curr_wide = (curr.pivot_table(index=["cik", "fiscal_year"],
                               columns="tag", values="scaled", aggfunc="mean")
                 .add_suffix("_t").reset_index())

lag_wide  = (lag.pivot_table(index=["cik", "fiscal_year"],
                              columns="tag", values="scaled", aggfunc="mean")
                .add_suffix("_t1").reset_index())

X = curr_wide.merge(lag_wide, on=["cik", "fiscal_year"], how="outer")

curr_tags = [c.replace("_t", "") for c in curr_wide.columns
             if c.endswith("_t") and c not in ["cik", "fiscal_year"]]

for tag in curr_tags:
    ct, lt = f"{tag}_t", f"{tag}_t1"
    if ct in X.columns and lt in X.columns:
        denom = X[lt].abs().replace(0, np.nan)
        X[f"{tag}_pct"] = (X[ct] - X[lt]) / denom

feat_cols = [c for c in X.columns if c not in ["cik", "fiscal_year"]]
X[feat_cols] = X[feat_cols].fillna(0).replace([np.inf, -np.inf], 0)

sub_info = sub[["cik", "adsh", "name", "sic", "period", "filed",
                "fiscal_year", "calendar_year", "form"]].copy()
X_v2 = X.merge(sub_info, on=["cik", "fiscal_year"], how="inner")

print(f"Feature matrix shape : {X_v2.shape}")
X_v2.to_parquet(PROCESSED_PATH / "feature_matrix_v2.parquet", index=False)
print("Saved feature_matrix_v2.parquet")


# =============================================================================
# STEP 4 — ML MODEL TRAINING (RANDOM FOREST + GRADIENT BOOSTING)
# =============================================================================

X_v2   = pd.read_parquet(PROCESSED_PATH / "feature_matrix_v2.parquet")
target = pd.read_parquet(PROCESSED_PATH / "target_variable.parquet")
link   = pd.read_parquet(PROCESSED_PATH / "compustat_link.parquet")
crsp   = pd.read_parquet(PROCESSED_PATH / "crsp_monthly.parquet")

link["cusip8"]  = link["cusip"].str[:8].str.strip()
target["cusip"] = target["cusip"].str.strip()
link_dedup      = link.drop_duplicates(subset="cusip8", keep="last")[["cusip8", "cik"]]

target_cik = target.merge(link_dedup, left_on="cusip", right_on="cusip8", how="inner")
target_cik["xbrl_fiscal_year"] = target_cik["fiscal_year_end"] - 1
target_cik["cik"] = target_cik["cik"].astype(str).str.strip().str.lstrip("0").str.zfill(10)
X_v2["cik"]       = X_v2["cik"].astype(str).str.strip().str.lstrip("0").str.zfill(10)

ml_v2 = X_v2.merge(
    target_cik[["cik", "xbrl_fiscal_year", "fiscal_year_end",
                "y", "eps", "delta_eps", "drift", "delta_eps_detrended"]],
    left_on=["cik", "fiscal_year"],
    right_on=["cik", "xbrl_fiscal_year"],
    how="inner"
)

crsp_cusips  = set(crsp["cusip"].str.strip().unique())
crsp_cik     = (pd.DataFrame({"cusip8": list(crsp_cusips)})
                  .merge(link_dedup, on="cusip8", how="inner"))
crsp_cik["cik"] = crsp_cik["cik"].astype(str).str.strip().str.lstrip("0").str.zfill(10)
crsp_cik_set    = set(crsp_cik["cik"].unique())

ml = ml_v2[ml_v2["cik"].isin(crsp_cik_set)].copy()
ml.to_parquet(PROCESSED_PATH / "ml_dataset_v2_final.parquet", index=False)
print(f"ML dataset saved : {ml.shape}")

NON_FEAT  = ["adsh", "cik", "name", "sic", "period", "filed",
             "fiscal_year", "calendar_year", "form",
             "xbrl_fiscal_year", "fiscal_year_end",
             "y", "eps", "delta_eps", "drift", "delta_eps_detrended"]
FEAT_COLS = [c for c in ml.columns if c not in NON_FEAT]
K_DEFAULT = int(math.sqrt(len(FEAT_COLS)))

TEST_YEARS = list(range(2015, 2024))

RF_GRID = {
    "n_estimators"    : [500, 1000, 2000],
    "max_features"    : [K_DEFAULT],
    "min_samples_leaf": [1, 5],
    "max_samples"     : [0.5]
}
GBM_GRID = {
    "n_estimators"    : [500, 1000, 2000],
    "learning_rate"   : [0.005, 0.01, 0.05],
    "max_depth"       : [1, 2],
    "min_samples_leaf": [10],
    "subsample"       : [0.5]
}

results_v2   = []
all_preds_v2 = []

for ty in TEST_YEARS:
    print(f"
Test year: {ty}")

    idx_tr = ml["fiscal_year"].isin([ty - 3, ty - 2])
    idx_va = ml["fiscal_year"] == ty - 1
    idx_te = ml["fiscal_year"] == ty

    X_tr = ml.loc[idx_tr, FEAT_COLS].values
    y_tr = ml.loc[idx_tr, "y"].values
    X_va = ml.loc[idx_va, FEAT_COLS].values
    y_va = ml.loc[idx_va, "y"].values
    X_te = ml.loc[idx_te, FEAT_COLS].values
    y_te = ml.loc[idx_te, "y"].values

    best_rf_auc, best_rf_clf, best_rf_p = -1, None, None
    for p in ParameterGrid(RF_GRID):
        clf = RandomForestClassifier(
            n_estimators=p["n_estimators"], max_features=p["max_features"],
            min_samples_leaf=p["min_samples_leaf"], max_samples=p["max_samples"],
            n_jobs=-1, random_state=42)
        clf.fit(X_tr, y_tr)
        auc = roc_auc_score(y_va, clf.predict_proba(X_va)[:, 1])
        if auc > best_rf_auc:
            best_rf_auc, best_rf_clf, best_rf_p = auc, clf, p
    rf_test = roc_auc_score(y_te, best_rf_clf.predict_proba(X_te)[:, 1])
    rf_prob = best_rf_clf.predict_proba(X_te)[:, 1]

    best_gbm_auc, best_gbm_clf, best_gbm_p = -1, None, None
    for p in ParameterGrid(GBM_GRID):
        clf = GradientBoostingClassifier(
            n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
            max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"],
            subsample=p["subsample"], random_state=42)
        clf.fit(X_tr, y_tr)
        auc = roc_auc_score(y_va, clf.predict_proba(X_va)[:, 1])
        if auc > best_gbm_auc:
            best_gbm_auc, best_gbm_clf, best_gbm_p = auc, clf, p
    gbm_test = roc_auc_score(y_te, best_gbm_clf.predict_proba(X_te)[:, 1])
    gbm_prob = best_gbm_clf.predict_proba(X_te)[:, 1]

    print(f"  RF  valid:{best_rf_auc:.4f}  test:{rf_test:.4f}")
    print(f"  GBM valid:{best_gbm_auc:.4f}  test:{gbm_test:.4f}")

    pred = ml.loc[idx_te, ["cik", "fiscal_year", "y"]].copy()
    pred["rf_prob"]  = rf_prob
    pred["gbm_prob"] = gbm_prob
    all_preds_v2.append(pred)

    results_v2.append({
        "test_year"    : ty,
        "rf_valid_auc" : best_rf_auc,  "rf_test_auc" : rf_test,
        "rf_params"    : str(best_rf_p),
        "gbm_valid_auc": best_gbm_auc, "gbm_test_auc": gbm_test,
        "gbm_params"   : str(best_gbm_p)
    })

    pd.DataFrame(results_v2).to_parquet(PROCESSED_PATH / "ml_results_v2.parquet", index=False)

avg_rf  = np.mean([r["rf_test_auc"]  for r in results_v2])
avg_gbm = np.mean([r["gbm_test_auc"] for r in results_v2])
print(f"
Avg RF AUC  : {avg_rf:.4f}")
print(f"Avg GBM AUC : {avg_gbm:.4f}")

pd.concat(all_preds_v2).to_parquet(PROCESSED_PATH / "ml_predictions_v2.parquet", index=False)
print("Saved ml_results_v2.parquet")
print("Saved ml_predictions_v2.parquet")
