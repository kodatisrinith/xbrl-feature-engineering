# XBRL Financial Feature Engineering for ML-Based Restatement Prediction

## Overview

I built this pipeline to extract structured financial statement data from SEC EDGAR
XBRL filings and construct a wide-format feature matrix for training machine learning
models to predict accounting restatements.

The pipeline ingests raw XBRL company facts, filters them to a research sample of
annual 10-K filings, and produces a panel dataset where each row is a firm-year
observation and each column is a scaled financial ratio at year t, its lagged value
at t-1, or a year-over-year percentage change.

## Key Features

- Large-scale XBRL parsing processes all US-GAAP unit types (USD, shares, pure)
  with a priority hierarchy to handle multi-unit tags correctly
- Strict filing-date discipline retains only filings submitted within three months
  of the fiscal period end date to reduce look-ahead bias
- Common-tag filtering restricts features to tags present in every year of the
  sample, ensuring a complete feature matrix without structural missingness
- Total-asset scaling normalises all balance sheet and income statement items by
  contemporaneous total assets for meaningful cross-firm comparison
- Percentage change features compute year-over-year changes with denominator guards,
  producing three feature variants per tag: level t, level t-1, and pct change
- Memory-efficient processing uses Parquet throughout and applies CIK-year filters
  early to keep only the relevant subset in memory

## Technologies

| Category | Tools |
|---|---|
| Data Source | SEC EDGAR XBRL Company Facts (JSON) |
| Processing | Python, pandas, NumPy |
| File Format | Parquet (pyarrow backend) |
| Filing Index | SEC quarterly submission text files |

## Requirements

```bash
pip install pandas numpy pyarrow tqdm pathlib
```

## Pipeline Steps

**Step 1 — XBRL Extraction**
Iterates over all EDGAR XBRL JSON files, extracts annual 10-K US-GAAP facts for
fiscal years 2008 to 2024, and saves to `xbrl_long_v2.parquet`

**Step 2 — Filing Index Construction**
Loads quarterly SEC submission files, filters to 10-K and 10-K/A forms, deduplicates
to one filing per firm per fiscal year, and saves to `sub_sample_extended.parquet`

**Step 3 — Feature Matrix Construction**
Filters to sample CIKs, identifies common tags, scales by total assets, pivots to
wide format with current and lagged values, computes percentage changes, merges
filing metadata, and saves to `feature_matrix_v2.parquet`

## Output Files

| File | Description |
|---|---|
| `xbrl_long_v2.parquet` | Long-format XBRL facts for all sample firms |
| `sub_sample_extended.parquet` | Filing index with fiscal and calendar year |
| `feature_matrix_v2.parquet` | Final wide-format feature matrix for ML |

## Research Context

This feature engineering work supports a research project on accounting restatement
prediction. The construction methodology follows prior literature that uses XBRL
structured data to build large-scale financial ratio panels for classification tasks.
