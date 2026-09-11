# ============================================================
# CLANAK EXTENDED ANALYSIS — COMPLETE 38-CELL MASTER CODE
# ============================================================
#
# FINAL SYNCHRONIZED VERSION
#
# ELM hidden neurons = 110
# LSTM hidden neurons = 128
# PCA maximum components = 6
# LSTM maximum epochs = 30
# Early stopping patience = 5
# Stratified 5-Fold CV
# Random seed = 42
#
# FINAL CLEAN DATA:
# Raw records                  = 16,000
# Blank/padded records removed = 5,193
# Exact duplicates removed     = 212
# Final dataset                = 10,595
#
# IMPORTANT:
# There is NO Dew Point variable in the original dataset.
#
# ============================================================

# ============================================================
# CELL 1 — INSTALL REQUIRED PACKAGES
# ============================================================

!pip -q install imbalanced-learn statsmodels openpyxl xgboost lightgbm

# ============================================================
# CELL 2 — IMPORTS AND GLOBAL SETTINGS
# ============================================================

import os
import glob
import time
import shutil
import zipfile
import warnings
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats

import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

SEED = 42

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

OUTPUT_DIR = "/content/CLANAK_Extended_Analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Output directory:", OUTPUT_DIR)
print("Random seed:", SEED)
print("ELM hidden neurons: 110")
print("LSTM neurons: 128")

# ============================================================
# CELL 3 — UPLOAD DATA
# ============================================================

from google.colab import files

uploaded = files.upload()

print("\nUploaded files:")
for f in uploaded:
    print(" -", f)

# ============================================================
# CELL 4 — LOAD AND STANDARDIZE THE FOUR SOURCE FILES
# ============================================================

CSV_FILES = [
    "sunrise_normal.csv",
    "sunrise_attack.csv",
    "sunset_normal.csv",
    "sunset_attack.csv"
]

available_files = glob.glob("/content/*.csv")

print("CSV files found:")
for f in available_files:
    print(os.path.basename(f))

frames = []

def normalize_column_name(c):
    c = str(c).strip().lower()
    c = c.replace(" ", "_")
    c = c.replace("-", "_")
    c = c.replace("/", "_")
    c = c.replace("(", "")
    c = c.replace(")", "")
    return c

COLUMN_ALIASES = {
    "temperature": "temperature",
    "temp": "temperature",

    "humidity": "humidity",
    "relative_humidity": "humidity",

    "pressure": "pressure",
    "atmospheric_pressure": "pressure",

    "wind_speed": "wind_speed",
    "windspeed": "wind_speed",

    "wind_direction": "wind_direction",
    "wind_dir": "wind_direction",

    "wind_gust": "wind_gust",
    "gust": "wind_gust",

    "rain": "rain",
    "rainfall": "rain",

    "handshake_time": "handshake_time",
    "handshake_duration": "handshake_time",
    "tls_handshake_time": "handshake_time",

    "time": "time",
    "timestamp": "time",

    "label": "label",
    "class": "label"
}

for path in available_files:

    fname = os.path.basename(path).lower()

    if not any(x in fname for x in ["sunrise", "sunset"]):
        continue

    df = pd.read_csv(path)

    rename_map = {}

    for c in df.columns:
        n = normalize_column_name(c)
        rename_map[c] = COLUMN_ALIASES.get(n, n)

    df = df.rename(columns=rename_map)

    if "attack" in fname:
        label = 1
    else:
        label = 0

    if "sunrise" in fname:
        period = "Sunrise"
    elif "sunset" in fname:
        period = "Sunset"
    else:
        period = "Unknown"

    condition = "Attack" if label == 1 else "Normal"

    df["source_period"] = period
    df["source_condition"] = condition
    df["label"] = label
    df["source_file"] = os.path.basename(path)

    frames.append(df)

if not frames:
    raise FileNotFoundError(
        "No sunrise/sunset CSV files were found in /content."
    )

data = pd.concat(frames, ignore_index=True)

print("\nRaw combined shape:", data.shape)
print("\nSource counts:")
print(data["source_file"].value_counts())

# ============================================================
# CELL 5 — FINAL REPRODUCIBLE DATA CLEANING
# ============================================================

ENVIRONMENTAL_FEATURES = [
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "wind_direction",
    "wind_gust",
    "rain"
]

OBSERVATION_FIELDS = ENVIRONMENTAL_FEATURES + [
    "handshake_time",
    "time"
]

for col in ENVIRONMENTAL_FEATURES + ["handshake_time"]:
    if col in data.columns:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

if "time" in data.columns:
    data["time"] = data["time"].replace(
        ["", " ", "nan", "NaN", "None", "none"],
        np.nan
    )

available_obs = [
    c for c in OBSERVATION_FIELDS
    if c in data.columns
]

raw_count = len(data)

# ------------------------------------------------------------
# Remove structurally blank / padded rows
# ------------------------------------------------------------

blank_mask = data[available_obs].isna().all(axis=1)

blank_removed = int(blank_mask.sum())

data = data.loc[~blank_mask].copy()

# ------------------------------------------------------------
# Remove exact duplicate observations within class
# ------------------------------------------------------------

duplicate_key = [
    c for c in available_obs
    if c in data.columns
] + ["label"]

before_dup = len(data)

data = data.drop_duplicates(
    subset=duplicate_key,
    keep="first"
).reset_index(drop=True)

duplicates_removed = before_dup - len(data)

final_count = len(data)

cleaning_audit = pd.DataFrame({
    "Stage": [
        "Raw records",
        "Blank/padded records removed",
        "Exact duplicate observations removed",
        "Final clean records"
    ],
    "Count": [
        raw_count,
        blank_removed,
        duplicates_removed,
        final_count
    ]
})

print(cleaning_audit)

cleaning_audit.to_csv(
    os.path.join(OUTPUT_DIR, "cleaning_audit.csv"),
    index=False
)

print("\nFinal class distribution:")
print(data["label"].value_counts())

print("\nFinal period distribution:")
print(data["source_period"].value_counts())

print("\nFinal period × condition distribution:")
print(
    pd.crosstab(
        data["source_period"],
        data["source_condition"]
    )
)

# Expected final dataset for the four-source dataset
if raw_count == 16000:
    assert blank_removed == 5193, (
        f"Expected 5193 blank/padded rows removed, "
        f"got {blank_removed}"
    )

    assert duplicates_removed == 212, (
        f"Expected 212 exact duplicates removed, "
        f"got {duplicates_removed}"
    )

    assert final_count == 10595, (
        f"Expected 10595 final records, "
        f"got {final_count}"
    )

print("\nCleaning verification passed.")

# ============================================================
# CELL 6 — CYCLIC TIME ENCODING
# ============================================================

data["parsed_time"] = pd.to_datetime(
    data["time"],
    format="%H:%M:%S",
    errors="coerce"
)

data["hour"] = data["parsed_time"].dt.hour.fillna(0)
data["minute"] = data["parsed_time"].dt.minute.fillna(0)
data["second"] = data["parsed_time"].dt.second.fillna(0)

data["time_seconds"] = (
    data["hour"] * 3600
    + data["minute"] * 60
    + data["second"]
)

SECONDS_PER_DAY = 86400.0

data["time_sin"] = np.sin(
    2 * np.pi * data["time_seconds"] / SECONDS_PER_DAY
)

data["time_cos"] = np.cos(
    2 * np.pi * data["time_seconds"] / SECONDS_PER_DAY
)

unit_circle_error = (
    data["time_sin"] ** 2
    + data["time_cos"] ** 2
    - 1
).abs().max()

print("Maximum unit-circle error:", unit_circle_error)

assert unit_circle_error < 1e-10

data = data.drop(
    columns=[
        "parsed_time",
        "hour",
        "minute",
        "second",
        "time_seconds"
    ],
    errors="ignore"
)

data.to_csv(
    os.path.join(OUTPUT_DIR, "cleaned_context_enriched_dataset.csv"),
    index=False
)

print("Cyclic time encoding completed.")

# ============================================================
# CELL 6 — CYCLIC TIME ENCODING
# ============================================================

data["parsed_time"] = pd.to_datetime(
    data["time"],
    format="%H:%M:%S",
    errors="coerce"
)

data["hour"] = data["parsed_time"].dt.hour.fillna(0)
data["minute"] = data["parsed_time"].dt.minute.fillna(0)
data["second"] = data["parsed_time"].dt.second.fillna(0)

data["time_seconds"] = (
    data["hour"] * 3600
    + data["minute"] * 60
    + data["second"]
)

SECONDS_PER_DAY = 86400.0

data["time_sin"] = np.sin(
    2 * np.pi * data["time_seconds"] / SECONDS_PER_DAY
)

data["time_cos"] = np.cos(
    2 * np.pi * data["time_seconds"] / SECONDS_PER_DAY
)

unit_circle_error = (
    data["time_sin"] ** 2
    + data["time_cos"] ** 2
    - 1
).abs().max()

print("Maximum unit-circle error:", unit_circle_error)

assert unit_circle_error < 1e-10

data = data.drop(
    columns=[
        "parsed_time",
        "hour",
        "minute",
        "second",
        "time_seconds"
    ],
    errors="ignore"
)

data.to_csv(
    os.path.join(OUTPUT_DIR, "cleaned_context_enriched_dataset.csv"),
    index=False
)

print("Cyclic time encoding completed.")

# ============================================================
# CELL 8 — OVERALL DATASET COMPOSITION
# ============================================================

composition = pd.DataFrame({
    "Total": [len(data)],
    "Normal": [(data["label"] == 0).sum()],
    "MitM_Attack": [(data["label"] == 1).sum()],
    "Daytime": [(data["source_period"] == "Sunrise").sum()],
    "Nighttime": [(data["source_period"] == "Sunset").sum()]
})

print(composition.T)

composition.to_csv(
    os.path.join(OUTPUT_DIR, "dataset_composition.csv")
)

# ============================================================
# CELL 9 — DESCRIPTIVE STATISTICS BY PERIOD AND CONDITION
# ============================================================

analysis_variables = [
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "wind_direction",
    "wind_gust",
    "rain",
    "handshake_time"
]

descriptive = (
    data.groupby(
        ["source_period", "source_condition"]
    )[analysis_variables]
    .agg(["count", "mean", "std", "min", "median", "max"])
)

print(descriptive)

descriptive.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "descriptive_statistics_period_condition.csv"
    )
)

# ============================================================
# CELL 10 — SUNRISE VS SUNSET WELCH T-TESTS
# ============================================================

rows = []

for var in analysis_variables:

    sunrise = data.loc[
        data["source_period"] == "Sunrise",
        var
    ].dropna()

    sunset = data.loc[
        data["source_period"] == "Sunset",
        var
    ].dropna()

    if len(sunrise) < 2 or len(sunset) < 2:
        continue

    t_stat, p_value = stats.ttest_ind(
        sunrise,
        sunset,
        equal_var=False
    )

    pooled_sd = np.sqrt(
        (
            sunrise.var(ddof=1)
            + sunset.var(ddof=1)
        ) / 2
    )

    cohens_d = (
        (sunrise.mean() - sunset.mean())
        / pooled_sd
        if pooled_sd > 0 else np.nan
    )

    rows.append({
        "Variable": var,
        "Sunrise_Mean": sunrise.mean(),
        "Sunset_Mean": sunset.mean(),
        "T_statistic": t_stat,
        "P_value": p_value,
        "Cohens_d": cohens_d
    })

sunset_tests = pd.DataFrame(rows)

if len(sunset_tests):
    sunset_tests["FDR_P"] = multipletests(
        sunset_tests["P_value"],
        method="fdr_bh"
    )[1]

print(sunset_tests)

sunset_tests.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "sunrise_vs_sunset_welch_tests.csv"
    ),
    index=False
)

# ============================================================
# CELL 11 — NORMAL VS MITM WELCH T-TESTS
# ============================================================

rows = []

for var in analysis_variables:

    normal = data.loc[
        data["label"] == 0,
        var
    ].dropna()

    attack = data.loc[
        data["label"] == 1,
        var
    ].dropna()

    if len(normal) < 2 or len(attack) < 2:
        continue

    t_stat, p_value = stats.ttest_ind(
        normal,
        attack,
        equal_var=False
    )

    pooled_sd = np.sqrt(
        (
            normal.var(ddof=1)
            + attack.var(ddof=1)
        ) / 2
    )

    cohens_d = (
        (normal.mean() - attack.mean())
        / pooled_sd
        if pooled_sd > 0 else np.nan
    )

    rows.append({
        "Variable": var,
        "Normal_Mean": normal.mean(),
        "Attack_Mean": attack.mean(),
        "T_statistic": t_stat,
        "P_value": p_value,
        "Cohens_d": cohens_d
    })

attack_tests = pd.DataFrame(rows)

if len(attack_tests):
    attack_tests["FDR_P"] = multipletests(
        attack_tests["P_value"],
        method="fdr_bh"
    )[1]

print(attack_tests)

attack_tests.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "normal_vs_attack_welch_tests.csv"
    ),
    index=False
)

# ============================================================
# CELL 12 — NORMALITY AND LEVENE TESTS
# ============================================================

normality_rows = []

rng = np.random.RandomState(SEED)

for var in analysis_variables:

    normal = data.loc[
        data["label"] == 0,
        var
    ].dropna()

    attack = data.loc[
        data["label"] == 1,
        var
    ].dropna()

    if len(normal) > 5000:
        normal = normal.sample(
            5000,
            random_state=SEED
        )

    if len(attack) > 5000:
        attack = attack.sample(
            5000,
            random_state=SEED
        )

    if len(normal) >= 3:
        sw_normal = stats.shapiro(normal)

        normality_rows.append({
            "Variable": var,
            "Group": "Normal",
            "W": sw_normal.statistic,
            "P_value": sw_normal.pvalue
        })

    if len(attack) >= 3:
        sw_attack = stats.shapiro(attack)

        normality_rows.append({
            "Variable": var,
            "Group": "Attack",
            "W": sw_attack.statistic,
            "P_value": sw_attack.pvalue
        })

normality_results = pd.DataFrame(normality_rows)

if len(normality_results):
    normality_results["FDR_P"] = multipletests(
        normality_results["P_value"],
        method="fdr_bh"
    )[1]

print(normality_results)

normality_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "normality_tests.csv"
    ),
    index=False
)

levene_rows = []

for var in analysis_variables:

    normal = data.loc[
        data["label"] == 0,
        var
    ].dropna()

    attack = data.loc[
        data["label"] == 1,
        var
    ].dropna()

    if len(normal) > 1 and len(attack) > 1:

        stat, p = stats.levene(
            normal,
            attack,
            center="median"
        )

        levene_rows.append({
            "Variable": var,
            "Levene_statistic": stat,
            "P_value": p
        })

levene_results = pd.DataFrame(levene_rows)

if len(levene_results):
    levene_results["FDR_P"] = multipletests(
        levene_results["P_value"],
        method="fdr_bh"
    )[1]

print(levene_results)

levene_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "levene_tests.csv"
    ),
    index=False
)

# ============================================================
# CELL 13 — ONE-WAY ANOVA ACROSS PERIOD × CONDITION GROUPS
# ============================================================

anova_rows = []

data["period_condition"] = (
    data["source_period"]
    + "_"
    + data["source_condition"]
)

groups = data["period_condition"].dropna().unique()

for var in analysis_variables:

    grouped = []

    for g in groups:

        values = data.loc[
            data["period_condition"] == g,
            var
        ].dropna()

        if len(values) > 1:
            grouped.append(values)

    if len(grouped) < 2:
        continue

    f_stat, p_value = stats.f_oneway(*grouped)

    valid = data[[var, "period_condition"]].dropna()

    grand_mean = valid[var].mean()

    ss_between = 0

    for g in valid["period_condition"].unique():

        vals = valid.loc[
            valid["period_condition"] == g,
            var
        ]

        ss_between += (
            len(vals)
            * (vals.mean() - grand_mean) ** 2
        )

    ss_total = (
        (valid[var] - grand_mean) ** 2
    ).sum()

    eta_squared = (
        ss_between / ss_total
        if ss_total > 0 else np.nan
    )

    anova_rows.append({
        "Variable": var,
        "F_statistic": f_stat,
        "P_value": p_value,
        "Eta_squared": eta_squared
    })

anova_results = pd.DataFrame(anova_rows)

if len(anova_results):
    anova_results["FDR_P"] = multipletests(
        anova_results["P_value"],
        method="fdr_bh"
    )[1]

print(anova_results)

anova_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "period_condition_anova.csv"
    ),
    index=False
)

# ============================================================
# CELL 14 — OVERALL SPEARMAN ENVIRONMENT VS HANDSHAKE TIME
# ============================================================

spearman_rows = []

for var in ENVIRONMENTAL_FEATURES:

    subset = data[
        [var, "handshake_time"]
    ].dropna()

    if len(subset) < 3:
        continue

    rho, p = stats.spearmanr(
        subset[var],
        subset["handshake_time"]
    )

    spearman_rows.append({
        "Variable": var,
        "Spearman_Rho": rho,
        "P_value": p,
        "N": len(subset)
    })

spearman_overall = pd.DataFrame(
    spearman_rows
)

if len(spearman_overall):
    spearman_overall["FDR_P"] = multipletests(
        spearman_overall["P_value"],
        method="fdr_bh"
    )[1]

print(spearman_overall)

spearman_overall.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "spearman_overall.csv"
    ),
    index=False
)

# ============================================================
# CELL 15 — SPEARMAN CORRELATION BY PERIOD
# ============================================================

rows = []

for period in data["source_period"].dropna().unique():

    subset_period = data[
        data["source_period"] == period
    ]

    for var in ENVIRONMENTAL_FEATURES:

        subset = subset_period[
            [var, "handshake_time"]
        ].dropna()

        if len(subset) < 3:
            continue

        rho, p = stats.spearmanr(
            subset[var],
            subset["handshake_time"]
        )

        rows.append({
            "Period": period,
            "Variable": var,
            "Spearman_Rho": rho,
            "P_value": p,
            "N": len(subset)
        })

spearman_period = pd.DataFrame(rows)

if len(spearman_period):
    spearman_period["FDR_P"] = multipletests(
        spearman_period["P_value"],
        method="fdr_bh"
    )[1]

print(spearman_period)

spearman_period.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "spearman_by_period.csv"
    ),
    index=False
)

# ============================================================
# CELL 16 — SPEARMAN CORRELATION BY CONDITION
# ============================================================

rows = []

for condition in data["source_condition"].dropna().unique():

    subset_condition = data[
        data["source_condition"] == condition
    ]

    for var in ENVIRONMENTAL_FEATURES:

        subset = subset_condition[
            [var, "handshake_time"]
        ].dropna()

        if len(subset) < 3:
            continue

        rho, p = stats.spearmanr(
            subset[var],
            subset["handshake_time"]
        )

        rows.append({
            "Condition": condition,
            "Variable": var,
            "Spearman_Rho": rho,
            "P_value": p,
            "N": len(subset)
        })

spearman_condition = pd.DataFrame(rows)

if len(spearman_condition):
    spearman_condition["FDR_P"] = multipletests(
        spearman_condition["P_value"],
        method="fdr_bh"
    )[1]

print(spearman_condition)

spearman_condition.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "spearman_by_condition.csv"
    ),
    index=False
)

# ============================================================
# CELL 17 — VARIANCE INFLATION FACTOR
# ============================================================

vif_data = data[
    ENVIRONMENTAL_FEATURES
].copy()

vif_data = vif_data.apply(
    pd.to_numeric,
    errors="coerce"
)

vif_data = vif_data.dropna()

vif_scaled = StandardScaler().fit_transform(
    vif_data
)

vif_scaled = pd.DataFrame(
    vif_scaled,
    columns=ENVIRONMENTAL_FEATURES
)

vif_rows = []

for i, feature in enumerate(
    ENVIRONMENTAL_FEATURES
):

    vif_value = variance_inflation_factor(
        vif_scaled.values,
        i
    )

    vif_rows.append({
        "Variable": feature,
        "VIF": vif_value
    })

vif_results = pd.DataFrame(vif_rows)

print(vif_results)

vif_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "environmental_VIF.csv"
    ),
    index=False
)

# ============================================================
# CELL 18 — ENVIRONMENTAL PCA
# ============================================================

env_complete = data[
    ENVIRONMENTAL_FEATURES
].dropna()

env_scaler = StandardScaler()

env_scaled = env_scaler.fit_transform(
    env_complete
)

environment_pca = PCA()

environment_pca_scores = environment_pca.fit_transform(
    env_scaled
)

explained = environment_pca.explained_variance_ratio_
cumulative = np.cumsum(explained)

n_95 = np.argmax(
    cumulative >= 0.95
) + 1

print("Explained variance:")
for i, value in enumerate(explained, 1):
    print(
        f"PC{i}: "
        f"{value:.6f} "
        f"({value*100:.2f}%)"
    )

print("\nComponents required for 95% variance:", n_95)

loadings = pd.DataFrame(
    environment_pca.components_.T,
    index=ENVIRONMENTAL_FEATURES,
    columns=[
        f"PC{i}"
        for i in range(1, len(ENVIRONMENTAL_FEATURES)+1)
    ]
)

print("\nPCA loadings:")
print(loadings)

pca_variance = pd.DataFrame({
    "Component": [
        f"PC{i}"
        for i in range(1, len(explained)+1)
    ],
    "Explained_Variance": explained,
    "Cumulative_Variance": cumulative
})

pca_variance.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "environmental_PCA_variance.csv"
    ),
    index=False
)

loadings.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "environmental_PCA_loadings.csv"
    )
)

# ============================================================
# CELL 19 — ENVIRONMENT-ADJUSTED OLS
# ============================================================

ols_data = data[
    ["handshake_time", "label", "source_period"]
    + ENVIRONMENTAL_FEATURES
].dropna().copy()

ols_data["sunset"] = (
    ols_data["source_period"] == "Sunset"
).astype(int)

env_std = StandardScaler().fit_transform(
    ols_data[ENVIRONMENTAL_FEATURES]
)

ols_pca = PCA(
    n_components=min(
        6,
        len(ENVIRONMENTAL_FEATURES)
    )
)

ols_pc_scores = ols_pca.fit_transform(
    env_std
)

for i in range(ols_pc_scores.shape[1]):
    ols_data[f"PC{i+1}"] = ols_pc_scores[:, i]

predictors = [
    "label",
    "sunset"
] + [
    f"PC{i+1}"
    for i in range(ols_pc_scores.shape[1])
]

X_ols = sm.add_constant(
    ols_data[predictors]
)

y_ols = ols_data["handshake_time"]

ols_model = sm.OLS(
    y_ols,
    X_ols
).fit()

print(ols_model.summary())

with open(
    os.path.join(
        OUTPUT_DIR,
        "environment_adjusted_OLS.txt"
    ),
    "w"
) as f:
    f.write(
        ols_model.summary().as_text()
    )

# ============================================================
# CELL 20 — ATTACK × ENVIRONMENT INTERACTION OLS
# ============================================================

interaction_data = ols_data.copy()

for i in range(
    1,
    ols_pc_scores.shape[1] + 1
):

    interaction_data[
        f"attack_PC{i}"
    ] = (
        interaction_data["label"]
        * interaction_data[f"PC{i}"]
    )

interaction_predictors = [
    "label",
    "sunset"
]

interaction_predictors += [
    f"PC{i}"
    for i in range(
        1,
        ols_pc_scores.shape[1] + 1
    )
]

interaction_predictors += [
    f"attack_PC{i}"
    for i in range(
        1,
        ols_pc_scores.shape[1] + 1
    )
]

X_interaction = sm.add_constant(
    interaction_data[
        interaction_predictors
    ]
)

y_interaction = interaction_data[
    "handshake_time"
]

interaction_model = sm.OLS(
    y_interaction,
    X_interaction
).fit()

print(interaction_model.summary())

with open(
    os.path.join(
        OUTPUT_DIR,
        "attack_environment_interaction_OLS.txt"
    ),
    "w"
) as f:
    f.write(
        interaction_model.summary().as_text()
    )

# ============================================================
# CELL 21 — CONVENTIONAL MACHINE LEARNING MODELS
# ============================================================

MODELS = {

    "LogisticRegression": LogisticRegression(
        max_iter=2000,
        random_state=SEED
    ),

    "DecisionTree": DecisionTreeClassifier(
        random_state=SEED
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        random_state=SEED,
        n_jobs=-1
    ),

    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=200,
        random_state=SEED,
        n_jobs=-1
    ),

    "SVM": SVC(
        probability=True,
        random_state=SEED
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "GradientBoosting": GradientBoostingClassifier(
        random_state=SEED
    ),

    "MLP": MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=500,
        random_state=SEED
    )
}

# Optional XGBoost
try:

    from xgboost import XGBClassifier

    MODELS["XGBoost"] = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1
    )

    print("XGBoost available.")

except Exception as e:

    print("XGBoost unavailable:", e)


# Optional LightGBM
try:

    from lightgbm import LGBMClassifier

    MODELS["LightGBM"] = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=SEED,
        verbosity=-1
    )

    print("LightGBM available.")

except Exception as e:

    print("LightGBM unavailable:", e)


print("\nModels:")
for name in MODELS:
    print(" -", name)

# ============================================================
# CELL 22 — GENERIC CONVENTIONAL ML EVALUATION FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_score=None
):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    result = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Specificity": specificity,
        "FPR": fpr,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }

    if y_score is not None:

        try:
            result["ROC_AUC"] = roc_auc_score(
                y_true,
                y_score
            )

        except Exception:
            result["ROC_AUC"] = np.nan

        try:
            result["PR_AUC"] = average_precision_score(
                y_true,
                y_score
            )

        except Exception:
            result["PR_AUC"] = np.nan

    else:

        result["ROC_AUC"] = np.nan
        result["PR_AUC"] = np.nan

    return result


def evaluate_conventional_model(
    model,
    X,
    y,
    model_name,
    config_name
):

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(X, y),
        start=1
    ):

        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()

        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()

        # TRAIN ONLY IMPUTATION
        imputer = SimpleImputer(
            strategy="mean"
        )

        X_train_imp = imputer.fit_transform(
            X_train
        )

        X_test_imp = imputer.transform(
            X_test
        )

        # TRAIN ONLY SCALING
        scaler = MinMaxScaler()

        X_train_scaled = scaler.fit_transform(
            X_train_imp
        )

        X_test_scaled = scaler.transform(
            X_test_imp
        )

        # TRAIN ONLY SMOTE
        smote = SMOTE(
            random_state=SEED,
            k_neighbors=5
        )

        X_balanced, y_balanced = smote.fit_resample(
            X_train_scaled,
            y_train
        )

        clf = clone(model)

        start_time = time.time()

        clf.fit(
            X_balanced,
            y_balanced
        )

        train_time = time.time() - start_time

        y_pred = clf.predict(
            X_test_scaled
        )

        if hasattr(clf, "predict_proba"):

            y_score = clf.predict_proba(
                X_test_scaled
            )[:, 1]

        elif hasattr(clf, "decision_function"):

            y_score = clf.decision_function(
                X_test_scaled
            )

        else:

            y_score = None

        metrics = calculate_metrics(
            y_test,
            y_pred,
            y_score
        )

        metrics.update({
            "Model": model_name,
            "Configuration": config_name,
            "Fold": fold,
            "Training_Time_sec": train_time
        })

        fold_results.append(metrics)

    return pd.DataFrame(fold_results)

# ============================================================
# CELL 23 — RUN CONVENTIONAL ML ON D1, D2 AND D3
# ============================================================

conventional_results = []

for config_name, features in CONFIGURATIONS.items():

    print("\n================================================")
    print(config_name)
    print("================================================")

    X = data[features].copy()
    y = data["label"].astype(int).copy()

    for model_name, model in MODELS.items():

        print("Running:", model_name)

        result = evaluate_conventional_model(
            model=model,
            X=X,
            y=y,
            model_name=model_name,
            config_name=config_name
        )

        conventional_results.append(result)

conventional_results = pd.concat(
    conventional_results,
    ignore_index=True
)

print(
    conventional_results.head()
)

conventional_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "conventional_ML_fold_results.csv"
    ),
    index=False
)

# ============================================================
# CELL 24 — AGGREGATE CONVENTIONAL ML RESULTS
# ============================================================

metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "Specificity",
    "FPR",
    "ROC_AUC",
    "PR_AUC",
    "Training_Time_sec"
]

conventional_summary = (
    conventional_results
    .groupby(
        ["Configuration", "Model"]
    )[metric_columns]
    .agg(["mean", "std"])
    .reset_index()
)

print(conventional_summary)

conventional_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "conventional_ML_summary.csv"
    ),
    index=False
)

# ============================================================
# CELL 25 — CONFIGURATION COMPARISON PIVOT
# ============================================================

config_comparison = (
    conventional_results
    .groupby(
        ["Configuration", "Model"]
    )[
        [
            "Accuracy",
            "F1",
            "FPR",
            "ROC_AUC",
            "PR_AUC"
        ]
    ]
    .mean()
    .reset_index()
)

print(config_comparison)

config_comparison.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "configuration_comparison.csv"
    ),
    index=False
)

accuracy_pivot = config_comparison.pivot(
    index="Model",
    columns="Configuration",
    values="Accuracy"
)

f1_pivot = config_comparison.pivot(
    index="Model",
    columns="Configuration",
    values="F1"
)

fpr_pivot = config_comparison.pivot(
    index="Model",
    columns="Configuration",
    values="FPR"
)

print("\nAccuracy:")
print(accuracy_pivot)

print("\nF1:")
print(f1_pivot)

print("\nFPR:")
print(fpr_pivot)

# ============================================================
# CELL 26 — CONTEXT IMPROVEMENT ANALYSIS
# ============================================================

context_rows = []

for model_name in config_comparison["Model"].unique():

    subset = config_comparison[
        config_comparison["Model"] == model_name
    ].set_index("Configuration")

    if (
        "D1_TLS_only" not in subset.index
        or "D2_TLS_Meteorological" not in subset.index
        or "D3_TLS_Meteorological_Time" not in subset.index
    ):
        continue

    d1 = subset.loc["D1_TLS_only"]
    d2 = subset.loc["D2_TLS_Meteorological"]
    d3 = subset.loc["D3_TLS_Meteorological_Time"]

    context_rows.append({
        "Model": model_name,

        "D1_Accuracy": d1["Accuracy"],
        "D2_Accuracy": d2["Accuracy"],
        "D3_Accuracy": d3["Accuracy"],

        "D2_minus_D1_Accuracy": (
            d2["Accuracy"] - d1["Accuracy"]
        ),

        "D3_minus_D1_Accuracy": (
            d3["Accuracy"] - d1["Accuracy"]
        ),

        "D1_F1": d1["F1"],
        "D2_F1": d2["F1"],
        "D3_F1": d3["F1"],

        "D3_minus_D1_F1": (
            d3["F1"] - d1["F1"]
        ),

        "D1_FPR": d1["FPR"],
        "D3_FPR": d3["FPR"],

        "D3_FPR_Reduction": (
            d1["FPR"] - d3["FPR"]
        ),

        "D3_FPR_Reduction_Percent": (
            (
                d1["FPR"] - d3["FPR"]
            ) / d1["FPR"] * 100
            if d1["FPR"] > 0
            else np.nan
        )
    })

context_improvement = pd.DataFrame(
    context_rows
)

print(context_improvement)

context_improvement.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "context_improvement_analysis.csv"
    ),
    index=False
)

# ============================================================
# CELL 27 — CROSS-PERIOD GENERALISATION
# ============================================================

def cross_period_evaluation(
    model,
    features,
    train_period,
    test_period,
    model_name,
    config_name
):

    train_data = data[
        data["source_period"] == train_period
    ]

    test_data = data[
        data["source_period"] == test_period
    ]

    X_train = train_data[features].copy()
    y_train = train_data["label"].astype(int)

    X_test = test_data[features].copy()
    y_test = test_data["label"].astype(int)

    imputer = SimpleImputer(
        strategy="mean"
    )

    X_train = imputer.fit_transform(
        X_train
    )

    X_test = imputer.transform(
        X_test
    )

    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_test = scaler.transform(
        X_test
    )

    smote = SMOTE(
        random_state=SEED,
        k_neighbors=5
    )

    X_train, y_train = smote.fit_resample(
        X_train,
        y_train
    )

    clf = clone(model)

    start = time.time()

    clf.fit(
        X_train,
        y_train
    )

    elapsed = time.time() - start

    y_pred = clf.predict(
        X_test
    )

    if hasattr(clf, "predict_proba"):

        y_score = clf.predict_proba(
            X_test
        )[:, 1]

    elif hasattr(clf, "decision_function"):

        y_score = clf.decision_function(
            X_test
        )

    else:

        y_score = None

    metrics = calculate_metrics(
        y_test,
        y_pred,
        y_score
    )

    metrics.update({
        "Model": model_name,
        "Configuration": config_name,
        "Train_Period": train_period,
        "Test_Period": test_period,
        "Training_Time_sec": elapsed
    })

    return metrics


cross_period_results = []

for config_name, features in CONFIGURATIONS.items():

    for model_name, model in MODELS.items():

        for train_period, test_period in [
            ("Sunrise", "Sunset"),
            ("Sunset", "Sunrise")
        ]:

            print(
                config_name,
                model_name,
                train_period,
                "->",
                test_period
            )

            result = cross_period_evaluation(
                model,
                features,
                train_period,
                test_period,
                model_name,
                config_name
            )

            cross_period_results.append(
                result
            )

cross_period_results = pd.DataFrame(
    cross_period_results
)

print(cross_period_results)

cross_period_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "cross_period_generalisation.csv"
    ),
    index=False
)

# ============================================================
# CELL 28 — DEFINE SYNCHRONIZED ELM WITH 110 HIDDEN NEURONS
# ============================================================

class SimpleELM:

    def __init__(
        self,
        hidden_size=110,
        random_state=42
    ):

        self.hidden_size = hidden_size
        self.random_state = random_state

        self.input_weights = None
        self.bias = None
        self.beta = None

    @staticmethod
    def sigmoid(x):

        x = np.clip(
            x,
            -50,
            50
        )

        return 1.0 / (
            1.0 + np.exp(-x)
        )

    def fit(
        self,
        X,
        y
    ):

        rng = np.random.RandomState(
            self.random_state
        )

        n_features = X.shape[1]

        self.input_weights = rng.normal(
            size=(
                n_features,
                self.hidden_size
            )
        )

        self.bias = rng.normal(
            size=self.hidden_size
        )

        H = self.sigmoid(
            X @ self.input_weights
            + self.bias
        )

        self.beta = (
            np.linalg.pinv(H)
            @ np.asarray(y).reshape(-1, 1)
        )

        return self

    def predict_proba(
        self,
        X
    ):

        H = self.sigmoid(
            X @ self.input_weights
            + self.bias
        )

        scores = H @ self.beta

        scores = np.clip(
            scores.ravel(),
            0,
            1
        )

        return np.column_stack([
            1 - scores,
            scores
        ])

    def predict(
        self,
        X
    ):

        probabilities = self.predict_proba(
            X
        )[:, 1]

        return (
            probabilities >= 0.5
        ).astype(int)


print(
    "SimpleELM configured with",
    "110 hidden neurons."
)

# ============================================================
# CELL 29 — ELM 110 BENCHMARK WITH TRAIN-ONLY IG
# ============================================================

elm_fold_results = []
elm_ig_results = []

for config_name, features in CONFIGURATIONS.items():

    X = data[features].copy()
    y = data["label"].astype(int).copy()

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(X, y),
        start=1
    ):

        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()

        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()

        # TRAIN ONLY IMPUTATION
        imputer = SimpleImputer(
            strategy="mean"
        )

        X_train_imp = imputer.fit_transform(
            X_train
        )

        X_test_imp = imputer.transform(
            X_test
        )

        # TRAIN ONLY SCALING
        scaler = MinMaxScaler()

        X_train_scaled = scaler.fit_transform(
            X_train_imp
        )

        X_test_scaled = scaler.transform(
            X_test_imp
        )

        # TRAIN ONLY SMOTE
        smote = SMOTE(
            random_state=SEED,
            k_neighbors=5
        )

        X_balanced, y_balanced = smote.fit_resample(
            X_train_scaled,
            y_train
        )

        # TRAIN-ONLY INFORMATION GAIN
        mi_scores = mutual_info_classif(
            X_balanced,
            y_balanced,
            random_state=SEED
        )

        mi_table = pd.DataFrame({
            "Feature": features,
            "Information_Gain": mi_scores
        }).sort_values(
            "Information_Gain",
            ascending=False
        )

        top_k = min(
            6,
            len(features)
        )

        selected_features = (
            mi_table.head(top_k)["Feature"]
            .tolist()
        )

        selected_indices = [
            features.index(f)
            for f in selected_features
        ]

        X_train_selected = (
            X_balanced[:, selected_indices]
        )

        X_test_selected = (
            X_test_scaled[:, selected_indices]
        )

        for _, row in mi_table.iterrows():

            elm_ig_results.append({
                "Configuration": config_name,
                "Fold": fold,
                "Feature": row["Feature"],
                "Information_Gain": row[
                    "Information_Gain"
                ],
                "Selected": (
                    row["Feature"]
                    in selected_features
                )
            })

        # ELM — EXACT 110-NEURON SETTING
        elm = SimpleELM(
            hidden_size=110,
            random_state=SEED + fold
        )

        start = time.time()

        elm.fit(
            X_train_selected,
            y_balanced
        )

        training_time = time.time() - start

        y_score = elm.predict_proba(
            X_test_selected
        )[:, 1]

        y_pred = (
            y_score >= 0.5
        ).astype(int)

        metrics = calculate_metrics(
            y_test,
            y_pred,
            y_score
        )

        metrics.update({
            "Model": "ELM_110",
            "Configuration": config_name,
            "Fold": fold,
            "Training_Time_sec": training_time
        })

        elm_fold_results.append(
            metrics
        )

elm_fold_results = pd.DataFrame(
    elm_fold_results
)

elm_ig_results = pd.DataFrame(
    elm_ig_results
)

print("\nELM results:")
print(elm_fold_results)

print("\nInformation Gain:")
print(elm_ig_results)

elm_fold_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ELM_110_fold_results.csv"
    ),
    index=False
)

elm_ig_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ELM_110_information_gain.csv"
    ),
    index=False
)

# ============================================================
# CELL 30 — LSTM 128 WITH TRAIN-ONLY PCA AND EARLY STOPPING
# ============================================================

def build_lstm(
    input_features,
    seed
):

    tf.keras.backend.clear_session()

    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    model = Sequential([
        Input(
            shape=(1, input_features)
        ),

        LSTM(128),

        Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy"
    )

    return model


lstm_fold_results = []
lstm_pca_results = []

for config_name, features in CONFIGURATIONS.items():

    X = data[features].copy()
    y = data["label"].astype(int).copy()

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(X, y),
        start=1
    ):

        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()

        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()

        # TRAIN ONLY IMPUTATION
        imputer = SimpleImputer(
            strategy="mean"
        )

        X_train_imp = imputer.fit_transform(
            X_train
        )

        X_test_imp = imputer.transform(
            X_test
        )

        # TRAIN ONLY SCALING
        scaler = MinMaxScaler()

        X_train_scaled = scaler.fit_transform(
            X_train_imp
        )

        X_test_scaled = scaler.transform(
            X_test_imp
        )

        # TRAIN ONLY SMOTE
        smote = SMOTE(
            random_state=SEED,
            k_neighbors=5
        )

        X_balanced, y_balanced = smote.fit_resample(
            X_train_scaled,
            y_train
        )

        # TRAIN ONLY PCA
        n_components = min(
            6,
            X_balanced.shape[1]
        )

        pca = PCA(
            n_components=n_components
        )

        X_train_pca = pca.fit_transform(
            X_balanced
        )

        X_test_pca = pca.transform(
            X_test_scaled
        )

        explained = (
            pca.explained_variance_ratio_
        )

        cumulative = np.cumsum(
            explained
        )

        for pc_i in range(
            len(explained)
        ):

            lstm_pca_results.append({
                "Configuration": config_name,
                "Fold": fold,
                "Component": pc_i + 1,
                "Explained_Variance": explained[pc_i],
                "Cumulative_Variance": cumulative[pc_i]
            })

        model = build_lstm(
            X_train_pca.shape[1],
            SEED + fold
        )

        early_stopping = EarlyStopping(
            monitor="loss",
            patience=5,
            restore_best_weights=True
        )

        start = time.time()

        model.fit(
            X_train_pca.reshape(
                -1,
                1,
                X_train_pca.shape[1]
            ),
            y_balanced,
            epochs=30,
            batch_size=64,
            callbacks=[early_stopping],
            verbose=0
        )

        training_time = time.time() - start

        y_score = model.predict(
            X_test_pca.reshape(
                -1,
                1,
                X_test_pca.shape[1]
            ),
            verbose=0
        ).ravel()

        # LSTM threshold is strictly > 0.5
        y_pred = (
            y_score > 0.5
        ).astype(int)

        metrics = calculate_metrics(
            y_test,
            y_pred,
            y_score
        )

        metrics.update({
            "Model": "LSTM_128",
            "Configuration": config_name,
            "Fold": fold,
            "Training_Time_sec": training_time,
            "Epochs_Max": 30
        })

        lstm_fold_results.append(
            metrics
        )

lstm_fold_results = pd.DataFrame(
    lstm_fold_results
)

lstm_pca_results = pd.DataFrame(
    lstm_pca_results
)

print("\nLSTM results:")
print(lstm_fold_results)

print("\nLSTM PCA:")
print(lstm_pca_results)

lstm_fold_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "LSTM_128_fold_results.csv"
    ),
    index=False
)

lstm_pca_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "LSTM_128_PCA_results.csv"
    ),
    index=False
)

# ============================================================
# CELL 31 — COMBINE CONVENTIONAL ML, ELM AND LSTM RESULTS
# ============================================================

all_model_results = []

# Conventional
conv_temp = conventional_results.copy()
conv_temp["Model_Type"] = "Conventional_ML"

all_model_results.append(
    conv_temp
)

# ELM
elm_temp = elm_fold_results.copy()
elm_temp["Model_Type"] = "ELM"

all_model_results.append(
    elm_temp
)

# LSTM
lstm_temp = lstm_fold_results.copy()
lstm_temp["Model_Type"] = "LSTM"

all_model_results.append(
    lstm_temp
)

all_model_results = pd.concat(
    all_model_results,
    ignore_index=True,
    sort=False
)

print(
    all_model_results[
        [
            "Model",
            "Configuration",
            "Fold",
            "Accuracy",
            "FPR"
        ]
    ].head()
)

all_model_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "all_model_fold_results.csv"
    ),
    index=False
)

# ============================================================
# CELL 32 — ELM-LSTM UNANIMOUS ENSEMBLE
# ============================================================

ensemble_results = []
ensemble_prediction_records = []

for config_name, features in CONFIGURATIONS.items():

    X = data[features].copy()
    y = data["label"].astype(int).copy()

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(X, y),
        start=1
    ):

        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()

        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()

        # ----------------------------------------------------
        # TRAIN-ONLY IMPUTATION
        # ----------------------------------------------------

        imputer = SimpleImputer(
            strategy="mean"
        )

        X_train_imp = imputer.fit_transform(
            X_train
        )

        X_test_imp = imputer.transform(
            X_test
        )

        # ----------------------------------------------------
        # TRAIN-ONLY MINMAX SCALING
        # ----------------------------------------------------

        scaler = MinMaxScaler()

        X_train_scaled = scaler.fit_transform(
            X_train_imp
        )

        X_test_scaled = scaler.transform(
            X_test_imp
        )

        # ----------------------------------------------------
        # TRAIN-ONLY SMOTE
        # ----------------------------------------------------

        smote = SMOTE(
            random_state=SEED,
            k_neighbors=5
        )

        X_balanced, y_balanced = smote.fit_resample(
            X_train_scaled,
            y_train
        )

        # ----------------------------------------------------
        # INFORMATION GAIN FOR ELM
        # ----------------------------------------------------

        mi_scores = mutual_info_classif(
            X_balanced,
            y_balanced,
            random_state=SEED
        )

        mi_table = pd.DataFrame({
            "Feature": features,
            "Information_Gain": mi_scores
        }).sort_values(
            "Information_Gain",
            ascending=False
        )

        top_k = min(
            6,
            len(features)
        )

        selected_features = (
            mi_table.head(top_k)["Feature"]
            .tolist()
        )

        selected_indices = [
            features.index(f)
            for f in selected_features
        ]

        X_train_elm = (
            X_balanced[:, selected_indices]
        )

        X_test_elm = (
            X_test_scaled[:, selected_indices]
        )

        # ----------------------------------------------------
        # ELM — 110 HIDDEN NEURONS
        # ----------------------------------------------------

        elm = SimpleELM(
            hidden_size=110,
            random_state=SEED + fold
        )

        elm.fit(
            X_train_elm,
            y_balanced
        )

        elm_score = elm.predict_proba(
            X_test_elm
        )[:, 1]

        elm_pred = (
            elm_score >= 0.5
        ).astype(int)

        # ----------------------------------------------------
        # LSTM PCA
        # ----------------------------------------------------

        n_components = min(
            6,
            X_balanced.shape[1]
        )

        pca = PCA(
            n_components=n_components
        )

        X_train_lstm = pca.fit_transform(
            X_balanced
        )

        X_test_lstm = pca.transform(
            X_test_scaled
        )

        # ----------------------------------------------------
        # LSTM
        # ----------------------------------------------------

        lstm = build_lstm(
            X_train_lstm.shape[1],
            SEED + fold
        )

        early_stopping = EarlyStopping(
            monitor="loss",
            patience=5,
            restore_best_weights=True
        )

        lstm.fit(
            X_train_lstm.reshape(
                -1,
                1,
                X_train_lstm.shape[1]
            ),
            y_balanced,
            epochs=30,
            batch_size=64,
            callbacks=[early_stopping],
            verbose=0
        )

        lstm_score = lstm.predict(
            X_test_lstm.reshape(
                -1,
                1,
                X_test_lstm.shape[1]
            ),
            verbose=0
        ).ravel()

        lstm_pred = (
            lstm_score > 0.5
        ).astype(int)

        # ----------------------------------------------------
        # UNANIMOUS AND ENSEMBLE
        # ----------------------------------------------------

        ensemble_pred = (
            (elm_pred == 1)
            &
            (lstm_pred == 1)
        ).astype(int)

        # Continuous ensemble score.
        #
        # IMPORTANT:
        # This product is a ranking/score quantity,
        # NOT a calibrated probability.
        ensemble_score = (
            elm_score * lstm_score
        )

        metrics = calculate_metrics(
            y_test,
            ensemble_pred,
            ensemble_score
        )

        metrics.update({
            "Model": "ELM110_LSTM128_Unanimous_AND",
            "Configuration": config_name,
            "Fold": fold
        })

        ensemble_results.append(
            metrics
        )

        for i, original_idx in enumerate(
            test_idx
        ):

            ensemble_prediction_records.append({
                "Configuration": config_name,
                "Fold": fold,
                "Index": original_idx,
                "True_Label": int(y_test.iloc[i]),
                "ELM_Score": elm_score[i],
                "ELM_Prediction": elm_pred[i],
                "LSTM_Score": lstm_score[i],
                "LSTM_Prediction": lstm_pred[i],
                "Ensemble_Score": ensemble_score[i],
                "Ensemble_Prediction": ensemble_pred[i]
            })

ensemble_results = pd.DataFrame(
    ensemble_results
)

ensemble_predictions = pd.DataFrame(
    ensemble_prediction_records
)

print("\nEnsemble fold results:")
print(ensemble_results)

ensemble_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ELM110_LSTM128_ensemble_fold_results.csv"
    ),
    index=False
)

ensemble_predictions.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ELM110_LSTM128_ensemble_predictions.csv"
    ),
    index=False
)

# ============================================================
# CELL 33 — FINAL MODEL COMPARISON
# ============================================================

model_summary_rows = []

for df_name, df in [
    ("Conventional_ML", conventional_results),
    ("ELM_110", elm_fold_results),
    ("LSTM_128", lstm_fold_results),
    ("ELM110_LSTM128_Ensemble", ensemble_results)
]:

    group_columns = [
        "Configuration"
    ]

    grouped = (
        df.groupby(group_columns)
        [
            [
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "Specificity",
                "FPR",
                "ROC_AUC",
                "PR_AUC"
            ]
        ]
        .mean()
        .reset_index()
    )

    grouped["Model"] = df_name

    model_summary_rows.append(
        grouped
    )

final_model_comparison = pd.concat(
    model_summary_rows,
    ignore_index=True
)

final_model_comparison = final_model_comparison[
    [
        "Model",
        "Configuration",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Specificity",
        "FPR",
        "ROC_AUC",
        "PR_AUC"
    ]
]

print(final_model_comparison)

final_model_comparison.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_model_comparison.csv"
    ),
    index=False
)

# ============================================================
# CELL 34 — PUBLICATION-READY MEAN ± SD TABLE
# ============================================================

publication_rows = []

datasets_for_publication = {
    "Conventional_ML": conventional_results,
    "ELM_110": elm_fold_results,
    "LSTM_128": lstm_fold_results,
    "ELM110_LSTM128_Ensemble": ensemble_results
}

for model_name, df in datasets_for_publication.items():

    for config_name in CONFIGURATIONS:

        subset = df[
            df["Configuration"]
            == config_name
        ]

        if len(subset) == 0:
            continue

        row = {
            "Model": model_name,
            "Configuration": config_name
        }

        for metric in [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "Specificity",
            "FPR",
            "ROC_AUC",
            "PR_AUC"
        ]:

            mean_value = subset[
                metric
            ].mean()

            sd_value = subset[
                metric
            ].std()

            row[
                metric + "_Mean_SD"
            ] = (
                f"{mean_value:.4f} ± "
                f"{sd_value:.4f}"
            )

        publication_rows.append(
            row
        )

publication_table = pd.DataFrame(
    publication_rows
)

print(publication_table)

publication_table.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "publication_mean_SD_table.csv"
    ),
    index=False
)

# ============================================================
# CELL 35 — CREATE MASTER EXCEL WORKBOOK
# ============================================================

MASTER_EXCEL = os.path.join(
    OUTPUT_DIR,
    "CLANAK_Extended_Master_Results.xlsx"
)


# ------------------------------------------------------------
# Helper function:
# Flatten MultiIndex columns before Excel export
# ------------------------------------------------------------
def flatten_columns(df):
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(
                str(level).strip()
                for level in col
                if str(level).strip() not in ("", "None", "nan")
            )
            for col in df.columns
        ]

    return df


# Flatten tables that may contain MultiIndex columns
conventional_summary_excel = flatten_columns(
    conventional_summary
)

final_model_comparison_excel = flatten_columns(
    final_model_comparison
)

publication_table_excel = flatten_columns(
    publication_table
)


# ------------------------------------------------------------
# Create Master Excel Workbook
# ------------------------------------------------------------
with pd.ExcelWriter(
    MASTER_EXCEL,
    engine="openpyxl"
) as writer:

    cleaning_audit.to_excel(
        writer,
        sheet_name="Cleaning_Audit",
        index=False
    )

    composition.to_excel(
        writer,
        sheet_name="Composition",
        index=False
    )

    descriptive.to_excel(
        writer,
        sheet_name="Descriptive_Stats"
    )

    sunset_tests.to_excel(
        writer,
        sheet_name="Sunrise_Sunset_Tests",
        index=False
    )

    attack_tests.to_excel(
        writer,
        sheet_name="Normal_Attack_Tests",
        index=False
    )

    normality_results.to_excel(
        writer,
        sheet_name="Normality",
        index=False
    )

    levene_results.to_excel(
        writer,
        sheet_name="Levene",
        index=False
    )

    anova_results.to_excel(
        writer,
        sheet_name="ANOVA",
        index=False
    )

    spearman_overall.to_excel(
        writer,
        sheet_name="Spearman_Overall",
        index=False
    )

    spearman_period.to_excel(
        writer,
        sheet_name="Spearman_Period",
        index=False
    )

    spearman_condition.to_excel(
        writer,
        sheet_name="Spearman_Condition",
        index=False
    )

    vif_results.to_excel(
        writer,
        sheet_name="VIF",
        index=False
    )

    pca_variance.to_excel(
        writer,
        sheet_name="Environmental_PCA",
        index=False
    )

    loadings.to_excel(
        writer,
        sheet_name="PCA_Loadings"
    )

    conventional_results.to_excel(
        writer,
        sheet_name="Conventional_Folds",
        index=False
    )

    # IMPORTANT:
    # Flattened MultiIndex version
    conventional_summary_excel.to_excel(
        writer,
        sheet_name="Conventional_Summary",
        index=False
    )

    context_improvement.to_excel(
        writer,
        sheet_name="Context_Improvement",
        index=False
    )

    cross_period_results.to_excel(
        writer,
        sheet_name="Cross_Period",
        index=False
    )

    elm_fold_results.to_excel(
        writer,
        sheet_name="ELM110_Folds",
        index=False
    )

    elm_ig_results.to_excel(
        writer,
        sheet_name="ELM110_Information_Gain",
        index=False
    )

    lstm_fold_results.to_excel(
        writer,
        sheet_name="LSTM128_Folds",
        index=False
    )

    lstm_pca_results.to_excel(
        writer,
        sheet_name="LSTM128_PCA",
        index=False
    )

    ensemble_results.to_excel(
        writer,
        sheet_name="Ensemble_Folds",
        index=False
    )

    ensemble_predictions.to_excel(
        writer,
        sheet_name="Ensemble_Predictions",
        index=False
    )

    # Flattened version in case this table also has MultiIndex columns
    final_model_comparison_excel.to_excel(
        writer,
        sheet_name="Final_Comparison",
        index=False
    )

    # Flattened version in case publication table has MultiIndex columns
    publication_table_excel.to_excel(
        writer,
        sheet_name="Publication_Table",
        index=False
    )


print(
    "Master Excel created successfully:"
)
print(MASTER_EXCEL)

# ============================================================
# CELL 36 — CREATE COMPLETE ZIP ARCHIVE
# ============================================================

ZIP_PATH = "/content/CLANAK_Extended_Analysis_COMPLETE.zip"

if os.path.exists(ZIP_PATH):
    os.remove(ZIP_PATH)

with zipfile.ZipFile(
    ZIP_PATH,
    "w",
    zipfile.ZIP_DEFLATED
) as zipf:

    for root, dirs, files_in_dir in os.walk(
        OUTPUT_DIR
    ):

        for filename in files_in_dir:

            filepath = os.path.join(
                root,
                filename
            )

            arcname = os.path.relpath(
                filepath,
                OUTPUT_DIR
            )

            zipf.write(
                filepath,
                arcname
            )

print(
    "ZIP archive created:",
    ZIP_PATH
)

print(
    "Archive size:",
    round(
        os.path.getsize(ZIP_PATH)
        / (1024 ** 2),
        2
    ),
    "MB"
)

# ============================================================
# CELL 37 — DOWNLOAD MASTER EXCEL
# ============================================================

from google.colab import files

files.download(
    MASTER_EXCEL
)

# ============================================================
# CELL 38 — DOWNLOAD COMPLETE ZIP
# ============================================================

from google.colab import files

files.download(
    ZIP_PATH
)

print("\n================================================")
print("CLANAK EXTENDED ANALYSIS COMPLETED")
print("================================================")

print(
    "Final dataset:",
    final_count
)

print(
    "ELM hidden neurons:",
    110
)

print(
    "LSTM neurons:",
    128
)

print(
    "Maximum LSTM epochs:",
    30
)

print(
    "Early stopping patience:",
    5
)

print(
    "Cross-validation:",
    "Stratified 5-Fold"
)

print(
    "Ensemble rule:",
    "ELM == 1 AND LSTM == 1"
)

print(
    "\nResults directory:",
    OUTPUT_DIR
)

print(
    "Master Excel:",
    MASTER_EXCEL
)

print(
    "Complete ZIP:",
    ZIP_PATH
)