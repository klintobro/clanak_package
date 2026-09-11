# ============================================================
# CORRECTED THESIS ELM-LSTM PIPELINE
# ============================================================
#
# PURPOSE:
# Leakage-controlled ELM-LSTM ensemble for MitM detection
# under TLS, meteorological and diurnal conditions.
#
# INPUT FILES:
#
# NORMAL:
#   1. sunrise_normal.csv
#   2. sunset_normal.csv
#
# ATTACK:
#   3. sunrise_attack.csv
#   4. sunset_attack.csv
#
# DATA CONFIGURATIONS:
#
# D1 = TLS Only
# D2 = TLS + Meteorological
# D3 = TLS + Meteorological + Time
#
# PIPELINE:
#
# Four CSV files
#       ↓
# Combine Normal Sunrise + Sunset
# Combine Attack Sunrise + Sunset
#       ↓
# Combine Normal + Attack
#       ↓
# Basic Cleaning
#       ↓
# Cyclic Time Encoding
#       ↓
# Stratified 5-Fold CV
#       ↓
# TRAIN / TEST
#       ↓
# Imputer fitted on TRAIN only
#       ↓
# MinMaxScaler fitted on TRAIN only
#       ↓
# SMOTE on TRAIN only
#       ↓
# Information Gain on TRAIN only → ELM
#       ↓
# PCA fitted on TRAIN only → LSTM
#       ↓
# ELM + LSTM
#       ↓
# Unanimous Ensemble
#       ↓
# TEST METRICS
#       ↓
# CSV + Excel + Confusion Matrices + ZIP
#
# ============================================================


# ============================================================
# 1. INSTALL REQUIRED LIBRARIES
# ============================================================

!pip install -q scikit-learn tensorflow imbalanced-learn matplotlib seaborn openpyxl


# ============================================================
# 2. IMPORT LIBRARIES
# ============================================================

import os
import random
import zipfile
import warnings
import shutil

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

from IPython.display import display


warnings.filterwarnings("ignore")


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

SEED = 42

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 90)
print("REPRODUCIBILITY")
print("=" * 90)
print("Random seed:", SEED)


# ============================================================
# 4. OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = "/content/corrected_thesis_results"

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\nOutput directory:")
print(OUTPUT_DIR)


# ============================================================
# 5. STANDARDIZE COLUMN NAMES
# ============================================================

def standardize_column_names(df):

    df = df.copy()

    rename_map = {}

    for col in df.columns:

        original = str(col).strip()
        lower = original.lower()

        # ----------------------------------------------------
        # TIME
        # ----------------------------------------------------

        if lower in [
            "time",
            "timestamp",
            "date_time",
            "datetime",
            "date time"
        ]:

            rename_map[col] = "Time"

        # ----------------------------------------------------
        # HANDSHAKE TIME
        # ----------------------------------------------------

        elif lower in [
            "handshake time",
            "handshake_time",
            "handshake-time",
            "tls handshake time",
            "tls_handshake_time",
            "tls handshake",
            "tls_handshake"
        ]:

            rename_map[col] = "handshake_time"

        # ----------------------------------------------------
        # TEMPERATURE
        # ----------------------------------------------------

        elif lower in [
            "temperature",
            "temp"
        ]:

            rename_map[col] = "temperature"

        # ----------------------------------------------------
        # HUMIDITY
        # ----------------------------------------------------

        elif lower in [
            "humidity",
            "relative humidity",
            "relative_humidity"
        ]:

            rename_map[col] = "humidity"

        # ----------------------------------------------------
        # PRESSURE
        # ----------------------------------------------------

        elif lower in [
            "pressure",
            "atmospheric pressure",
            "atmospheric_pressure"
        ]:

            rename_map[col] = "pressure"

        # ----------------------------------------------------
        # WIND SPEED
        # ----------------------------------------------------

        elif lower in [
            "wind speed",
            "wind_speed",
            "windspeed"
        ]:

            rename_map[col] = "wind_speed"

        # ----------------------------------------------------
        # WIND DIRECTION
        # ----------------------------------------------------

        elif lower in [
            "wind direction",
            "wind_direction",
            "winddirection"
        ]:

            rename_map[col] = "wind_direction"

        # ----------------------------------------------------
        # WIND GUST
        # ----------------------------------------------------

        elif lower in [
            "wind gust",
            "wind_gust",
            "windgust"
        ]:

            rename_map[col] = "wind_gust"

        # ----------------------------------------------------
        # RAIN
        # ----------------------------------------------------

        elif lower in [
            "rain",
            "rainfall",
            "precipitation"
        ]:

            rename_map[col] = "rain"

    df = df.rename(columns=rename_map)

    return df


# ============================================================
# 6. READ ONE CSV FILE
# ============================================================

def read_csv_file(filename):

    df = pd.read_csv(filename)

    df = standardize_column_names(df)

    return df


# ============================================================
# 7. UPLOAD FOUR DATA FILES
# ============================================================

def load_data():

    print("\n")
    print("=" * 90)
    print("STEP 1: UPLOAD ALL FOUR CSV FILES")
    print("=" * 90)

    print("""
Please select ALL FOUR files at once:

    sunrise_normal.csv
    sunset_normal.csv
    sunrise_attack.csv
    sunset_attack.csv

You can select multiple files in the upload window.
""")

    uploaded = files.upload()

    if len(uploaded) < 4:

        raise ValueError(
            f"\nExpected 4 CSV files, but only {len(uploaded)} "
            f"file(s) were uploaded."
        )

    filenames = list(uploaded.keys())

    print("\nUploaded files:")

    for filename in filenames:
        print(" -", filename)


    # ========================================================
    # IDENTIFY FILES AUTOMATICALLY
    # ========================================================

    def find_file(keyword1, keyword2):

        matches = [
            f for f in filenames
            if keyword1 in f.lower()
            and keyword2 in f.lower()
        ]

        if len(matches) == 1:
            return matches[0]

        if len(matches) == 0:
            return None

        raise ValueError(
            f"Multiple files matched {keyword1} + {keyword2}: "
            f"{matches}"
        )


    sunrise_normal_file = find_file(
        "sunrise",
        "normal"
    )

    sunset_normal_file = find_file(
        "sunset",
        "normal"
    )

    sunrise_attack_file = find_file(
        "sunrise",
        "attack"
    )

    sunset_attack_file = find_file(
        "sunset",
        "attack"
    )


    # ========================================================
    # CHECK FILES
    # ========================================================

    required_files = {

        "sunrise_normal":
            sunrise_normal_file,

        "sunset_normal":
            sunset_normal_file,

        "sunrise_attack":
            sunrise_attack_file,

        "sunset_attack":
            sunset_attack_file

    }


    missing = [
        key
        for key, value in required_files.items()
        if value is None
    ]


    if missing:

        raise ValueError(
            "\nCould not identify the following required files:\n"
            + "\n".join(
                f" - {x}"
                for x in missing
            )
            +
            "\n\nExpected filenames similar to:\n"
            "sunrise_normal.csv\n"
            "sunset_normal.csv\n"
            "sunrise_attack.csv\n"
            "sunset_attack.csv"
        )


    print("\n")
    print("=" * 90)
    print("FILES IDENTIFIED")
    print("=" * 90)

    print(
        "Sunrise Normal :",
        sunrise_normal_file
    )

    print(
        "Sunset Normal  :",
        sunset_normal_file
    )

    print(
        "Sunrise Attack :",
        sunrise_attack_file
    )

    print(
        "Sunset Attack  :",
        sunset_attack_file
    )


    # ========================================================
    # READ FILES
    # ========================================================

    sunrise_normal = read_csv_file(
        sunrise_normal_file
    )

    sunset_normal = read_csv_file(
        sunset_normal_file
    )

    sunrise_attack = read_csv_file(
        sunrise_attack_file
    )

    sunset_attack = read_csv_file(
        sunset_attack_file
    )


    # ========================================================
    # ADD SOURCE INFORMATION
    # ========================================================

    sunrise_normal["source_period"] = "Sunrise"
    sunset_normal["source_period"] = "Sunset"

    sunrise_attack["source_period"] = "Sunrise"
    sunset_attack["source_period"] = "Sunset"


    sunrise_normal["source_condition"] = "Normal"
    sunset_normal["source_condition"] = "Normal"

    sunrise_attack["source_condition"] = "Attack"
    sunset_attack["source_condition"] = "Attack"


    # ========================================================
    # ASSIGN LABELS
    # ========================================================

    sunrise_normal["label"] = 0
    sunset_normal["label"] = 0

    sunrise_attack["label"] = 1
    sunset_attack["label"] = 1


    # ========================================================
    # DISPLAY INDIVIDUAL FILE SIZES
    # ========================================================

    print("\n")
    print("=" * 90)
    print("INDIVIDUAL DATASET SIZES")
    print("=" * 90)

    print(
        "Sunrise Normal:",
        len(sunrise_normal)
    )

    print(
        "Sunset Normal:",
        len(sunset_normal)
    )

    print(
        "Sunrise Attack:",
        len(sunrise_attack)
    )

    print(
        "Sunset Attack:",
        len(sunset_attack)
    )


    # ========================================================
    # COMBINE NORMAL FILES
    # ========================================================

    normal_df = pd.concat(

        [
            sunrise_normal,
            sunset_normal
        ],

        ignore_index=True

    )


    # ========================================================
    # COMBINE ATTACK FILES
    # ========================================================

    attack_df = pd.concat(

        [
            sunrise_attack,
            sunset_attack
        ],

        ignore_index=True

    )


    # ========================================================
    # COMBINE ALL DATA
    # ========================================================

    df = pd.concat(

        [
            normal_df,
            attack_df
        ],

        ignore_index=True

    )


    # ========================================================
    # DATA QUALITY CLEANING
    # ========================================================
    #
    # IMPORTANT:
    # 1. Remove structurally blank/padded observations.
    # 2. Audit duplicates BEFORE removing them.
    # 3. Remove exact duplicate observations within the same
    #    class using the actual observation fields + label.
    #
    # The final number of observations is CALCULATED from the
    # uploaded files. It is NOT hard-coded to any target size.
    # ========================================================

    observation_columns = [
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "wind_direction",
        "wind_gust",
        "rain",
        "handshake_time",
        "Time"
    ]

    observation_columns = [
        c for c in observation_columns
        if c in df.columns
    ]

    if "label" not in df.columns:
        raise KeyError(
            "Label column is required before data cleaning."
        )

    # --------------------------------------------------------
    # CONVERT NUMERIC COLUMNS BEFORE CLEANING
    # --------------------------------------------------------

    possible_numeric = [
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "wind_direction",
        "wind_gust",
        "rain",
        "handshake_time"
    ]

    for col in possible_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # Normalize blank Time values to missing
    if "Time" in df.columns:
        df["Time"] = df["Time"].replace(
            r"^\\s*$",
            np.nan,
            regex=True
        )

    cleaning_start_rows = len(df)

    # --------------------------------------------------------
    # 1. IDENTIFY STRUCTURALLY BLANK / PADDED ROWS
    # --------------------------------------------------------

    blank_mask = df[observation_columns].isna().all(axis=1)

    blank_rows = df.loc[blank_mask].copy()

    df = df.loc[~blank_mask].copy()

    blank_rows_removed = len(blank_rows)

    if blank_rows_removed > 0:
        blank_rows.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "removed_blank_padded_rows.csv"
            ),
            index=False
        )

    # --------------------------------------------------------
    # 2. AUDIT EXACT DUPLICATES BEFORE REMOVAL
    # --------------------------------------------------------

    duplicate_subset = [
        c for c in observation_columns
        if c in df.columns
    ]

    duplicate_subset_with_label = duplicate_subset + ["label"]

    duplicate_mask = df.duplicated(
        subset=duplicate_subset_with_label,
        keep=False
    )

    duplicate_rows_before_removal = df.loc[
        duplicate_mask
    ].copy()

    exact_duplicate_rows = int(
        df.duplicated(
            subset=duplicate_subset_with_label,
            keep="first"
        ).sum()
    )

    # Save all duplicate groups before removing them
    if len(duplicate_rows_before_removal) > 0:
        duplicate_rows_before_removal.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "duplicate_rows_before_removal.csv"
            ),
            index=False
        )

    # --------------------------------------------------------
    # 3. REMOVE EXACT DUPLICATES WITHIN SAME CLASS
    # --------------------------------------------------------
    #
    # Label is deliberately included. Therefore, an observation
    # is only removed as an exact duplicate when its observation
    # values AND class label are identical.
    # --------------------------------------------------------

    df = df.drop_duplicates(
        subset=duplicate_subset_with_label,
        keep="first"
    ).copy()

    duplicates_removed = exact_duplicate_rows

    # --------------------------------------------------------
    # 4. CLEANING AUDIT
    # --------------------------------------------------------

    cleaning_final_rows = len(df)

    cleaning_audit = pd.DataFrame({
        "Metric": [
            "Raw combined rows",
            "Structurally blank/padded rows removed",
            "Exact duplicate rows removed",
            "Final clean rows",
            "Final normal observations",
            "Final attack observations"
        ],
        "Value": [
            cleaning_start_rows,
            blank_rows_removed,
            duplicates_removed,
            cleaning_final_rows,
            int((df["label"] == 0).sum()),
            int((df["label"] == 1).sum())
        ]
    })

    cleaning_audit.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "data_cleaning_audit.csv"
        ),
        index=False
    )

    print("\n")
    print("=" * 90)
    print("DATA QUALITY CLEANING AUDIT")
    print("=" * 90)

    print(
        "Raw combined rows:",
        cleaning_start_rows
    )

    print(
        "Structurally blank/padded rows removed:",
        blank_rows_removed
    )

    print(
        "Exact duplicate rows removed:",
        duplicates_removed
    )

    print(
        "Final clean rows:",
        cleaning_final_rows
    )

    print("\nFinal class distribution:")
    print(
        df["label"].value_counts().sort_index()
    )

    print("\nFinal period × condition distribution:")
    print(
        pd.crosstab(
            df["source_period"],
            df["source_condition"]
        )
    )


    # ========================================================
    # DATA SUMMARY
    # ========================================================
    # ========================================================
    # DATA SUMMARY
    # ========================================================

    print("\n")
    print("=" * 90)
    print("COMBINED DATASET")
    print("=" * 90)

    print(
        "Rows before duplicate removal:",
        cleaning_start_rows - blank_rows_removed
    )

    print(
        "Duplicates removed:",
        duplicates_removed
    )

    print(
        "Final rows:",
        len(df)
    )

    print(
        "\nColumns:"
    )

    print(
        df.columns.tolist()
    )


    print("\nClass distribution:")

    print(
        df["label"].value_counts().sort_index()
    )


    print("\nPeriod distribution:")

    print(
        df["source_period"].value_counts()
    )


    print("\nCondition distribution:")

    print(
        df["source_condition"].value_counts()
    )


    print("\nMissing values:")

    print(
        df.isnull().sum()
    )


    return df


# ============================================================
# 8. CYCLIC TIME ENCODING
# ============================================================

def encode_cyclic_time(
    df,
    time_col="Time"
):

    df = df.copy()


    if time_col not in df.columns:

        raise KeyError(

            f"\nTime column '{time_col}' was not found.\n\n"

            f"Available columns:\n"
            f"{df.columns.tolist()}\n\n"

            "Please check the time column in your CSV files."

        )


    # ========================================================
    # PARSE TIME
    # ========================================================

    parsed_time = pd.to_datetime(

        df[time_col],

        errors="coerce"

    )


    # ========================================================
    # EXTRACT TIME COMPONENTS
    # ========================================================

    hours = parsed_time.dt.hour.fillna(0)

    minutes = parsed_time.dt.minute.fillna(0)

    seconds = parsed_time.dt.second.fillna(0)


    # ========================================================
    # TOTAL SECONDS
    # ========================================================

    total_seconds = (

        hours * 3600
        +
        minutes * 60
        +
        seconds

    )


    seconds_in_day = 86400


    # ========================================================
    # CYCLIC ENCODING
    # ========================================================

    df["time_sin"] = np.sin(

        2 * np.pi *
        total_seconds /
        seconds_in_day

    )


    df["time_cos"] = np.cos(

        2 * np.pi *
        total_seconds /
        seconds_in_day

    )


    # ========================================================
    # REMOVE ORIGINAL TIME
    # ========================================================

    df = df.drop(

        columns=[time_col]

    )


    return df


# ============================================================
# 9. EXTREME LEARNING MACHINE
# ============================================================

class ELM:

    def __init__(
        self,
        input_size,
        hidden_size=100,
        random_state=42
    ):

        rng = np.random.RandomState(
            random_state
        )


        self.input_weights = rng.randn(

            input_size,
            hidden_size

        )


        self.bias = rng.randn(

            hidden_size

        )


        self.output_weights = None


    # ========================================================
    # SIGMOID
    # ========================================================

    def sigmoid(self, x):

        x = np.clip(

            x,

            -500,
            500

        )


        return 1.0 / (

            1.0 +
            np.exp(-x)

        )


    # ========================================================
    # FIT
    # ========================================================

    def fit(
        self,
        X,
        y
    ):

        H = self.sigmoid(

            np.dot(
                X,
                self.input_weights
            )
            +
            self.bias

        )


        self.output_weights = np.dot(

            np.linalg.pinv(H),

            np.asarray(y)

        )


        return self


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        X
    ):

        H = self.sigmoid(

            np.dot(
                X,
                self.input_weights
            )
            +
            self.bias

        )


        y_pred = np.dot(

            H,
            self.output_weights

        )


        return (

            y_pred >= 0.5

        ).astype(int)


# ============================================================
# 10. METRICS
# ============================================================

def get_metrics(
    y_true,
    y_pred
):

    cm = confusion_matrix(

        y_true,
        y_pred,

        labels=[0, 1]

    )


    tn, fp, fn, tp = cm.ravel()


    accuracy = accuracy_score(

        y_true,
        y_pred

    )


    precision = precision_score(

        y_true,
        y_pred,

        average="weighted",

        zero_division=0

    )


    recall = recall_score(

        y_true,
        y_pred,

        average="weighted",

        zero_division=0

    )


    f1 = f1_score(

        y_true,
        y_pred,

        average="weighted",

        zero_division=0

    )


    specificity = (

        tn / (tn + fp)

        if (tn + fp) > 0

        else 0.0

    )


    fpr = (

        fp / (fp + tn)

        if (fp + tn) > 0

        else 0.0

    )


    return {

        "Accuracy":
            float(accuracy),

        "Precision":
            float(precision),

        "Recall":
            float(recall),

        "F1 Score":
            float(f1),

        "Specificity":
            float(specificity),

        "FPR":
            float(fpr),

        "TN":
            int(tn),

        "FP":
            int(fp),

        "FN":
            int(fn),

        "TP":
            int(tp)

    }


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(

    y_true,
    y_pred,
    title,
    filename

):

    cm = confusion_matrix(

        y_true,
        y_pred,

        labels=[0, 1]

    )


    plt.figure(

        figsize=(6, 5)

    )


    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues",

        xticklabels=[
            "Normal",
            "Attack"
        ],

        yticklabels=[
            "Normal",
            "Attack"
        ]

    )


    plt.title(title)

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "Actual Class"
    )


    plt.tight_layout()


    plt.savefig(

        filename,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()


# ============================================================
# 12. IDENTIFY FEATURES
# ============================================================

def identify_features(df):


    # ========================================================
    # TLS FEATURES
    # ========================================================

    tls_features = [

        col

        for col in df.columns

        if (

            "tls" in col.lower()

            or

            "handshake" in col.lower()

        )

    ]


    # ========================================================
    # METEOROLOGICAL FEATURES
    # ========================================================

    meteo_names = [

        "temperature",

        "humidity",

        "pressure",

        "wind_speed",

        "wind_direction",

        "wind_gust",

        "rain"

    ]


    meteo_features = [

        col

        for col in df.columns

        if col.lower() in meteo_names

    ]


    # ========================================================
    # TIME FEATURES
    # ========================================================

    time_features = [

        "time_sin",

        "time_cos"

    ]


    # ========================================================
    # DATA CONFIGURATIONS
    # ========================================================

    datasets = {

        "D1_TLS_Only":

            tls_features,


        "D2_TLS_Meteorological":

            list(

                dict.fromkeys(

                    tls_features
                    +
                    meteo_features

                )

            ),


        "D3_TLS_Meteorological_Time":

            list(

                dict.fromkeys(

                    tls_features
                    +
                    meteo_features
                    +
                    time_features

                )

            )

    }


    print("\n")
    print("=" * 90)
    print("FEATURE CONFIGURATIONS")
    print("=" * 90)


    for name, feature_list in datasets.items():

        print("\n", name)

        print("-" * 70)

        print(
            "Number of features:",
            len(feature_list)
        )

        print(
            "Features:",
            feature_list
        )


    return datasets


# ============================================================
# 13. AVERAGE METRICS
# ============================================================

def average_metrics(metric_list):

    metric_names = [

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score",

        "Specificity",

        "FPR"

    ]


    averaged = {}


    for metric in metric_names:

        values = [

            m[metric]

            for m in metric_list

        ]


        averaged[metric] = float(

            np.mean(values)

        )


    return averaged


# ============================================================
# 14. MAIN TRAINING AND EVALUATION
# ============================================================

def train_and_evaluate_all():


    # ========================================================
    # LOAD DATA
    # ========================================================

    df = load_data()


    # ========================================================
    # SAVE COMBINED CLEANED DATA
    # ========================================================

    cleaned_raw_path = os.path.join(

        OUTPUT_DIR,

        "combined_cleaned_dataset.csv"

    )


    df.to_csv(

        cleaned_raw_path,

        index=False

    )


    # ========================================================
    # TIME ENCODING
    # ========================================================

    df = encode_cyclic_time(

        df,

        "Time"

    )


    # ========================================================
    # CHECK LABEL
    # ========================================================

    if "label" not in df.columns:

        raise KeyError(
            "Label column was not created."
        )


    y = df["label"].astype(int)


    # ========================================================
    # IDENTIFY FEATURES
    # ========================================================

    datasets = identify_features(
        df
    )


    # ========================================================
    # RESULT CONTAINERS
    # ========================================================

    results_summary = {}

    all_fold_results = []

    selected_features_results = []

    pca_results = []


    # ========================================================
    # LOOP THROUGH DATA CONFIGURATIONS
    # ========================================================

    for dataset_name, features in datasets.items():


        print("\n\n")

        print("=" * 100)

        print(
            f"DATASET CONFIGURATION: {dataset_name}"
        )

        print("=" * 100)


        # ====================================================
        # VERIFY FEATURES
        # ====================================================

        missing_features = [

            f

            for f in features

            if f not in df.columns

        ]


        if missing_features:

            print(
                "\nWARNING: Missing features:",
                missing_features
            )

            print(
                "Configuration skipped."
            )

            continue


        if len(features) == 0:

            print(
                "\nWARNING: No features found."
            )

            continue


        # ====================================================
        # FEATURE MATRIX
        # ====================================================

        X = df[features].copy()


        X = X.apply(

            pd.to_numeric,

            errors="coerce"

        )


        # ====================================================
        # STRATIFIED 5-FOLD CV
        # ====================================================

        skf = StratifiedKFold(

            n_splits=5,

            shuffle=True,

            random_state=SEED

        )


        metrics_elm_list = []

        metrics_lstm_list = []

        metrics_ensemble_list = []


        # ====================================================
        # FOLD LOOP
        # ====================================================

        for fold_number, (train_idx, test_idx) in enumerate(

            skf.split(X, y),

            start=1

        ):


            print("\n")

            print("-" * 90)

            print(
                f"{dataset_name} - FOLD {fold_number}"
            )

            print("-" * 90)


            # =================================================
            # TRAIN / TEST
            # =================================================

            X_train = X.iloc[
                train_idx
            ].copy()


            X_test = X.iloc[
                test_idx
            ].copy()


            y_train = y.iloc[
                train_idx
            ].copy()


            y_test = y.iloc[
                test_idx
            ].copy()


            print(
                "Training samples:",
                len(X_train)
            )

            print(
                "Testing samples:",
                len(X_test)
            )


            # =================================================
            # IMPUTATION
            # TRAIN ONLY
            # =================================================

            imputer = SimpleImputer(

                strategy="mean"

            )


            X_train_imputed = imputer.fit_transform(

                X_train

            )


            X_test_imputed = imputer.transform(

                X_test

            )


            # =================================================
            # MINMAX SCALING
            # TRAIN ONLY
            # =================================================

            scaler = MinMaxScaler()


            X_train_scaled = scaler.fit_transform(

                X_train_imputed

            )


            X_test_scaled = scaler.transform(

                X_test_imputed

            )


            # =================================================
            # SMOTE
            # TRAIN ONLY
            # =================================================

            smote = SMOTE(

                random_state=SEED,

                k_neighbors=5

            )


            X_train_balanced, y_train_balanced = (

                smote.fit_resample(

                    X_train_scaled,

                    y_train

                )

            )


            print("\nClass distribution after SMOTE:")

            unique_classes, class_counts = np.unique(

                y_train_balanced,

                return_counts=True

            )


            for class_value, count in zip(

                unique_classes,

                class_counts

            ):

                print(

                    f"Class {class_value}: {count}"

                )


            # =================================================
            # INFORMATION GAIN
            # TRAINING DATA ONLY
            # =================================================

            ig_scores = mutual_info_classif(

                X_train_balanced,

                y_train_balanced,

                random_state=SEED

            )


            # =================================================
            # SELECT TOP FEATURES
            # =================================================

            top_k = min(

                6,

                len(ig_scores)

            )


            top_k_idx = np.argsort(

                ig_scores

            )[-top_k:]


            top_k_idx = top_k_idx[

                np.argsort(

                    ig_scores[top_k_idx]

                )[::-1]

            ]


            selected_features = [

                features[i]

                for i in top_k_idx

            ]


            print("\nELM selected features:")

            print(
                selected_features
            )


            # =================================================
            # SAVE INFORMATION GAIN
            # =================================================

            for feature, score in zip(

                features,

                ig_scores

            ):

                selected_features_results.append({

                    "Dataset":
                        dataset_name,

                    "Fold":
                        fold_number,

                    "Feature":
                        feature,

                    "Information_Gain":
                        float(score),

                    "Selected":
                        feature in selected_features

                })


            # =================================================
            # ELM
            # =================================================

            elm = ELM(

                input_size=top_k,

                hidden_size=100,

                random_state=SEED + fold_number

            )


            elm.fit(

                X_train_balanced[:, top_k_idx],

                y_train_balanced

            )


            y_pred_elm = elm.predict(

                X_test_scaled[:, top_k_idx]

            )


            # =================================================
            # PCA FOR LSTM
            # TRAINING DATA ONLY
            # =================================================

            n_comp = min(

                6,

                X_train_balanced.shape[1]

            )


            pca = PCA(

                n_components=n_comp,

                random_state=SEED

            )


            X_train_pca = pca.fit_transform(

                X_train_balanced

            )


            X_test_pca = pca.transform(

                X_test_scaled

            )


            # =================================================
            # PCA EXPLAINED VARIANCE
            # =================================================

            explained_variance = (

                pca.explained_variance_ratio_

            )


            cumulative_variance = np.cumsum(

                explained_variance

            )


            for component_number in range(

                len(explained_variance)

            ):

                pca_results.append({

                    "Dataset":
                        dataset_name,

                    "Fold":
                        fold_number,

                    "Component":
                        component_number + 1,

                    "Explained_Variance":
                        float(
                            explained_variance[
                                component_number
                            ]
                        ),

                    "Cumulative_Variance":
                        float(
                            cumulative_variance[
                                component_number
                            ]
                        )

                })


            print(

                "\nPCA components:",
                n_comp

            )


            print(

                "Cumulative explained variance:",
                round(
                    float(
                        cumulative_variance[-1]
                    ),
                    6
                )

            )


            # =================================================
            # RESHAPE FOR LSTM
            # =================================================

            X_train_lstm = X_train_pca.reshape(

                X_train_pca.shape[0],

                1,

                X_train_pca.shape[1]

            )


            X_test_lstm = X_test_pca.reshape(

                X_test_pca.shape[0],

                1,

                X_test_pca.shape[1]

            )


            # =================================================
            # CLEAR TENSORFLOW SESSION
            # =================================================

            tf.keras.backend.clear_session()


            tf.random.set_seed(

                SEED + fold_number

            )


            # =================================================
            # LSTM MODEL
            # =================================================

            lstm_model = Sequential([

                Input(

                    shape=(

                        1,

                        n_comp

                    )

                ),

                LSTM(

                    128

                ),

                Dense(

                    1,

                    activation="sigmoid"

                )

            ])


            lstm_model.compile(

                optimizer="adam",

                loss="binary_crossentropy",

                metrics=[

                    "accuracy"

                ]

            )


            # =================================================
            # EARLY STOPPING
            #
            # Monitor training loss because no validation
            # set is being used.
            # =================================================

            early_stopping = EarlyStopping(

                monitor="loss",

                patience=5,

                restore_best_weights=True

            )


            # =================================================
            # TRAIN LSTM
            # =================================================

            lstm_model.fit(

                X_train_lstm,

                y_train_balanced,

                epochs=30,

                batch_size=64,

                verbose=0,

                callbacks=[

                    early_stopping

                ]

            )


            # =================================================
            # LSTM PREDICTION
            # =================================================

            y_probs_lstm = lstm_model.predict(

                X_test_lstm,

                verbose=0

            )


            y_pred_lstm = (

                y_probs_lstm > 0.5

            ).astype(int).flatten()


            # =================================================
            # UNANIMOUS ENSEMBLE
            #
            # Attack ONLY when:
            #
            # ELM = 1
            # AND
            # LSTM = 1
            #
            # =================================================

            y_pred_ensemble = np.where(

                (

                    (y_pred_elm == 1)

                    &

                    (y_pred_lstm == 1)

                ),

                1,

                0

            )


            # =================================================
            # METRICS
            # =================================================

            m_elm = get_metrics(

                y_test,

                y_pred_elm

            )


            m_lstm = get_metrics(

                y_test,

                y_pred_lstm

            )


            m_ensemble = get_metrics(

                y_test,

                y_pred_ensemble

            )


            metrics_elm_list.append(
                m_elm
            )

            metrics_lstm_list.append(
                m_lstm
            )

            metrics_ensemble_list.append(
                m_ensemble
            )


            # =================================================
            # STORE FOLD RESULTS
            # =================================================

            model_results = [

                (
                    "ELM",
                    m_elm
                ),

                (
                    "LSTM",
                    m_lstm
                ),

                (
                    "Ensemble",
                    m_ensemble
                )

            ]


            for model_name, metrics in model_results:

                row = {

                    "Dataset":
                        dataset_name,

                    "Fold":
                        fold_number,

                    "Model":
                        model_name

                }


                row.update(metrics)


                all_fold_results.append(
                    row
                )


            # =================================================
            # FILE NAME
            # =================================================

            base_name = (

                dataset_name

                .replace(
                    " ",
                    "_"
                )

                .replace(
                    "+",
                    "plus"
                )

            )


            # =================================================
            # CONFUSION MATRICES
            # =================================================

            save_confusion_matrix(

                y_test,

                y_pred_elm,

                (
                    f"{dataset_name} "
                    f"- Fold {fold_number} "
                    f"- ELM"
                ),

                os.path.join(

                    OUTPUT_DIR,

                    (
                        f"{base_name}_"
                        f"Fold_{fold_number}_"
                        f"ELM.png"
                    )

                )

            )


            save_confusion_matrix(

                y_test,

                y_pred_lstm,

                (
                    f"{dataset_name} "
                    f"- Fold {fold_number} "
                    f"- LSTM"
                ),

                os.path.join(

                    OUTPUT_DIR,

                    (
                        f"{base_name}_"
                        f"Fold_{fold_number}_"
                        f"LSTM.png"
                    )

                )

            )


            save_confusion_matrix(

                y_test,

                y_pred_ensemble,

                (
                    f"{dataset_name} "
                    f"- Fold {fold_number} "
                    f"- Ensemble"
                ),

                os.path.join(

                    OUTPUT_DIR,

                    (
                        f"{base_name}_"
                        f"Fold_{fold_number}_"
                        f"Ensemble.png"
                    )

                )

            )


            # =================================================
            # PRINT FOLD RESULTS
            # =================================================

            print("\nELM")

            print(m_elm)


            print("\nLSTM")

            print(m_lstm)


            print("\nUnanimous Ensemble")

            print(m_ensemble)


        # ====================================================
        # SAVE AVERAGED RESULTS FOR CONFIGURATION
        # ====================================================

        results_summary[dataset_name] = {

            "ELM":

                average_metrics(
                    metrics_elm_list
                ),

            "LSTM":

                average_metrics(
                    metrics_lstm_list
                ),

            "Ensemble":

                average_metrics(
                    metrics_ensemble_list
                )

        }


    # ========================================================
    # AVERAGED RESULTS DATAFRAME
    # ========================================================

    averaged_rows = []


    for dataset_name, model_results in (

        results_summary.items()

    ):

        for model_name, metrics in (

            model_results.items()

        ):

            row = {

                "Dataset":
                    dataset_name,

                "Model":
                    model_name

            }


            row.update(metrics)


            averaged_rows.append(
                row
            )


    averaged_df = pd.DataFrame(
        averaged_rows
    )


    # ========================================================
    # FOLD RESULTS DATAFRAME
    # ========================================================

    fold_df = pd.DataFrame(
        all_fold_results
    )


    # ========================================================
    # INFORMATION GAIN DATAFRAME
    # ========================================================

    ig_df = pd.DataFrame(
        selected_features_results
    )


    # ========================================================
    # PCA DATAFRAME
    # ========================================================

    pca_df = pd.DataFrame(
        pca_results
    )


    # ========================================================
    # SAVE CSV FILES
    # ========================================================

    averaged_path = os.path.join(

        OUTPUT_DIR,

        "corrected_averaged_results.csv"

    )

    averaged_df.to_csv(

        averaged_path,

        index=False

    )


    fold_path = os.path.join(

        OUTPUT_DIR,

        "corrected_fold_results.csv"

    )

    fold_df.to_csv(

        fold_path,

        index=False

    )


    ig_path = os.path.join(

        OUTPUT_DIR,

        "information_gain_results.csv"

    )

    ig_df.to_csv(

        ig_path,

        index=False

    )


    pca_path = os.path.join(

        OUTPUT_DIR,

        "pca_results.csv"

    )

    pca_df.to_csv(

        pca_path,

        index=False

    )


    # ========================================================
    # DISPLAY FINAL RESULTS
    # ========================================================

    print("\n\n")

    print("=" * 110)

    print(
        "CORRECTED FIVE-FOLD CROSS-VALIDATION RESULTS"
    )

    print("=" * 110)


    display(

        averaged_df.round(6)

    )


    return (

        results_summary,

        all_fold_results,

        averaged_df,

        fold_df,

        ig_df,

        pca_df

    )


# ============================================================
# 15. RUN COMPLETE ANALYSIS
# ============================================================

(

    results_summary,

    all_fold_results,

    averaged_df,

    fold_df,

    ig_df,

    pca_df

) = train_and_evaluate_all()


# ============================================================
# 16. PUBLICATION-READY RESULTS
# ============================================================

paper_summary = averaged_df.copy()


percentage_columns = [

    "Accuracy",

    "Precision",

    "Recall",

    "F1 Score",

    "Specificity",

    "FPR"

]


for col in percentage_columns:

    paper_summary[col] = (

        paper_summary[col] * 100

    ).round(4)


print("\n\n")

print("=" * 110)

print(
    "PUBLICATION-READY RESULTS (%)"
)

print("=" * 110)


display(
    paper_summary
)


# ============================================================
# SAVE PUBLICATION SUMMARY
# ============================================================

paper_summary_path = os.path.join(

    OUTPUT_DIR,

    "publication_summary_percent.csv"

)


paper_summary.to_csv(

    paper_summary_path,

    index=False

)


# ============================================================
# 17. MASTER EXCEL WORKBOOK
# ============================================================

excel_path = os.path.join(

    OUTPUT_DIR,

    "Corrected_Thesis_Analysis_Results.xlsx"

)


with pd.ExcelWriter(

    excel_path,

    engine="openpyxl"

) as writer:


    averaged_df.to_excel(

        writer,

        sheet_name="Averaged_Results",

        index=False

    )


    fold_df.to_excel(

        writer,

        sheet_name="Fold_Results",

        index=False

    )


    ig_df.to_excel(

        writer,

        sheet_name="Information_Gain",

        index=False

    )


    pca_df.to_excel(

        writer,

        sheet_name="PCA",

        index=False

    )


    paper_summary.to_excel(

        writer,

        sheet_name="Publication_Summary",

        index=False

    )


print("\n")

print("=" * 110)

print(
    "MASTER EXCEL WORKBOOK CREATED"
)

print("=" * 110)

print(
    excel_path
)


# ============================================================
# 18. CREATE ZIP PACKAGE
# ============================================================

zip_path = (

    "/content/"

    "Corrected_Thesis_Analysis_Results.zip"

)


with zipfile.ZipFile(

    zip_path,

    "w",

    zipfile.ZIP_DEFLATED

) as zipf:


    for root, dirs, filenames in os.walk(

        OUTPUT_DIR

    ):

        for filename in filenames:

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


print("\n")

print("=" * 110)

print(
    "RESULTS PACKAGE CREATED"
)

print("=" * 110)

print(
    zip_path
)


# ============================================================
# 19. DISPLAY GENERATED FILES
# ============================================================

print("\n")

print("=" * 110)

print(
    "FILES GENERATED"
)

print("=" * 110)


for root, dirs, filenames in os.walk(

    OUTPUT_DIR

):

    for filename in sorted(filenames):

        print(

            os.path.join(
                root,
                filename
            )

        )


# ============================================================
# 20. DOWNLOAD ZIP
# ============================================================

print("\n")

print("=" * 110)

print(
    "DOWNLOADING RESULTS ZIP..."
)

print("=" * 110)


files.download(
    zip_path
)


# ============================================================
# END OF CORRECTED THESIS PIPELINE
# ============================================================