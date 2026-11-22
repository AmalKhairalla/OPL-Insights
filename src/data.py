"""Load the ALREADY-FIXED train/validation split and prepare inputs without leakage."""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CATEGORICAL_FEATURES,
    NAME_COLUMN,
    NUMERIC_FEATURES,
    SPLIT_COLUMN,
    TARGET,
)


@dataclass
class PreparedData:
    X_train: np.ndarray
    X_val: np.ndarray
    y_train: np.ndarray            # Kilograms: use for metric reporting
    y_val: np.ndarray              # Kilograms
    y_train_scaled: np.ndarray     # Standardized using TRAIN only
    preprocessor: ColumnTransformer
    y_scaler: StandardScaler
    train_count: int
    val_count: int


def load_fixed_split(path_or_url: Union[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the saved fixed-split CSV(.gz). Test rows are NOT returned."""
    df = pd.read_csv(path_or_url, compression="infer")
    required = {NAME_COLUMN, SPLIT_COLUMN, TARGET, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    allowed = {"train", "validation", "test"}
    actual = set(df[SPLIT_COLUMN].dropna().unique())
    if actual != allowed or df[SPLIT_COLUMN].isna().any():
        raise ValueError(f"Expected exactly {allowed} in '{SPLIT_COLUMN}'; got {actual}")

    train_df = df.loc[df[SPLIT_COLUMN] == "train"].copy()
    val_df = df.loc[df[SPLIT_COLUMN] == "validation"].copy()
    # Test data is deliberately not exposed by this function.

    # Name grouping is not a perfect person identifier, but prevent exact normalized-name overlap.
    normalized_names = df[NAME_COLUMN].astype("string").str.strip().str.casefold()
    if normalized_names.isna().any() or normalized_names.eq("").any():
        raise ValueError("Missing or empty athlete names; cannot verify grouping.")
    memberships = df.assign(_group=normalized_names).groupby("_group")[SPLIT_COLUMN].nunique()
    if memberships.max() > 1:
        raise ValueError("Name leakage: at least one normalized name appears in multiple splits.")

    return train_df, val_df


def prepare_data(train_df: pd.DataFrame, val_df: pd.DataFrame) -> PreparedData:
    """Fit preprocessing exclusively on training records; reproduce Colab baseline rules."""
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    for frame, name in [(train_df, "training"), (val_df, "validation")]:
        if frame.empty:
            raise ValueError(f"The {name} split is empty.")
        missing = set(features + [TARGET]) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing {name} columns: {sorted(missing)}")
        if frame[TARGET].isna().any():
            raise ValueError(f"Missing target values in {name} split.")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("sex", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])
    X_train = preprocessor.fit_transform(train_df[features]).astype(np.float32)
    X_val = preprocessor.transform(val_df[features]).astype(np.float32)

    y_train = train_df[TARGET].to_numpy(dtype=np.float64)
    y_val = val_df[TARGET].to_numpy(dtype=np.float64)
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).astype(np.float32)

    return PreparedData(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        y_train_scaled=y_train_scaled,
        preprocessor=preprocessor,
        y_scaler=y_scaler,
        train_count=len(train_df),
        val_count=len(val_df),
    )
