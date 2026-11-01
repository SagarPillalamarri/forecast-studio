"""src/data.py
Dataset configurations and data loading utilities.
App reads ONLY the cleaned CSVs produced by prepare_data.py.
"""

import os
import pandas as pd
import streamlit as st

# Base directory for datasets
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

DATASET_CONFIGS = {
    "ae_monthly": {
        "id": "ae_monthly",
        "name": "NHS England Monthly A&E Attendances",
        "short_name": "NHS A&E Attendances",
        "file_path": os.path.join(DATA_DIR, "ae_monthly.csv"),
        "frequency": "MS",
        "seasonal_period": 12,
        "default_test_size": 12,
        "unit": "Attendances",
        "period_label": "Month",
        "description": "Monthly national total A&E attendances in England (Aug 2010 to Apr 2026). Exhibiting long-term upward trend, sharp COVID-19 pandemic dip, and annual winter/summer seasonality."
    },
    "electricity_daily": {
        "id": "electricity_daily",
        "name": "UK Electricity Daily Demand (NESO)",
        "short_name": "UK Electricity Demand",
        "file_path": os.path.join(DATA_DIR, "electricity_daily.csv"),
        "frequency": "D",
        "seasonal_period": 7,
        "default_test_size": 60,
        "unit": "Daily Energy (GWh)",
        "period_label": "Day of Week",
        "description": "National Grid / NESO Great Britain daily aggregate electricity demand (2021 to 2024). Exhibiting distinct weekly cycles (lower on weekends) and seasonal weather patterns."
    }
}


@st.cache_data(show_spinner=False)
def load_dataset(dataset_key: str) -> pd.DataFrame:
    """
    Loads and validates cleaned dataset CSV.
    Guarantees 'date' datetime index with fixed frequency and single 'value' column.
    """
    if dataset_key not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset key: {dataset_key}")
    
    cfg = DATASET_CONFIGS[dataset_key]
    file_path = cfg["file_path"]
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Cleaned dataset file not found at '{file_path}'. "
            "Please run 'python prepare_data.py' first."
        )
    
    df = pd.read_csv(file_path)
    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError(f"Dataset {file_path} must have exactly 'date' and 'value' columns.")
    
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().sort_values("date").reset_index(drop=True)
    
    # Set DatetimeIndex with inferred or configured frequency
    df = df.set_index("date")
    freq = cfg["frequency"]
    
    # Resample or set freq to ensure continuous time index
    if freq == "MS":
        # Ensure month start dates
        df.index = df.index.to_period("M").to_timestamp(how="start")
        df = df[~df.index.duplicated(keep="first")]
        df = df.asfreq("MS")
    elif freq == "D":
        df = df[~df.index.duplicated(keep="first")]
        df = df.asfreq("D")
    
    # Forward fill or interpolate if any single missing step
    if df["value"].isna().any():
        df["value"] = df["value"].interpolate(method="time")
        
    # CSV stores sums of half-hourly MW readings: MW × 0.5 h / 1000 = GWh.
    # Keep the source CSV unchanged and convert once at the shared loader.
    if dataset_key == "electricity_daily":
        df["value"] = df["value"] * 0.5 / 1000

    return df


def get_train_test_split(df: pd.DataFrame, test_size: int):
    """
    Splits the time series dataframe into train and test sets.
    """
    if test_size <= 0 or test_size >= len(df):
        raise ValueError(f"Invalid test size {test_size} for series of length {len(df)}")
    
    train = df.iloc[:-test_size]
    test = df.iloc[-test_size:]
    return train, test
