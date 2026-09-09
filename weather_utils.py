import pandas as pd

# Required columns
REQUIRED_COLUMNS = [
    "date",
    "temperature",
    "humidity",
    "rainfall",
    "wind_speed",
    "pressure"
]

# Features used by the ML model
FEATURE_COLUMNS = [
    "year",
    "month",
    "day",
    "day_of_year",
    "humidity",
    "rainfall",
    "wind_speed",
    "pressure"
]

TARGET_COLUMN = "temperature"


def clean_data(df):
    """
    Clean real weather data obtained from Open-Meteo.
    """

    df = df.copy()

    # Check required columns
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Convert date
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    # Convert weather columns to numeric
    numeric_columns = [
        "temperature",
        "humidity",
        "rainfall",
        "wind_speed",
        "pressure"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Remove rows where date or temperature is missing
    df = df.dropna(
        subset=["date", "temperature"]
    )

    # Remove duplicate dates
    df = df.drop_duplicates(
        subset=["date"]
    )

    # Fill missing predictor values with median
    predictor_columns = [
        "humidity",
        "rainfall",
        "wind_speed",
        "pressure"
    ]

    for col in predictor_columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(
                df[col].median()
            )

    # Sort chronologically
    df = df.sort_values(
        "date"
    ).reset_index(drop=True)

    return df


def create_features(df):
    """
    Create calendar-based features from the date.
    """

    df = df.copy()

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_year"] = df["date"].dt.dayofyear

    return df


def prepare_dataset(data):
    """
    Load, clean, and prepare the weather dataset.
    """

    # If a file path is provided, load the CSV
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data.copy()

    df = clean_data(df)
    df = create_features(df)

    return df

def month_name_from_number(month):
    """
    Convert month number to month name.
    """

    months = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December"
    }

    return months.get(month, "Unknown")