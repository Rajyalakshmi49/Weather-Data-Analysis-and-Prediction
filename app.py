"""
app.py
-------
Streamlit dashboard for the "Weather Data Analysis and Temperature
Prediction" project.

Pages:
    1. Home                  - project overview
    2. Dataset                - view and explore the raw/cleaned data
    3. Data Analysis           - trends, seasonal patterns, correlations
    4. Model Performance       - compare MAE/MSE/RMSE/R2 for all models
    5. Temperature Prediction  - enter weather info and get a prediction

Run with:
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
from datetime import date
from weather_api import get_current_weather

from weather_utils import (
    prepare_dataset,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    month_name_from_number,
)

# ---------------------------------------------------------------------
# Page configuration (must be the first Streamlit command)
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Weather Data Analysis & Temperature Prediction",
    page_icon="🌦️",
    layout="wide",
)

# ============================================================
# PROFESSIONAL UI STYLING
# ============================================================

st.markdown("""
<style>
    /* Main application background */
    .stApp {
        background-color: #f8fafc;
    }

    /* Main headings */
    h1, h2, h3 {
        font-weight: 700;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #eef2f7;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: white;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }

    /* Info / success / warning boxes */
    div[data-testid="stAlert"] {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

sns.set_theme(style="whitegrid")

DATA_PATH = os.path.join("data", "weather_data.csv")
MODELS_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
MODEL_INFO_PATH = os.path.join(MODELS_DIR, "model_info.pkl")


# ---------------------------------------------------------------------
# Cached loaders so the app stays fast (data/model are only loaded once)
# ---------------------------------------------------------------------
@st.cache_data
def get_dataset():
    """Loads and cleans the dataset. Returns (dataframe, error_message)."""
    try:
        df = prepare_dataset(DATA_PATH)
        return df, None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def get_model_and_info():
     """Loads the saved model and model_info. Returns (model, info, error)."""
     if not os.path.exists(BEST_MODEL_PATH) or not os.path.exists(MODEL_INFO_PATH):
        return None, None, (
            "No trained model was found. Please run 'python train_model.py' "
            "first to train and save a model."
        )
     try:
        model = joblib.load(BEST_MODEL_PATH)
        info = joblib.load(MODEL_INFO_PATH)
        return model, info, None
     except Exception as e:
        return None, None, f"Could not load the saved model: {e}"

@st.cache_data(ttl=600)
def get_live_weather(city):
    """Fetch current weather from Open-Meteo."""
    try:
        weather = get_current_weather(city)
        return weather, None
    except Exception as e:
        return None, str(e)


df, data_error = get_dataset()
model, model_info, model_error = get_model_and_info()


# ---------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------
st.sidebar.title("🌦️ Navigation")
page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Current Weather",
        "Dataset",
        "Data Analysis",
        "Model Performance",
        "Temperature Prediction"
    ]
)


st.sidebar.markdown("---")
st.sidebar.caption(
    "Weather Data Analysis and Temperature Prediction\n\n"
    "Built with Python, Pandas, Scikit-learn and Streamlit."
)


# =======================================================================
# PAGE 1: HOME
# =======================================================================
if page == "Home":

    st.title("🌦️ Weather Data Analysis and Temperature Prediction")

    st.markdown(
        """
        Welcome! This dashboard is an end-to-end **Machine Learning project**
        that analyzes historical weather data and predicts **temperature**
        using regression models.
        """
    )

    # ============================================================
    # PROJECT OVERVIEW METRICS
    # ============================================================

    st.markdown("### 📌 Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📊 Weather Records",
            value="1,826",
            delta="5 Years"
        )

    with col2:
        st.metric(
            label="🌡️ Weather Features",
            value="5",
            delta="Real API Data"
        )

    with col3:
        st.metric(
            label="🤖 ML Models",
            value="3",
            delta="Regression"
        )

    with col4:
        st.metric(
            label="🏆 Best R² Score",
            value="0.884",
            delta="Gradient Boosting"
        )

    st.markdown("---")

    # ============================================================
    # WHAT THIS PROJECT DOES
    # ============================================================

    st.subheader("🔍 What this project does")

    st.markdown(
        """
        - Loads and cleans a historical daily weather dataset
        - Explores trends in temperature, humidity, rainfall and wind speed
        - Engineers simple date-based features such as year, month, day and day-of-year
        - Trains and compares three regression models:
            - Linear Regression
            - Random Forest Regressor
            - Gradient Boosting Regressor
        - Selects the best model automatically
        - Uses the trained model to estimate temperature from weather conditions and a date
        """
    )

    # ============================================================
    # DASHBOARD FEATURES
    # ============================================================

    st.subheader("🧭 Dashboard Features")

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            "📊 **Dataset**\n\n"
            "View and explore the historical weather dataset."
        )

        st.info(
            "📈 **Data Analysis**\n\n"
            "Explore trends, seasonality and relationships between weather variables."
        )

        st.info(
            "🤖 **Model Performance**\n\n"
            "Compare MAE, MSE, RMSE and R² scores for all trained models."
        )

    with col2:
        st.info(
            "🌤️ **Current Weather**\n\n"
            "Get the latest weather conditions using the Open-Meteo API."
        )

        st.info(
            "🌡️ **Temperature Prediction**\n\n"
            "Enter weather conditions and use the trained model to estimate temperature."
        )

    st.markdown("---")

    # ============================================================
    # DATASET SUMMARY
    # ============================================================

    if data_error:
        st.error(f"Dataset issue: {data_error}")
    else:
        st.subheader("📋 Dataset Summary")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Total Records",
            f"{len(df):,}"
        )

        c2.metric(
            "Date Range",
            f"{df['date'].min().date()} → {df['date'].max().date()}"
        )

        c3.metric(
            "Avg Temperature",
            f"{df['temperature'].mean():.1f} °C"
        )

        c4.metric(
            "Avg Humidity",
            f"{df['humidity'].mean():.1f} %"
        )

# =======================================================================
# PAGE 2: CURRENT WEATHER
# =======================================================================
elif page == "Current Weather":
    st.title("🌤️ Current Weather")

    st.markdown(
        """
        Get the latest weather conditions for any city using
        **real-time data from the Open-Meteo API**.
        """
    )

    st.markdown("---")

    city = st.text_input(
        "Enter City",
        value="Hyderabad",
        help="Enter the city for which you want current weather."
    )

    if st.button("Get Current Weather", type="primary"):
        if not city.strip():
            st.error("Please enter a city name.")
        else:
            with st.spinner("Fetching current weather..."):
                weather, weather_error = get_live_weather(city.strip())

            if weather_error:
                st.error(
                    f"Could not retrieve weather data: {weather_error}"
                )
            else:
                st.success(
                    f"🌍 Current weather for {weather['city']}"
                )

                st.markdown("---")

                st.subheader("📊 Current Conditions")

                c1, c2, c3, c4, c5 = st.columns(5)

                c1.metric(
                    "🌡️ Temperature",
                    f"{weather['temperature']:.1f} °C"
                )

                c2.metric(
                    "💧 Humidity",
                    f"{weather['humidity']} %"
                )

                c3.metric(
                    "🌧️ Rainfall",
                    f"{weather['rainfall']:.1f} mm"
                )

                c4.metric(
                    "💨 Wind Speed",
                    f"{weather['wind_speed']:.1f} km/h"
                )

                c5.metric(
                    "🎚️ Pressure",
                    f"{weather['pressure']:.1f} hPa"
                )

                st.markdown("---")

                st.subheader("📍 Weather Details")
                st.write(
                    f"**Location:** {weather['city']}"
                )

                st.write(
                    f"**Recorded Time:** {weather['time']}"
                )

                st.info(
                       "🌐 Weather information is retrieved in real time "
                       "from the Open-Meteo API."
                )

# =======================================================================
# PAGE 3: DATASET
# =======================================================================
elif page == "Dataset":

    st.title("📊 Weather Dataset")

    st.markdown(
        """
        This section displays the historical weather dataset used for
        analysis and machine learning. The data was collected from the
        Open-Meteo historical weather API.
        """
    )

    st.markdown("---")

    # Check whether dataset loaded successfully
    if data_error:
        st.error(f"Could not load dataset: {data_error}")
        st.stop()

    # Dataset information
    st.subheader("📋 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "📊 Total Records",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "📅 Start Date",
            df["date"].min().strftime("%d-%m-%Y")
        )

    with col3:
        st.metric(
            "📅 End Date",
            df["date"].max().strftime("%d-%m-%Y")
        )

    with col4:
        st.metric(
            "🌡️ Avg Temperature",
            f"{df['temperature'].mean():.1f} °C"
        )

    st.markdown("---")

    # Dataset Preview
    st.subheader("🔎 Dataset Preview")

    st.caption(
        "Showing the first 20 records from the cleaned historical weather dataset."
    )

    st.dataframe(
        df.head(20),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # Statistical Summary
    st.subheader("📈 Statistical Summary")

    st.caption(
        "Descriptive statistics for the numerical weather variables."
    )

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.markdown("---")

    # Download Dataset
    st.subheader("⬇️ Download Dataset")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Weather Dataset",
        data=csv,
        file_name="weather_data.csv",
        mime="text/csv"
    )

    st.info(
        "The dataset contains real historical weather observations "
        "retrieved from the Open-Meteo historical weather API."
    )

# =======================================================================
# PAGE 4: DATA ANALYSIS
# =======================================================================
elif page == "Data Analysis":

    st.title("📈 Weather Data Analysis")

    st.markdown(
        """
        Explore temperature, humidity, rainfall and wind patterns from
        the historical weather dataset. These visualizations help identify
        trends, seasonal variations and relationships between weather variables.
        """
    )

    st.markdown("---")

    # Check whether dataset loaded successfully
    if data_error:
        st.error(f"Could not load dataset: {data_error}")
        st.stop()

    # -------------------------------------------------------------------
    # KEY STATISTICS
    # -------------------------------------------------------------------
    st.subheader("📊 Key Weather Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🌡️ Average Temperature",
            f"{df['temperature'].mean():.1f} °C"
        )

    with col2:
        st.metric(
            "🔥 Maximum Temperature",
            f"{df['temperature'].max():.1f} °C"
        )

    with col3:
        st.metric(
            "❄️ Minimum Temperature",
            f"{df['temperature'].min():.1f} °C"
        )

    with col4:
        st.metric(
            "💧 Average Humidity",
            f"{df['humidity'].mean():.1f} %"
        )

    st.markdown("---")

    # -------------------------------------------------------------------
    # TEMPERATURE TREND
    # -------------------------------------------------------------------
    st.subheader("🌡️ Temperature Trend Over Time")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        df["date"],
        df["temperature"],
        linewidth=1.2
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Historical Temperature Trend")
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "The graph shows daily temperature variations across the complete "
        "historical dataset."
    )

    st.markdown("---")

    # -------------------------------------------------------------------
    # MONTHLY TEMPERATURE
    # -------------------------------------------------------------------
    st.subheader("📅 Average Temperature by Month")

    monthly_temp = (
        df.groupby(df["date"].dt.month)["temperature"]
        .mean()
        .reset_index()
    )

    monthly_temp["Month"] = monthly_temp["date"].apply(
        month_name_from_number
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(
        monthly_temp["Month"],
        monthly_temp["temperature"]
    )

    ax.set_xlabel("Month")
    ax.set_ylabel("Average Temperature (°C)")
    ax.set_title("Average Temperature by Month")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Monthly averages help identify seasonal temperature patterns."
    )

    st.markdown("---")

    # -------------------------------------------------------------------
    # HUMIDITY TREND
    # -------------------------------------------------------------------
    st.subheader("💧 Humidity Trend Over Time")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        df["date"],
        df["humidity"],
        linewidth=1.2
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Humidity (%)")
    ax.set_title("Historical Humidity Trend")
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Daily relative humidity values recorded in the historical dataset."
    )

    st.markdown("---")

    # -------------------------------------------------------------------
    # RAINFALL ANALYSIS
    # -------------------------------------------------------------------
    st.subheader("🌧️ Rainfall Analysis")

    col1, col2 = st.columns(2)

    with col1:

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.plot(
            df["date"],
            df["rainfall"],
            linewidth=1.0
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Rainfall (mm)")
        ax.set_title("Rainfall Over Time")
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close(fig)

    with col2:

        monthly_rainfall = (
            df.groupby(df["date"].dt.month)["rainfall"]
            .mean()
            .reset_index()
        )

        monthly_rainfall["Month"] = monthly_rainfall["date"].apply(
            month_name_from_number
        )

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.bar(
            monthly_rainfall["Month"],
            monthly_rainfall["rainfall"]
        )

        ax.set_xlabel("Month")
        ax.set_ylabel("Average Rainfall (mm)")
        ax.set_title("Average Rainfall by Month")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)

        st.pyplot(fig)
        plt.close(fig)

    st.caption(
        "Rainfall analysis shows both daily rainfall variation and monthly averages."
    )

    st.markdown("---")

    # -------------------------------------------------------------------
    # WIND SPEED
    # -------------------------------------------------------------------
    st.subheader("💨 Wind Speed Analysis")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        df["date"],
        df["wind_speed"],
        linewidth=1.2
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Wind Speed (km/h)")
    ax.set_title("Historical Wind Speed Trend")
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Daily maximum wind speed recorded in the historical dataset."
    )

    st.markdown("---")

    # -------------------------------------------------------------------
    # CORRELATION HEATMAP
    # -------------------------------------------------------------------
    st.subheader("🔗 Weather Feature Correlation")

    correlation_columns = [
        "temperature",
        "humidity",
        "rainfall",
        "wind_speed",
        "pressure"
    ]

    correlation = df[correlation_columns].corr()

    fig, ax = plt.subplots(figsize=(9, 6))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        ax=ax
    )

    ax.set_title("Correlation Between Weather Variables")

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Correlation values range from -1 to +1. Values closer to +1 "
        "indicate a strong positive relationship, while values closer "
        "to -1 indicate a strong negative relationship."
    )

    st.markdown("---")

    # -------------------------------------------------------------------
    # ANALYSIS SUMMARY
    # -------------------------------------------------------------------
    st.subheader("📝 Analysis Summary")

    st.info(
        """
        **Key observations from the dataset:**

        • Temperature varies significantly across different dates and months.

        • Monthly averages help identify seasonal temperature patterns.

        • Humidity, rainfall and wind speed show considerable variation over time.

        • The correlation matrix helps understand relationships between
          temperature and other weather variables.

        • These historical patterns provide useful features for the
          machine learning temperature prediction model.
        """
    )


# =======================================================================
# PAGE 5: MODEL PERFORMANCE
# =======================================================================
elif page == "Model Performance":

    st.title("🤖 Machine Learning Model Performance")

    st.markdown(
        """
        This section compares the performance of the machine learning
        models trained on historical weather data. The models are evaluated
        using MAE, MSE, RMSE and R² Score.
        """
    )

    st.markdown("---")

    # Check whether model loaded successfully
    if model_error:
        st.error(model_error)
        st.stop()

    # -------------------------------------------------------------------
    # BEST MODEL
    # -------------------------------------------------------------------
    best_model_name = model_info["best_model_name"]

    st.success(
        f"🏆 Best Model Selected: **{best_model_name}**"
    )

    st.markdown(
        """
        The best model is selected based on the highest **R² Score**,
        while MAE and RMSE are used to understand prediction error.
        """
    )

    st.markdown("---")

    # -------------------------------------------------------------------
    # MODEL COMPARISON
    # -------------------------------------------------------------------
    st.subheader("📊 Comparison of All Trained Models")

    results = model_info["all_results"]

    results_df = pd.DataFrame(results).T

    results_df = results_df.rename(
        columns={
            "MAE": "MAE (°C)",
            "MSE": "MSE",
            "RMSE": "RMSE (°C)",
            "R2": "R² Score"
        }
    )

    # Round numerical values
    results_df = results_df.round(4)

    st.dataframe(
        results_df,
        use_container_width=True
    )

    st.caption(
        "Lower MAE, MSE and RMSE indicate lower prediction error, "
        "while a higher R² Score indicates better model performance."
    )

    st.markdown("---")

    # -------------------------------------------------------------------
    # PERFORMANCE METRICS
    # -------------------------------------------------------------------
    st.subheader("🎯 Best Model Metrics")

    best_result = model_info["all_results"][best_model_name]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "📉 MAE",
            f"{best_result['MAE']:.3f} °C"
        )

    with col2:
        st.metric(
            "📊 MSE",
            f"{best_result['MSE']:.3f}"
        )

    with col3:
        st.metric(
            "📉 RMSE",
            f"{best_result['RMSE']:.3f} °C"
        )

    with col4:
        st.metric(
            "🏆 R² Score",
            f"{best_result['R2']:.3f}"
        )

    st.markdown("---")

    # -------------------------------------------------------------------
    # VISUAL COMPARISON
    # -------------------------------------------------------------------
    st.subheader("📈 Visual Model Comparison")

    col1, col2 = st.columns(2)

    model_names = results_df.index.tolist()

    with col1:

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.bar(
            model_names,
            results_df["RMSE (°C)"]
        )

        ax.set_title("RMSE Comparison")
        ax.set_xlabel("Model")
        ax.set_ylabel("RMSE (°C)")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)

        st.pyplot(fig)
        plt.close(fig)

    with col2:

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.bar(
            model_names,
            results_df["R² Score"]
        )

        ax.set_title("R² Score Comparison")
        ax.set_xlabel("Model")
        ax.set_ylabel("R² Score")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)

        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    # -------------------------------------------------------------------
    # ACTUAL VS PREDICTED
    # -------------------------------------------------------------------
    st.subheader("🌡️ Actual vs Predicted Temperature")

    st.caption(
        "Comparison of actual and model-estimated temperatures on the "
        "chronological test dataset."
    )

    try:

        test_df = df.copy()

        split_index = int(len(test_df) * 0.8)

        test_data = test_df.iloc[split_index:].copy()

        X_test = test_data[FEATURE_COLUMNS]
        y_test = test_data[TARGET_COLUMN]

        y_pred = model.predict(X_test)

        comparison_df = pd.DataFrame(
            {
                "Date": test_data["date"].values,
                "Actual Temperature": y_test.values,
                "Predicted Temperature": y_pred
            }
        )

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(
            comparison_df["Date"],
            comparison_df["Actual Temperature"],
            label="Actual Temperature",
            linewidth=1.5
        )

        ax.plot(
            comparison_df["Date"],
            comparison_df["Predicted Temperature"],
            label="Predicted Temperature",
            linewidth=1.5
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Temperature (°C)")
        ax.set_title("Actual vs Predicted Temperature")
        ax.legend()
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:

        st.error(
            f"Unable to generate Actual vs Predicted graph: {e}"
        )

    st.markdown("---")

    # -------------------------------------------------------------------
    # TRAINING DETAILS
    # -------------------------------------------------------------------
    st.subheader("📚 Training Details")

    split_index = int(len(df) * 0.8)

    train_data = df.iloc[:split_index]
    test_data = df.iloc[split_index:]

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            f"""
            **Training Dataset**

            • Records: **{len(train_data):,}**

            • Start Date: **{train_data['date'].min().strftime('%d-%m-%Y')}**

            • End Date: **{train_data['date'].max().strftime('%d-%m-%Y')}**
            """
        )

    with col2:

        st.info(
            f"""
            **Testing Dataset**

            • Records: **{len(test_data):,}**

            • Start Date: **{test_data['date'].min().strftime('%d-%m-%Y')}**

            • End Date: **{test_data['date'].max().strftime('%d-%m-%Y')}**
            """
        )

    st.markdown("---")

    st.subheader("💡 Evaluation Method")

    st.info(
        """
        The dataset was divided chronologically into **80% training data**
        and **20% testing data**. Earlier weather records were used for
        training, while later records were used for testing.

        **MAE:** Average absolute prediction error.

        **MSE:** Average squared prediction error.

        **RMSE:** Square root of MSE, expressed in the same unit as temperature.

        **R² Score:** Measures how well the model explains variation in
        temperature values. A higher value indicates better performance.
        """
    )

    st.caption(
        "Note: The R² score represents model performance on the test data; "
        "it should not be interpreted as a percentage of forecasting accuracy."
    )


# =======================================================================
# PAGE 6: TEMPERATURE PREDICTION
# =======================================================================
elif page == "Temperature Prediction":
    st.title("🌡️ Temperature Prediction")

    st.markdown(
        """
        Use the trained machine learning model to estimate temperature
        based on the selected date and weather conditions.
        """
    )

    st.markdown("---")

    if model_error:
        st.error(model_error)
        st.stop()

    st.subheader("📝 Enter Weather Conditions")

    st.caption(
        "Provide the weather conditions below. The trained Gradient Boosting "
        "model will use these values to estimate temperature."
    )

    # Input section
    col1, col2 = st.columns(2)

    with col1:
        input_date = st.date_input(
            "📅 Date",
            value=date.today(),
            min_value=date(2000, 1, 1),
            max_value=date(2100, 12, 31)
        )

        humidity = st.slider(
            "💧 Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            step=1.0
        )

        rainfall = st.number_input(
            "🌧️ Rainfall (mm)",
            min_value=0.0,
            value=0.0,
            step=0.1
        )

    with col2:
        wind_speed = st.number_input(
            "💨 Wind Speed (km/h)",
            min_value=0.0,
            value=10.0,
            step=0.1
        )

        pressure = st.number_input(
            "🎚️ Pressure (hPa)",
            min_value=800.0,
            max_value=1100.0,
            value=950.0,
            step=0.1
        )

    st.markdown("---")

    # Prediction button
    predict_clicked = st.button(
        "🤖 Predict Temperature",
        type="primary",
        use_container_width=True
    )

    if predict_clicked:

        input_row = {
            "year": input_date.year,
            "month": input_date.month,
            "day": input_date.day,
            "day_of_year": input_date.timetuple().tm_yday,
            "humidity": float(humidity),
            "rainfall": float(rainfall),
            "wind_speed": float(wind_speed),
            "pressure": float(pressure),
        }

        input_df = pd.DataFrame([input_row])[FEATURE_COLUMNS]

        try:
            prediction = model.predict(input_df)[0]

            st.markdown("---")
            st.subheader("🎯 Prediction Result")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "🌡️ Estimated Temperature",
                    f"{prediction:.2f} °C"
                )

            with col2:
                st.metric(
                    "💧 Humidity",
                    f"{humidity:.0f} %"
                )

            with col3:
                st.metric(
                    "🌧️ Rainfall",
                    f"{rainfall:.1f} mm"
                )

            st.success(
                f"🌡️ Estimated temperature for **{input_date.strftime('%d-%m-%Y')}** "
                f"is **{prediction:.2f} °C**."
            )

            st.markdown("---")

            st.subheader("📋 Input Conditions")

            result_df = pd.DataFrame({
                "Parameter": [
                    "Date",
                    "Humidity",
                    "Rainfall",
                    "Wind Speed",
                    "Pressure"
                ],
                "Value": [
                    input_date.strftime("%d-%m-%Y"),
                    f"{humidity:.1f} %",
                    f"{rainfall:.1f} mm",
                    f"{wind_speed:.1f} km/h",
                    f"{pressure:.1f} hPa"
                ]
            })

            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True
            )

        except Exception as e:
            st.error(
                f"Unable to generate the temperature prediction: {e}"
            )

    st.markdown("---")

    st.subheader("ℹ️ About This Prediction")

    st.info(
        """
        The prediction is generated using the trained **Gradient Boosting
        Regression** model.

        The model was trained using real historical weather data retrieved
        from the **Open-Meteo API**.

        **Important:** This is a machine learning estimate based on the
        supplied weather conditions. It should not be treated as an official
        weather forecast.
        """
    )

    st.caption(
        "Model inputs: date, humidity, rainfall, wind speed and pressure."
    )