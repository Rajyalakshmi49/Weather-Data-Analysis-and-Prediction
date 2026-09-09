# 🌦️ Weather Data Analysis and Temperature Prediction

A complete, beginner-friendly Machine Learning project that analyzes
historical weather data and predicts **temperature** using regression
models, wrapped in an interactive Streamlit dashboard.

Built for an engineering internship project — every file is fully
commented so you can explain each part in an interview.

---

## 📁 Folder Structure

```
weather_ml_project/
│
├── data/
│   └── weather_data.csv        # Historical weather dataset (date, temperature,
│                                # humidity, rainfall, wind_speed, pressure)
│
├── models/
│   ├── best_model.pkl          # Best trained model, saved with Joblib
│   └── model_info.pkl          # Saved metrics + metadata for the dashboard
│
├── generate_dataset.py         # Creates a realistic sample dataset (optional,
│                                # skip if you already have your own CSV)
├── weather_utils.py            # Shared data cleaning & feature engineering
│                                # functions (used by both training and the app)
├── train_model.py              # Trains, evaluates and saves the best model
├── app.py                      # Streamlit dashboard (the main application)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🧠 What This Project Does

1. Loads a historical daily weather CSV (`date, temperature, humidity,
   rainfall, wind_speed, pressure`)
2. Cleans the data — fixes the date column, converts numeric columns,
   removes duplicate rows, fixes/removes unrealistic outlier values, and
   fills missing values using the column median
3. Engineers simple, explainable features from the date: `year`, `month`,
   `day`, `day_of_year`
4. Performs exploratory data analysis with visualizations:
   - Temperature trend over time
   - Monthly and yearly average temperature
   - Humidity and rainfall seasonal patterns
   - Wind speed distribution
   - Correlation heatmap between all weather variables
5. Trains **three regression models** to predict temperature:
   - Linear Regression
   - Random Forest Regressor
   - Gradient Boosting Regressor
6. Splits the data **chronologically** (earlier dates → train, most recent
   dates → test) because this is time-series data — shuffling would leak
   future information into training
7. Evaluates every model using **MAE, MSE, RMSE and R²**
8. Automatically selects the best model (highest R²) and saves it with
   **Joblib**
9. Serves everything through a clean **Streamlit dashboard** with 5 pages:
   Home, Dataset, Data Analysis, Model Performance, and Temperature
   Prediction (where you can type in weather conditions and get a live
   prediction)

---

## ⚙️ Requirements

- Python 3.9 or higher
- pip

---

## 🚀 Installation & Usage (Windows Commands)

Open **Command Prompt (cmd)** or **PowerShell** and follow these steps
exactly.

### 1. Go to the project folder

```
cd path\to\weather_ml_project
```

### 2. (Recommended) Create and activate a virtual environment

```
python -m venv venv
venv\Scripts\activate
```

> If PowerShell blocks the activation script, run this once as
> administrator, then try again:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### 3. Install the required packages

```
pip install -r requirements.txt
```

### 4. Generate the sample dataset

> Skip this step only if you already placed your own weather CSV at
> `data\weather_data.csv` with columns:
> `date, temperature, humidity, rainfall, wind_speed, pressure`

```
python generate_dataset.py
```

### 5. Train the models

This trains all three models, evaluates them, and saves the best one.

```
python train_model.py
```

You should see output like:

```
Training Linear Regression...
Training Random Forest Regressor...
Training Gradient Boosting Regressor...
Best model: Gradient Boosting Regressor (R2 = 0.92)
Saved model to: models/best_model.pkl
```

### 6. Launch the Streamlit dashboard

```
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`. If it
doesn't, open that link manually.

---

## 🖥️ Using the Dashboard

| Page | What it shows |
|---|---|
| **Home** | Project overview and quick dataset stats |
| **Dataset** | Preview of the cleaned data, summary statistics, and a CSV download button |
| **Data Analysis** | Temperature trends, monthly/yearly averages, humidity & rainfall patterns, wind speed distribution, correlation heatmap |
| **Model Performance** | Side-by-side MAE / MSE / RMSE / R² comparison of all three models, with charts |
| **Temperature Prediction** | Enter a date, humidity, rainfall, wind speed and pressure to get a predicted temperature |

---

## 🔁 Using Your Own Dataset

Replace `data/weather_data.csv` with your own CSV as long as it has these
column names (case-sensitive):

```
date, temperature, humidity, rainfall, wind_speed, pressure
```

Then simply re-run:

```
python train_model.py
streamlit run app.py
```

The cleaning code in `weather_utils.py` will automatically handle missing
values, wrong data types, duplicate rows and out-of-range values in your
own data too.

---

## 🧩 Understanding the Code (for your interview)

- **`weather_utils.py`** — the single source of truth for data cleaning
  and feature engineering. Both the training script and the Streamlit app
  import from here, so predictions always use the exact same
  transformations that were used during training.
- **`train_model.py`** — a plain, linear script: load → clean → split
  chronologically → train 3 models → evaluate → save the best one. No
  hidden magic, every step is printed to the console.
- **`app.py`** — a multi-page Streamlit app. Each page is a simple
  `if/elif` block. Data and the model are loaded once and cached with
  `@st.cache_data` / `@st.cache_resource` for speed.
- Why **chronological split** instead of `train_test_split`? Weather data
  is a time series — using future dates to help predict the past would be
  unrealistic ("data leakage"). Splitting by date mimics real-world
  forecasting.
- Why does **Linear Regression** usually perform worse here? Temperature
  follows a seasonal, curved (non-linear) pattern across the year. A
  straight-line model struggles with that curve, while Random Forest and
  Gradient Boosting can learn non-linear seasonal patterns — a good talking
  point about model selection.

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push this project to a **GitHub repository** (see next section).
2. Go to [https://share.streamlit.io](https://share.streamlit.io) and
   sign in with your GitHub account.
3. Click **"New app"**.
4. Select your repository, branch (usually `main`), and set the main file
   path to `app.py`.
5. Click **Deploy**. Streamlit Cloud will install everything from
   `requirements.txt` automatically and build your app.
6. **Important:** make sure `data/weather_data.csv`, `models/best_model.pkl`
   and `models/model_info.pkl` are committed to the repository (i.e. not
   in `.gitignore`), otherwise the deployed app won't find them. If you'd
   rather not commit the trained model, add a `train_model.py` run step,
   or simply run `python train_model.py` locally once before pushing so
   the `.pkl` files exist and get committed.

---

## 🐙 Pushing This Project to GitHub (Windows Commands)

```
cd path\to\weather_ml_project
git init
git add .
git commit -m "Initial commit: Weather Data Analysis and Temperature Prediction"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

Replace `<your-username>` and `<your-repo-name>` with your actual GitHub
username and repository name.

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---|---|
| `'streamlit' is not recognized` | Make sure your virtual environment is activated, or run `pip install -r requirements.txt` again |
| `No trained model was found` (in the app) | Run `python train_model.py` before launching the app |
| `Could not find the dataset` | Run `python generate_dataset.py`, or place your own CSV at `data\weather_data.csv` |
| PowerShell won't activate venv | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once, as described above |
| Port already in use | Run `streamlit run app.py --server.port 8502` to use a different port |

---

## 📦 Tech Stack

- **Python** — core language
- **Pandas / NumPy** — data loading, cleaning, feature engineering
- **Matplotlib / Seaborn** — data visualizations
- **Scikit-learn** — Linear Regression, Random Forest, Gradient Boosting,
  and evaluation metrics
- **Streamlit** — interactive web dashboard
- **Joblib** — saving and loading the trained model

No deep learning frameworks, no external APIs, and no API keys are used
anywhere in this project.