# 🇱🇰 LankaPriceAI – Sri Lankan Rice Price Prediction

### Rice Price Analysis and Prediction Using Ensemble Learning

LankaPriceAI is a machine learning project developed to analyse historical rice prices across Sri Lankan markets and predict rice prices using different machine learning and ensemble learning techniques.

The project combines data preprocessing, feature engineering, machine learning model development, performance evaluation, and an interactive Streamlit dashboard.

---

## 📌 Project Overview

The main objective of this project is to explore historical rice price patterns and evaluate machine learning approaches for predicting rice prices in Sri Lanka.

**Key features:**

- Historical rice price analysis across Sri Lankan markets.
- Data preprocessing and feature engineering.
- Comparison of multiple machine learning models.
- Hybrid Random Forest and ARIMA prediction approach.
- Model performance evaluation.
- Interactive Streamlit dashboard.
- Visualisation of dataset information and model comparisons.

## 📊 Dataset Information

The dataset was obtained from the World Bank Microdata Library.

| Attribute | Details |
|---|---|
| Country | Sri Lanka |
| Data source | World Bank Microdata Library |
| Original records | 10,516 |
| Original market labels | 44 |
| Market Average | Excluded from modelled markets |
| Modelled markets | 43 |
| Processed records | 9,761 |
| Target variable | `c_rice_various` |
| Data frequency | Monthly |

**Dataset:** [World Bank – Sri Lanka Retail Food Prices](https://microdata.worldbank.org/catalog/6169)

**DOI:** 10.48529/2ZH0-JF55

**Dataset licence:** CC BY 4.0

## ⚙️ Data Preprocessing and Feature Engineering

The dataset was prepared before model training.

The main preprocessing steps included:

- Removing the aggregate Market Average records from the modelled market set.
- Organising the data by market and date.
- Handling missing values in the selected model features.
- Creating historical price lag features.
- Calculating rolling mean features.
- Preparing the final dataset for model training and evaluation.

### Features Used

| Feature | Description |
|---|---|
| `mkt_name` | Market name |
| `year` | Year |
| `month` | Month |
| `lag_1` | Previous month's rice price |
| `lag_2` | Rice price from two months earlier |
| `lag_3` | Rice price from three months earlier |
| `lag_6` | Rice price from six months earlier |
| `lag_12` | Rice price from twelve months earlier |
| `rolling_mean_3` | Three-month rolling average |
| `rolling_mean_6` | Six-month rolling average |

**Target variable:** `c_rice_various`

## 🤖 Machine Learning Models

Six machine learning and ensemble approaches were evaluated.

1. Random Forest
2. XGBoost
3. LightGBM
4. CatBoost
5. Stacking Ensemble
6. Hybrid Random Forest + ARIMA

### Model Evaluation

The models were evaluated using a chronological 80/20 train-test split based on unique dates.

The following evaluation metrics were used:

- **MAE (Mean Absolute Error):** Measures the average absolute prediction error.
- **RMSE (Root Mean Squared Error):** Measures prediction error while giving greater weight to larger errors.
- **R² Score:** Measures how well the model explains the variation in the target values.

### Model Performance Comparison

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 7.4735 | 9.4045 | 0.1979 |
| XGBoost | 11.3663 | 14.4626 | -0.8970 |
| LightGBM | 10.4455 | 13.2750 | -0.5983 |
| CatBoost | 13.9544 | 17.0704 | -1.6428 |
| Stacking Ensemble | 98.2278 | 100.5332 | -90.6632 |
| **Hybrid Random Forest + ARIMA** | **5.0338** | **6.7181** | **0.5907** |

### Best Performing Model

**Hybrid Random Forest + ARIMA**

| Metric | Result |
|---|---:|
| MAE | 5.0338 |
| RMSE | 6.7181 |
| R² | 0.5907 |

The Hybrid Random Forest + ARIMA model achieved the best overall performance among the evaluated approaches based on the reported MAE, RMSE, and R² results.

The hybrid evaluation was based on 1,978 matched predictions.

*Note: These results represent the experiments conducted in this project. Performance may vary with different data, markets, and evaluation conditions.*

## 📈 Interactive Dashboard

An interactive dashboard was developed using Streamlit to present the project results in an accessible format.

### Dashboard Features

- **Overview:** Summary of the project, dataset, and best evaluated model.
- **Dataset & EDA:** Dataset information, descriptive statistics, and price distribution.
- **Model Performance:** Comparison of machine learning models using evaluation metrics.
- **Price Analysis:** Exploration of historical rice price information.
- **Generative AI Acknowledgement:** Disclosure of dashboard development assistance.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Ensemble Learning | Random Forest, XGBoost, LightGBM, CatBoost |
| Time Series Modelling | ARIMA, Statsmodels |
| Data Visualisation | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| Development Environment | Jupyter Notebook, VS Code |
| Version Control | Git, GitHub |

## 📁 Project Structure

```text
LankaPriceAI/
│
├── app.py
│
├── LankaPrice_AI_Ensemble_Learning.ipynb
│
├── requirements.txt
│
├── README.md
│
├── .gitignore
│
├── .gitattributes
│
├── .streamlit/
│   └── config.toml
│
└── lankaprice_models/
    ├── final_random_forest.pkl
    ├── final_xgboost.pkl
    ├── final_lightgbm.pkl
    ├── final_catboost.pkl
    ├── market_history.csv
    └── model_comparison.csv
```

## 🚀 Installation and Setup

Follow these steps to run the dashboard locally.

### 1. Clone the Repository

```bash
git clone https://github.com/ThishaniDissanayake/LankaPriceAI.git
```

### 2. Navigate to the Project Folder

```bash
cd LankaPriceAI
```

### 3. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your default web browser.

**Important:** The trained Random Forest model is stored using Git LFS. Make sure Git LFS is installed and the model file has been downloaded successfully before running the dashboard.

## 📓 Jupyter Notebook

The complete machine learning workflow is available in:

`LankaPrice_AI_Ensemble_Learning.ipynb`

It contains:

- Dataset loading and exploration.
- Data preprocessing.
- Feature engineering.
- Model training.
- Ensemble learning experiments.
- Hybrid model development.
- Model evaluation and comparison.

## 📚 References

1. World Bank Microdata Library – Sri Lanka Retail Food Prices Dataset.

   https://microdata.worldbank.org/catalog/6169

2. Dataset DOI: https://doi.org/10.48529/2ZH0-JF55

3. Scikit-learn Documentation: https://scikit-learn.org/

4. Streamlit Documentation: https://docs.streamlit.io/

## 👩‍💻 Author

**Thishani Dissanayake**

GitHub: [@ThishaniDissanayake](https://github.com/ThishaniDissanayake)

Project: LankaPriceAI

---

## Generative AI Acknowledgement

ChatGPT by OpenAI was used only to assist with the design and development of the Streamlit dashboard.
