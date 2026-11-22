
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

# =============================================================================
# LankaPrice | Sri Lanka Rice Price Intelligence
# =============================================================================

st.set_page_config(
    page_title="LankaPrice | Rice Market Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODELS_DIR = "lankaprice_models"

HERO_IMAGE = (
    "https://images.unsplash.com/photo-1500382017468-9049fed747ef"
    "?auto=format&fit=crop&w=2200&q=85"
)

# =============================================================================
# Notebook evaluation results
# =============================================================================

NOTEBOOK_METRICS = pd.DataFrame(
    [
        ["Random Forest", 7.4735, 9.4045, 0.1979],
        ["XGBoost", 11.3663, 14.4626, -0.8970],
        ["LightGBM", 10.4455, 13.2750, -0.5983],
        ["CatBoost", 13.9544, 17.0704, -1.6428],
        ["Stacking", 98.2278, 100.5332, -90.6632],
        ["Hybrid RF + ARIMA", 5.0338, 6.7181, 0.5907],
    ],
    columns=["Model", "MAE", "RMSE", "R2"],
)

FEATURES = [
    "mkt_name",
    "year",
    "month",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "rolling_mean_3",
    "rolling_mean_6",
]

MODEL_FILES = {
    "Random Forest": "final_random_forest.pkl",
    "XGBoost": "final_xgboost.pkl",
    "LightGBM": "final_lightgbm.pkl",
    "CatBoost": "final_catboost.pkl",
}

# Display full name for the best model card
MODEL_DISPLAY_NAMES = {
    "Hybrid RF + ARIMA": "Hybrid Random Forest + ARIMA",
    "Hybrid Random Forest + ARIMA": "Hybrid Random Forest + ARIMA",
}

# =============================================================================
# Styling
# =============================================================================

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {{
        --ink: #17251f;
        --muted: #6b7b72;
        --green: #1f5b43;
        --green-2: #2f7657;
        --cream: #f5f6f1;
        --line: #e4e9e2;
        --card: #ffffff;
    }}

    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif;
        color: var(--ink);
    }}

    .stApp {{
        background:
          linear-gradient(
            180deg,
            rgba(246,248,243,.96) 0%,
            rgba(246,248,243,.99) 34%
          ),
          url('{HERO_IMAGE}') center top / cover fixed;
    }}

    [data-testid="stHeader"] {{
        background: transparent;
    }}

    [data-testid="stSidebar"] {{
        display: none;
    }}

    .block-container {{
        max-width: 1500px;
        padding-top: 1.1rem;
        padding-bottom: 3rem;
    }}

    h1, h2, h3, h4 {{
        font-family: 'Manrope', sans-serif !important;
        letter-spacing: -0.035em;
        color: #17251f;
    }}

    .brandline {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 0 0 1.1rem 0;
    }}

    .brandmark {{
        width: 42px;
        height: 42px;
        border-radius: 13px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #1f5b43;
        color: white;
        font-size: 22px;
        box-shadow: 0 7px 20px rgba(31,91,67,.18);
    }}

    .brandname {{
        font: 800 1.12rem 'Manrope';
        letter-spacing: -.04em;
    }}

    .branddesc {{
        font-size: .72rem;
        color: #728078;
        letter-spacing: .12em;
        text-transform: uppercase;
    }}

    .hero {{
        position: relative;
        overflow: hidden;
        border-radius: 24px;
        min-height: 260px;
        padding: 36px 42px;
        margin: 8px 0 24px;
        color: white;
        background:
          linear-gradient(
            90deg,
            rgba(10,39,27,.91) 0%,
            rgba(16,57,39,.76) 48%,
            rgba(16,57,39,.18) 100%
          ),
          url('{HERO_IMAGE}') center 48% / cover;
        box-shadow: 0 18px 45px rgba(22,54,38,.13);
    }}

    .hero-kicker {{
        color: #d5e8d8;
        font-size: .73rem;
        font-weight: 700;
        letter-spacing: .18em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }}

    .hero h1 {{
        color: #fff !important;
        font-size: clamp(2rem,4vw,3.1rem);
        line-height: 1.08;
        max-width: 760px;
        margin: 0 0 12px;
    }}

    .hero p {{
        color: #e0ebe3;
        max-width: 680px;
        font-size: 1rem;
        line-height: 1.65;
        margin: 0;
    }}

    .hero-foot {{
        margin-top: 22px;
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }}

    .hero-pill {{
        border: 1px solid rgba(255,255,255,.28);
        background: rgba(255,255,255,.10);
        border-radius: 99px;
        padding: 7px 12px;
        font-size: .75rem;
        color: #f4faf5;
    }}

    .section-kicker {{
        color: #668072;
        text-transform: uppercase;
        letter-spacing: .15em;
        font-size: .68rem;
        font-weight: 700;
        margin-bottom: 5px;
    }}

    .section-title {{
        font: 800 1.55rem 'Manrope';
        margin: 0 0 5px;
        color: #17251f;
    }}

    .section-sub {{
        color: #75837b;
        font-size: .9rem;
        margin: 0 0 18px;
    }}

    div[data-testid="stMetric"] {{
        background: rgba(255,255,255,.92);
        border: 1px solid var(--line);
        border-radius: 17px;
        padding: 17px 18px;
        box-shadow: 0 5px 18px rgba(30,55,39,.035);
    }}

    div[data-testid="stMetricLabel"] p {{
        color: #75837b !important;
        font-size: .78rem;
        font-weight: 600;
    }}

    div[data-testid="stMetricValue"] {{
        color: #1d3f2e !important;
        font-family: 'Manrope';
    }}

    div[data-testid="stMetricDelta"] {{
        font-size: .75rem;
    }}

    div[data-testid="stTabs"] button {{
        font-weight: 600;
        border-radius: 10px 10px 0 0;
    }}

    .info-card {{
        background: white;
        border: 1px solid #e5ebe4;
        border-radius: 17px;
        padding: 18px 20px;
        height: 100%;
    }}

    .info-label {{
        font-size: .72rem;
        color: #7a887f;
        text-transform: uppercase;
        letter-spacing: .1em;
        font-weight: 700;
    }}

    .info-value {{
        font: 800 1.45rem 'Manrope';
        color: #214b36;
        margin-top: 6px;
    }}

    .info-note {{
        font-size: .8rem;
        color: #718078;
        margin-top: 5px;
        line-height: 1.5;
    }}

    .note {{
        background: #edf4ed;
        border-left: 4px solid #4f8967;
        border-radius: 9px;
        padding: 12px 15px;
        color: #345441;
        font-size: .86rem;
        margin: 10px 0 16px;
    }}

    .warning {{
        background: #fff6e7;
        border-left: 4px solid #d49a3b;
        border-radius: 9px;
        padding: 12px 15px;
        color: #74551f;
        font-size: .86rem;
        margin: 10px 0 16px;
    }}

    .footer {{
        border-top: 1px solid #e1e8df;
        margin-top: 42px;
        padding-top: 16px;
        color: #819087;
        font-size: .76rem;
    }}

    .stButton button,
    .stDownloadButton button {{
        border-radius: 10px;
        font-weight: 600;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Data loaders
# =============================================================================

@st.cache_data
def load_history():
    path = os.path.join(MODELS_DIR, "market_history.csv")

    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    df["price_date"] = pd.to_datetime(df["price_date"])

    return df.sort_values(
        ["mkt_name", "price_date"]
    ).reset_index(drop=True)


@st.cache_data
def load_metrics():
    path = os.path.join(MODELS_DIR, "model_comparison.csv")

    if os.path.exists(path):
        df = pd.read_csv(path)
        return df.sort_values("MAE").reset_index(drop=True)

    return NOTEBOOK_METRICS.copy().sort_values(
        "MAE"
    ).reset_index(drop=True)


@st.cache_resource
def load_models():
    models = {}

    for name, filename in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, filename)

        if os.path.exists(path):
            models[name] = joblib.load(path)

    return models


# =============================================================================
# Forecast function
# =============================================================================

def forecast_future_prices(
    market_df,
    market_name,
    horizon,
    rf_model,
    extra_models
):
    prices = market_df["c_rice_various"].astype(float).tolist()
    last_date = market_df["price_date"].max()

    if len(prices) < 12:
        raise ValueError(
            "At least 12 monthly observations are required."
        )

    try:
        arima_fit = ARIMA(
            prices,
            order=(1, 1, 1)
        ).fit()

        arima_values = list(
            arima_fit.forecast(steps=horizon)
        )

    except Exception:
        arima_values = [prices[-1]] * horizon

    rf_hist = prices.copy()
    hybrid_hist = prices.copy()

    extra_hist = {
        name: prices.copy()
        for name in extra_models
    }

    rows = []

    def make_features(series, date):
        return pd.DataFrame(
            [{
                "mkt_name": market_name,
                "year": date.year,
                "month": date.month,
                "lag_1": series[-1],
                "lag_2": series[-2],
                "lag_3": series[-3],
                "lag_6": series[-6],
                "lag_12": series[-12],
                "rolling_mean_3": float(
                    np.mean(series[-3:])
                ),
                "rolling_mean_6": float(
                    np.mean(series[-6:])
                ),
            }],
            columns=FEATURES
        )

    for step in range(horizon):

        date = last_date + pd.DateOffset(
            months=step + 1
        )

        arima_pred = float(arima_values[step])

        row = {
            "price_date": date,
            "ARIMA Component": arima_pred
        }

        if rf_model is not None:

            rf_pred = float(
                rf_model.predict(
                    make_features(rf_hist, date)
                )[0]
            )

            rf_hist.append(rf_pred)

            rf_on_hybrid = float(
                rf_model.predict(
                    make_features(hybrid_hist, date)
                )[0]
            )

            hybrid_pred = (
                0.5 * rf_on_hybrid
                + 0.5 * arima_pred
            )

            hybrid_hist.append(hybrid_pred)

            row["Random Forest"] = rf_pred
            row["Hybrid RF + ARIMA"] = hybrid_pred

        for name, model in extra_models.items():

            if name == "Random Forest":
                continue

            pred = float(
                model.predict(
                    make_features(extra_hist[name], date)
                )[0]
            )

            extra_hist[name].append(pred)
            row[name] = pred

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Helper functions
# =============================================================================

def section_heading(kicker, title, subtitle=""):
    st.markdown(
        f'<div class="section-kicker">{kicker}</div>'
        f'<div class="section-title">{title}</div>'
        f'<div class="section-sub">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def card(label, value, note):
    st.markdown(
        f'<div class="info-card">'
        f'<div class="info-label">{label}</div>'
        f'<div class="info-value">{value}</div>'
        f'<div class="info-note">{note}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# Load dashboard data
# =============================================================================

history_df = load_history()
metrics_df = load_metrics()
saved_models = load_models()

if history_df is None:

    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-kicker">
            Sri Lanka • Food commodity intelligence
          </div>
          <h1>LankaPrice</h1>
          <p>
            Rice price trends, model evaluation and market-level
            forecasts in one clear view.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.error(
        "The dashboard data files were not found. "
        "Run the final save/export cells in "
        "LankaPrice_AI_Ensemble_Learning.ipynb first. "
        "The app expects the lankaprice_models folder "
        "and its exported files."
    )

    st.stop()


all_markets = sorted(
    history_df["mkt_name"].dropna().unique().tolist()
)

latest_date = history_df["price_date"].max()

best_model = metrics_df.sort_values(
    "MAE"
).iloc[0]

best_name = str(best_model["Model"])

best_name_display = MODEL_DISPLAY_NAMES.get(
    best_name,
    best_name
)


# =============================================================================
# Header
# =============================================================================

st.markdown(
    """
    <div class="brandline">
      <div class="brandmark">🌾</div>
      <div>
        <div class="brandname">LankaPrice</div>
        <div class="branddesc">
          Sri Lanka rice market intelligence
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# Header navigation - no sidebar and no Explainability tab.

pages = [
    "Overview",
    "Dataset & EDA",
    "Model Performance",
    "Price Forecast",
    "Market Explorer",
]

selected_page = st.radio(
    "Main navigation",
    pages,
    horizontal=True,
    label_visibility="collapsed",
    key="main_navigation",
)


# =============================================================================
# Hero section
# =============================================================================

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">
        Sri Lanka · World Bank RTFP dataset · 2007–2026
      </div>

      <h1>
        Understanding rice prices.<br>
        Planning with better insight.
      </h1>

      <p>
        A market intelligence dashboard built from historical
        rice price observations, machine learning comparisons
        and a hybrid forecasting approach.
      </p>

      <div class="hero-foot">
        <span class="hero-pill">43 modelled markets</span>
        <span class="hero-pill">6 evaluated approaches</span>
        <span class="hero-pill">Monthly price history</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Market selector
# =============================================================================

head_left, head_right = st.columns([1.2, 2.8])

with head_left:

    default_market = (
        "Colombo City"
        if "Colombo City" in all_markets
        else all_markets[0]
    )

    selected_market = st.selectbox(
        "Focus market",
        all_markets,
        index=all_markets.index(default_market)
    )


with head_right:
    st.markdown(
        f"""
        <div style="
            text-align:right;
            padding-top:30px;
            color:#77857c;
            font-size:.82rem;">
            Data through <b>{latest_date:%B %Y}</b>
            &nbsp;·&nbsp;
            Prices shown in Sri Lankan Rupees (LKR)
        </div>
        """,
        unsafe_allow_html=True
    )


market_df = history_df[
    history_df["mkt_name"] == selected_market
].sort_values("price_date")

latest_price = float(
    market_df["c_rice_various"].iloc[-1]
)

previous_price = (
    float(market_df["c_rice_various"].iloc[-2])
    if len(market_df) > 1
    else latest_price
)

mom_delta = latest_price - previous_price

mom_pct = (
    mom_delta / previous_price * 100
    if previous_price
    else 0
)


# =============================================================================
# OVERVIEW
# =============================================================================

if selected_page == "Overview":

    section_heading(
        "Dashboard overview",
        "Rice price intelligence at a glance",
        "A high-level view of the dataset, current market position "
        "and the notebook's evaluation results."
    )

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        f"Latest {selected_market} price",
        f"Rs. {latest_price:,.2f}",
        f"{mom_delta:+.2f} LKR · {mom_pct:+.1f}% vs previous month",
        delta_color="inverse"
    )

    k2.metric(
        "Markets covered",
        f"{history_df['mkt_name'].nunique():,}",
        "Market Average excluded from modelled market set"
    )

    k3.metric(
        "Monthly records",
        f"{len(history_df):,}",
        "Feature-ready market history"
    )

    # Full model name with smaller font size.
        # Full model name with smaller font size.
    with k4:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-label">Best evaluated model</div>
                <div style="font-size:0.94rem; font-weight:800;">
                    {best_name_display}
                </div>
                <div class="info-note">
                    MAE {best_model['MAE']:.2f}
                    · R² {best_model['R2']:.3f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("")

    left, right = st.columns([1.55, 1])

    with left:

        section_heading(
            "Market movement",
            f"{selected_market} · historical rice prices",
            "Monthly price history from the exported model-ready dataset."
        )

        fig = px.area(
            market_df,
            x="price_date",
            y="c_rice_various",
            labels={
                "price_date": "Date",
                "c_rice_various": "Price (LKR/kg)"
            },
        )

        fig.update_traces(
            line_color="#2f7657",
            fillcolor="rgba(47,118,87,.13)"
        )

        fig.update_layout(
            height=360,
            margin=dict(l=8, r=8, t=15, b=8),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with right:

        section_heading(
            "Model snapshot",
            "How the approaches compare",
            "Lower MAE indicates smaller average absolute prediction error."
        )

        preview = metrics_df.sort_values("MAE").copy()

        fig = px.bar(
            preview,
            x="MAE",
            y="Model",
            orientation="h",
            color="MAE",
            color_continuous_scale=[
                "#3c8a65",
                "#d9a34a"
            ],
            labels={
                "MAE": "MAE (LKR)",
                "Model": ""
            }
        )

        fig.update_layout(
            height=360,
            margin=dict(l=8, r=8, t=15, b=8),
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    st.markdown("")

    section_heading(
        "Project at a glance",
        "What this project contains",
        "The dashboard follows the workflow implemented "
        "in the supplied Jupyter notebook."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        card(
            "Dataset source",
            "World Bank RTFP",
            "Sri Lanka monthly food-price data. "
            "Notebook reference: LKA_2021_RTFP_v02_M."
        )

    with c2:
        card(
            "Prediction target",
            "c_rice_various",
            "The notebook uses this column as the target rice price series."
        )

    with c3:
        card(
            "Forecast approach",
            "RF + ARIMA",
            "The hybrid combines Random Forest and ARIMA forecasts "
            "with equal 50% weights."
        )


    st.markdown(
        """
        <div class="note">
            <b>Reading the results:</b>
            The notebook reports the Hybrid RF + ARIMA approach
            as the strongest evaluated model
            (MAE 5.0338, RMSE 6.7181, R² 0.5907).
            Stacking has a much larger error in the recorded evaluation
            and should remain visible as a comparison, not be presented
            as the best model.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# DATASET & EDA
# =============================================================================

elif selected_page == "Dataset & EDA":

    section_heading(
        "Data profile",
        "Dataset overview & exploratory analysis",
        "Explore the model-ready monthly history, market coverage, "
        "price distribution and data preparation."
    )

    raw1, raw2, raw3, raw4 = st.columns(4)

    raw1.metric(
        "Original dataset rows",
        "10,516",
        "Notebook dataset before feature filtering"
    )

    raw2.metric(
        "Original columns",
        "46",
        "Loaded from the World Bank CSV"
    )

    raw3.metric(
        "Model-ready rows",
        f"{len(history_df):,}",
        "After lag and rolling-window feature preparation"
    )

    raw4.metric(
        "Modelled markets",
        f"{history_df['mkt_name'].nunique()}",
        "Market Average excluded"
    )


    # Price distribution - Records by Market chart removed.

    st.markdown("")

    section_heading(
        "Price distribution",
        "Rice price range",
        "Distribution of c_rice_various across all exported market records."
    )

    a, b = st.columns([1.4, 1])

    with a:

        fig = px.histogram(
            history_df,
            x="c_rice_various",
            nbins=40,
            labels={
                "c_rice_various": "Rice price (LKR/kg)"
            },
            color_discrete_sequence=["#438362"]
        )

        fig.update_layout(
            height=350,
            margin=dict(l=5, r=5, t=10, b=5),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with b:

        stats = history_df["c_rice_various"].describe()

        st.dataframe(
            pd.DataFrame({
                "Statistic": [
                    "Count",
                    "Mean",
                    "Std. deviation",
                    "Minimum",
                    "Median",
                    "Maximum"
                ],
                "Value": [
                    f"{stats['count']:,.0f}",
                    f"{stats['mean']:.2f}",
                    f"{stats['std']:.2f}",
                    f"{stats['min']:.2f}",
                    f"{stats['50%']:.2f}",
                    f"{stats['max']:.2f}"
                ],
            }),
            hide_index=True,
            use_container_width=True
        )


    # Feature engineering

    st.markdown("")

    section_heading(
        "Feature engineering",
        "What goes into the models",
        "The notebook creates historical lag variables and "
        "rolling averages separately for each market."
    )

    feature_info = pd.DataFrame(
        [
            [
                "Market and calendar",
                "mkt_name, year, month",
                "Market identity and target month"
            ],
            [
                "Lag features",
                "lag_1, lag_2, lag_3",
                "Prices from 1, 2 and 3 months earlier"
            ],
            [
                "Longer lags",
                "lag_6, lag_12",
                "Prices from 6 and 12 months earlier"
            ],
            [
                "Rolling averages",
                "rolling_mean_3, rolling_mean_6",
                "Previous 3- and 6-month mean prices"
            ],
            [
                "Target",
                "c_rice_various",
                "Rice price to be predicted"
            ],
        ],
        columns=[
            "Feature group",
            "Columns",
            "Purpose"
        ]
    )

    st.dataframe(
        feature_info,
        hide_index=True,
        use_container_width=True
    )


    st.markdown(
        """
        <div class="note">
            <b>Notebook data preparation:</b>
            The original source contains 10,516 rows,
            44 market labels and 46 columns.
            The notebook removes the aggregate “Market Average”,
            creates lag and rolling features, and drops rows
            without the required feature history.
            The resulting model dataset contains 9,761 rows
            across 43 markets, with 227 records per market
            and no missing values in the selected model features.
        </div>
        """,
        unsafe_allow_html=True,
    )


    # Recent records

    section_heading(
        "Explore records",
        f"Recent records · {selected_market}",
        "Inspect the latest observations in the exported historical dataset."
    )

    recent = market_df[
        [
            "price_date",
            "mkt_name",
            "c_rice_various"
        ]
    ].sort_values(
        "price_date",
        ascending=False
    ).head(30).rename(
        columns={
            "price_date": "Date",
            "mkt_name": "Market",
            "c_rice_various": "Rice price (LKR/kg)"
        }
    )

    st.dataframe(
        recent,
        hide_index=True,
        use_container_width=True
    )


# =============================================================================
# MODEL PERFORMANCE
# =============================================================================

elif selected_page == "Model Performance":

    section_heading(
        "Evaluation",
        "Six forecasting approaches. One consistent comparison.",
        "The notebook evaluates models using MAE, RMSE and R² "
        "on its chronological test split."
    )

    sorted_metrics = metrics_df.sort_values(
        "MAE"
    ).reset_index(drop=True)

    top = sorted_metrics.iloc[0]

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Lowest MAE",
        f"{top['MAE']:.4f} LKR",
        str(top["Model"])
    )

    m2.metric(
        "Lowest RMSE",
        f"{sorted_metrics['RMSE'].min():.4f} LKR",
        str(
            sorted_metrics.loc[
                sorted_metrics["RMSE"].idxmin(),
                "Model"
            ]
        )
    )

    m3.metric(
        "Highest R²",
        f"{sorted_metrics['R2'].max():.4f}",
        str(
            sorted_metrics.loc[
                sorted_metrics["R2"].idxmax(),
                "Model"
            ]
        )
    )


    st.markdown("")

    p1, p2 = st.columns(2)

    with p1:

        section_heading(
            "Error metrics",
            "MAE and RMSE",
            "Both metrics are in LKR; lower values indicate better predictions."
        )

        melt = sorted_metrics.melt(
            id_vars="Model",
            value_vars=["MAE", "RMSE"],
            var_name="Metric",
            value_name="Error (LKR)"
        )

        fig = px.bar(
            melt,
            x="Model",
            y="Error (LKR)",
            color="Metric",
            barmode="group",
            text_auto=".2f",
            color_discrete_map={
                "MAE": "#2f7657",
                "RMSE": "#d6a04b"
            }
        )

        fig.update_layout(
            height=410,
            xaxis_tickangle=-22,
            margin=dict(l=5, r=5, t=15, b=5),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with p2:

        section_heading(
            "Goodness of fit",
            "R² score",
            "Higher is better. Negative values indicate performance "
            "below a mean-baseline reference."
        )

        fig = px.bar(
            sorted_metrics,
            x="Model",
            y="R2",
            color="R2",
            color_continuous_scale=[
                "#d58a64",
                "#3b8060"
            ],
            text_auto=".3f"
        )

        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="#a64d3d"
        )

        fig.update_layout(
            height=410,
            xaxis_tickangle=-22,
            margin=dict(l=5, r=5, t=15, b=5),
            coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # Full evaluation table

    section_heading(
        "Full results",
        "Model evaluation table",
        "Values are the notebook's recorded evaluation outputs."
    )

    st.dataframe(
        sorted_metrics.style.format({
            "MAE": "{:.4f}",
            "RMSE": "{:.4f}",
            "R2": "{:.4f}"
        }),
        hide_index=True,
        use_container_width=True
    )


    st.download_button(
        "Download evaluation metrics",
        data=metrics_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="lankaprice_model_comparison.csv",
        mime="text/csv",
    )


    st.markdown(
        """
        <div class="warning">
            <b>Important interpretation:</b>
            Stacking recorded MAE 98.2278,
            RMSE 100.5332 and R² −90.6632.
            The notebook notes that the Ridge meta-model
            extrapolated poorly across the regime shift.
            Keep this result in the comparison for transparency.
            The Hybrid RF + ARIMA result is based on
            1,978 matched predictions.
        </div>
        """,
        unsafe_allow_html=True,
    )


    with st.expander("How to read the metrics"):

        st.markdown(
            "- **MAE:** average absolute difference between actual and predicted prices.\n"
            "- **RMSE:** error metric that penalises larger prediction errors more strongly.\n"
            "- **R²:** indicates how much variation is explained relative to a mean-baseline model; "
            "a negative value is a warning that the model performed worse than that baseline."
        )


# =============================================================================
# PRICE FORECAST
# =============================================================================

elif selected_page == "Price Forecast":

    section_heading(
        "Forecast studio",
        "Explore the next months of rice prices",
        "Choose a forecasting model and horizon. Forecasts are generated "
        "from the selected market's history."
    )

    if "Random Forest" not in saved_models:

        st.error(
            "The saved Random Forest model is missing. "
            "Run the notebook's final model-export cell."
        )

        st.stop()


    f1, f2 = st.columns([1, 1])

    with f1:

        model_options = [
            "Hybrid RF + ARIMA",
            "Random Forest"
        ] + [
            name
            for name in saved_models
            if name not in ("Random Forest",)
        ]

        forecast_model = st.selectbox(
            "Forecast model",
            model_options
        )


    with f2:

        horizon = st.slider(
            "Forecast horizon (months)",
            1,
            12,
            6
        )


    try:

        forecast_df = forecast_future_prices(
            market_df,
            selected_market,
            horizon,
            saved_models.get("Random Forest"),
            saved_models
        )

    except Exception as exc:

        st.error(
            f"Forecast could not be generated: {exc}"
        )

        st.stop()


    if forecast_model not in forecast_df.columns:

        st.warning(
            f"{forecast_model} is not available in the forecast output."
        )

        st.stop()


    next_value = float(
        forecast_df[forecast_model].iloc[0]
    )

    delta = next_value - latest_price

    delta_pct = (
        delta / latest_price * 100
        if latest_price
        else 0
    )


    q1, q2, q3 = st.columns(3)

    q1.metric(
        "Latest observed price",
        f"Rs. {latest_price:,.2f}"
    )

    q2.metric(
        f"Next month · {forecast_df['price_date'].iloc[0]:%b %Y}",
        f"Rs. {next_value:,.2f}",
        f"{delta:+.2f} LKR · {delta_pct:+.1f}%",
        delta_color="inverse"
    )

    q3.metric(
        "Forecast horizon",
        f"{horizon} months",
        forecast_model
    )


    # Forecast chart

    hist_window = market_df.tail(72)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=hist_window["price_date"],
            y=hist_window["c_rice_various"],
            mode="lines",
            name="Observed price",
            line=dict(
                color="#315f49",
                width=2.6
            )
        )
    )

    bridge_x = (
        [market_df["price_date"].iloc[-1]]
        + list(forecast_df["price_date"])
    )

    bridge_y = (
        [latest_price]
        + list(forecast_df[forecast_model])
    )

    fig.add_trace(
        go.Scatter(
            x=bridge_x,
            y=bridge_y,
            mode="lines+markers",
            name=f"{forecast_model} forecast",
            line=dict(
                color="#d39a42",
                width=3,
                dash="dash"
            ),
            marker=dict(size=7)
        )
    )

    fig.update_layout(
        height=470,
        hovermode="x unified",
        xaxis_title="Month",
        yaxis_title="Rice price (LKR/kg)",
        margin=dict(l=10, r=10, t=25, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # Forecast details

    section_heading(
        "Forecast detail",
        "Month-by-month predictions",
        "Download the generated values for further analysis."
    )

    display_forecast = forecast_df.copy()

    display_forecast["Month"] = (
        display_forecast["price_date"].dt.strftime("%b %Y")
    )

    display_forecast = display_forecast.drop(
        columns=["price_date"]
    )

    display_forecast = display_forecast[
        ["Month"]
        + [
            c
            for c in display_forecast.columns
            if c != "Month"
        ]
    ]

    st.dataframe(
        display_forecast.style.format({
            c: "{:.2f}"
            for c in display_forecast.columns
            if c != "Month"
        }),
        hide_index=True,
        use_container_width=True
    )


    st.download_button(
        "Download forecast CSV",
        data=display_forecast.to_csv(
            index=False
        ).encode("utf-8"),
        file_name=(
            f"lankaprice_"
            f"{selected_market.lower().replace(' ', '_')}"
            f"_forecast.csv"
        ),
        mime="text/csv",
    )


    with st.expander("Forecast methodology"):

        st.markdown(
            "- **Random Forest:** recursively predicts future prices using market, calendar, lag and rolling-mean features.\n"
            "- **ARIMA:** fits an ARIMA(1, 1, 1) time-series model to the selected market's historical price series.\n"
            "- **Hybrid RF + ARIMA:** averages the Random Forest and ARIMA forecasts with equal 50% weights.\n"
            "- Future recursive predictions use earlier predicted values as lag inputs."
        )


# =============================================================================
# MARKET EXPLORER
# =============================================================================

elif selected_page == "Market Explorer":

    section_heading(
        "Across Sri Lanka",
        "Compare rice prices between markets",
        "Compare historical price trajectories and identify markets "
        "with the highest and lowest latest prices."
    )


    defaults = [
        m
        for m in [
            "Colombo City",
            "Kandy",
            "Jaffna",
            "Galle",
            "Anuradhapura"
        ]
        if m in all_markets
    ]

    if selected_market not in defaults:
        defaults = [selected_market] + defaults[:4]


    compare = st.multiselect(
        "Markets to compare",
        all_markets,
        default=defaults
    )


    if compare:

        subset = history_df[
            history_df["mkt_name"].isin(compare)
        ]

        fig = px.line(
            subset,
            x="price_date",
            y="c_rice_various",
            color="mkt_name",
            labels={
                "price_date": "Date",
                "c_rice_various": "Rice price (LKR/kg)",
                "mkt_name": "Market"
            }
        )

        fig.update_layout(
            height=470,
            hovermode="x unified",
            margin=dict(l=8, r=8, t=15, b=8),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "Select at least one market to display the comparison."
        )


    # Latest market ranking

    latest_snapshot = history_df[
        history_df["price_date"] == latest_date
    ].sort_values(
        "c_rice_various",
        ascending=False
    )


    hi, lo = st.columns(2)

    with hi:

        section_heading(
            "Higher prices",
            f"Top 10 markets · {latest_date:%b %Y}",
            "Markets with the highest latest recorded prices."
        )

        fig = px.bar(
            latest_snapshot.head(10).sort_values(
                "c_rice_various"
            ),
            x="c_rice_various",
            y="mkt_name",
            orientation="h",
            labels={
                "c_rice_various": "Price (LKR/kg)",
                "mkt_name": "Market"
            },
            color="c_rice_various",
            color_continuous_scale=[
                "#f0c6a6",
                "#a94f36"
            ],
            text_auto=".2f"
        )

        fig.update_layout(
            height=420,
            margin=dict(l=5, r=5, t=10, b=5),
            coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with lo:

        section_heading(
            "Lower prices",
            f"Bottom 10 markets · {latest_date:%b %Y}",
            "Markets with the lowest latest recorded prices."
        )

        fig = px.bar(
            latest_snapshot.tail(10).sort_values(
                "c_rice_various",
                ascending=False
            ),
            x="c_rice_various",
            y="mkt_name",
            orientation="h",
            labels={
                "c_rice_various": "Price (LKR/kg)",
                "mkt_name": "Market"
            },
            color="c_rice_various",
            color_continuous_scale=[
                "#b9d9c2",
                "#2d6d4b"
            ],
            text_auto=".2f"
        )

        fig.update_layout(
            height=420,
            margin=dict(l=5, r=5, t=10, b=5),
            coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    st.markdown(
        "#### Latest market price ranking"
    )

    st.dataframe(
        latest_snapshot.rename(
            columns={
                "mkt_name": "Market",
                "price_date": "Date",
                "c_rice_various": "Rice price (LKR/kg)"
            }
        ),
        hide_index=True,
        use_container_width=True
    )


# =============================================================================
# FOOTER
# =============================================================================

st.markdown(
    """
    <div class="footer">
      <b>LankaPrice</b> · Sri Lanka Rice Price Intelligence

      <span style="float:right">
        Research dashboard · World Bank RTFP data
      </span>
    </div>
    """,
    unsafe_allow_html=True,
)
