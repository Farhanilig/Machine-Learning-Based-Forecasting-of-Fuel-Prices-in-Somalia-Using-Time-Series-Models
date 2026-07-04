import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import torch
import torch.nn as nn
from statsmodels.tsa.arima.model import ARIMAResults
from prophet import Prophet

# Page Config
st.set_page_config(
    page_title="Somalia Fuel Price Forecasting Dashboard",
    page_icon="🇸🇴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Sleek Theme)
st.markdown("""
<style>
    .main-banner {
        background: linear-gradient(135deg, #1e3a8a, #7c3aed);
        color: white;
        padding: 24px 30px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    }
    .main-banner h1 {
        color: white !important;
        margin: 0;
        font-size: 26px;
        font-weight: 700;
    }
    .main-banner p {
        margin: 6px 0 0 0;
        opacity: 0.9;
        font-size: 14px;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.05);
    }
    .metric-title {
        font-size: 13px;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 22px;
        font-weight: 700;
        color: #0f172a;
    }
    .badge-up {
        background-color: #dcfce7;
        color: #15803d;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }
    .badge-down {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }
    .badge-stable {
        background-color: #fef3c7;
        color: #b45309;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Define LSTM Class
class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

# Cache Loaders to avoid redundant loads
@st.cache_data
def load_data():
    ts = pd.read_csv("somalia_fuel_monthly_clean.csv", index_col="date", parse_dates=True).squeeze()
    ts.index = pd.DatetimeIndex(ts.index, freq='MS')
    return ts

@st.cache_resource
def load_models():
    # Load ARIMA
    arima_res = ARIMAResults.load('arima_model.pkl')
    
    # Load LSTM Config
    with open('lstm_config.json', 'r') as f:
        lstm_cfg = json.load(f)
        
    # Load LSTM Weights
    lstm_model = LSTMForecaster(hidden_size=32, num_layers=2, dropout=0.1)
    lstm_model.load_state_dict(torch.load('lstm_model.pth', map_location=torch.device('cpu')))
    lstm_model.eval()
    
    return arima_res, lstm_model, lstm_cfg

# Init App Files
try:
    ts = load_data()
    arima_res, lstm_model, lstm_cfg = load_models()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.info("Please verify that somalia_fuel_monthly_clean.csv, arima_model.pkl, lstm_model.pth, and lstm_config.json exist in e:/Fuel.")
    st.stop()

# Fit Prophet on the fly (cached)
@st.cache_resource
def fit_prophet_full(ts_df):
    df_p = pd.DataFrame({'ds': ts_df.index, 'y': ts_df.values})
    m = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='additive',
        changepoint_prior_scale=0.05
    )
    m.fit(df_p)
    return m

prophet_model = fit_prophet_full(ts)

# Generate predictions for 2026 (Jan 2026 - Dec 2026)
forecast_dates = pd.date_range('2026-01-01', periods=12, freq='MS')
last_price = ts.iloc[-1]
last_date = ts.index[-1]

# ── ARIMA Predictions ──
arima_fc_obj = arima_res.get_forecast(steps=12)
arima_mean = arima_fc_obj.predicted_mean.values
arima_lo = arima_fc_obj.conf_int(alpha=0.05).iloc[:, 0].values
arima_hi = arima_fc_obj.conf_int(alpha=0.05).iloc[:, 1].values

# ── Prophet Predictions ──
future_12 = prophet_model.make_future_dataframe(periods=12, freq='MS')
forecast_p = prophet_model.predict(future_12)
prophet_rows = forecast_p[forecast_p['ds'] > '2025-12-01'].reset_index(drop=True)
prophet_mean = prophet_rows['yhat'].values
prophet_lo = prophet_rows['yhat_lower'].values
prophet_hi = prophet_rows['yhat_upper'].values

# ── LSTM Predictions (Recursive) ──
ts_min = lstm_cfg['ts_min']
ts_max = lstm_cfg['ts_max']
std_resid = lstm_cfg['std_resid']
window = lstm_cfg['window']

ts_arr = ts.values.astype(np.float32)
ts_norm = (ts_arr - ts_min) / (ts_max - ts_min)
current_window = list(ts_norm[-window:])
lstm_fc_norm = []

for _ in range(12):
    x_in = torch.tensor(current_window, dtype=torch.float32).view(1, window, 1)
    with torch.no_grad():
        p_norm = lstm_model(x_in).item()
    lstm_fc_norm.append(p_norm)
    current_window = current_window[1:] + [p_norm]

lstm_mean = np.array(lstm_fc_norm) * (ts_max - ts_min) + ts_min

lstm_lo = []
lstm_hi = []
for k in range(1, 13):
    margin = 1.96 * std_resid * np.sqrt(k)
    lstm_lo.append(lstm_mean[k-1] - margin)
    lstm_hi.append(lstm_mean[k-1] + margin)
lstm_lo = np.array(lstm_lo)
lstm_hi = np.array(lstm_hi)


# ── STREAMLIT UI LAYOUT ──

# Sidebar Controls
st.sidebar.markdown("### 🇸🇴 Forecast Parameters")
model_choice = st.sidebar.selectbox(
    "Select Forecasting Model:",
    ["LSTM ★ (Best)", "Prophet", "ARIMA(2,1,1)"],
    index=0
)
months_ahead = st.sidebar.slider(
    "Forecast Horizon (Months):",
    min_value=1,
    max_value=12,
    value=3,
    step=1
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### Model Insights
* **LSTM:** Evaluated recursively over 2026. Captures complex, non-linear sequences. Achieves the **lowest out-of-sample forecasting error** among the available models (MAPE: 4.02%).
* **Prophet:** Additive regression model focusing on overall trend components. Highly flexible.
* **ARIMA(2,1,1):** Standard linear Box-Jenkins methodology. Captures short-term auto-dependencies.

*Note: Linear Regression (Lag Baseline) is omitted from selection because it is a static baseline requiring true historical lags, making it unsuitable for multi-step future forecasting.*
""")

# Main Banner
st.markdown("""
<div class="main-banner">
    <h1>Somalia Fuel Price Forecasting Dashboard</h1>
    <p>Decision support tool for monitoring and predicting retail national average fuel prices (USD/litre). Base Data: 2020–2025.</p>
</div>
""", unsafe_allow_html=True)

# Select variables based on chosen model
if model_choice == "LSTM ★ (Best)":
    fc_mean = lstm_mean
    fc_lo = lstm_lo
    fc_hi = lstm_hi
    color_fc = '#7c3aed'
    model_label = 'LSTM'
elif model_choice == "Prophet":
    fc_mean = prophet_mean
    fc_lo = prophet_lo
    fc_hi = prophet_hi
    color_fc = '#f97316'
    model_label = 'Prophet'
else:
    fc_mean = arima_mean
    fc_lo = arima_lo
    fc_hi = arima_hi
    color_fc = '#2563eb'
    model_label = 'ARIMA(2,1,1)'

target_idx = months_ahead - 1
target_date = forecast_dates[target_idx]
target_price = fc_mean[target_idx]
target_lo = fc_lo[target_idx]
target_hi = fc_hi[target_idx]

# Compute Direction & Percentage Change
pct_change = ((target_price - last_price) / last_price) * 100
if pct_change > 1.5:
    badge_html = f'<span class="badge-up">📈 Price Increase (+{pct_change:.1f}%)</span>'
elif pct_change < -1.5:
    badge_html = f'<span class="badge-down">📉 Price Decrease ({pct_change:.1f}%)</span>'
else:
    badge_html = f'<span class="badge-stable">↔ Stable Price (±{abs(pct_change):.1f}%)</span>'

# Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Current Price (Dec 2025)</div>
        <div class="metric-value">${last_price:.3f}</div>
        <div style="font-size:12px; color:#64748b; margin-top:4px;">Historical Baseline</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Forecast Price ({target_date.strftime('%b %Y')})</div>
        <div class="metric-value">${target_price:.3f}</div>
        <div style="font-size:12px; color:#64748b; margin-top:4px;">{months_ahead}-Month Horizon</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Market Trend / Direction</div>
        <div class="metric-value" style="font-size:18px; padding-top:4px;">{badge_html}</div>
        <div style="font-size:12px; color:#64748b; margin-top:8px;">Expected price movement</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">95% Confidence Interval</div>
        <div class="metric-value" style="font-size:18px; padding-top:4px;">${target_lo:.3f} - ${target_hi:.3f}</div>
        <div style="font-size:12px; color:#64748b; margin-top:8px;">Error Range Bounds</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Plot & CI Visualizer
pcol1, pcol2 = st.columns([3, 1])

with pcol1:
    fig, ax_chart = plt.subplots(figsize=(11, 5))
    ax_chart.plot(ts.index, ts.values, color='#334155', linewidth=2, label='Historical Actuals (2020-2025)', zorder=3)
    ax_chart.plot(forecast_dates, fc_mean, color=color_fc, linewidth=1.5, linestyle='--', alpha=0.4, zorder=2)
    ax_chart.plot(forecast_dates[:months_ahead], fc_mean[:months_ahead], color=color_fc, linewidth=2.5, linestyle='--', label=f'{model_label} Forecast', zorder=3)
    ax_chart.fill_between(forecast_dates[:months_ahead], fc_lo[:months_ahead], fc_hi[:months_ahead], alpha=0.15, color=color_fc, label='95% Confidence Interval')
    ax_chart.scatter([target_date], [target_price], color=color_fc, s=120, zorder=5, edgecolors='black', label=f'Target: {target_date.strftime("%b %Y")}')
    ax_chart.axvline(last_date, color='gray', linestyle=':', alpha=0.6)
    ax_chart.text(last_date, ts.min() + 0.01, ' Dec 2025', color='gray', fontsize=8)
    ax_chart.set_title(f'Somalia Fuel Price Forecast Trend ({model_label})', fontsize=12, fontweight='bold', pad=10)
    ax_chart.set_ylabel('USD/litre')
    ax_chart.legend(loc='upper left', fontsize=9)
    ax_chart.grid(True, alpha=0.2)
    ax_chart.set_ylim(0.75, 1.25)
    st.pyplot(fig)

with pcol2:
    # 95% Confidence Interval Visualizer Bar
    st.markdown("<h4 style='text-align: center; margin-top: 10px; font-size: 15px;'>Confidence Range</h4>", unsafe_allow_html=True)
    fig_ci, ax_ci = plt.subplots(figsize=(3, 5))
    ax_ci.set_xlim(0, 1)
    ax_ci.set_ylim(0, 1)
    ax_ci.axis('off')
    
    ax_ci.text(0.5, 0.90, '95% CI Range', ha='center', va='center', fontsize=11, fontweight='bold', color='#334155')
    
    ci_range = target_hi - target_lo
    ci_center = (target_price - target_lo) / ci_range if ci_range > 0 else 0.5
    
    # Render horizontal bar
    ax_ci.barh(0.55, 1.0, height=0.15, color='#e2e8f0', left=0)
    ax_ci.barh(0.55, ci_center, height=0.15, color=color_fc + '60', left=0)
    ax_ci.scatter([ci_center], [0.55], s=120, color=color_fc, edgecolors='black', zorder=5)
    
    ax_ci.text(0.05, 0.38, f"${target_lo:.3f}", ha='left', va='center', fontsize=11, fontweight='bold', color='#64748b')
    ax_ci.text(0.95, 0.38, f"${target_hi:.3f}", ha='right', va='center', fontsize=11, fontweight='bold', color='#64748b')
    ax_ci.text(0.5, 0.15, f"Forecast Value:\\n${target_price:.3f}", ha='center', va='center', fontsize=12, fontweight='bold', color='#0f172a')
    
    st.pyplot(fig_ci)

# Detailed Table expander
with st.expander("📊 View Complete 12-Month Forecast Table"):
    fc_df = pd.DataFrame({
        'Month': [d.strftime('%B %Y') for d in forecast_dates],
        'Forecast Price (USD)': [f"${v:.3f}" for v in fc_mean],
        'Lower Bound (95% CI)': [f"${v:.3f}" for v in fc_lo],
        'Upper Bound (95% CI)': [f"${v:.3f}" for v in fc_hi],
        'Expected Change': [f"{((v-last_price)/last_price)*100:+.1f}%" for v in fc_mean]
    })
    st.dataframe(fc_df, use_container_width=True)

# Academic Comparison Card
st.markdown("---")
st.markdown("### 🏆 Model Comparison & Academic Performance")
st.markdown("""
The models are evaluated against the historical out-of-sample **10-month test period (March 2025 – December 2025)** and benchmarked against the published regional literature of **Hussein & Abdillahi (2025)** (Somalia CPI ARIMA model, MAPE = 6.18%).
""")

# Markdown table of results
st.markdown("""
| Model Rank | Model Type | Test RMSE | Test MAPE | vs. Benchmark (6.18% MAPE) | Status |
|---|---|---|---|---|---|
| 🥇 **1st** | **Linear Regression (Lag Baseline)** | **0.0398** | **3.00%** | **Beats by 3.18 pp (51% reduction in error)** | ✅ Beat |
| 🥈 **2nd** | **LSTM (PyTorch Deep Learning)** | **0.0509** | **4.02%** | **Beats by 2.16 pp (35% reduction in error)** | ✅ Beat |
| 🥉 **3rd** | **Facebook Prophet** | **0.0515** | **4.15%** | **Beats by 2.03 pp (33% reduction in error)** | ✅ Beat |
| 4th | **ARIMA(2,1,1)** | **0.0571** | **4.29%** | **Beats by 1.89 pp (31% reduction in error)** | ✅ Beat |
| 5th | **ARIMA(1,1,2)** | **0.0583** | **4.41%** | **Beats by 1.77 pp (29% reduction in error)** | ✅ Beat |
| 6th | **ARIMA(1,1,1)** | **0.0618** | **4.79%** | **Beats by 1.39 pp (22% reduction in error)** | ✅ Beat |
""")

st.markdown("""
> **Key Academic Takeaway:** All 6 time-series models successfully beat the Hussein & Abdillahi (2025) baseline. 
> The PyTorch LSTM network represents the primary machine learning contribution of the thesis, producing an extremely low test MAPE of 4.02%, which proves the capacity of deep learning recurrent networks to forecast highly volatile fuel prices in a data-scarce environment.
""")