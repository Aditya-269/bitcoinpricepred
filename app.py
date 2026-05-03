import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import scipy.stats as stats
from arch import arch_model
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import json
from pathlib import Path

# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================

st.set_page_config(
    page_title="AlphaI Bitcoin Forecaster",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    h1 {
        color: #00d4ff;
        text-align: center;
        font-size: 3em;
        margin-bottom: 0.2em;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
    }
    
    h2 {
        color: #00d4ff;
        border-bottom: 2px solid #00d4ff;
        padding-bottom: 0.5em;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(0, 212, 255, 0.05) 100%);
        border-left: 4px solid #00d4ff;
        padding: 1.5em;
        border-radius: 0.75em;
        margin: 0.5em 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .stat-good {
        color: #00ff88;
        font-weight: bold;
        font-size: 1.1em;
    }
    
    .stat-warning {
        color: #ffaa00;
        font-weight: bold;
        font-size: 1.1em;
    }
    
    .stat-bad {
        color: #ff4444;
        font-weight: bold;
        font-size: 1.1em;
    }
    
    .info-box {
        background: rgba(0, 212, 255, 0.15);
        border: 1px solid #00d4ff;
        border-radius: 0.5em;
        padding: 1em;
        margin: 1em 0;
    }
    
    .success-box {
        background: rgba(0, 255, 136, 0.15);
        border: 1px solid #00ff88;
        border-radius: 0.5em;
        padding: 1em;
        margin: 1em 0;
    }
    
    .warning-box {
        background: rgba(255, 170, 0, 0.15);
        border: 1px solid #ffaa00;
        border-radius: 0.5em;
        padding: 1em;
        margin: 1em 0;
    }
    
    .stMetric {
        background: rgba(26, 26, 46, 0.8);
        padding: 1.5em;
        border-radius: 0.75em;
        border: 1px solid rgba(0, 212, 255, 0.2);
    }
    
    button {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        color: white;
        border: none;
        padding: 0.75em 1.5em;
        border-radius: 0.5em;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 212, 255, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

BACKTEST_CONFIG = {
    'coverage_95': 0.9417,
    'mean_winkler': 1723.74,
    'mean_width': 1139.30,
    'n_predictions': 720,
    'backtest_period': '30 days',
    'model': 'GARCH(1,1) + Student-t + GBM',
}

MODEL_CONFIG = {
    'train_window': 504,
    'n_sims': 10_000,
    'confidence': 0.95,
    'fetch_hours': 500,
}

DB_PATH = str(Path(__file__).resolve().parent / "predictions.db")

# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def init_db():
    """Initialize SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            btc_price REAL,
            low_95 REAL,
            high_95 REAL,
            actual_price REAL,
            in_range BOOLEAN
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(btc_price, low_95, high_95):
    """Save prediction to database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO predictions (btc_price, low_95, high_95)
        VALUES (?, ?, ?)
    ''', (btc_price, low_95, high_95))
    conn.commit()
    conn.close()

def get_prediction_history(limit=100):
    """Fetch prediction history."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        'SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?',
        conn,
        params=(limit,)
    )
    conn.close()
    if len(df) > 0:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp').reset_index(drop=True)
    return None

def get_live_stats():
    """Count of rows in predictions table."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM predictions')
    total_saved = int(c.fetchone()[0])
    conn.close()
    return {'total_saved': total_saved}

# ============================================================================
# DATA FETCHING
# ============================================================================

@st.cache_data(ttl=300)
def fetch_btcusdt_data(hours=500):
    """Fetch BTCUSDT 1-hour candles from Binance."""
    url = 'https://data-api.binance.vision/api/v3/klines'
    params = {
        'symbol': 'BTCUSDT',
        'interval': '1h',
        'limit': hours
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'qav', 'num_trades', 'taker_base', 'taker_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['close'])
        return df.set_index('timestamp')[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        st.error(f"❌ Error fetching data: {str(e)}")
        return None

# ============================================================================
# VOLATILITY & FORECASTING
# ============================================================================

def rolling_entropy(x, window=60, bins=20):
    """Calculate rolling entropy."""
    def ent(v):
        p, _ = np.histogram(v, bins=bins, density=True)
        p = p[p > 0]
        if len(p) == 0:
            return 0
        return -np.sum(p * np.log(p))
    return x.rolling(window).apply(ent, raw=True)

def get_live_prediction(prices, n_sims=10_000):
    """Generate next-hour 95% CI using GARCH + GBM."""
    try:
        log_ret = np.log(prices / prices.shift(1)).dropna()
        
        if len(log_ret) < 100:
            st.error("❌ Not enough historical data")
            return None, None, None, None, None
        
        # Fit GARCH(1,1)
        am = arch_model(log_ret * 100, vol='Garch', p=1, q=1, dist='studentst')
        res = am.fit(disp='off', show_warning=False)
        
        sigma = res.conditional_volatility.iloc[-1] / 100
        mu = log_ret.mean()
        
        # Student-t for fat tails
        resid = (log_ret * 100 - res.params['mu']) / res.conditional_volatility
        nu = max(4, stats.t.fit(resid, floc=0, fscale=1)[0])
        
        # Monte Carlo
        S0 = prices.iloc[-1]
        Z = np.random.standard_t(nu, size=n_sims) * np.sqrt((nu - 2) / nu)
        S_t1 = S0 * np.exp((mu - 0.5 * sigma**2) + sigma * Z)
        
        low, high = np.percentile(S_t1, [2.5, 97.5])
        
        return float(S0), float(low), float(high), float(sigma), float(nu)
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return None, None, None, None, None

# ============================================================================
# ADVANCED CHARTING FUNCTIONS
# ============================================================================

# Binance-style chart palette (dark + gold line)
_BINANCE_BG = '#1e2329'
_BINANCE_PANEL = '#2b3139'
_BINANCE_GRID = '#363c45'
_BINANCE_TEXT = '#eaecef'
_BINANCE_MUTED = '#848e9c'
_BINANCE_GOLD = '#F0B90B'


def create_main_chart(df, current_price, pred_low, pred_high, n_bars=168):
    """Binance-like dark chart: gold close line, crosshair, 95% forecast band."""
    want = min(max(1, int(n_bars)), len(df))
    d = df.iloc[-want:]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.72, 0.28],
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": False}]
        ]
    )

    next_time = d.index[-1] + timedelta(hours=1)

    # Forecast band (under the price line)
    fig.add_trace(
        go.Scatter(
            x=[d.index[-1], next_time, next_time, d.index[-1]],
            y=[current_price, pred_high, pred_low, current_price],
            fill='toself',
            fillcolor='rgba(240, 185, 11, 0.12)',
            line=dict(color='rgba(0,0,0,0)', width=0),
            name='95% CI (next hour)',
            hoverinfo='skip',
            showlegend=True,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[d.index[-1], next_time],
            y=[current_price, pred_high],
            mode='lines',
            line=dict(color='rgba(14, 203, 129, 0.85)', width=1.5, dash='dash'),
            name=f'High {pred_high:,.0f}',
            hovertemplate='95% high\n$%{y:,.2f}<extra></extra>',
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[d.index[-1], next_time],
            y=[current_price, pred_low],
            mode='lines',
            line=dict(color='rgba(246, 70, 93, 0.85)', width=1.5, dash='dash'),
            name=f'Low {pred_low:,.0f}',
            hovertemplate='95% low\n$%{y:,.2f}<extra></extra>',
        ),
        row=1, col=1,
    )

    # Area under price: flat baseline just below window low (avoid fill-to-zero)
    _lo, _hi = float(d['low'].min()), float(d['high'].max())
    _pad = max((_hi - _lo) * 0.04, _hi * 1e-6)
    y_fill_floor = _lo - _pad
    fig.add_trace(
        go.Scatter(
            x=d.index,
            y=np.full(len(d), y_fill_floor),
            mode='lines',
            line=dict(width=0),
            hoverinfo='skip',
            showlegend=False,
        ),
        row=1, col=1,
    )

    # Gold line + gradient fill — plain-text hover (Streamlit can escape HTML in hovertemplate)
    _price_hover = [
        (
            f"{ts.strftime('%Y-%m-%d')}  {ts.strftime('%I:%M:%S %p')} UTC\n"
            f"$ {float(row['close']):,.2f}\n"
            f"Hourly close · O {float(row['open']):,.2f} · H {float(row['high']):,.2f} · "
            f"L {float(row['low']):,.2f}"
        )
        for ts, row in d.iterrows()
    ]
    _show_markers = len(d) <= 48
    fig.add_trace(
        go.Scatter(
            x=d.index,
            y=d['close'],
            mode='lines+markers',
            name='BTC / USDT',
            line=dict(color=_BINANCE_GOLD, width=2),
            fill='tonexty',
            fillcolor='rgba(240, 185, 11, 0.22)',
            marker=dict(
                size=11,
                color=_BINANCE_GOLD,
                line=dict(color=_BINANCE_TEXT, width=1),
                opacity=1 if _show_markers else 0,
            ),
            hovertext=_price_hover,
            hoverlabel=dict(
                bgcolor=_BINANCE_PANEL,
                bordercolor=_BINANCE_GRID,
                font=dict(color=_BINANCE_TEXT, size=13),
                align='left',
            ),
            hovertemplate='%{text}<extra></extra>',
        ),
        row=1, col=1,
    )

    _vol_hover = [
        f"{ts.strftime('%Y-%m-%d %H:%M')} UTC\nVol {float(row['volume']):,.2f} BTC"
        for ts, row in d.iterrows()
    ]
    fig.add_trace(
        go.Bar(
            x=d.index,
            y=d['volume'],
            name='Volume',
            marker_color='rgba(132, 142, 156, 0.35)',
            hovertext=_vol_hover,
            hoverlabel=dict(bgcolor=_BINANCE_PANEL, bordercolor=_BINANCE_GRID, font_color=_BINANCE_TEXT),
            hovertemplate='%{text}<extra></extra>',
        ),
        row=2, col=1,
    )

    spike_line = dict(
        showspikes=True,
        spikethickness=1,
        spikecolor='rgba(255, 255, 255, 0.45)',
        spikesnap='cursor',
        spikemode='across',
        spikedash='dot',
    )

    fig.update_layout(
        title=dict(
            text=(
                f'<b style="color:{_BINANCE_TEXT}">BTCUSDT</b> '
                f'<span style="color:{_BINANCE_MUTED};font-weight:normal">· 1h · GARCH forecast</span>'
            ),
            x=0.01,
            xanchor='left',
            font=dict(size=18),
        ),
        margin=dict(t=56, l=56, r=24, b=48),
        xaxis_title=None,
        yaxis_title=None,
        height=720,
        hovermode='closest',
        dragmode='zoom',
        font=dict(
            family='ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif',
            size=12,
            color=_BINANCE_TEXT,
        ),
        plot_bgcolor=_BINANCE_BG,
        paper_bgcolor=_BINANCE_BG,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            bgcolor='rgba(30, 35, 41, 0.92)',
            bordercolor=_BINANCE_GRID,
            borderwidth=1,
            font=dict(size=11, color=_BINANCE_TEXT),
        ),
        xaxis_rangeslider_visible=False,
    )

    axis_style = dict(
        showgrid=True,
        gridcolor=_BINANCE_GRID,
        gridwidth=1,
        zeroline=False,
        linecolor=_BINANCE_GRID,
        tickfont=dict(color=_BINANCE_MUTED, size=11),
        **spike_line,
    )

    fig.update_xaxes(axis_style, row=1, col=1)
    fig.update_xaxes(axis_style, row=2, col=1)

    fig.update_yaxes(
        showgrid=True,
        gridcolor=_BINANCE_GRID,
        gridwidth=1,
        zeroline=False,
        linecolor=_BINANCE_GRID,
        tickfont=dict(color=_BINANCE_MUTED, size=11),
        tickprefix='$',
        tickformat=',.0f',
        side='right',
        **spike_line,
        row=1, col=1,
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        linecolor=_BINANCE_GRID,
        tickfont=dict(color=_BINANCE_MUTED, size=11),
        title=dict(text='Volume', font=dict(color=_BINANCE_MUTED, size=11)),
        showspikes=False,
        row=2, col=1,
    )

    return fig

def create_volatility_chart(df):
    """Create volatility over time chart."""
    log_ret = np.log(df['close'] / df['close'].shift(1)).dropna()
    volatility = log_ret.rolling(window=24).std() * np.sqrt(365 * 24) * 100
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=volatility.index[-100:],
            y=volatility.iloc[-100:],
            name='Realized Volatility (24H)',
            line=dict(color='#00d4ff', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 212, 255, 0.2)',
            hovertemplate='%{x|%H:%M}\nVolatility: %{y:.2f}%<extra></extra>'
        )
    )
    
    fig.update_layout(
        title='<b>24-Hour Realized Volatility</b>',
        xaxis_title='Time (UTC)',
        yaxis_title='Volatility (%)',
        template='plotly_dark',
        height=350,
        hovermode='x unified',
        plot_bgcolor='rgba(15, 15, 35, 0.5)',
        paper_bgcolor='rgba(26, 26, 46, 1)',
    )
    
    return fig

def create_history_chart(history_df):
    """Create prediction history chart."""
    if history_df is None or len(history_df) == 0:
        return None
    
    fig = go.Figure()
    
    # Actual prices
    actual_available = history_df[history_df['actual_price'].notna()]
    if len(actual_available) > 0:
        fig.add_trace(
            go.Scatter(
                x=actual_available['timestamp'],
                y=actual_available['actual_price'],
                name='Actual Price',
                mode='markers',
                marker=dict(color='#ffaa00', size=8, symbol='circle'),
                hovertemplate='%{x|%H:%M}\nActual: $%{y:,.2f}<extra></extra>'
            )
        )
    
    # Predicted highs
    fig.add_trace(
        go.Scatter(
            x=history_df['timestamp'],
            y=history_df['high_95'],
            name='95% High',
            mode='lines',
            line=dict(color='rgba(0, 255, 136, 0.5)', width=1, dash='dash'),
            hovertemplate='%{x|%H:%M}\nHigh: $%{y:,.2f}<extra></extra>'
        )
    )
    
    # Predicted lows
    fig.add_trace(
        go.Scatter(
            x=history_df['timestamp'],
            y=history_df['low_95'],
            name='95% Low',
            mode='lines',
            line=dict(color='rgba(255, 68, 68, 0.5)', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(0, 212, 255, 0.1)',
            hovertemplate='%{x|%H:%M}\nLow: $%{y:,.2f}<extra></extra>'
        )
    )
    
    fig.update_layout(
        title='<b>Prediction History with Actuals</b>',
        xaxis_title='Time (UTC)',
        yaxis_title='Price (USD)',
        template='plotly_dark',
        height=400,
        hovermode='x unified',
        plot_bgcolor='rgba(15, 15, 35, 0.5)',
        paper_bgcolor='rgba(26, 26, 46, 1)',
    )
    
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    init_db()
    
    # Header
    st.markdown("# 🎯 Bitcoin Next-Hour Forecaster")
    st.markdown("**AlphaI × Polaris Challenge** | Real-time Prediction Engine")
    st.divider()
    
    # Fetch data
    with st.spinner("📡 Fetching latest Bitcoin data..."):
        df = fetch_btcusdt_data(hours=MODEL_CONFIG['fetch_hours'])
    
    if df is None or len(df) < 100:
        st.error("Failed to fetch data")
        return
    
    # Get prediction
    with st.spinner("🧮 Computing prediction..."):
        current_price, pred_low, pred_high, sigma, nu = get_live_prediction(
            df['close'],
            n_sims=MODEL_CONFIG['n_sims']
        )
    
    if current_price is None:
        st.error("Failed to generate prediction")
        return
    
    # ========================================================================
    # LIVE PREDICTION SECTION
    # ========================================================================
    
    st.subheader("📊 Next-Hour Prediction (95% Confidence)")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Current Price",
            f"${current_price:,.2f}",
            delta="BTCUSDT"
        )
    with col2:
        change_pct = ((pred_low - current_price) / current_price * 100)
        st.metric(
            "95% Low",
            f"${pred_low:,.2f}",
            delta=f"{change_pct:+.2f}%"
        )
    with col3:
        change_pct = ((pred_high - current_price) / current_price * 100)
        st.metric(
            "95% High",
            f"${pred_high:,.2f}",
            delta=f"{change_pct:+.2f}%"
        )
    with col4:
        width = pred_high - pred_low
        width_pct = (width / current_price * 100)
        st.metric(
            "Range Width",
            f"${width:,.2f}",
            delta=f"{width_pct:.2f}% of price"
        )
    
    st.divider()
    
    # ========================================================================
    # INTERACTIVE CHARTS - TAB BASED
    # ========================================================================
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Main Chart", 
        "📊 Volatility", 
        "📚 History", 
        "ℹ️ Details"
    ])
    
    with tab1:
        st.markdown("### Price chart")
        _tf = st.radio(
            "Timeframe",
            ["1D", "3D", "7D", "14D", "All"],
            index=2,
            horizontal=True,
            label_visibility="collapsed",
            key="main_chart_timeframe",
        )
        _tf_bars = {"1D": 24, "3D": 72, "7D": 168, "14D": 336, "All": len(df)}
        fig_main = create_main_chart(
            df, current_price, pred_low, pred_high, n_bars=_tf_bars[_tf]
        )
        st.plotly_chart(fig_main, use_container_width=True, config={
            'displayModeBar': True,
            'displaylogo': False,
            'scrollZoom': True,
            'doubleClick': 'reset',
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
        })
        st.caption(
            "Hover for date, time, close · O/H/L/C on the same candle · "
            "Crosshair follows the cursor · Scroll to zoom · Double-click resets"
        )
    
    with tab2:
        st.markdown("### Realized Volatility (Last 100 Hours)")
        fig_vol = create_volatility_chart(df)
        st.plotly_chart(fig_vol, use_container_width=True, config={
            'displayModeBar': True,
            'displaylogo': False,
        })
    
    with tab3:
        st.markdown("### Prediction History (Part C)")
        history_df = get_prediction_history(limit=100)
        if history_df is not None and len(history_df) > 0:
            fig_hist = create_history_chart(history_df)
            if fig_hist:
                st.plotly_chart(fig_hist, use_container_width=True, config={
                    'displayModeBar': True,
                    'displaylogo': False,
                })
            
            # History stats
            st.markdown("### Recent Predictions")
            st.dataframe(
                history_df[[
                    'timestamp', 'btc_price', 'low_95', 'high_95', 'actual_price'
                ]].head(20),
                use_container_width=True
            )
        else:
            st.info("💡 No prediction history yet. Save predictions to populate this section.")
    
    with tab4:
        st.markdown("### Model Parameters & Information")
        
        # FIX: Compute log_ret locally for display
        log_ret_display = np.log(df['close'] / df['close'].shift(1)).dropna()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Volatility Model**")
            st.json({
                'Type': 'GARCH(1,1)',
                'Distribution': 'Student-t',
                'DoF (ν)': f'{nu:.2f}',
                'Current σ': f'{sigma*100:.3f}%',
                'Trend (μ)': f'{log_ret_display.mean()*100:.4f}%'
            })
        
        with col2:
            st.markdown("**Simulation Parameters**")
            st.json({
                'Method': 'Geometric Brownian Motion',
                'Paths': f'{MODEL_CONFIG["n_sims"]:,}',
                'Confidence': f'{MODEL_CONFIG["confidence"]:.0%}',
                'Time Horizon': '1 Hour',
                'Data Source': 'Binance Vision API'
            })
        
        st.markdown("**Data Window**")
        st.info(f"""
        - **Training Period**: Last {MODEL_CONFIG['train_window']} hours ({MODEL_CONFIG['train_window']/24:.1f} days)
        - **Lookback Data**: Last {MODEL_CONFIG['fetch_hours']} hours ({MODEL_CONFIG['fetch_hours']/24:.1f} days)
        - **Current Time**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
        - **Data Update Frequency**: Every 5 minutes
        """)
    
    st.divider()
    
    # ========================================================================
    # BACKTEST METRICS
    # ========================================================================
    
    st.subheader("📈 30-Day Backtest Metrics (Part A)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        coverage = BACKTEST_CONFIG['coverage_95']
        color = "🟢" if 0.93 <= coverage <= 0.97 else "🟡" if coverage > 0.97 else "🔴"
        st.metric(
            f"{color} Coverage",
            f"{coverage:.2%}",
            delta="✓ On target" if 0.93 <= coverage <= 0.97 else "⚠ Needs tuning"
        )
    
    with col2:
        st.metric(
            "Mean Width",
            f"${BACKTEST_CONFIG['mean_width']:,.2f}",
            delta=f"{(BACKTEST_CONFIG['mean_width']/current_price*100):.2f}% of price"
        )
    
    with col3:
        st.metric(
            "Winkler Score",
            f"{BACKTEST_CONFIG['mean_winkler']:.2f}",
            delta="Lower is better ↓"
        )
    
    with col4:
        st.metric(
            "Total Predictions",
            f"{BACKTEST_CONFIG['n_predictions']}",
            delta=BACKTEST_CONFIG['backtest_period']
        )
    
    st.divider()
    
    # ========================================================================
    # SAVE PREDICTION & HISTORY
    # ========================================================================
    
    st.subheader("💾 Save & Track Predictions (Part C)")
    
    if st.session_state.pop("_prediction_just_saved", False):
        st.success("✅ Prediction saved to database!")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col2:
        if st.button("💾 Save Prediction", use_container_width=True, key="save_btn"):
            try:
                save_prediction(current_price, pred_low, pred_high)
                st.session_state["_prediction_just_saved"] = True
                st.rerun()
            except Exception as e:
                st.error(f"Could not save: {e}")
    
    with col3:
        if st.button("🗑️ Clear History", use_container_width=True, key="clear_btn"):
            try:
                Path(DB_PATH).unlink()
                init_db()
                st.success("✅ History cleared!")
                st.rerun()
            except:
                st.error("Could not clear history")
    
    stats = get_live_stats()
    st.metric("Predictions in DB", stats['total_saved'])

if __name__ == "__main__":
    main()