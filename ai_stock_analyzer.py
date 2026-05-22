"""
Code Gym - AI 股票趨勢分析系統
================================
使用 Financial Modeling Prep API 獲取股票數據
使用 Google Gemini 2.5 Pro / 2.5 Flash 進行 AI 技術分析
使用 Plotly Graph Objects 繪製專業 K 線圖

執行方式：streamlit run ai_stock_analyzer.py
"""

# ── 標準函式庫 ──────────────────────────────
import json
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ── 第三方套件 ──────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.genai as genai
from google.genai import types as gtypes

# ════════════════════════════════════════════
# 頁面基本設定
# ════════════════════════════════════════════
st.set_page_config(
    page_title="AI 股票趨勢分析系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════
# F-001: 側邊欄 UI 設計
# ════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ 分析設定")
    st.divider()

    # 股票代碼輸入
    symbol = st.text_input(
        "📌 股票代碼",
        value="AAPL",
        placeholder="例如：AAPL、MSFT、GOOGL、TSLA",
        help="請輸入美股股票代碼（英文大寫）"
    ).strip().upper()

    # FMP API Key
    fmp_api_key = st.text_input(
        "🔑 FMP API Key",
        type="password",
        placeholder="請輸入 Financial Modeling Prep API Key",
        help="前往 https://financialmodelingprep.com 免費申請"
    )

    # Gemini API Key
    gemini_api_key = st.text_input(
        "🤖 Gemini API Key",
        type="password",
        placeholder="請輸入 Google Gemini API Key",
        help="前往 https://aistudio.google.com/apikey 免費申請（訂閱 Gemini Advanced 可使用 Pro 模型）"
    )

    # Gemini 模型選擇
    st.markdown("**🧠 AI 分析模型**")
    gemini_model = st.selectbox(
        "選擇 Gemini 模型",
        options=["gemini-2.5-pro", "gemini-2.5-flash"],
        index=0,
        help="gemini-2.5-pro：深度分析（需訂閱 Gemini Advanced）\ngemini-2.5-flash：快速分析（免費額度充足）"
    )

    # 模型說明標籤
    if gemini_model == "gemini-2.5-pro":
        st.success("✨ Pro 模型：深度推理，分析更精細")
    else:
        st.info("⚡ Flash 模型：高速回應，性價比最佳")

    st.divider()

    # 日期範圍選擇
    st.markdown("**📅 分析期間**")
    default_end   = datetime.today().date()
    default_start = (datetime.today() - timedelta(days=90)).date()

    start_date = st.date_input(
        "起始日期",
        value=default_start,
        max_value=default_end,
        help="分析起始日期（預設為 90 天前）"
    )
    end_date = st.date_input(
        "結束日期",
        value=default_end,
        min_value=start_date,
        max_value=default_end,
        help="分析結束日期（預設為今天）"
    )

    st.divider()

    # 執行按鈕
    analyze_btn = st.button(
        "🔍 分析",
        type="primary",
        use_container_width=True
    )

    st.divider()

    # API 申請指引
    with st.expander("📖 API Key 申請說明"):
        st.markdown("""
**FMP API Key（股票數據）**
- 前往 [financialmodelingprep.com](https://financialmodelingprep.com)
- 免費方案每日 250 次請求

**Gemini API Key（AI 分析）**
- 前往 [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- 免費方案可使用 Flash 模型
- 訂閱 Gemini Advanced 可使用 Pro 模型
        """)

    # F-009: 免責聲明
    st.markdown("""
---
### 📢 免責聲明
本系統僅供學術研究與教育用途，AI 提供的數據與分析結果僅供參考，**不構成投資建議或財務建議**。
請使用者自行判斷投資決策，並承擔相關風險。本系統作者不對任何投資行為負責，亦不承擔任何損失責任。
""")


# ════════════════════════════════════════════
# 頁面主標題
# ════════════════════════════════════════════
st.title("📈 AI 股票趨勢分析系統")
st.divider()

# 歡迎說明
if not analyze_btn:
    st.markdown("""
### 👋 歡迎使用 AI 股票趨勢分析系統

本系統整合 **Financial Modeling Prep** 股票數據與 **Google Gemini AI**，提供專業的技術分析報告。

**使用步驟：**
1. 在左側輸入**股票代碼**（如 `AAPL`、`MSFT`、`GOOGL`）
2. 輸入 **FMP API Key** 和 **Gemini API Key**
3. 選擇 **AI 分析模型**（Pro 深度分析 / Flash 快速分析）
4. 設定**分析期間**（預設近 90 天）
5. 點擊「🔍 分析」按鈕

**技術指標說明：**
| 指標 | 說明 |
|------|------|
| MA5  | 5 日移動平均線，反映短期趨勢 |
| MA10 | 10 日移動平均線，反映短中期趨勢 |
| MA20 | 20 日移動平均線，反映中期趨勢 |
| MA60 | 60 日移動平均線，反映長期趨勢 |
    """)
    st.stop()


# ════════════════════════════════════════════
# F-008: 輸入驗證
# ════════════════════════════════════════════
def validate_inputs() -> bool:
    """驗證所有必填輸入項目"""
    if not symbol:
        st.error("❌ 請輸入股票代碼（例如：AAPL、MSFT、GOOGL）")
        return False
    if not symbol.isalpha():
        st.error("❌ 股票代碼格式錯誤，請輸入英文字母（例如：AAPL）")
        return False
    if not fmp_api_key:
        st.error("❌ 請輸入 FMP API Key。前往 https://financialmodelingprep.com 免費申請")
        return False
    if not gemini_api_key:
        st.error("❌ 請輸入 Gemini API Key。前往 https://aistudio.google.com/apikey 免費申請")
        return False
    if start_date >= end_date:
        st.error("❌ 起始日期必須早於結束日期，請重新選擇")
        return False
    days_diff = (end_date - start_date).days
    if days_diff > 365 * 3:
        st.warning("⚠️ 分析期間超過 3 年，數據量較大，處理時間可能較長")
    return True

if not validate_inputs():
    st.stop()


# ════════════════════════════════════════════
# F-002: 數據獲取函數
# ════════════════════════════════════════════
@st.cache_data(ttl=300, show_spinner=False)
def get_stock_data(sym: str, api_key: str) -> pd.DataFrame:
    """
    從 Financial Modeling Prep API 獲取股票完整歷史數據
    API: https://financialmodelingprep.com/stable/historical-price-eod/full
    """
    url = (
        f"https://financialmodelingprep.com/stable/historical-price-eod/full"
        f"?symbol={sym}&apikey={api_key}"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        # 檢查 API 回傳格式
        if isinstance(data, dict) and data.get("Error Message"):
            raise ValueError(f"API 錯誤：{data['Error Message']}")
        if isinstance(data, dict) and "historicalStockList" in data:
            # 部分方案回傳格式不同
            raise ValueError("您的 FMP 方案不支援此 API，請升級或確認 API Key 正確")
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError(f"找不到股票代碼 '{sym}' 的數據，請確認代碼是否正確")

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date", ascending=True).reset_index(drop=True)
        return df

    except requests.exceptions.ConnectionError:
        raise ConnectionError("網路連線失敗，請檢查網路連線後重試")
    except requests.exceptions.Timeout:
        raise TimeoutError("API 請求逾時，請稍後重試")
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 401:
            raise ValueError("FMP API Key 無效或已過期，請確認金鑰是否正確")
        elif resp.status_code == 403:
            raise ValueError("FMP API Key 權限不足，請確認方案是否支援此功能")
        else:
            raise ValueError(f"API 請求失敗（HTTP {resp.status_code}）：{str(e)}")


def filter_by_date_range(df: pd.DataFrame,
                          s_date: datetime,
                          e_date: datetime) -> pd.DataFrame:
    """根據日期範圍過濾數據"""
    mask = (df["date"] >= pd.Timestamp(s_date)) & \
           (df["date"] <= pd.Timestamp(e_date))
    filtered = df[mask].copy().reset_index(drop=True)
    return filtered


# ════════════════════════════════════════════
# F-003: 技術指標計算函數
# ════════════════════════════════════════════
def get_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算 MA5、MA10、MA20、MA60 移動平均線
    使用 Pandas rolling 函數，min_periods=1 確保數據不足時仍有值
    """
    result = df.copy()
    for window in [5, 10, 20, 60]:
        result[f"MA{window}"] = (
            result["close"]
            .rolling(window=window, min_periods=1)
            .mean()
            .round(4)
        )
    return result


# ════════════════════════════════════════════
# F-006: AI 分析函數
# ════════════════════════════════════════════
def generate_ai_insights(sym: str, stock_data: pd.DataFrame,
                          api_key: str, model: str) -> str:
    """
    使用 Google Gemini 進行股票技術分析
    model: 'gemini-2.5-pro' 或 'gemini-2.5-flash'
    """
    # 準備分析數據（最多傳送 120 筆，避免 Token 過多）
    analysis_df = stock_data.tail(120).copy()
    analysis_df["date"] = analysis_df["date"].dt.strftime("%Y-%m-%d")

    # 選取關鍵欄位
    cols = ["date", "open", "high", "low", "close", "volume",
            "MA5", "MA10", "MA20", "MA60"]
    available_cols = [c for c in cols if c in analysis_df.columns]
    data_json = analysis_df[available_cols].round(2).to_json(
        orient="records", date_format="iso"
    )

    # 基本統計
    first_date  = analysis_df["date"].iloc[0]
    last_date   = analysis_df["date"].iloc[-1]
    start_price = analysis_df["close"].iloc[0]
    end_price   = analysis_df["close"].iloc[-1]
    price_change = (end_price - start_price) / start_price * 100

    # ── System Message ──────────────────────
    system_message = """你是一位專業的技術分析師，專精於股票技術分析和歷史數據解讀。你的職責包括：

1. 客觀描述股票價格的歷史走勢和技術指標狀態
2. 解讀歷史市場數據和交易量變化模式
3. 識別技術面的歷史支撐阻力位
4. 提供純教育性的技術分析知識

重要原則：
- 僅提供歷史數據分析和技術指標解讀，絕不提供任何投資建議或預測
- 保持完全客觀中立的分析態度
- 使用專業術語但保持易懂
- 所有分析僅供教育和研究目的
- 強調技術分析的局限性和不確定性
- 使用繁體中文回答

嚴格的表達方式要求：
- 使用「歷史數據顯示」、「技術指標反映」、「過去走勢呈現」等客觀描述
- 避免「可能性」、「預期」、「建議」、「關注」等暗示性用詞
- 禁用「如果...則...」的假設句型，改用「歷史上當...時，曾出現...現象」
- 不提供具體價位的操作參考點，僅描述技術位階的歷史表現
- 強調「歷史表現不代表未來結果」
- 避免任何可能被解讀為操作指引的表達

免責聲明：所提供的分析內容純粹基於歷史數據的技術解讀，僅供教育和研究參考，不構成任何投資建議或未來走勢預測。歷史表現不代表未來結果。"""

    # ── User Prompt ─────────────────────────
    user_prompt = f"""請基於以下股票歷史數據進行深度技術分析：

### 基本資訊
- 股票代號：{sym}
- 分析期間：{first_date} 至 {last_date}
- 期間價格變化：{price_change:.2f}%（從 ${start_price:.2f} 變化到 ${end_price:.2f}）

### 完整交易數據
以下是該期間的完整交易數據，包含日期、開盤價、最高價、最低價、收盤價、成交量和移動平均線：
{data_json}

### 分析架構：技術面完整分析

#### 1. 趨勢分析
- 整體趨勢方向（上升、下降、盤整）
- 關鍵支撐位和阻力位識別
- 趨勢強度評估

#### 2. 技術指標分析
- 移動平均線分析（短期與長期 MA 的關係）
- 價格與移動平均線的相對位置
- 成交量與價格變動的關聯性

#### 3. 價格行為分析
- 重要的價格突破點
- 波動性評估
- 關鍵的轉折點識別

#### 4. 風險評估
- 當前價位的風險等級
- 潛在的支撐和阻力區間
- 市場情緒指標

#### 5. 市場觀察
- 短期技術面觀察（1-2 週）
- 中期技術面觀察（1-3 個月）
- 關鍵價位觀察點
- 技術面風險因子

### 綜合評估要求
#### 輸出格式要求
- 條理清晰，分段論述
- 提供具體的數據支撐
- 避免過於絕對的預測，強調分析的局限性
- 在適當位置使用表格或重點標記

分析目標：{sym}"""

    # ── 呼叫 Gemini API ──────────────────────
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=[
                gtypes.Content(
                    role="user",
                    parts=[gtypes.Part(text=f"{system_message}\n\n{user_prompt}")]
                )
            ],
        )
        return response.text
    except Exception as e:
        err = str(e)
        if "API_KEY_INVALID" in err or "invalid" in err.lower():
            raise ValueError("Gemini API Key 無效，請確認金鑰是否正確")
        elif "quota" in err.lower() or "RESOURCE_EXHAUSTED" in err:
            raise ValueError("Gemini API 配額已用盡，請稍後再試或切換至 Flash 模型")
        elif "permission" in err.lower() or "PERMISSION_DENIED" in err:
            raise ValueError(
                f"Gemini API 權限不足：{gemini_model} 需要訂閱 Gemini Advanced，"
                "請切換至 gemini-2.5-flash 或確認訂閱狀態"
            )
        else:
            raise ValueError(f"Gemini AI 分析失敗：{err}")


# ════════════════════════════════════════════
# F-004: K 線圖繪製函數
# ════════════════════════════════════════════
def plot_candlestick_chart(df: pd.DataFrame, sym: str,
                            s_date, e_date) -> go.Figure:
    """
    使用 Plotly Graph Objects 繪製專業 K 線圖
    包含 MA5、MA10、MA20、MA60 移動平均線與成交量
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.03,
        subplot_titles=(
            f"{sym} 股價 K 線圖（{s_date} ～ {e_date}）",
            "成交量"
        )
    )

    # ── K 線圖 ───────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="K線",
            increasing_line_color="#EF5350",   # 上漲：紅色
            decreasing_line_color="#26A69A",   # 下跌：綠色
            increasing_fillcolor="#EF5350",
            decreasing_fillcolor="#26A69A",
        ),
        row=1, col=1
    )

    # ── 移動平均線 ───────────────────────────
    ma_config = {
        "MA5":  {"color": "#FFD700", "width": 1.5, "dash": "solid"},
        "MA10": {"color": "#FF9800", "width": 1.5, "dash": "solid"},
        "MA20": {"color": "#2196F3", "width": 2.0, "dash": "solid"},
        "MA60": {"color": "#9C27B0", "width": 2.0, "dash": "solid"},
    }
    for ma, cfg in ma_config.items():
        if ma in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df[ma],
                    name=ma,
                    line=dict(
                        color=cfg["color"],
                        width=cfg["width"],
                        dash=cfg["dash"]
                    ),
                    opacity=0.9,
                    hovertemplate=f"{ma}: %{{y:.2f}}<extra></extra>"
                ),
                row=1, col=1
            )

    # ── 成交量長條圖 ─────────────────────────
    vol_colors = [
        "#EF5350" if c >= o else "#26A69A"
        for c, o in zip(df["close"], df["open"])
    ]
    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=df["volume"],
            name="成交量",
            marker_color=vol_colors,
            opacity=0.7,
            hovertemplate="成交量: %{y:,.0f}<extra></extra>"
        ),
        row=2, col=1
    )

    # ── 圖表版面設定 ─────────────────────────
    fig.update_layout(
        template="plotly_dark",
        height=620,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1
        ),
        margin=dict(l=50, r=30, t=80, b=30),
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False
    )
    fig.update_yaxes(title_text="價格（USD）", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)

    return fig


# ════════════════════════════════════════════
# 主程式執行流程
# ════════════════════════════════════════════

# ── 步驟 1：獲取股票數據 ─────────────────────
st.info(f"⏳ 正在從 Financial Modeling Prep 獲取 **{symbol}** 的歷史數據...")

try:
    with st.spinner("數據獲取中，請稍候..."):
        raw_df = get_stock_data(symbol, fmp_api_key)
except (ValueError, ConnectionError, TimeoutError) as e:
    st.error(f"❌ 數據獲取失敗：{str(e)}")
    st.stop()
except Exception as e:
    st.error(f"❌ 發生未預期的錯誤：{str(e)}")
    st.stop()

# ── 步驟 2：日期範圍過濾 ─────────────────────
filtered_df = filter_by_date_range(raw_df, start_date, end_date)

if filtered_df.empty:
    st.error(
        f"❌ 在 {start_date} ～ {end_date} 期間找不到 **{symbol}** 的交易數據。\n"
        "請確認日期範圍是否包含有效交易日，或嘗試擴大日期範圍。"
    )
    st.stop()

# ── 步驟 3：計算技術指標 ─────────────────────
st.info("⚙️ 正在計算技術指標（MA5、MA10、MA20、MA60）...")
stock_df = get_moving_averages(filtered_df)

data_count = len(stock_df)
if data_count < 20:
    st.warning(
        f"⚠️ 分析期間內僅有 {data_count} 筆交易數據，"
        "部分移動平均線（MA20、MA60）可能因數據不足而不夠準確，建議擴大分析期間。"
    )

st.success(
    f"✅ 成功獲取 **{symbol}** 共 **{data_count}** 筆交易數據"
    f"（{stock_df['date'].iloc[0].strftime('%Y-%m-%d')} ～ "
    f"{stock_df['date'].iloc[-1].strftime('%Y-%m-%d')}）"
)

# ════════════════════════════════════════════
# F-005: 基本統計資訊展示
# ════════════════════════════════════════════
st.subheader(f"📊 {symbol} 基本統計資訊")

start_price  = stock_df["close"].iloc[0]
end_price    = stock_df["close"].iloc[-1]
price_diff   = end_price - start_price
price_pct    = price_diff / start_price * 100
period_high  = stock_df["high"].max()
period_low   = stock_df["low"].min()
avg_volume   = stock_df["volume"].mean()
latest_ma20  = stock_df["MA20"].iloc[-1]
latest_ma60  = stock_df["MA60"].iloc[-1] if data_count >= 10 else None

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="📅 起始價格",
        value=f"${start_price:.2f}",
        help=f"分析期間第一個交易日（{stock_df['date'].iloc[0].strftime('%Y-%m-%d')}）收盤價"
    )

with col2:
    st.metric(
        label="📅 結束價格",
        value=f"${end_price:.2f}",
        delta=f"{price_diff:+.2f} ({price_pct:+.2f}%)",
        help=f"分析期間最後一個交易日（{stock_df['date'].iloc[-1].strftime('%Y-%m-%d')}）收盤價"
    )

with col3:
    direction = "📈 上漲" if price_pct >= 0 else "📉 下跌"
    st.metric(
        label="📊 期間漲跌幅",
        value=f"{price_pct:+.2f}%",
        delta=f"{direction} ${abs(price_diff):.2f}",
        delta_color="normal" if price_pct >= 0 else "inverse"
    )

# 第二列：更多統計
col4, col5, col6 = st.columns(3)
with col4:
    st.metric("📈 期間最高價", f"${period_high:.2f}")
with col5:
    st.metric("📉 期間最低價", f"${period_low:.2f}")
with col6:
    # 成交量格式化
    if avg_volume >= 1e9:
        vol_str = f"{avg_volume/1e9:.2f}B"
    elif avg_volume >= 1e6:
        vol_str = f"{avg_volume/1e6:.2f}M"
    elif avg_volume >= 1e3:
        vol_str = f"{avg_volume/1e3:.2f}K"
    else:
        vol_str = f"{avg_volume:.0f}"
    st.metric("📦 平均日成交量", vol_str)

st.divider()

# ════════════════════════════════════════════
# F-004: K 線圖與技術指標
# ════════════════════════════════════════════
st.subheader("📉 股價 K 線圖與技術指標")

fig = plot_candlestick_chart(stock_df, symbol, start_date, end_date)
st.plotly_chart(fig, use_container_width=True)

# 均線多空狀態說明
latest = stock_df.iloc[-1]
st.markdown("**📌 最新均線狀態**")
ma_col1, ma_col2, ma_col3, ma_col4 = st.columns(4)

def ma_status(price, ma_val, ma_name):
    if pd.isna(ma_val):
        return f"**{ma_name}**：資料不足"
    diff_pct = (price - ma_val) / ma_val * 100
    arrow = "▲" if price > ma_val else "▼"
    color = "🟢" if price > ma_val else "🔴"
    return f"{color} **{ma_name}**：${ma_val:.2f}（{arrow}{abs(diff_pct):.1f}%）"

ma_col1.markdown(ma_status(latest["close"], latest.get("MA5"),  "MA5"))
ma_col2.markdown(ma_status(latest["close"], latest.get("MA10"), "MA10"))
ma_col3.markdown(ma_status(latest["close"], latest.get("MA20"), "MA20"))
ma_col4.markdown(ma_status(latest["close"], latest.get("MA60"), "MA60"))

st.divider()

# ════════════════════════════════════════════
# F-006: AI 技術分析
# ════════════════════════════════════════════
st.subheader(f"🤖 AI 技術分析報告（{gemini_model}）")

with st.spinner(f"AI 正在分析中，使用 {gemini_model}，請稍候（約 20-40 秒）..."):
    try:
        ai_report = generate_ai_insights(
            symbol, stock_df, gemini_api_key, gemini_model
        )
        st.success("✅ AI 分析完成")
        st.markdown(ai_report)
    except ValueError as e:
        st.error(f"❌ AI 分析失敗：{str(e)}")
        st.info("💡 建議：若使用 gemini-2.5-pro 遇到權限問題，請切換至 gemini-2.5-flash")
    except Exception as e:
        st.error(f"❌ AI 分析發生未預期錯誤：{str(e)}")

st.divider()

# ════════════════════════════════════════════
# F-007: 歷史數據表格
# ════════════════════════════════════════════
st.subheader("📋 最近 10 筆交易數據")

# 按日期降序排列，取最近 10 筆
display_df = stock_df.copy()
display_df = display_df.sort_values("date", ascending=False).head(10)
display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")

# 選取顯示欄位並重命名
show_cols = {
    "date":   "日期",
    "open":   "開盤價",
    "high":   "最高價",
    "low":    "最低價",
    "close":  "收盤價",
    "volume": "成交量",
    "MA5":    "MA5",
    "MA10":   "MA10",
    "MA20":   "MA20",
    "MA60":   "MA60",
}
available = {k: v for k, v in show_cols.items() if k in display_df.columns}
display_df = display_df[list(available.keys())].rename(columns=available)

# 格式化數值
price_cols = ["開盤價", "最高價", "最低價", "收盤價", "MA5", "MA10", "MA20", "MA60"]
price_cols_exist = [c for c in price_cols if c in display_df.columns]

st.dataframe(
    display_df.style.format(
        {c: "${:.2f}" for c in price_cols_exist} |
        {"成交量": "{:,.0f}"}
    ).background_gradient(
        subset=["收盤價"] if "收盤價" in display_df.columns else [],
        cmap="RdYlGn"
    ),
    use_container_width=True,
    hide_index=True
)

# ── 數據下載 ─────────────────────────────────
st.divider()
full_display = stock_df.copy()
full_display["date"] = full_display["date"].dt.strftime("%Y-%m-%d")
csv_data = full_display.to_csv(index=False, encoding="utf-8-sig")

st.download_button(
    label=f"💾 下載 {symbol} 完整數據（CSV）",
    data=csv_data,
    file_name=f"{symbol}_{start_date}_{end_date}_analysis.csv",
    mime="text/csv",
    help="下載包含所有技術指標的完整歷史數據"
)
