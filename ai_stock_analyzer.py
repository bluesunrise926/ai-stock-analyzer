"""
Code Gym - AI 股票趨勢分析系統（雙市場版）
==========================================
美股：使用 Financial Modeling Prep API 獲取數據
台股：使用 FinMind API 獲取數據
AI：使用 Google Gemini 2.5 Pro / 2.5 Flash 進行技術分析
圖表：使用 Plotly Graph Objects 繪製專業 K 線圖

執行方式：streamlit run ai_stock_analyzer.py
"""

# ── 標準函式庫 ──────────────────────────────
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
# 市場判斷函數
# ════════════════════════════════════════════
def detect_market(symbol: str) -> str:
    """
    自動判斷股票代碼屬於台股或美股
    台股規則：純數字代碼（如 2330、0050）
    美股規則：英文字母代碼（如 AAPL、MSFT）
    """
    cleaned = symbol.strip().upper()
    if cleaned.isdigit():
        return "TW"   # 台股
    elif cleaned.isalpha():
        return "US"   # 美股
    else:
        return "UNKNOWN"

# ════════════════════════════════════════════
# F-001: 側邊欄 UI 設計
# ════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ 分析設定")
    st.divider()

    # 市場選擇
    st.markdown("**🌏 市場選擇**")
    market_choice = st.radio(
        "選擇市場",
        options=["🇺🇸 美股", "🇹🇼 台股"],
        index=0,
        horizontal=True,
        help="選擇美股使用 FMP API，選擇台股使用 FinMind API"
    )
    is_tw = (market_choice == "🇹🇼 台股")

    # 股票代碼輸入
    if is_tw:
        symbol_raw = st.text_input(
            "📌 台股代碼",
            value="2330",
            placeholder="例如：2330、0050、2317、2454",
            help="請輸入台股代碼（純數字）"
        ).strip()
        symbol = symbol_raw
    else:
        symbol_raw = st.text_input(
            "📌 美股代碼",
            value="AAPL",
            placeholder="例如：AAPL、MSFT、GOOGL、TSLA",
            help="請輸入美股代碼（英文大寫）"
        ).strip().upper()
        symbol = symbol_raw

    st.divider()

    # ── 從 Streamlit Secrets 自動讀取 API Key ──
    _fmp_secret     = st.secrets.get("FMP_API_KEY", "")
    _gemini_secret  = st.secrets.get("GEMINI_API_KEY", "")
    _finmind_secret = st.secrets.get("FINMIND_API_TOKEN", "")

    # FMP API Key（美股專用）
    if not is_tw:
        st.markdown("**🔑 數據來源：Financial Modeling Prep**")
        if _fmp_secret:
            fmp_api_key = _fmp_secret
            st.success("🔑 FMP API Key：已從 Secrets 自動載入 ✅")
        else:
            fmp_api_key = st.text_input(
                "FMP API Key",
                type="password",
                placeholder="請輸入 FMP API Key",
                help="前往 https://financialmodelingprep.com 免費申請"
            )
    else:
        fmp_api_key = ""

    # FinMind API Token（台股專用）
    if is_tw:
        st.markdown("**🔑 數據來源：FinMind**")
        if _finmind_secret:
            finmind_token = _finmind_secret
            st.success("🔑 FinMind Token：已從 Secrets 自動載入 ✅")
        else:
            finmind_token = ""
        # 無論 Secrets 是否有值，都顯示輸入框讓使用者可手動覆蓋
        manual_token = st.text_input(
            "FinMind API Token（選填）",
            type="password",
            value="",
            help="請至 finmindtrade.com 免費註冊取得。若查詢當日數據或熱門股，未帶 Token 會被伺服器拒絕（HTTP 400）。"
        )
        if manual_token:
            finmind_token = manual_token
    else:
        finmind_token = ""

    st.divider()

    # Gemini API Key
    st.markdown("**🤖 AI 分析：Google Gemini**")
    if _gemini_secret:
        gemini_api_key = _gemini_secret
        st.success("🤖 Gemini API Key：已從 Secrets 自動載入 ✅")
    else:
        gemini_api_key = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="請輸入 Google Gemini API Key",
            help="前往 https://aistudio.google.com/apikey 免費申請"
        )

    # Gemini 模型選擇
    st.markdown("**🧠 AI 分析模型**")
    gemini_model = st.selectbox(
        "選擇 Gemini 模型",
        options=["gemini-2.5-pro", "gemini-2.5-flash"],
        index=0,
        help="gemini-2.5-pro：深度分析（需訂閱 Gemini Advanced）\ngemini-2.5-flash：快速分析（免費額度充足）"
    )

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
        "🔍 開始分析",
        type="primary",
        use_container_width=True
    )

    st.divider()

    # API 申請指引
    with st.expander("📖 API Key 申請說明"):
        st.markdown("""
**FMP API Key（美股數據）**
- 前往 [financialmodelingprep.com](https://financialmodelingprep.com)
- 免費方案每日 250 次請求

**FinMind API Token（台股數據）**
- 前往 [finmindtrade.com](https://finmindtrade.com) 免費註冊
- 免費方案每小時 600 次請求
- 不填 Token 仍可使用，但額度較低

**Gemini API Key（AI 分析）**
- 前往 [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- 免費方案可使用 Flash 模型
- 訂閱 Gemini Advanced 可使用 Pro 模型
        """)

    # 免責聲明
    st.markdown("""
---
### 📢 免責聲明
本系統僅供學術研究與教育用途，AI 提供的數據與分析結果僅供參考，**不構成投資建議或財務建議**。
請使用者自行判斷投資決策，並承擔相關風險。
""")


# ════════════════════════════════════════════
# 頁面主標題
# ════════════════════════════════════════════
market_label = "🇹🇼 台股" if is_tw else "🇺🇸 美股"
st.title(f"📈 AI 股票趨勢分析系統｜{market_label}")
st.divider()

# 歡迎說明
if not analyze_btn:
    st.markdown(f"""
### 👋 歡迎使用 AI 股票趨勢分析系統

本系統支援 **美股（FMP API）** 與 **台股（FinMind API）** 雙市場分析，結合 **Google Gemini AI** 提供專業技術分析報告。

**使用步驟：**
1. 在左側選擇**市場**（美股 / 台股）
2. 輸入**股票代碼**（美股如 `AAPL`；台股如 `2330`）
3. 若 API Key 已設定於 Streamlit Secrets，**系統將自動載入**
4. 選擇 **AI 分析模型**（Pro 深度分析 / Flash 快速分析）
5. 設定**分析期間**（預設近 90 天）
6. 點擊「🔍 開始分析」按鈕

**支援市場與數據來源：**
| 市場 | 數據來源 | 代碼範例 |
|------|---------|---------|
| 🇺🇸 美股 | Financial Modeling Prep | AAPL、MSFT、GOOGL、TSLA |
| 🇹🇼 台股 | FinMind API | 2330、0050、2317、2454 |

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
        st.error("❌ 請輸入股票代碼")
        return False

    if is_tw:
        if not symbol.isdigit():
            st.error("❌ 台股代碼格式錯誤，請輸入純數字（例如：2330、0050）")
            return False
    else:
        if not symbol.isalpha():
            st.error("❌ 美股代碼格式錯誤，請輸入英文字母（例如：AAPL）")
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
# F-002A: 美股數據獲取（FMP API）
# ════════════════════════════════════════════
@st.cache_data(ttl=300, show_spinner=False)
def get_us_stock_data(sym: str, api_key: str) -> pd.DataFrame:
    """從 Financial Modeling Prep API 獲取美股歷史數據"""
    url = (
        f"https://financialmodelingprep.com/stable/historical-price-eod/full"
        f"?symbol={sym}&apikey={api_key}"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, dict) and data.get("Error Message"):
            raise ValueError(f"API 錯誤：{data['Error Message']}")
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError(f"找不到股票代碼 '{sym}' 的數據，請確認代碼是否正確")

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date", ascending=True).reset_index(drop=True)
        # 統一欄位名稱
        df = df.rename(columns={
            "open": "open", "high": "high", "low": "low",
            "close": "close", "volume": "volume"
        })
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


# ════════════════════════════════════════════
# F-002B: 台股數據獲取（FinMind API）
# ════════════════════════════════════════════
@st.cache_data(ttl=300, show_spinner=False)
def get_tw_stock_data(sym: str, token: str,
                       s_date: str, e_date: str) -> pd.DataFrame:
    """
    從 FinMind API v4 獲取台股歷史數據
    - 不使用 raise_for_status()，改用 response.json() 解析錯誤訊息
    - token 透過 params 傳遞（非 Authorization header）
    """
    url = "https://api.finmindtrade.com/api/v4/data"
    params = {
        "dataset": "TaiwanStockPrice",
        "data_id": sym,
        "start_date": s_date,
        "end_date": e_date,
    }
    # Token 透過 params 傳遞（FinMind v4 標準方式）
    if token:
        params["token"] = token

    try:
        response = requests.get(url, params=params, timeout=20)
        # 不使用 raise_for_status()，直接解析 JSON 取得真正錯誤原因
        res_json = response.json()

        # FinMind v4 標準成功狀態碼為 200
        if response.status_code != 200 or res_json.get("status") != 200:
            server_msg = res_json.get("msg", "未知伺服器錯誤")
            raise ValueError(
                f"FinMind 拒絕請求原因：{server_msg} "
                f"(HTTP {response.status_code})。"
                f"請確認 Token 是否正確，或嘗試擴大日期範圍避開非交易日。"
            )

        records = res_json.get("data", [])
        if not records:
            raise ValueError(
                f"此時間區間內找不到 '{sym}' 的交易資料，"
                "請確認股票代碼正確，或調整日期範圍避開週末與國定假日。"
            )

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date", ascending=True).reset_index(drop=True)

        # 統一欄位名稱（FinMind 使用 max/min 與 Trading_Volume）
        df = df.rename(columns={
            "max": "high",
            "min": "low",
            "Trading_Volume": "volume",
        })

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df[["date", "open", "high", "low", "close", "volume"]]

    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"網路連線或解析失敗：{str(e)}")


def filter_by_date_range(df: pd.DataFrame,
                          s_date, e_date) -> pd.DataFrame:
    """根據日期範圍過濾數據"""
    mask = (df["date"] >= pd.Timestamp(s_date)) & \
           (df["date"] <= pd.Timestamp(e_date))
    return df[mask].copy().reset_index(drop=True)


# ════════════════════════════════════════════
# F-003: 技術指標計算函數
# ════════════════════════════════════════════
def get_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """計算 MA5、MA10、MA20、MA60 移動平均線"""
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
                          api_key: str, model: str,
                          market: str) -> str:
    """使用 Google Gemini 進行股票技術分析"""
    analysis_df = stock_data.tail(120).copy()
    analysis_df["date"] = analysis_df["date"].dt.strftime("%Y-%m-%d")

    cols = ["date", "open", "high", "low", "close", "volume",
            "MA5", "MA10", "MA20", "MA60"]
    available_cols = [c for c in cols if c in analysis_df.columns]
    data_json = analysis_df[available_cols].round(2).to_json(
        orient="records", date_format="iso"
    )

    first_date  = analysis_df["date"].iloc[0]
    last_date   = analysis_df["date"].iloc[-1]
    start_price = analysis_df["close"].iloc[0]
    end_price   = analysis_df["close"].iloc[-1]
    price_change = (end_price - start_price) / start_price * 100

    currency = "TWD（新台幣）" if market == "TW" else "USD（美元）"
    market_name = "台灣股票市場（TWSE/TPEx）" if market == "TW" else "美國股票市場（NYSE/NASDAQ）"

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
- 禁用「如果...則...」的假設句型
- 不提供具體價位的操作參考點，僅描述技術位階的歷史表現
- 強調「歷史表現不代表未來結果」

免責聲明：所提供的分析內容純粹基於歷史數據的技術解讀，僅供教育和研究參考，不構成任何投資建議或未來走勢預測。"""

    user_prompt = f"""請基於以下股票歷史數據進行深度技術分析：

### 基本資訊
- 股票代號：{sym}
- 所屬市場：{market_name}
- 計價幣別：{currency}
- 分析期間：{first_date} 至 {last_date}
- 期間價格變化：{price_change:.2f}%（從 {start_price:.2f} 變化到 {end_price:.2f}）

### 完整交易數據
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
- 技術面風險因子

分析目標：{sym}（{market_name}）"""

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
                            s_date, e_date,
                            currency_symbol: str = "$") -> go.Figure:
    """使用 Plotly Graph Objects 繪製專業 K 線圖"""
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

    # K 線圖
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="K線",
            increasing_line_color="#EF5350",
            decreasing_line_color="#26A69A",
            increasing_fillcolor="#EF5350",
            decreasing_fillcolor="#26A69A",
        ),
        row=1, col=1
    )

    # 移動平均線
    ma_config = {
        "MA5":  {"color": "#FFD700", "width": 1.5},
        "MA10": {"color": "#FF9800", "width": 1.5},
        "MA20": {"color": "#2196F3", "width": 2.0},
        "MA60": {"color": "#9C27B0", "width": 2.0},
    }
    for ma, cfg in ma_config.items():
        if ma in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df[ma],
                    name=ma,
                    line=dict(color=cfg["color"], width=cfg["width"]),
                    opacity=0.9,
                    hovertemplate=f"{ma}: %{{y:.2f}}<extra></extra>"
                ),
                row=1, col=1
            )

    # 成交量
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

    # 版面設定
    price_unit = "價格（TWD）" if currency_symbol == "NT$" else "價格（USD）"
    fig.update_layout(
        template="plotly_dark",
        height=620,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1
        ),
        margin=dict(l=50, r=30, t=80, b=30),
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False)
    fig.update_yaxes(title_text=price_unit, row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)

    return fig


# ════════════════════════════════════════════
# 主程式執行流程
# ════════════════════════════════════════════
market = "TW" if is_tw else "US"
currency_symbol = "NT$" if is_tw else "$"
currency_label  = "TWD" if is_tw else "USD"

# ── 步驟 1：獲取股票數據 ─────────────────────
if is_tw:
    st.info(f"⏳ 正在從 **FinMind** 獲取台股 **{symbol}** 的歷史數據...")
    try:
        with st.spinner("台股數據獲取中，請稍候..."):
            raw_df = get_tw_stock_data(
                symbol, finmind_token,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )
        stock_df = get_moving_averages(raw_df)
    except (ValueError, ConnectionError, TimeoutError) as e:
        st.error(f"❌ 數據獲取失敗：{str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"❌ 發生未預期的錯誤：{str(e)}")
        st.stop()
else:
    st.info(f"⏳ 正在從 **Financial Modeling Prep** 獲取美股 **{symbol}** 的歷史數據...")
    try:
        with st.spinner("美股數據獲取中，請稍候..."):
            raw_df = get_us_stock_data(symbol, fmp_api_key)
    except (ValueError, ConnectionError, TimeoutError) as e:
        st.error(f"❌ 數據獲取失敗：{str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"❌ 發生未預期的錯誤：{str(e)}")
        st.stop()

    # 美股需要再做日期過濾
    filtered_df = filter_by_date_range(raw_df, start_date, end_date)
    if filtered_df.empty:
        st.error(
            f"❌ 在 {start_date} ～ {end_date} 期間找不到 **{symbol}** 的交易數據。\n"
            "請確認日期範圍是否包含有效交易日，或嘗試擴大日期範圍。"
        )
        st.stop()
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

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        label="📅 起始價格",
        value=f"{currency_symbol}{start_price:.2f}",
        help=f"分析期間第一個交易日（{stock_df['date'].iloc[0].strftime('%Y-%m-%d')}）收盤價"
    )
with col2:
    st.metric(
        label="📅 結束價格",
        value=f"{currency_symbol}{end_price:.2f}",
        delta=f"{price_diff:+.2f} ({price_pct:+.2f}%)",
        help=f"分析期間最後一個交易日（{stock_df['date'].iloc[-1].strftime('%Y-%m-%d')}）收盤價"
    )
with col3:
    direction = "📈 上漲" if price_pct >= 0 else "📉 下跌"
    st.metric(
        label="📊 期間漲跌幅",
        value=f"{price_pct:+.2f}%",
        delta=f"{direction} {currency_symbol}{abs(price_diff):.2f}",
        delta_color="normal" if price_pct >= 0 else "inverse"
    )

col4, col5, col6 = st.columns(3)
with col4:
    st.metric("📈 期間最高價", f"{currency_symbol}{period_high:.2f}")
with col5:
    st.metric("📉 期間最低價", f"{currency_symbol}{period_low:.2f}")
with col6:
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

fig = plot_candlestick_chart(stock_df, symbol, start_date, end_date, currency_symbol)
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
    return f"{color} **{ma_name}**：{ma_val:.2f}（{arrow}{abs(diff_pct):.1f}%）"

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
            symbol, stock_df, gemini_api_key, gemini_model, market
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

display_df = stock_df.copy()
display_df = display_df.sort_values("date", ascending=False).head(10)
display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")

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

price_cols = ["開盤價", "最高價", "最低價", "收盤價", "MA5", "MA10", "MA20", "MA60"]
price_cols_exist = [c for c in price_cols if c in display_df.columns]
price_fmt = f"{currency_symbol}{{:.2f}}"

st.dataframe(
    display_df.style.format(
        {c: price_fmt for c in price_cols_exist} |
        {"成交量": "{:,.0f}"}
    ).background_gradient(
        subset=["收盤價"] if "收盤價" in display_df.columns else [],
        cmap="RdYlGn"
    ),
    use_container_width=True,
    hide_index=True
)

# 數據下載
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
