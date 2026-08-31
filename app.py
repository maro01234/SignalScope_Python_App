"""Streamlit user interface for technical signal research."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data_source import download_prices, generate_sample_prices, load_price_csv
from signal_engine import StrategyConfig, backtest, component_snapshot, generate_signals


st.set_page_config(
    page_title="SignalScope | 売買シグナル調査",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(160deg, #07101d 0%, #0b1424 55%, #101b2e 100%);}
    [data-testid="stSidebar"] {background: #07101d; border-right: 1px solid #20324a;}
    .hero {padding: 1.25rem 1.5rem; border: 1px solid #223955; border-radius: 18px;
           background: linear-gradient(120deg, rgba(22,45,73,.95), rgba(10,20,36,.95));
           box-shadow: 0 12px 35px rgba(0,0,0,.22); margin-bottom: 1rem;}
    .hero h1 {margin: 0; letter-spacing: .02em; font-size: 2.1rem;}
    .hero p {margin: .45rem 0 0; color: #a9bad0;}
    .signal-card {padding: 1.1rem 1.25rem; border-radius: 14px; border-left: 5px solid #5aa7ff;
                  background: rgba(18,32,52,.82); margin: .4rem 0 1rem;}
    .signal-card h3 {margin: 0 0 .35rem;}
    .muted {color: #9db0c6; font-size: .92rem;}
    div[data-testid="stMetric"] {background: rgba(15,29,48,.72); border: 1px solid #203652;
                                 padding: .8rem 1rem; border-radius: 13px;}
    .footer-note {font-size: .82rem; color: #8497ad; padding-top: 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=900, show_spinner=False)
def cached_download(ticker: str, start: date, end: date) -> pd.DataFrame:
    return download_prices(ticker, start, end)


def format_price(value: float) -> str:
    decimals = 0 if abs(value) >= 1_000 else 2
    return f"{value:,.{decimals}f}"


def action_label(latest: pd.Series) -> tuple[str, str]:
    if latest["Signal"] == "BUY":
        return "買い転換", "新たに買い条件が成立"
    if latest["Signal"] == "SELL":
        return "売り転換", "保有解除条件が成立"
    if int(latest["Position"]) == 1:
        return "保有継続", "過去の買い条件を維持"
    return "様子見", "現在はノーポジション"


def technical_chart(frame: pd.DataFrame, symbol: str) -> go.Figure:
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.58, 0.20, 0.22],
        subplot_titles=(f"{symbol} 価格・移動平均・ボリンジャーバンド", "RSI", "MACD"),
    )
    figure.add_trace(
        go.Candlestick(
            x=frame.index,
            open=frame["Open"], high=frame["High"], low=frame["Low"], close=frame["Close"],
            name="OHLC",
            increasing_line_color="#37d99b", decreasing_line_color="#ff6b7a",
        ), row=1, col=1,
    )
    figure.add_trace(go.Scatter(x=frame.index, y=frame["SMA_Fast"], name="短期SMA", line=dict(color="#55b6ff", width=1.5)), row=1, col=1)
    figure.add_trace(go.Scatter(x=frame.index, y=frame["SMA_Slow"], name="長期SMA", line=dict(color="#ffbf69", width=1.5)), row=1, col=1)
    figure.add_trace(go.Scatter(x=frame.index, y=frame["BB_Upper"], name="BB上限", line=dict(color="#778da9", width=1, dash="dot")), row=1, col=1)
    figure.add_trace(go.Scatter(x=frame.index, y=frame["BB_Lower"], name="BB下限", line=dict(color="#778da9", width=1, dash="dot"), fill="tonexty", fillcolor="rgba(119,141,169,.08)"), row=1, col=1)

    buys = frame.loc[frame["Signal"] == "BUY"]
    sells = frame.loc[frame["Signal"] == "SELL"]
    figure.add_trace(go.Scatter(x=buys.index, y=buys["Low"] * 0.985, mode="markers", name="BUY", marker=dict(symbol="triangle-up", size=13, color="#34d399")), row=1, col=1)
    figure.add_trace(go.Scatter(x=sells.index, y=sells["High"] * 1.015, mode="markers", name="SELL", marker=dict(symbol="triangle-down", size=13, color="#fb7185")), row=1, col=1)

    figure.add_trace(go.Scatter(x=frame.index, y=frame["RSI"], name="RSI", line=dict(color="#c084fc", width=1.6)), row=2, col=1)
    figure.add_hline(y=70, line_dash="dot", line_color="#fb7185", row=2, col=1)
    figure.add_hline(y=30, line_dash="dot", line_color="#34d399", row=2, col=1)
    histogram_colors = ["#34d399" if value >= 0 else "#fb7185" for value in frame["MACD_Hist"].fillna(0)]
    figure.add_trace(go.Bar(x=frame.index, y=frame["MACD_Hist"], name="MACD差", marker_color=histogram_colors, opacity=.55), row=3, col=1)
    figure.add_trace(go.Scatter(x=frame.index, y=frame["MACD"], name="MACD", line=dict(color="#55b6ff", width=1.5)), row=3, col=1)
    figure.add_trace(go.Scatter(x=frame.index, y=frame["MACD_Signal"], name="シグナル", line=dict(color="#ffbf69", width=1.3)), row=3, col=1)
    figure.update_layout(
        height=820, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(7,16,29,.45)",
        margin=dict(l=20, r=20, t=60, b=20), hovermode="x unified", xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    figure.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], gridcolor="#1c2e45")
    figure.update_yaxes(gridcolor="#1c2e45")
    figure.update_yaxes(range=[0, 100], row=2, col=1)
    return figure


def equity_chart(frame: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=frame.index, y=(frame["StrategyEquity"] - 1) * 100, name="戦略", line=dict(color="#55b6ff", width=2.2)))
    figure.add_trace(go.Scatter(x=frame.index, y=(frame["BuyHoldEquity"] - 1) * 100, name="買い持ち", line=dict(color="#a9bad0", width=1.5, dash="dot")))
    figure.update_layout(
        height=430, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(7,16,29,.45)",
        margin=dict(l=20, r=20, t=35, b=20), hovermode="x unified", yaxis_title="累積リターン（%）",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    figure.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], gridcolor="#1c2e45")
    figure.update_yaxes(gridcolor="#1c2e45")
    return figure


def make_config() -> StrategyConfig:
    return StrategyConfig(
        sma_fast=int(st.session_state.sma_fast),
        sma_slow=int(st.session_state.sma_slow),
        ema_fast=int(st.session_state.ema_fast),
        ema_slow=int(st.session_state.ema_slow),
        macd_signal=int(st.session_state.macd_signal),
        rsi_period=int(st.session_state.rsi_period),
        rsi_oversold=float(st.session_state.rsi_oversold),
        rsi_overbought=float(st.session_state.rsi_overbought),
        bb_period=int(st.session_state.bb_period),
        bb_std=float(st.session_state.bb_std),
        score_threshold=int(st.session_state.score_threshold),
        fee_bps=float(st.session_state.fee_bps),
    )


with st.sidebar:
    st.markdown("## ⚙️ 分析条件")
    source = st.radio("データソース", ["Yahoo Finance", "CSVアップロード", "オフライン・デモ"])
    with st.form("data_form"):
        if source == "Yahoo Finance":
            ticker = st.text_input("銘柄コード", "7203.T", help="日本株は例: 7203.T、米国株は例: AAPL")
            date_columns = st.columns(2)
            start = date_columns[0].date_input("開始日", date.today() - timedelta(days=3 * 365))
            end = date_columns[1].date_input("終了日", date.today())
            uploaded = None
        elif source == "CSVアップロード":
            uploaded = st.file_uploader("OHLCV CSV", type=["csv"], help="DateとCloseは必須です")
            ticker = "CSV"
            start = date.today() - timedelta(days=3 * 365)
            end = date.today()
        else:
            ticker = "DEMO"
            start = date.today() - timedelta(days=3 * 365)
            end = date.today()
            uploaded = None
            st.caption("ネット接続なしで動作する疑似価格データです。")
        load_button = st.form_submit_button("価格データを読み込む", type="primary", width="stretch")

    st.markdown("### 指標パラメータ")
    sma_columns = st.columns(2)
    sma_columns[0].number_input("短期SMA", min_value=2, max_value=200, value=20, key="sma_fast")
    sma_columns[1].number_input("長期SMA", min_value=3, max_value=400, value=50, key="sma_slow")
    ema_columns = st.columns(2)
    ema_columns[0].number_input("短期EMA", min_value=2, max_value=100, value=12, key="ema_fast")
    ema_columns[1].number_input("長期EMA", min_value=3, max_value=200, value=26, key="ema_slow")
    st.number_input("MACDシグナル", min_value=2, max_value=100, value=9, key="macd_signal")
    st.number_input("RSI期間", min_value=2, max_value=100, value=14, key="rsi_period")
    rsi_columns = st.columns(2)
    rsi_columns[0].number_input("RSI売られすぎ", min_value=1, max_value=49, value=30, key="rsi_oversold")
    rsi_columns[1].number_input("RSI買われすぎ", min_value=51, max_value=99, value=70, key="rsi_overbought")
    bb_columns = st.columns(2)
    bb_columns[0].number_input("BB期間", min_value=2, max_value=200, value=20, key="bb_period")
    bb_columns[1].number_input("BB標準偏差", min_value=.5, max_value=4.0, value=2.0, step=.1, key="bb_std")
    st.slider("売買スコア閾値", min_value=1, max_value=4, value=2, key="score_threshold", help="高いほどシグナルが厳選されます")
    st.number_input("片道コスト（bps）", min_value=0.0, max_value=500.0, value=10.0, step=1.0, key="fee_bps", help="10 bps = 0.10%")

if load_button:
    try:
        with st.spinner("価格データを読み込んでいます…"):
            if source == "Yahoo Finance":
                loaded_prices = cached_download(ticker.strip().upper(), start, end)
                loaded_symbol = ticker.strip().upper()
            elif source == "CSVアップロード":
                if uploaded is None:
                    raise ValueError("CSVファイルを選択してください。")
                loaded_prices = load_price_csv(uploaded)
                loaded_symbol = uploaded.name
            else:
                loaded_prices = generate_sample_prices()
                loaded_symbol = "DEMO"
        st.session_state["prices"] = loaded_prices
        st.session_state["symbol"] = loaded_symbol
        st.session_state["source_name"] = source
    except Exception as exc:
        st.error(f"読み込みに失敗しました: {exc}")

if "prices" not in st.session_state:
    st.session_state["prices"] = generate_sample_prices()
    st.session_state["symbol"] = "DEMO"
    st.session_state["source_name"] = "オフライン・デモ"

st.markdown(
    """
    <div class="hero">
      <h1>SignalScope</h1>
      <p>テクニカル指標を組み合わせて、売買シグナルの根拠と過去成績を同じ画面で調査します。</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    config = make_config()
    signals = generate_signals(st.session_state["prices"], config)
    tested, metrics, trades = backtest(signals, config)
except Exception as exc:
    st.error(f"分析条件を確認してください: {exc}")
    st.stop()

symbol = st.session_state["symbol"]
latest = signals.iloc[-1]
previous_close = signals["Close"].iloc[-2]
daily_change = latest["Close"] / previous_close - 1
action, action_detail = action_label(latest)

metric_columns = st.columns(5)
metric_columns[0].metric("現在の判定", action)
metric_columns[1].metric("終値", format_price(float(latest["Close"])), f"{daily_change:+.2%}")
metric_columns[2].metric("合成スコア", f"{int(latest['Score']):+d} / 4")
metric_columns[3].metric("RSI", f"{latest['RSI']:.1f}")
metric_columns[4].metric("戦略累積", f"{metrics['total_return']:+.1%}", f"買い持ち {metrics['benchmark_return']:+.1%}", delta_color="normal")

tabs = st.tabs(["現在の判定", "テクニカルチャート", "シグナル履歴", "バックテスト", "複数銘柄スキャン", "使い方"])

with tabs[0]:
    st.markdown(
        f"""
        <div class="signal-card">
          <h3>{action} <span class="muted">— {action_detail}</span></h3>
          <div>基準日: {signals.index[-1].date()}　合成スコア: {int(latest['Score']):+d}　現在ポジション: {'保有' if latest['Position'] == 1 else 'なし'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([1, 1.3])
    with left:
        st.subheader("指標別の判定")
        st.dataframe(component_snapshot(signals), hide_index=True, width="stretch")
    with right:
        st.subheader("判定根拠")
        st.write(latest["Reason"])
        st.caption("各指標を +1 / 0 / -1 で採点し、合計が閾値に達したときだけ売買を切り替えます。")
        recent_events = signals.loc[signals["Signal"].isin(["BUY", "SELL"]), ["Close", "Signal", "Score", "Reason"]].tail(5).copy()
        recent_events.index = recent_events.index.date
        st.dataframe(recent_events.rename(columns={"Close": "終値", "Signal": "イベント", "Score": "点数", "Reason": "根拠"}), width="stretch")

with tabs[1]:
    chart_options = sorted(set([126, 252, 504, 756, len(signals)]))
    display_rows = st.select_slider("表示期間（営業日）", options=chart_options, value=min(504, len(signals)), key="chart_rows")
    st.plotly_chart(technical_chart(signals.tail(display_rows), symbol), width="stretch", config={"displaylogo": False})

with tabs[2]:
    events = signals.loc[signals["Signal"].isin(["BUY", "SELL"]), ["Close", "Signal", "Score", "RSI", "Reason"]].copy()
    events.index.name = "Date"
    shown_events = events.reset_index().sort_values("Date", ascending=False).rename(columns={"Date": "日付", "Close": "終値", "Signal": "売買", "Score": "点数", "Reason": "根拠"})
    st.dataframe(shown_events, hide_index=True, width="stretch", column_config={"RSI": st.column_config.NumberColumn(format="%.1f"), "終値": st.column_config.NumberColumn(format="%.2f")})
    st.download_button("シグナル履歴をCSV保存", events.to_csv().encode("utf-8-sig"), file_name=f"{symbol}_signals.csv", mime="text/csv")

with tabs[3]:
    backtest_columns = st.columns(6)
    backtest_columns[0].metric("累積リターン", f"{metrics['total_return']:+.1%}")
    backtest_columns[1].metric("年率換算", f"{metrics['cagr']:+.1%}")
    backtest_columns[2].metric("最大ドローダウン", f"{metrics['max_drawdown']:.1%}")
    backtest_columns[3].metric("シャープレシオ", f"{metrics['sharpe']:.2f}")
    backtest_columns[4].metric("エントリー数", str(metrics["trade_count"]))
    backtest_columns[5].metric("勝率（決済済み）", f"{metrics['win_rate']:.1%}")
    st.plotly_chart(equity_chart(tested), width="stretch", config={"displaylogo": False})
    st.caption("仮定: ロングのみ、シグナルは当日終値で確定し翌営業日から反映、税・スリッページは除外、指定した片道コストを売買ごとに控除。配当は調整価格に依存します。")
    if not trades.empty:
        trade_view = trades.copy()
        trade_view["EntryDate"] = trade_view["EntryDate"].dt.date
        trade_view["ExitDate"] = trade_view["ExitDate"].dt.date
        trade_view = trade_view.rename(columns={"EntryDate": "買付日", "ExitDate": "売却/基準日", "EntryPrice": "買値", "ExitPrice": "売値/現在値", "GrossReturn": "粗収益率", "NetReturn": "コスト後収益率", "Bars": "保有営業日", "Status": "状態"})
        st.dataframe(trade_view, hide_index=True, width="stretch", column_config={"粗収益率": st.column_config.NumberColumn(format="percent"), "コスト後収益率": st.column_config.NumberColumn(format="percent")})
    else:
        st.info("選択期間には成立した取引がありません。")

with tabs[4]:
    st.write("最大10銘柄を同じ設定で比較します。日本株は `.T` を付けてください。")
    scan_text = st.text_input("銘柄コード（カンマ区切り）", "7203.T, 6758.T, AAPL, MSFT")
    if st.button("スキャン開始", type="primary"):
        symbols = list(dict.fromkeys(item.strip().upper() for item in scan_text.split(",") if item.strip()))[:10]
        rows = []
        progress = st.progress(0, text="価格を取得しています…")
        for index, scan_symbol in enumerate(symbols, start=1):
            try:
                scan_prices = cached_download(scan_symbol, date.today() - timedelta(days=3 * 365), date.today())
                scan_signals = generate_signals(scan_prices, config)
                _, scan_metrics, _ = backtest(scan_signals, config)
                scan_latest = scan_signals.iloc[-1]
                scan_action, _ = action_label(scan_latest)
                rows.append({"銘柄": scan_symbol, "基準日": scan_signals.index[-1].date(), "判定": scan_action, "点数": int(scan_latest["Score"]), "終値": float(scan_latest["Close"]), "RSI": float(scan_latest["RSI"]), "戦略累積": scan_metrics["total_return"], "最大DD": scan_metrics["max_drawdown"], "エラー": ""})
            except Exception as exc:
                rows.append({"銘柄": scan_symbol, "エラー": str(exc)})
            progress.progress(index / max(len(symbols), 1), text=f"{index}/{len(symbols)} 完了")
        progress.empty()
        st.session_state["scan_result"] = pd.DataFrame(rows)
    if "scan_result" in st.session_state:
        scan_result = st.session_state["scan_result"]
        if "点数" in scan_result.columns:
            scan_result = scan_result.sort_values("点数", ascending=False, na_position="last")
        st.dataframe(scan_result, hide_index=True, width="stretch", column_config={"戦略累積": st.column_config.NumberColumn(format="percent"), "最大DD": st.column_config.NumberColumn(format="percent"), "RSI": st.column_config.NumberColumn(format="%.1f"), "終値": st.column_config.NumberColumn(format="%.2f")})

with tabs[5]:
    st.markdown(
        """
        ### 判定ロジック

        - 短期SMAが長期SMAより上なら `+1`、下なら `-1`
        - MACDがシグナルより上なら `+1`、下なら `-1`
        - RSIが売られすぎなら `+1`、買われすぎなら `-1`、それ以外は `0`
        - 終値がボリンジャー下限以下なら `+1`、上限以上なら `-1`、バンド内は `0`
        - ノーポジション時に合計が正の閾値以上でBUY、保有中に負の閾値以下でSELL

        ### 入力例

        日本株は `7203.T`、米国株は `AAPL`、指数は `^N225` など、Yahoo Financeの銘柄コードを使います。
        CSVでは `Date, Open, High, Low, Close, Volume` を推奨し、最低限 `Date` と `Close` が必要です。

        ### 注意

        これは過去データを調査する教育・研究用ツールで、投資助言や将来収益の保証ではありません。実際の売買前には、企業業績、流動性、決算日、ニュース、注文条件も別途確認してください。
        """
    )

st.markdown(f"<div class='footer-note'>データ: {st.session_state['source_name']} / 表示銘柄: {symbol} / 教育・研究用途</div>", unsafe_allow_html=True)
