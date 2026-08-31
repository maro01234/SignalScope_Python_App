"""Technical indicator, signal, and backtest engine.

The module contains no UI or network code, so its calculations can be tested
independently and reused from notebooks or other applications.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    """Parameters for the composite long-only strategy."""

    sma_fast: int = 20
    sma_slow: int = 50
    ema_fast: int = 12
    ema_slow: int = 26
    macd_signal: int = 9
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    bb_period: int = 20
    bb_std: float = 2.0
    score_threshold: int = 2
    fee_bps: float = 10.0

    def __post_init__(self) -> None:
        integer_periods = (
            self.sma_fast,
            self.sma_slow,
            self.ema_fast,
            self.ema_slow,
            self.macd_signal,
            self.rsi_period,
            self.bb_period,
        )
        if any(period < 2 for period in integer_periods):
            raise ValueError("指標の期間は2以上にしてください。")
        if self.sma_fast >= self.sma_slow:
            raise ValueError("短期SMAは長期SMAより短くしてください。")
        if self.ema_fast >= self.ema_slow:
            raise ValueError("短期EMAは長期EMAより短くしてください。")
        if not 0 < self.rsi_oversold < self.rsi_overbought < 100:
            raise ValueError("RSI閾値は 0 < 売られすぎ < 買われすぎ < 100 にしてください。")
        if self.bb_std <= 0:
            raise ValueError("ボリンジャーバンドの標準偏差倍率は正数にしてください。")
        if not 1 <= self.score_threshold <= 4:
            raise ValueError("スコア閾値は1〜4にしてください。")
        if self.fee_bps < 0:
            raise ValueError("取引コストは0以上にしてください。")

    @property
    def minimum_history(self) -> int:
        return max(
            self.sma_slow,
            self.ema_slow + self.macd_signal,
            self.rsi_period,
            self.bb_period,
        ) + 5


def prepare_ohlcv(prices: pd.DataFrame) -> pd.DataFrame:
    """Normalize an OHLCV DataFrame and perform basic validation."""

    if prices is None or prices.empty:
        raise ValueError("価格データが空です。")

    frame = prices.copy()
    frame.columns = [str(column).strip().title() for column in frame.columns]
    if "Close" not in frame.columns:
        raise ValueError("Close列が必要です。")

    for column in ("Open", "High", "Low"):
        if column not in frame.columns:
            frame[column] = frame["Close"]
    if "Volume" not in frame.columns:
        frame["Volume"] = 0.0

    wanted = ["Open", "High", "Low", "Close", "Volume"]
    frame = frame[wanted]
    for column in wanted:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame.loc[~frame.index.isna()]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame = frame.dropna(subset=["Close"])
    frame[["Open", "High", "Low"]] = frame[["Open", "High", "Low"]].fillna(
        frame["Close"], axis=0
    )
    frame["Volume"] = frame["Volume"].fillna(0.0)

    if frame.empty:
        raise ValueError("有効な終値データがありません。")
    return frame


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder-style RSI using exponentially smoothed gains and losses."""

    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    relative_strength = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + relative_strength))
    result = result.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    result = result.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    result = result.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    return result


def add_indicators(prices: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Add all strategy indicators to a normalized price DataFrame."""

    frame = prepare_ohlcv(prices)
    if len(frame) < config.minimum_history:
        raise ValueError(
            f"データが不足しています。最低{config.minimum_history}営業日分が必要です。"
        )

    close = frame["Close"]
    frame["SMA_Fast"] = close.rolling(config.sma_fast).mean()
    frame["SMA_Slow"] = close.rolling(config.sma_slow).mean()
    frame["EMA_Fast"] = close.ewm(span=config.ema_fast, adjust=False).mean()
    frame["EMA_Slow"] = close.ewm(span=config.ema_slow, adjust=False).mean()
    frame["MACD"] = frame["EMA_Fast"] - frame["EMA_Slow"]
    frame["MACD_Signal"] = frame["MACD"].ewm(
        span=config.macd_signal, adjust=False
    ).mean()
    frame["MACD_Hist"] = frame["MACD"] - frame["MACD_Signal"]
    frame["RSI"] = _rsi(close, config.rsi_period)
    frame["BB_Middle"] = close.rolling(config.bb_period).mean()
    rolling_std = close.rolling(config.bb_period).std(ddof=0)
    frame["BB_Upper"] = frame["BB_Middle"] + config.bb_std * rolling_std
    frame["BB_Lower"] = frame["BB_Middle"] - config.bb_std * rolling_std
    return frame


def _reason_text(row: pd.Series, config: StrategyConfig) -> str:
    if row["TrendScore"] > 0:
        trend = "短期線が長期線を上回る"
    elif row["TrendScore"] < 0:
        trend = "短期線が長期線を下回る"
    else:
        trend = "短期線と長期線が同値"
    if row["MACDScore"] > 0:
        macd = "MACDがシグナルを上回る"
    elif row["MACDScore"] < 0:
        macd = "MACDがシグナルを下回る"
    else:
        macd = "MACDとシグナルが同値"
    if row["RSIScore"] > 0:
        rsi = f"RSIが売られすぎ圏（≤{config.rsi_oversold:g}）"
    elif row["RSIScore"] < 0:
        rsi = f"RSIが買われすぎ圏（≥{config.rsi_overbought:g}）"
    else:
        rsi = "RSIは中立圏"
    if row["BBScore"] > 0:
        band = "終値がボリンジャー下限以下"
    elif row["BBScore"] < 0:
        band = "終値がボリンジャー上限以上"
    else:
        band = "終値はボリンジャーバンド内"
    return " / ".join((trend, macd, rsi, band))


def generate_signals(prices: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Calculate component scores and stateful BUY/SELL events.

    The strategy is long-only. A BUY is emitted while flat when the composite
    score reaches the positive threshold. A SELL is emitted while invested
    when it reaches the negative threshold. Otherwise the prior position is
    maintained.
    """

    frame = add_indicators(prices, config)
    ready_columns = [
        "SMA_Fast",
        "SMA_Slow",
        "MACD",
        "MACD_Signal",
        "RSI",
        "BB_Upper",
        "BB_Lower",
    ]
    frame["Ready"] = frame[ready_columns].notna().all(axis=1)

    frame["TrendScore"] = np.select(
        [frame["SMA_Fast"] > frame["SMA_Slow"], frame["SMA_Fast"] < frame["SMA_Slow"]],
        [1, -1],
        default=0,
    )
    frame["MACDScore"] = np.select(
        [frame["MACD"] > frame["MACD_Signal"], frame["MACD"] < frame["MACD_Signal"]],
        [1, -1],
        default=0,
    )
    frame["RSIScore"] = np.select(
        [frame["RSI"] <= config.rsi_oversold, frame["RSI"] >= config.rsi_overbought],
        [1, -1],
        default=0,
    )
    frame["BBScore"] = np.select(
        [frame["Close"] <= frame["BB_Lower"], frame["Close"] >= frame["BB_Upper"]],
        [1, -1],
        default=0,
    )
    score_columns = ["TrendScore", "MACDScore", "RSIScore", "BBScore"]
    frame.loc[~frame["Ready"], score_columns] = 0
    frame["Score"] = frame[score_columns].sum(axis=1).astype(int)
    frame["Confidence"] = frame["Score"].abs() / 4.0

    position = 0
    positions: list[int] = []
    events: list[str] = []
    for ready, score in zip(frame["Ready"], frame["Score"], strict=True):
        event = "WAIT" if not ready else "HOLD"
        if ready and position == 0 and score >= config.score_threshold:
            position = 1
            event = "BUY"
        elif ready and position == 1 and score <= -config.score_threshold:
            position = 0
            event = "SELL"
        positions.append(position)
        events.append(event)

    frame["Signal"] = events
    frame["Position"] = positions
    frame["Reason"] = "指標計算待ち"
    ready_mask = frame["Ready"]
    frame.loc[ready_mask, "Reason"] = frame.loc[ready_mask].apply(
        _reason_text, axis=1, config=config
    )
    return frame


def extract_trades(signals: pd.DataFrame, fee_bps: float = 0.0) -> pd.DataFrame:
    """Pair BUY and SELL events into a trade ledger."""

    trades: list[dict[str, Any]] = []
    entry_date: pd.Timestamp | None = None
    entry_price: float | None = None
    one_way_cost = fee_bps / 10_000.0

    for date, row in signals.iterrows():
        if row["Signal"] == "BUY" and entry_date is None:
            entry_date = pd.Timestamp(date)
            entry_price = float(row["Close"])
        elif row["Signal"] == "SELL" and entry_date is not None and entry_price is not None:
            exit_price = float(row["Close"])
            gross_return = exit_price / entry_price - 1
            net_return = (exit_price / entry_price) * (1 - one_way_cost) ** 2 - 1
            trades.append(
                {
                    "EntryDate": entry_date,
                    "ExitDate": pd.Timestamp(date),
                    "EntryPrice": entry_price,
                    "ExitPrice": exit_price,
                    "GrossReturn": gross_return,
                    "NetReturn": net_return,
                    "Bars": int(signals.loc[entry_date:date].shape[0] - 1),
                    "Status": "CLOSED",
                }
            )
            entry_date = None
            entry_price = None

    if entry_date is not None and entry_price is not None:
        final_date = pd.Timestamp(signals.index[-1])
        final_price = float(signals["Close"].iloc[-1])
        trades.append(
            {
                "EntryDate": entry_date,
                "ExitDate": final_date,
                "EntryPrice": entry_price,
                "ExitPrice": final_price,
                "GrossReturn": final_price / entry_price - 1,
                "NetReturn": (final_price / entry_price) * (1 - one_way_cost) - 1,
                "Bars": int(signals.loc[entry_date:].shape[0] - 1),
                "Status": "OPEN",
            }
        )

    columns = [
        "EntryDate",
        "ExitDate",
        "EntryPrice",
        "ExitPrice",
        "GrossReturn",
        "NetReturn",
        "Bars",
        "Status",
    ]
    return pd.DataFrame(trades, columns=columns)


def _safe_sharpe(returns: pd.Series) -> float:
    volatility = returns.std(ddof=0)
    if volatility == 0 or np.isnan(volatility):
        return 0.0
    return float(np.sqrt(252) * returns.mean() / volatility)


def _cagr(equity: pd.Series) -> float:
    if equity.empty or equity.iloc[-1] <= 0:
        return 0.0
    calendar_days = max((equity.index[-1] - equity.index[0]).days, 1)
    years = calendar_days / 365.25
    return float(equity.iloc[-1] ** (1 / years) - 1)


def backtest(signals: pd.DataFrame, config: StrategyConfig) -> tuple[pd.DataFrame, dict[str, float | int], pd.DataFrame]:
    """Backtest signals without applying today's decision to today's return."""

    frame = signals.copy()
    frame["MarketReturn"] = frame["Close"].pct_change().fillna(0.0)
    frame["HeldPosition"] = frame["Position"].shift(1).fillna(0).astype(int)
    frame["Turnover"] = frame["Position"].diff().abs().fillna(frame["Position"].abs())
    frame["TradingCost"] = frame["Turnover"] * (config.fee_bps / 10_000.0)
    frame["StrategyReturn"] = frame["HeldPosition"] * frame["MarketReturn"] - frame["TradingCost"]
    frame["StrategyEquity"] = (1 + frame["StrategyReturn"]).cumprod()
    frame["BuyHoldEquity"] = (1 + frame["MarketReturn"]).cumprod()
    running_peak = frame["StrategyEquity"].cummax()
    frame["Drawdown"] = frame["StrategyEquity"] / running_peak - 1

    trades = extract_trades(frame, config.fee_bps)
    closed_trades = trades.loc[trades["Status"] == "CLOSED"] if not trades.empty else trades
    win_rate = (
        float((closed_trades["NetReturn"] > 0).mean())
        if not closed_trades.empty
        else 0.0
    )
    metrics: dict[str, float | int] = {
        "total_return": float(frame["StrategyEquity"].iloc[-1] - 1),
        "benchmark_return": float(frame["BuyHoldEquity"].iloc[-1] - 1),
        "cagr": _cagr(frame["StrategyEquity"]),
        "max_drawdown": float(frame["Drawdown"].min()),
        "sharpe": _safe_sharpe(frame["StrategyReturn"]),
        "trade_count": int((frame["Signal"] == "BUY").sum()),
        "closed_trade_count": int(len(closed_trades)),
        "win_rate": win_rate,
    }
    return frame, metrics, trades


def component_snapshot(signals: pd.DataFrame) -> pd.DataFrame:
    """Return a Japanese-labelled explanation table for the latest bar."""

    latest = signals.iloc[-1]
    rows = [
        ("SMAトレンド", int(latest["TrendScore"]), "短期SMAと長期SMA"),
        ("MACD", int(latest["MACDScore"]), "MACDラインとシグナル"),
        ("RSI", int(latest["RSIScore"]), "買われすぎ・売られすぎ"),
        ("ボリンジャーバンド", int(latest["BBScore"]), "終値と上下バンド"),
    ]
    labels = {1: "買い寄り", 0: "中立", -1: "売り寄り"}
    return pd.DataFrame(
        [
            {"指標": name, "判定": labels[score], "点数": score, "比較内容": basis}
            for name, score, basis in rows
        ]
    )
