"""Price data adapters for Yahoo Finance, CSV files, and an offline demo."""

from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO
from typing import BinaryIO

import numpy as np
import pandas as pd

from signal_engine import prepare_ohlcv


def normalize_yahoo_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize both single-level and MultiIndex yfinance responses."""

    if raw is None or raw.empty:
        raise ValueError(f"{ticker} の価格データを取得できませんでした。")

    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        last_level = frame.columns.get_level_values(-1)
        first_level = frame.columns.get_level_values(0)
        if ticker in last_level:
            frame = frame.xs(ticker, axis=1, level=-1, drop_level=True)
        elif ticker in first_level:
            frame = frame.xs(ticker, axis=1, level=0, drop_level=True)
        else:
            known = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
            if known.intersection(first_level):
                frame.columns = first_level
            else:
                frame.columns = last_level
    frame = frame.rename(columns={"Adj Close": "Close"})
    frame = frame.loc[:, ~frame.columns.duplicated(keep="first")]
    return prepare_ohlcv(frame)


def download_prices(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Download adjusted daily prices with yfinance.

    yfinance treats end as exclusive, so one calendar day is added to include
    the date selected in the UI.
    """

    symbol = ticker.strip().upper()
    if not symbol:
        raise ValueError("銘柄コードを入力してください。")
    if start >= end:
        raise ValueError("開始日は終了日より前にしてください。")

    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinanceが未インストールです。requirements.txtをインストールしてください。") from exc

    kwargs = {
        "tickers": symbol,
        "start": start.isoformat(),
        "end": (end + timedelta(days=1)).isoformat(),
        "interval": "1d",
        "auto_adjust": True,
        "progress": False,
        "threads": False,
        "group_by": "column",
    }
    try:
        raw = yf.download(**kwargs, multi_level_index=False)
    except TypeError:
        raw = yf.download(**kwargs)
    return normalize_yahoo_frame(raw, symbol)


def load_price_csv(uploaded_file: BinaryIO | BytesIO) -> pd.DataFrame:
    """Load a CSV with Date plus OHLCV columns (Close is mandatory)."""

    raw = pd.read_csv(uploaded_file)
    if raw.empty:
        raise ValueError("CSVが空です。")

    date_candidates = [column for column in raw.columns if str(column).strip().lower() in {"date", "datetime", "日付"}]
    if not date_candidates:
        raise ValueError("CSVに Date、Datetime、または日付列が必要です。")
    date_column = date_candidates[0]
    raw[date_column] = pd.to_datetime(raw[date_column], errors="coerce")
    raw = raw.set_index(date_column)
    column_map = {
        column: str(column).strip().title()
        for column in raw.columns
    }
    raw = raw.rename(columns=column_map)
    return prepare_ohlcv(raw)


def generate_sample_prices(days: int = 756, seed: int = 24) -> pd.DataFrame:
    """Create deterministic OHLCV demo data for offline exploration."""

    if days < 80:
        raise ValueError("サンプル期間は80営業日以上にしてください。")
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    regimes = np.resize(np.array([0.0008] * 90 + [-0.0006] * 60 + [0.00025] * 80), days)
    shocks = rng.normal(regimes, 0.016, days)
    close = 1_500 * np.exp(np.cumsum(shocks))
    open_ = close * (1 + rng.normal(0, 0.004, days))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.015, days))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.015, days))
    volume = rng.integers(150_000, 2_000_000, days)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=index,
    )

