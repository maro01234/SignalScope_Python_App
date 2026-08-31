# SignalScope — Python売買シグナル調査アプリ

SMA、MACD、RSI、ボリンジャーバンドを組み合わせ、現在の判定、判定根拠、過去の売買イベント、簡易バックテストを調査するStreamlitアプリです。日本株・米国株などのYahoo Finance銘柄コード、手元のCSV、オフラインのデモデータに対応します。

## 主な機能

- 4種類の指標を `+1 / 0 / -1` で採点する透明な合成シグナル
- BUY/SELLマーカー付きローソク足、RSI、MACDチャート
- シグナル履歴の表とCSV書き出し
- 当日の判定を翌営業日から反映する、先読みを避けたロング専用バックテスト
- 片道取引コスト、累積収益、CAGR、最大ドローダウン、シャープレシオ、勝率
- 最大10銘柄の一括スキャン
- Yahoo Finance、OHLCV CSV、オフライン・デモの3データソース

## Macでの起動

Python 3.10以上を推奨します。初回はインターネット接続が必要です。

1. ZIPを展開します。
2. ターミナルで展開先へ移動します。
3. 次を実行します。

```bash
chmod +x run_app.command
./run_app.command
```

初回だけ仮想環境の作成とライブラリのインストールを行い、その後ブラウザでアプリが開きます。終了はターミナルで `Control + C` です。

通常の起動方法はこちらです。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Windowsでは仮想環境を有効にした後、最後の2行を実行してください。

## 銘柄コード例

| 対象 | 入力例 |
| --- | --- |
| トヨタ自動車 | `7203.T` |
| ソニーグループ | `6758.T` |
| Apple | `AAPL` |
| 日経平均 | `^N225` |
| USD/JPY | `JPY=X` |

Yahoo Financeに存在するコードを入力します。日足データ取得時は、選択した終了日を含めるようにアプリ側で補正しています。

## CSV形式

最低限 `Date` と `Close` が必要です。推奨形式は次の通りです。

```csv
Date,Open,High,Low,Close,Volume
2026-08-20,1000,1020,990,1015,500000
2026-08-21,1012,1035,1005,1030,620000
```

Open、High、LowがなければCloseで補完し、Volumeがなければ0で補完します。

## 判定ロジック

| 指標 | +1 | 0 | -1 |
| --- | --- | --- | --- |
| SMA | 短期 > 長期 | 同値 | 短期 < 長期 |
| MACD | MACD > Signal | 同値 | MACD < Signal |
| RSI | 売られすぎ | 中立 | 買われすぎ |
| Bollinger | 終値 ≤ 下限 | バンド内 | 終値 ≥ 上限 |

ノーポジション時に合計スコアが正の閾値以上ならBUY、保有中に負の閾値以下ならSELLです。閾値を高くすると判定回数が減り、条件が厳しくなります。

## テスト

```bash
source .venv/bin/activate
python -m pytest -q
```

テストでは、指標列とスコア範囲、将来価格が過去判定を変えないこと、バックテストが前営業日のポジションを使うこと、入力検証を確認します。

## 重要な注意

本アプリは教育・研究用です。投資助言、売買の自動執行、将来収益の保証を行うものではありません。バックテストは税金、スリッページ、約定不能、出来高制約を考慮せず、データ品質にも依存します。実際の取引では企業業績、決算日、ニュース、流動性、注文条件も確認してください。

データ取得は [yfinance](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)、画面表示は [Streamlit](https://docs.streamlit.io/) を利用します。
