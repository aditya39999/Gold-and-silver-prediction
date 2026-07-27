# Gold and Silver Prediction

A Streamlit forecasting platform for gold and silver futures, converted from
the original notebook prototype into a standalone, deployable app.

## What it does

- Downloads historical Gold (GC=F) and Silver (SI=F) futures data via `yfinance`
- Engineers technical indicators (moving averages, RSI, MACD, Bollinger Bands, ATR, momentum, lag returns)
- Trains a Random Forest / Gradient Boosting ensemble per metal to predict next-day log returns
- Produces a recursive multi-day forecast, dashboard metrics, trading signal, and confidence score
- Includes investment calculator, portfolio optimizer, comparison, price alerts, and strategy backtest tools
- Pulls recent news headlines and scores sentiment with TextBlob
- Generates a downloadable PDF report and CSV/Excel forecast exports

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Notes

- Currency conversion uses a static, clearly labeled FX table. Update the
  `CURRENCIES` dictionary in `app.py` if you want fresher rates.
- Market data downloads and model training are cached (`st.cache_data` /
  `st.cache_resource`) so the app stays responsive after the first load.
- This tool is for educational and informational purposes only. It is not
  financial advice.

## Deploying

The app is ready to deploy as-is on [Streamlit Community Cloud](https://streamlit.io/cloud):
push `app.py` and `requirements.txt` to a GitHub repo, then point Streamlit
Cloud at `app.py` as the entry point.
