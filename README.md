# 💲 MoneyPal — Analyse Global Financial Instruments 
<img width="940" height="399" alt="image" src="https://github.com/user-attachments/assets/753603d7-a1ab-48da-a474-5e8c0ffd3bac" />

**MoneyPal** is a comprehensive dashboard for analysing global financial instruments. It combines live market data, technical analysis with 50+ indicators, fundamental analysis, options-chain intelligence, quantitative risk metrics, volatility analysis, and strategy payoff visualisation in a polished, dark-themed UI.

Link to live terminal - https://moneypal.streamlit.app/

> ⚠️ **Disclaimer:** This project is for educational and informational purposes only. It is **not financial advice**. Always do your own research before making investment decisions.

---

## Features

### 1. Live Scrolling Ticker Bar
- Real-time price and percentage change for major Indian indices, US indices, commodities, and more.
- Fixed glass-morphism ticker bar with seamless scrolling animation.
- Auto-refresh control with cache clearing.

### 2. Cross-Market Comparison
- Compare normalised performance across multiple instruments on a single chart.
- Individual price charts with dual-axis support.
- Drawdown analysis and returns statistics summary.

### 3. Technical Analysis Engine (50+ Indicators)
- **Trend:** SMA, EMA, DEMA, TEMA, WMA, HMA, Supertrend, Ichimoku, VWAP, Parabolic SAR.
- **Momentum:** RSI, MACD, Stochastic, Stochastic RSI, Williams %R, CCI, ROC, MFI, Ultimate Oscillator, Awesome Oscillator, TSI, PPO.
- **Volatility:** Bollinger Bands, ATR, Keltner Channel, Donchian Channel, Chaikin Volatility, Historical Volatility, Normalised ATR.
- **Volume:** OBV, VWAP, A/D Line, CMF, Force Index, EFI, Volume SMA.
- **Overlay:** Pivot Points, Fibonacci Retracement.
- Interactive candlestick charts with overlays, oscillator subplots, and data tables.
<img width="940" height="352" alt="image" src="https://github.com/user-attachments/assets/85de27b5-adc5-495b-b46e-4cba39c78e14" />

### 4. Fundamental Analysis
- Key metrics cards: Market Cap, P/E, EPS, Dividend Yield, PEG, ROE, ROA, Profit Margin, Revenue Growth, Beta.
- Company profile with business summary, sector, and industry.
- Historical performance analysis and peer comparison.

### 5. Options Trading Intelligence
- **Live Option Chain** from NSE India for `NIFTY`, `BANKNIFTY`, `FINNIFTY`, `SENSEX`, `MIDCPNIFTY`, `NIFTYIT`.
  - ATM strike filtering, PCR, aggregate OI, OI change, volume, IV.
  - Styled chain table with colour-coded CALL / PUT sides.
  - OI and IV smile charts.
- **Black-Scholes Pricing & Greeks**
  - Call / Put prices, Delta, Gamma, Theta, Vega, Rho.
  - Sensitivity charts: Delta vs Spot, Gamma vs Spot, Option Price vs Spot, Theta decay.
- **Strategy Payoff Calculator**
  - Long Call, Long Put, Covered Call, Protective Put, Bull Call Spread, Bear Put Spread, Long Straddle, Long Strangle.
  - P/L zones, max profit / loss, and break-even points.

### 6. Quantitative Strategies & Risk Metrics
- Rolling correlation heatmap and risk metrics table.
- Rolling volatility, Sharpe, and drawdown analysis.
- Buy / sell signal generation based on configurable moving-average crossovers.

### 7. Volatility Analysis
- Rolling annualised volatility comparison across instruments.
- Volatility regime detection using GARCH-like rolling thresholds and volatility cones.

### 8. Market Snapshot
- Cross-asset performance bar chart (1D, 1W, 1M, 3M, 6M, 1Y).
- Heatmap-style performance cards for quick scanning.
<img width="940" height="344" alt="image" src="https://github.com/user-attachments/assets/b00e06c2-a86e-4e18-b74f-6a76c2821167" />

---

## Tech Stack

| Layer | Libraries |
|-------|-----------|
| App Framework | [Streamlit](https://streamlit.io/) |
| Data | [yfinance](https://pypi.org/project/yfinance/), [pandas](https://pandas.pydata.org/), [numpy](https://numpy.org/) |
| Technical Indicators | [pandas-ta](https://github.com/twopirllc/pandas-ta) |
| Options Math | [SciPy](https://scipy.org/) (`scipy.stats.norm`) |
| Visualisation | [Plotly](https://plotly.com/python/), [Altair](https://altair-viz.github.io/) |
| Web Data | [requests](https://requests.readthedocs.io/) (NSE option chain API) |

---

## Installation

1. **Navigate to the MoneyPal folder:**

   ```bash
   cd "p2.1"
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Run the Streamlit application from the `p2.1` folder:

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`.

### Workflow
1. **Select Asset Classes** in the sidebar (Indian Indices, Indian Stocks, US Stocks, Commodities, Crypto, International Indices).
2. **Choose Instruments** and **Time Period** for analysis.
3. **Pick Technical Indicators** from expandable groups.
4. Explore the dashboard sections:
   - Cross-Market Comparison
   - Technical Analysis Engine
   - Fundamental Analysis
   - Options Trading Intelligence
   - Quantitative Strategies & Risk Metrics
   - Volatility Analysis
   - Market Snapshot

---

## 📈 Supported Instruments

### Indian Indices
- NIFTY 50, SENSEX, BANK NIFTY, NIFTY IT, NIFTY MIDCAP 50

### Indian Stocks
- Reliance, Bandhan Bank, Kotak Bank, HDFC Bank, TCS, Infosys, ICICI Bank, SBI, Bharti Airtel, ITC, L&T

### US Stocks
- Apple, Google, Microsoft, Nvidia, Amazon, Tesla, Meta

### Commodities
- Gold, Silver, Crude Oil, Natural Gas, Copper

### Crypto
- Bitcoin, Ethereum, Solana

### International Indices
- S&P 500, Dow Jones, NASDAQ, FTSE 100, Nikkei 225

---

## 🧮 Black-Scholes Assumptions

The options pricing module uses the standard **Black-Scholes-Merton** model:

$$
d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}
$$

- $S$ = Spot price
- $K$ = Strike price
- $T$ = Time to expiry (years)
- $r$ = Risk-free rate
- $\sigma$ = Implied volatility

Greeks are computed analytically from the closed-form partial derivatives.

---

## Author

**Engineer**  
📧 contact: [ggengineerco@gmail.com](mailto:ggengineerco@gmail.com)

---

## 📜 License

This project is released for educational purposes. Use at your own risk.
