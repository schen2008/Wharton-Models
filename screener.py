import yfinance as yf
import pandas as pd
import numpy as np
import time
from tqdm import tqdm  # install with: pip install tqdm


# ============================================================
# Step 1: Load all NASDAQ + NYSE tickers
# ============================================================
def load_tickers():
    nasdaq_url = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
    otherlisted_url = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"

    # Load NASDAQ-listed tickers
    nasdaq = pd.read_csv(nasdaq_url, sep="|")
    nasdaq = nasdaq[nasdaq["Test Issue"] == "N"]

    # Load NYSE + other-listed tickers
    other = pd.read_csv(otherlisted_url, sep="|")
    other = other[(other["Test Issue"] == "N") & (other["Exchange"] == "N")]  # Only NYSE

    # Combine NASDAQ + NYSE
    tickers = pd.concat([nasdaq["Symbol"], other["ACT Symbol"]]).dropna().unique().tolist()
    print(f"✅ Loaded {len(tickers)} tickers from NASDAQ + NYSE.")
    return tickers


# ============================================================
# Step 2: Fetch financial data
# ============================================================
def get_metrics(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        market_cap = info.get("marketCap")
        country = info.get("country", "")
        sector = info.get("sector", "Unknown")

        # Only include U.S. companies over $2B
        if not market_cap or market_cap < 2e9 or country != "United States":
            return None

        fcf = info.get("freeCashflow")
        fcf_yield = (fcf / market_cap) if (fcf and market_cap) else np.nan

        return {
            "Ticker": ticker,
            "Sector": sector,
            "Market Cap": market_cap,
            # Profitability
            "ROE": info.get("returnOnEquity"),
            "Profit Margin": info.get("profitMargins"),
            "Operating Margin": info.get("operatingMargins"),
            "ROA": info.get("returnOnAssets"),
            # Valuation
            "FCF Yield": fcf_yield,
            "PEG": info.get("pegRatio"),
            "EV/EBITDA": info.get("enterpriseToEbitda"),
            "Price/Book": info.get("priceToBook"),
            # Growth
            "EPS Growth": info.get("earningsQuarterlyGrowth"),
            # Risk/Liquidity
            "Debt/Equity": info.get("debtToEquity"),
            "Quick Ratio": info.get("quickRatio"),
            "Beta": info.get("beta")
        }
    except Exception:
        return None


# ============================================================
# Step 3: Scoring logic
# ============================================================
def score_metric(value, weight, inverse=False):
    if value is None or pd.isna(value):
        return 0
    if inverse:
        return weight * (1 / (value + 1e-6))
    return weight * value


def score_stock(metrics):
    scores = {}
    # Profitability (26 pts)
    scores["ROE"] = score_metric(metrics.get("ROE"), 6)
    scores["Profit Margin"] = score_metric(metrics.get("Profit Margin"), 6)
    scores["Operating Margin"] = score_metric(metrics.get("Operating Margin"), 7)
    scores["ROA"] = score_metric(metrics.get("ROA"), 7)
    # Valuation (26 pts)
    scores["FCF Yield"] = score_metric(metrics.get("FCF Yield"), 7)
    scores["PEG"] = score_metric(metrics.get("PEG"), 6, inverse=True)
    scores["EV/EBITDA"] = score_metric(metrics.get("EV/EBITDA"), 7, inverse=True)
    scores["Price/Book"] = score_metric(metrics.get("Price/Book"), 6, inverse=True)
    # Growth (18 pts)
    scores["EPS Growth"] = score_metric(metrics.get("EPS Growth"), 18)
    # Risk/Liquidity (30 pts)
    scores["Debt/Equity"] = score_metric(metrics.get("Debt/Equity"), 10, inverse=True)
    scores["Quick Ratio"] = score_metric(metrics.get("Quick Ratio"), 10)
    scores["Beta"] = score_metric(metrics.get("Beta"), 10, inverse=True)

    total_score = round(min(sum(scores.values()), 100.0), 2)
    return total_score, scores


# ============================================================
# Step 4: Main process
# ============================================================
def main(top_per_sector=10):
    tickers = load_tickers()
    results = []

    print(f"\n🔍 Scanning {len(tickers)} NASDAQ + NYSE tickers...\n")
    for ticker in tqdm(tickers, desc="Fetching & scoring", unit="ticker"):
        metrics = get_metrics(ticker)
        if metrics:
            total, breakdown = score_stock(metrics)
            for key, val in breakdown.items():
                metrics[f"Score_{key}"] = val
            metrics["Total Score"] = total
            results.append(metrics)
        time.sleep(0.2)  # Prevent Yahoo rate limits

    df = pd.DataFrame(results)
    df = df[df["Sector"].notna()]

    print(f"\n✅ Found {len(df)} U.S. stocks above $2B market cap.")

    # Sort and select top per sector
    top_by_sector = (
        df.sort_values(["Sector", "Total Score"], ascending=[True, False])
        .groupby("Sector")
        .head(top_per_sector)
        .reset_index(drop=True)
    )

    # Save all results
    df.to_csv("all_us_stocks_over_2b.csv", index=False)

    # Save top stocks by sector
    top_by_sector.to_csv("top_by_sector_with_scores.csv", index=False)

    # Write explanation
    with open("score_explanation.txt", "w") as f:
        f.write("""
SCORE CALCULATION METHOD (Total = 100 points max)

Profitability (26 pts total)
  ROE ................. ×6
  Profit Margin ........ ×6
  Operating Margin ..... ×7
  ROA .................. ×7

Valuation (26 pts total)
  FCF Yield ............ ×7
  PEG (inverse) ........ ×6
  EV/EBITDA (inverse) .. ×7
  Price/Book (inverse) . ×6

Growth (18 pts total)
  EPS Growth ........... ×18

Risk & Liquidity (30 pts total)
  Debt/Equity (inverse)  ×10
  Quick Ratio .......... ×10
  Beta (inverse) ....... ×10

Notes:
- "inverse" = lower is better (score = weight / (value + 1e-6))
- Missing values = 0
- Final Total Score = min(sum(scores), 100)
""")

    print("\n✅ Export complete!")
    print("📊 'all_us_stocks_over_2b.csv' — full dataset")
    print("🏆 'top_by_sector_with_scores.csv' — top 10 per sector")
    print("📘 'score_explanation.txt' — scoring details")

    return top_by_sector


# ============================================================
# Run script
# ============================================================
if __name__ == "__main__":
    top_stocks = main(top_per_sector=10)

