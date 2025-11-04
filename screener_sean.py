#!/usr/bin/env python3
# industry_grouped_growth_tilted_fast.py
#
# ✔ FAST
# ✔ Clean console (no delisted warnings)
# ✔ Skips bad tickers instantly
# ✔ Cache + --refresh
# ✔ PNG chart + Excel Top 10
#
# Requirements:
#   pip install yfinance pandas numpy openpyxl tqdm matplotlib

import io
import os
import re
import sys
import time
import pickle
import urllib.request
import warnings
from typing import List

warnings.filterwarnings("ignore")   # <-- remove yfinance spam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import yfinance as yf

# ------------------------------------------------------
# SETTINGS
# ------------------------------------------------------
CACHE_FILE   = "fundamentals_cache.pkl"
SKIP_LOG     = "skipped_symbols.txt"
OUT_EXCEL    = "growth_fast_top10.xlsx"
CHART_FILE   = "ranked_results_chart.png"

USE_CACHE    = True     # set False or run with --refresh to refetch
EXPORT_CHART = True

EXCHANGES_ALLOWED = {"NYSE", "NasdaqGS", "NasdaqGM", "NasdaqCM"}
SECTORS_ALLOWED   = {"Financials", "Industrials", "Energy", "Materials", "Technology", "Consumer Staples"}
MIN_MARKET_CAP    = 2_000_000_000
MIN_PRICE         = 5.0

# Weights:
W_PROF = 26
W_VAL  = 26
W_GROW = 18
W_RISK = 30

PROF_METRICS = ["roe", "profit_margin", "operating_margin", "roa"]
VAL_METRICS  = ["fcf_yield", "peg_effective", "ev_ebitda", "pb"]
RISK_METRICS = ["debt_to_equity", "quick_ratio", "beta"]

BAD_SUFFIXES = (".U", ".W", ".R", "-U", "-W", "-R")

# ------------------------------------------------------
def safe_div(a, b):
    try:
        if b is None or b == 0 or pd.isna(b):
            return np.nan
        return a / b
    except:
        return np.nan

def normalize_symbol_to_yahoo(sym: str) -> str:
    s = sym.strip()

    # convert BRK.B → BRK-B, RDS.A → RDS-A
    if "." in s:
        s = s.replace(".", "-")

    # convert IVR$C → IVR-PRC
    if "$" in s:
        base, ser = s.split("$", 1)
        ser = re.sub(r"[^A-Za-z0-9]", "", ser).upper()
        if ser:
            return f"{base}-PR{ser}"
        return base

    return s.replace(" ", "")

# ------------------------------------------------------
def fetch_symbol_lists() -> pd.DataFrame:
    nasdaq_url = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
    other_url  = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"

    def load(url):
        with urllib.request.urlopen(url) as resp:
            return pd.read_csv(io.BytesIO(resp.read()), sep="|")

    nasdaq = load(nasdaq_url)
    other  = load(other_url)

    nasdaq = nasdaq[nasdaq["Test Issue"] == "N"].copy()
    other  = other[(other["Test Issue"] == "N") & (other["Exchange"] == "N")].copy()

    nasdaq["Symbol"]    = nasdaq["Symbol"].astype(str).str.strip()
    other["ACT Symbol"] = other["ACT Symbol"].astype(str).str.strip()

    df = pd.concat([
        pd.DataFrame({"symbol_raw": nasdaq["Symbol"]}),
        pd.DataFrame({"symbol_raw": other["ACT Symbol"]})
    ], ignore_index=True).drop_duplicates("symbol_raw")

    # remove known non-stock classes
    df = df[~df["symbol_raw"].str.endswith(BAD_SUFFIXES)].copy()

    df["symbol"] = df["symbol_raw"].apply(normalize_symbol_to_yahoo)
    return df.drop_duplicates("symbol")

# ------------------------------------------------------
def fetch_fundamentals(symbols: List[str]) -> pd.DataFrame:
    if USE_CACHE and os.path.exists(CACHE_FILE):
        print("✅ Loading cached fundamentals...")
        return pickle.load(open(CACHE_FILE, "rb"))

    print("⏳ Downloading fundamentals from Yahoo Finance...")
    rows = []
    skipped = []

    for sym in tqdm(symbols, desc="Pulling", unit="stk"):
        # quick skip of junk tickers
        if sym.endswith(BAD_SUFFIXES):
            skipped.append((sym, "unit/warrant/rights"))
            continue

        ysym = normalize_symbol_to_yahoo(sym)

        try:
            info = yf.Ticker(ysym).info
        except:
            skipped.append((ysym, "info fetch failed"))
            continue

        price = info.get("regularMarketPrice") or info.get("previousClose")
        if price is None or pd.isna(price):
            skipped.append((ysym, "no price / likely delisted"))
            continue

        rows.append({
            "symbol": ysym,
            "name": info.get("longName") or info.get("shortName"),
            "price": price,
            "exchange": info.get("exchange"),
            "sector": info.get("sector"),
            "industry": info.get("industry") or "Unknown",
            "marketCap": info.get("marketCap"),
            "roe": info.get("returnOnEquity"),
            "profit_margin": info.get("profitMargins"),
            "operating_margin": info.get("operatingMargins"),
            "roa": info.get("returnOnAssets"),
            "fcf_yield": safe_div(info.get("freeCashflow"), info.get("marketCap")),
            "ev_ebitda": safe_div(info.get("enterpriseValue"), info.get("ebitda")),
            "peg_effective": info.get("pegRatio") or safe_div(info.get("trailingPE"), info.get("earningsGrowth")),
            "pb": info.get("priceToBook"),
            "eps_growth": info.get("earningsGrowth"),
            "debt_to_equity": info.get("debtToEquity"),
            "quick_ratio": info.get("quickRatio"),
            "beta": info.get("beta"),
        })

    df = pd.DataFrame(rows)
    pickle.dump(df, open(CACHE_FILE, "wb"))
    print(f"✅ Saved cache → {CACHE_FILE}")

    if skipped:
        with open(SKIP_LOG, "w") as f:
            for s, r in skipped:
                f.write(f"{s}\t{r}\n")
        print(f"⚠️ Skipped {len(skipped)} bad symbols → {SKIP_LOG}")

    return df

# ------------------------------------------------------
def normalize_exchanges(df):
    conv = {"NYQ":"NYSE","NYSE":"NYSE","NMS":"NasdaqGS","NGM":"NasdaqGM","NCM":"NasdaqCM"}
    df["exchange_std"] = df["exchange"].map(conv).fillna(df["exchange"])
    return df

def apply_filters(df):
    df = normalize_exchanges(df)
    f = df[
        (df["marketCap"] >= MIN_MARKET_CAP) &
        (df["price"] >= MIN_PRICE) &
        (df["exchange_std"].isin(EXCHANGES_ALLOWED)) &
        (df["sector"].isin(SECTORS_ALLOWED))
    ].copy()
    f["industry_group"] = f["industry"].fillna("Unknown")
    return f

def pct_group(df, col, invert=False):
    result = pd.Series(np.nan, index=df.index)
    for ind, g in df.groupby("industry_group"):
        ok = g[col].notna()
        if ok.sum() > 0:
            pct = g.loc[ok, col].rank(pct=True)
            result.loc[pct.index] = (1 - pct) if invert else pct
    return result

# ------------------------------------------------------
def score(df):
    out = df.copy()

    for m in PROF_METRICS:
        out[f"pct_{m}"] = pct_group(out, m)

    out["pct_fcf_yield"]     = pct_group(out, "fcf_yield")
    out["pct_peg_effective"] = pct_group(out, "peg_effective", invert=True)
    out["pct_ev_ebitda"]     = pct_group(out, "ev_ebitda", invert=True)
    out["pct_pb"]            = pct_group(out, "pb", invert=True)

    out["pct_eps_growth"] = pct_group(out, "eps_growth")

    out["pct_debt_to_equity"] = pct_group(out, "debt_to_equity", invert=True)
    out["pct_quick_ratio"]    = pct_group(out, "quick_ratio")
    out["pct_beta"]           = pct_group(out, "beta", invert=True)

    out["prof_raw"]   = np.nanmean([1 + out[f"pct_{m}"] * 5 for m in PROF_METRICS], axis=0)
    out["val_raw"]    = np.nanmean([1 + out[f"pct_{m}"] * 5 for m in VAL_METRICS], axis=0)
    out["growth_raw"] = 1 + out["pct_eps_growth"] * 17
    out["risk_raw"]   = np.nanmean([
        1 + out["pct_debt_to_equity"] * 9,
        1 + out["pct_quick_ratio"] * 9,
        1 + out["pct_beta"] * 9
    ], axis=0)

    out["score"] = (
        out["prof_raw"] * (W_PROF/6) +
        out["val_raw"]  * (W_VAL/6) +
        out["growth_raw"] * (W_GROW/18) +
        out["risk_raw"] * (W_RISK/10)
    )

    return out.sort_values("score", ascending=False).reset_index(drop=True)

# ------------------------------------------------------
def show_terminal_chart(df):
    print("\n=== TOP 20 BREAKDOWN ===")
    for _, row in df.head(20).iterrows():
        print(f"\n{row['symbol']} | Score: {row['score']:.2f}")
        print(f"  Profitability: {row['prof_raw']:.2f}/6 ")
        print(f"     → ROE: {row['roe']}, Profit Margin: {row['profit_margin']}, Operating Margin: {row['operating_margin']}, ROA: {row['roa']}")
        print(f"  Valuation:     {row['val_raw']:.2f}/6 ")
        print(f"     → FCF Yield: {row['fcf_yield']}, PEG: {row['peg_effective']}, EV/EBITDA: {row['ev_ebitda']}, P/B: {row['pb']}")
        print(f"  Growth:        {row['growth_raw']:.2f}/18 (EPS Growth: {row['eps_growth']})")
        print(f"  Risk/Liquidity:{row['risk_raw']:.2f}/10 ")
        print(f"     → Debt/Equity: {row['debt_to_equity']}, Quick Ratio: {row['quick_ratio']}, Beta: {row['beta']}")

def export_full_chart(df):
    if not EXPORT_CHART:
        return
    plt.figure(figsize=(12, 6))
    plt.bar(df["symbol"], df["score"])
    plt.xticks(rotation=90)
    plt.title("Ranked Stocks")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(CHART_FILE)
    print(f"📈 Chart saved → {CHART_FILE}")

# ------------------------------------------------------
def main():
    global USE_CACHE
    if "--refresh" in sys.argv:
        USE_CACHE = False

    print("Building universe...")
    universe = fetch_symbol_lists()
    symbols = universe["symbol"].tolist()

    print("Getting fundamentals...")
    df = fetch_fundamentals(symbols)

    print("Applying filters...")
    f = apply_filters(df)
    if f.empty:
        print("❌ No valid stocks passed filters")
        return

    print("Scoring...")
    ranked = score(f)

    print("\n✅ Saving CSV files...")
    ranked.to_csv("all_ranked.csv", index=False)
    ranked.head(10).to_csv("top10.csv", index=False)
    print("✅ Saved: all_ranked.csv and top10.csv")

    show_terminal_chart(ranked)
    export_full_chart(ranked)

    print("\n✅ DONE")

if __name__ == "__main__":
    main()