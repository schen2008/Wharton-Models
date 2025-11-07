#!/usr/bin/env python3
# screener_sean.py
#
# ✅ No terminal commands needed
# ✅ Output always saved safely (no FileNotFound errors)
# ✅ Editable flags: USE_CACHE, SECTOR_FILTER
# ✅ Table-based scoring (Profitability, Valuation, Growth, Risk)
# ✅ Top 10 overall + Top 10 per sector + breakdown printout

import io
import os
import re
import pickle
import urllib.request
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# tqdm fallback (so missing tqdm won't break code)
try:
    from tqdm import tqdm
except:
    def tqdm(x, **k): return x

import yfinance as yf

# ============================================
# USER EDITABLE SETTINGS
# ============================================
USE_CACHE = True          # True = load cached fundamentals; False = refresh from Yahoo
SECTOR_FILTER = None      # e.g., "Technology", or None for ALL sectors

# ============================================
# GUARANTEED OUTPUT DIRECTORY (works even if parents missing)
# ============================================
from pathlib import Path

try:
    # Try to save right next to script / current working directory
    OUTPUT_DIR = Path(os.getcwd()) / "results"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    # Fallback to Documents
    OUTPUT_DIR = Path.home() / "Documents" / "Wharton-Models" / "results"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_FILE = OUTPUT_DIR / "fundamentals_cache.pkl"
SKIP_LOG   = OUTPUT_DIR / "skipped_symbols.txt"# ============================================
# FILTER RULES
# ============================================
MIN_MARKET_CAP = 2_000_000_000
MIN_PRICE      = 5.0

EXCHANGES_ALLOWED = {"NYSE", "NasdaqGS", "NasdaqGM", "NasdaqCM"}
SECTORS_ALLOWED   = {"Financials", "Industrials", "Energy", "Materials", "Technology", "Consumer Staples"}
BAD_SUFFIXES      = (".U", ".W", ".R", "-U", "-W", "-R")

# ============================================
# Helpers
# ============================================
def normalize_symbol(sym):
    if sym is None:
        return ""
    s = str(sym).strip().replace(" ", "")
    if "." in s: s = s.replace(".", "-")
    if "$" in s:
        base, ser = s.split("$", 1)
        ser = re.sub(r"[^A-Za-z0-9]", "", ser).upper()
        s = f"{base}-PR{ser}" if ser else base
    return s

def safe_div(a, b):
    if a is None or b is None: return None
    try:
        if pd.isna(a) or pd.isna(b) or b == 0: return None
    except: pass
    try: return a / b
    except: return None

def to_num(v):
    if v is None: return None
    try:
        if pd.isna(v): return None
    except: pass
    try: return float(v)
    except: return None

# ============================================
# TABLE-BASED SCORING
# ============================================
def score_profitability(roe, pm, om, roa):
    vals = [to_num(roe), to_num(pm), to_num(om), to_num(roa)]
    if any(v is not None and v < 0 for v in vals): return 1
    if all(v is not None and v > 0.15 for v in vals): return 6
    if all(v is not None and v > 0.05 for v in vals): return 3
    return 2

def score_valuation(fcf_yield, peg, ev_ebitda, pb):
    fy = to_num(fcf_yield)
    pg = to_num(peg)
    ee = to_num(ev_ebitda)
    pbv = to_num(pb)
    score = 0
    if fy is not None and fy > 0.05: score += 2
    if pg is not None and pg < 1: score += 2
    if ee is not None and ee < 10: score += 1
    if pbv is not None and pbv < 3: score += 1
    return max(1, min(score, 6))

def score_growth(eps_growth):
    g = to_num(eps_growth)
    if g is None: return 9
    if g < 0: return 1
    if g < 0.05: return 9
    if g > 0.20: return 18
    return 12

def score_risk(dte, qr, beta):
    d = to_num(dte)
    q = to_num(qr)
    b = to_num(beta)
    score = 0
    if d is not None and d < 1: score += 4
    if q is not None and q > 1: score += 3
    if b is not None and b < 1: score += 3
    return max(1, min(score, 10))

# ============================================
# NASDAQ symbol list
# ============================================
def fetch_symbol_lists():
    nasdaq_url = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
    other_url  = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"

    def load(url):
        with urllib.request.urlopen(url) as resp:
            return pd.read_csv(io.BytesIO(resp.read()), sep="|")

    nasdaq = load(nasdaq_url)
    other  = load(other_url)

    nasdaq = nasdaq[nasdaq["Test Issue"] == "N"]
    other  = other[(other["Test Issue"] == "N") & (other["Exchange"] == "N")]

    nasdaq["Symbol"]    = nasdaq["Symbol"].astype(str).str.strip()
    other["ACT Symbol"] = other["ACT Symbol"].astype(str).str.strip()

    df = pd.concat([
        pd.DataFrame({"symbol_raw": nasdaq["Symbol"]}),
        pd.DataFrame({"symbol_raw": other["ACT Symbol"]})
    ], ignore_index=True).drop_duplicates()

    df["symbol_raw"] = df["symbol_raw"].astype(str).str.strip()
    df = df[~df["symbol_raw"].str.endswith(BAD_SUFFIXES, na=False)]

    df["symbol"] = df["symbol_raw"].apply(normalize_symbol)
    df = df[df["symbol"] != ""]
    return df.drop_duplicates("symbol")

# ============================================
# FUNDAMENTALS (with caching)
# ============================================
def fetch_fundamentals(symbols):
    if USE_CACHE and os.path.exists(CACHE_FILE):
        print("✅ Loaded cached fundamentals")
        return pickle.load(open(CACHE_FILE, "rb"))

    print("⏳ Downloading fundamentals...")
    rows, skipped = [], []

    for sym in tqdm(symbols, desc="Pulling", unit="stk"):
        ysym = normalize_symbol(sym)
        if not ysym or any(ysym.endswith(suf) for suf in BAD_SUFFIXES):
            skipped.append((ysym, "bad suffix"))
            continue

        try:
            info = yf.Ticker(ysym).info
        except:
            skipped.append((ysym, "info failure"))
            continue

        price = to_num(info.get("regularMarketPrice") or info.get("previousClose"))
        if price is None:
            skipped.append((ysym, "no price"))
            continue

        rows.append({
            "symbol": ysym,
            "name": info.get("longName") or info.get("shortName"),
            "price": price,
            "exchange": info.get("exchange"),
            "sector": info.get("sector"),
            "industry": info.get("industry") or "Unknown",
            "marketCap": to_num(info.get("marketCap")),

            "roe": to_num(info.get("returnOnEquity")),
            "profit_margin": to_num(info.get("profitMargins")),
            "operating_margin": to_num(info.get("operatingMargins")),
            "roa": to_num(info.get("returnOnAssets")),

            "fcf_yield": safe_div(to_num(info.get("freeCashflow")), to_num(info.get("marketCap"))),
            "peg": to_num(info.get("pegRatio")),
            "ev_ebitda": safe_div(to_num(info.get("enterpriseValue")), to_num(info.get("ebitda"))),
            "pb": to_num(info.get("priceToBook")),

            "eps_growth": to_num(info.get("earningsGrowth")),

            "debt_to_equity": to_num(info.get("debtToEquity")),
            "quick_ratio": to_num(info.get("quickRatio")),
            "beta": to_num(info.get("beta")),
        })

    df = pd.DataFrame(rows)

    try:
        pickle.dump(df, open(CACHE_FILE, "wb"))
        print(f"✅ Cache saved → {CACHE_FILE}")
    except:
        print("⚠ Could not save cache file")

    if skipped:
        try:
            with open(SKIP_LOG, "w", encoding="utf-8") as f:
                for s, r in skipped:
                    f.write(f"{s}\t{r}\n")
            print(f"⚠ Skipped {len(skipped)} symbols → {SKIP_LOG}")
        except:
            print("⚠ Could not write skip log")

    return df

# ============================================
# FILTER + SCORE
# ============================================
def apply_filters(df):
    conv = {
        "NYQ": "NYSE",
        "NYSE": "NYSE",
        "NMS": "NasdaqGS",
        "NasdaqGS": "NasdaqGS",
        "NGM": "NasdaqGM",
        "NasdaqGM": "NasdaqGM",
        "NCM": "NasdaqCM",
        "NasdaqCM": "NasdaqCM",
    }

    df["exchange_std"] = df["exchange"].map(conv).fillna(df["exchange"])

    f = df[
        (df["marketCap"].apply(lambda x: x is not None and x >= MIN_MARKET_CAP)) &
        (df["price"].apply(lambda x: x is not None and x >= MIN_PRICE)) &
        (df["exchange_std"].isin(EXCHANGES_ALLOWED)) &
        (df["sector"].isin(SECTORS_ALLOWED))
    ].copy()

    return f

def score(df):
    out = df.copy()

    out["profitability"] = out.apply(lambda r: score_profitability(
        r["roe"], r["profit_margin"], r["operating_margin"], r["roa"]), axis=1)

    out["valuation"] = out.apply(lambda r: score_valuation(
        r["fcf_yield"], r["peg"], r["ev_ebitda"], r["pb"]), axis=1)

    out["growth"] = out["eps_growth"].apply(score_growth)

    out["risk"] = out.apply(lambda r: score_risk(
        r["debt_to_equity"], r["quick_ratio"], r["beta"]), axis=1)

    out["score"] = (
        out["profitability"] * (26/6) +
        out["valuation"]     * (26/6) +
        out["growth"]        * (18/18) +
        out["risk"]          * (30/10)
    )

    return out.sort_values("score", ascending=False).reset_index(drop=True)

# ============================================
# PRINT BREAKDOWN
# ============================================
def print_top10_breakdown(df):
    print("\n=== TOP 10 BREAKDOWN ===")
    for _, r in df.head(10).iterrows():
        print(f"\n{r['symbol']} — Score: {r['score']:.2f}")
        print(f"  Profitability: {r['profitability']}/6")
        print(f"    ROE: {r['roe']}, PM: {r['profit_margin']}, OM: {r['operating_margin']}, ROA: {r['roa']}")
        print(f"  Valuation: {r['valuation']}/6")
        print(f"    FCF Yield: {r['fcf_yield']}, PEG: {r['peg']}, EV/EBITDA: {r['ev_ebitda']}, P/B: {r['pb']}")
        print(f"  Growth: {r['growth']}/18 (EPS Growth: {r['eps_growth']})")
        print(f"  Risk/Liquidity: {r['risk']}/10")
        print(f"    D/E: {r['debt_to_equity']}, Quick: {r['quick_ratio']}, Beta: {r['beta']}")

# ============================================
# MAIN
# ============================================
def main():
    universe = fetch_symbol_lists()
    symbols = universe["symbol"].tolist()

    df = fetch_fundamentals(symbols)
    filtered = apply_filters(df)

    if filtered.empty:
        print("❌ No stocks passed filters.")
        return

    # Single sector only?
    if SECTOR_FILTER is not None:
        s = filtered[filtered["sector"] == SECTOR_FILTER]
        if s.empty:
            print(f"❌ No stocks in sector: {SECTOR_FILTER}")
            return

        ranked = score(s)
        fpath = os.path.join(OUTPUT_DIR, f"top10_{SECTOR_FILTER.replace(' ', '_')}.csv")
        ranked.to_csv(fpath, index=False)
        print_top10_breakdown(ranked)
        print(f"\n✅ DONE — saved → {fpath}")
        return

    # ALL sectors
    ranked = score(filtered)
    ranked.to_csv(os.path.join(OUTPUT_DIR, "all_ranked.csv"), index=False)
    ranked.head(10).to_csv(os.path.join(OUTPUT_DIR, "top10.csv"), index=False)

    # Top 10 per sector
    for sector, g in ranked.groupby("sector", sort=False):
        fpath = os.path.join(OUTPUT_DIR, f"top10_{sector.replace(' ', '_')}.csv")
        g.head(10).to_csv(fpath, index=False)

    print_top10_breakdown(ranked)
    print(f"\n✅ DONE — files saved in {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()