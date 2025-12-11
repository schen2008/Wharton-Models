import yfinance as yf
import requests
from bs4 import BeautifulSoup
import re

TICKERS = [
    "ARGX","ARLP","BZ","CALM","CART","COCO",
    "ERIC","GBDC","GLPI","GTX","INVA","KRYS",
    "IEF","TIP","TLT"
]

ETF_TICKERS = {"IEF","TIP","TLT"}
BDC_TICKERS = {"GBDC"}

# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------

def extract_num(text):
    if not text:
        return None
    text = text.replace(",", "").replace("$", "").strip()
    try:
        return float(text)
    except:
        m = re.search(r"\d+\.?\d*", text)
        return float(m.group()) if m else None


# -----------------------------------------------------------
# Google Finance Scraper (WORKS for ERIC)
# Example: https://www.google.com/finance/quote/ERIC:NASDAQ
# -----------------------------------------------------------

def google_finance_ev_ebitda(ticker):
    try:
        url = f"https://www.google.com/finance/quote/{ticker}:NASDAQ"
        html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")

        rows = soup.find_all("div", class_="gyFHrc")

        ev = None
        ebitda = None
        ratio = None

        for row in rows:
            label = row.find("div", class_="mfs7Fc")
            value = row.find("div", class_="P6K39c")
            if not label or not value:
                continue

            label = label.text.strip().lower()
            value = value.text.strip()

            if label == "enterprise value":
                ev = extract_num(value)

            if label == "ebitda":
                ebitda = extract_num(value)

            if label in ("ev/ebitda","ev to ebitda","ev to ebitda ttm"):
                ratio = extract_num(value)

        if ratio:
            return ratio, "Google Finance (direct ratio)"

        if ev and ebitda:
            return ev / ebitda, "Google Finance"

        return None
    except:
        return None


# -----------------------------------------------------------
# Yahoo attempt
# -----------------------------------------------------------

def yahoo_ev_ebitda(ticker):
    try:
        info = yf.Ticker(ticker).info
        ev = info.get("enterpriseValue")
        ebitda = info.get("ebitda")
        if ev and ebitda:
            return ev / ebitda, "Yahoo"
        return None
    except:
        return None


# -----------------------------------------------------------
# Main fetcher
# -----------------------------------------------------------

def fetch_ratio(ticker):
    t = ticker.upper()

    # ETFs never have EBITDA
    if t in ETF_TICKERS:
        return None, "N/A (ETF, no EBITDA exists)"

    # BDCs often report no EBITDA
    if t in BDC_TICKERS:
        return None, "N/A (BDC, EBITDA not reported)"

    # 1. Yahoo
    r = yahoo_ev_ebitda(t)
    if r:
        return r

    # 2. Google Finance (works for ERIC)
    r = google_finance_ev_ebitda(t)
    if r:
        return r

    # Fallback
    return None, "Missing from all accessible sources"


# -----------------------------------------------------------
# Run
# -----------------------------------------------------------

if __name__ == "__main__":
    results = {}
    missing = {}

    for t in TICKERS:
        ratio, src = fetch_ratio(t)
        if ratio:
            results[t] = (round(ratio, 4), src)
        else:
            missing[t] = src

    print("\n========= EV/EBITDA RESULTS =========")
    for t, (val, src) in results.items():
        print(f"{t}: {val}  (source: {src})")

    print("\n========= MISSING OR N/A =========")
    for t, msg in missing.items():
        print(f"{t}: {msg}")