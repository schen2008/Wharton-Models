import yfinance as yf

# -----------------------------------------------------
#  EDIT THIS ARRAY ONLY
# -----------------------------------------------------
TICKERS = [
    "ARGX",
    "ARLP",
    "BZ",
    "CALM",
    "CART",
    "COCO",
    "ERIC",
    "GBDC",
    "GLPI",
    "GTX",
    "INVA",
    "KRYS",
    "IEF",
    "TIP",
    "TLT",
]
# -----------------------------------------------------


def fetch_yahoo(ticker):
    """Fetch EV and EBITDA from Yahoo Finance. Works for ETFs too."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Check if we got valid data
        ev = info.get("enterpriseValue", None)
        ebitda = info.get("ebitda", None)

        if ev is None or ebitda is None:
            print(f"Warning: EV or EBITDA missing for {ticker}")
        return ev, ebitda
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None, None

def ev_ebitda(ev, ebitda):
    """Safe EV/EBITDA calculation."""
    if ev is None or ebitda is None:
        return None
    if ebitda == 0:
        return None
    return ev / ebitda


def calculate_ev_ebitda(tickers):
    results = {}
    missing = {}

    for t in tickers:
        t_upper = t.upper()

        ev, ebitda = fetch_yahoo(t_upper)

        if ev is None or ebitda is None:
            missing[t_upper] = "Missing EV or EBITDA from Yahoo Finance."
            continue

        ratio = ev_ebitda(ev, ebitda)

        if ratio is None:
            missing[t_upper] = "Invalid EV or EBITDA values."
            continue

        results[t_upper] = float(f"{ratio:.6f}")

    return {"results": results, "missing": missing}


if __name__ == "__main__":
    output = calculate_ev_ebitda(TICKERS)

    print("\n========= EV/EBITDA RESULTS =========")
    for t, value in output["results"].items():
        print(f"{t}: {value}")

    print("\n========= MISSING DATA =========")
    for t, msg in output["missing"].items():
        print(f"{t}: {msg}")