import os, io, math, requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
import pytz
##import pandas_market_calendars as mcal

ATHENS = pytz.timezone("Europe/Athens")
#XETRA_CAL = mcal.get_calendar("XETR")

PRED_CSV = "predictions.csv"
ONE_PAGER_PNG = "one_pager.png"
REPORT_TXT = "report.txt"

def athens_today() -> date:
    return datetime.now(ATHENS).date()

def is_xetra_trading_day(d: date) -> bool:
#    valid = XETRA_CAL.valid_days(start_date=d.isoformat(), end_date=d.isoformat())
#    return len(valid) > 0
    return d.weekday() < 5

def fetch_stooq_history(symbol: str = "vuaa.de") -> pd.DataFrame:
    base = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    variants = [
        base,
        base + f"/{symbol}_d.csv",
        base + f"/{symbol.replace('.', '_')}_d.csv",
    ]
    headers = {"User-Agent": "Mozilla/5.0"}
    last_err = None
    for url in variants:
        try:
            r = requests.get(url, headers=headers, timeout=20)
            if r.status_code != 200 or not r.text or "Date,Open,High,Low,Close,Volume" not in r.text:
                continue
            df = pd.read_csv(io.StringIO(r.text))
            df["Date"] = pd.to_datetime(df["Date"]).dt.date
            return df.sort_values("Date").reset_index(drop=True)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not fetch Stooq CSV for {symbol}. Last error: {last_err}")

def compute_close_to_close_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_pct"] = out["Close"].pct_change() * 100.0
    return out

def simple_forecast_model(last_ret: float, vol20: float) -> float:
    if last_ret is None or math.isnan(last_ret):
        return 0.05
    drift = 0.05
    mean_rev = -0.15 * last_ret
    raw = drift + mean_rev
    if vol20 and not math.isnan(vol20) and vol20 > 0:
        damp = min(1.0, 0.9 / vol20)
        raw *= damp
    return float(np.clip(raw, -2.0, 2.0))

def load_predictions() -> pd.DataFrame:
    if not os.path.exists(PRED_CSV):
        return pd.DataFrame(columns=[
            "trade_date", "forecast_pct", "actual_pct", "error_pp",
            "hit_direction", "created_at_athens"
        ])
    df = pd.read_csv(PRED_CSV)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df

def save_predictions(df: pd.DataFrame):
    out = df.copy()
    out["trade_date"] = out["trade_date"].astype(str)
    out.to_csv(PRED_CSV, index=False)

def month_to_date_filter(df: pd.DataFrame, asof: date) -> pd.DataFrame:
    start = asof.replace(day=1)
    return df[df["trade_date"].between(start, asof)]

def make_one_pager(asof: date, preds: pd.DataFrame):
    done = preds.dropna(subset=["actual_pct", "error_pp"])
    mtd = month_to_date_filter(done, asof)

    def stats_block(x: pd.DataFrame):
        if len(x) == 0:
            return (0, 0, np.nan, pd.DataFrame())
        hits = int(x["hit_direction"].sum())
        misses = int(len(x) - hits)
        mae = float(np.mean(np.abs(x["error_pp"])))
        big = x[np.abs(x["error_pp"]) > 1.0][["trade_date", "forecast_pct", "actual_pct", "error_pp"]]
        return hits, misses, mae, big

    hits_all, miss_all, mae_all, _ = stats_block(done)
    hits_m, miss_m, mae_m, big_m = stats_block(mtd)

    N = 30
    last_done = done.sort_values("trade_date").tail(N)

    fig = plt.figure(figsize=(13.33, 7.5), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(0.03, 0.94, "VUAA.DE Forecast — Daily Benchmark One-Pager", fontsize=20, weight="bold", va="top")
    ax.text(0.03, 0.905, f"As of (Athens): {asof.isoformat()}", fontsize=10, va="top")

    ax.text(0.03, 0.85,
            f"MTD direction: {hits_m} hits / {miss_m} misses | MTD MAE: "
            f"{'n/a' if np.isnan(mae_m) else f'{mae_m:.2f}'} pp",
            fontsize=12, weight="bold", va="top")

    ax.text(0.03, 0.81,
            f"All-time direction: {hits_all} hits / {miss_all} misses | All-time MAE: "
            f"{'n/a' if np.isnan(mae_all) else f'{mae_all:.2f}'} pp",
            fontsize=11, va="top")

    chart = fig.add_axes([0.06, 0.18, 0.88, 0.55])
    chart.axhline(0, linewidth=1)
    if len(last_done) > 0:
        chart.plot(last_done["trade_date"], last_done["actual_pct"], label="Actual %", linewidth=1.5)
        chart.plot(last_done["trade_date"], last_done["forecast_pct"], label="Forecast %", linewidth=1.5)
        chart.set_title(f"Last {len(last_done)} evaluated trading days", fontsize=11)
        chart.set_ylabel("%")
        chart.tick_params(axis="x", labelrotation=45, labelsize=8)
        chart.tick_params(axis="y", labelsize=9)
        chart.legend(frameon=False, fontsize=9)
    else:
        chart.text(0.5, 0.5, "No evaluated days yet (need at least 2 trading days).", ha="center", va="center")

    ax.text(0.03, 0.12, "MTD big misses (|error| > 1.0 pp):", fontsize=10, weight="bold", va="top")
    if len(big_m) == 0:
        ax.text(0.03, 0.095, "None (or not enough data yet).", fontsize=10, va="top")
    else:
        lines = []
        for _, r in big_m.sort_values("trade_date").tail(6).iterrows():
            lines.append(f"{r['trade_date']}: fc {r['forecast_pct']:.2f}% | act {r['actual_pct']:.2f}% | err {r['error_pp']:.2f}pp")
        ax.text(0.03, 0.095, "\n".join(lines), fontsize=9, va="top")

    plt.savefig(ONE_PAGER_PNG, bbox_inches="tight")
    plt.close(fig)

def write_report(lines: list[str]):
    txt = "\n".join(lines) + "\n"
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(txt)
    for ln in lines:
        print(ln)

def run_daily():
    run_at = datetime.now(ATHENS)
    today = run_at.date()
    open_today = is_xetra_trading_day(today)

    preds = load_predictions()
    log_rows = int(len(preds))
    evaluated_rows = int(len(preds.dropna(subset=["actual_pct", "error_pp"])))

    repo = os.getenv("GITHUB_REPOSITORY", "")
    branch = os.getenv("GITHUB_REF_NAME", "main")
    report_url = f"https://raw.githubusercontent.com/{repo}/{branch}/{REPORT_TXT}" if repo else ""
    one_pager_url = f"https://raw.githubusercontent.com/{repo}/{branch}/{ONE_PAGER_PNG}" if repo else ""

    lines = [
        "STATE",
        f"run_at_athens: {run_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"today_athens: {today.isoformat()}",
        f"is_xetra_trading_day: {'YES' if open_today else 'NO'}",
        "data_last_close: n/a",
        "data_prev_close: n/a",
        f"log_rows: {log_rows}",
        f"evaluated_rows: {evaluated_rows}",
        f"notes: report_url={report_url} one_pager_url={one_pager_url}".strip(),
        "END_STATE",
    ]

    if not open_today:
        lines.append("market closed—no forecast")
        write_report(lines)
        return

    raw = fetch_stooq_history("vuaa.de")
    px = compute_close_to_close_returns(raw)
    last_px_date = px["Date"].iloc[-1]
    prev_px_date = px["Date"].iloc[-2]

    lines[4] = f"data_last_close: {last_px_date}"
    lines[5] = f"data_prev_close: {prev_px_date}"

    ret_map = dict(zip(px["Date"], px["ret_pct"]))
    preds["actual_pct"] = preds.apply(
        lambda r: ret_map.get(r["trade_date"], np.nan) if pd.isna(r.get("actual_pct", np.nan)) else r["actual_pct"],
        axis=1
    )
    preds["error_pp"] = preds.apply(
        lambda r: (r["forecast_pct"] - r["actual_pct"]) if (pd.notna(r["forecast_pct"]) and pd.notna(r["actual_pct"])) else r.get("error_pp", np.nan),
        axis=1
    )
    preds["hit_direction"] = preds.apply(
        lambda r: (np.sign(r["forecast_pct"]) == np.sign(r["actual_pct"])) if (pd.notna(r["forecast_pct"]) and pd.notna(r["actual_pct"])) else r.get("hit_direction", np.nan),
        axis=1
    )

    if not (preds["trade_date"] == today).any():
        last_ret = float(px.loc[px["Date"] == last_px_date, "ret_pct"].iloc[0])
        vol20 = float(px["ret_pct"].tail(20).std())
        fc = simple_forecast_model(last_ret=last_ret, vol20=vol20)
        new_row = {
            "trade_date": today,
            "forecast_pct": fc,
            "actual_pct": np.nan,
            "error_pp": np.nan,
            "hit_direction": np.nan,
            "created_at_athens": run_at.strftime("%Y-%m-%d %H:%M:%S"),
        }
        preds = pd.concat([preds, pd.DataFrame([new_row])], ignore_index=True)

    preds["hit_direction"] = preds["hit_direction"].astype("float")
    save_predictions(preds)

    make_one_pager(asof=prev_px_date, preds=preds)

    done = preds.dropna(subset=["actual_pct", "error_pp"]).sort_values("trade_date")
    mtd = month_to_date_filter(done, prev_px_date)
    hits = int(mtd["hit_direction"].sum()) if len(mtd) else 0
    misses = int(len(mtd) - hits) if len(mtd) else 0
    mae = float(np.mean(np.abs(mtd["error_pp"]))) if len(mtd) else np.nan

    fc_today = float(preds[preds["trade_date"] == today].iloc[0]["forecast_pct"])

    lines += [
        f"Latest close date in data: {last_px_date} (prev: {prev_px_date})",
        f"Today ({today}) forecast: {fc_today:+.2f}%",
        f"MTD score (as of {prev_px_date}): {hits} hits / {misses} misses | MAE: {'n/a' if np.isnan(mae) else f'{mae:.2f}'} pp",
        f"One-pager saved to: {ONE_PAGER_PNG}",
        f"Log saved to: {PRED_CSV}",
    ]
    write_report(lines)

if __name__ == "__main__":
    run_daily()
