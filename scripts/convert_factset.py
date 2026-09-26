"""Convert FactSet Price History exports (.xlsx/.csv) into the long-format
parquet used by the backtester, with columns ``['date', 'ticker', 'open', 'close']``.

Usage:
    python scripts/convert_factset.py [--total-return] <file_or_dir> [...] <output.parquet>
"""
import re
import sys
from pathlib import Path
import pandas as pd

#: Price History column with the total-return index (dividends reinvested on
#: the ex-date). It equals the price on the first day.
TOTAL_RETURN_COLUMN = "total return (gross)"


def parse_single_factset_file(file_path: Path, total_return: bool = False) -> pd.DataFrame:
    """Parse a single FactSet Price History export (.xlsx or .csv).

    Returns a long DataFrame with columns ``['date', 'ticker', 'open', 'close']``.
    The ticker is read from the 'Price History: TICKER' header cell, falling
    back to the file name.

    With ``total_return=True`` prices are adjusted for dividends: the close
    becomes the total-return index and the open is scaled by the same daily
    factor (``TR / price``), so the dividend shows up in the overnight gap on
    the ex-date and the intraday move is unchanged.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Read only the first rows to locate the ticker and the table header.
    if file_path.suffix.lower() in [".xlsx", ".xls"]:
        xls = pd.ExcelFile(file_path)
        sheet = xls.sheet_names[0]
        raw = pd.read_excel(xls, sheet_name=sheet, nrows=6, header=None)
        read_fn = lambda skip: pd.read_excel(xls, sheet_name=sheet, skiprows=skip)
    else:
        raw = pd.read_csv(file_path, nrows=6, header=None)
        read_fn = lambda skip: pd.read_csv(file_path, skiprows=skip)

    # Ticker from cell (0, 0), e.g. 'Price History: SPY'; otherwise the file name.
    cell_0_0 = str(raw.iloc[0, 0])
    match = re.search(r"Price History:\s*([A-Za-z0-9._\-^]+)", cell_0_0)
    ticker = match.group(1).strip().upper() if match else file_path.stem.upper()

    # The table header is the first row containing 'Date' and 'Price' (or 'Close').
    header_idx = None
    for idx, row in raw.iterrows():
        row_vals = [str(v).strip().lower() for v in row.values]
        if "date" in row_vals and ("price" in row_vals or "close" in row_vals):
            header_idx = idx
            break

    if header_idx is None:
        raise ValueError(f"No header row with 'Date' and 'Price' found in {file_path}")

    df = read_fn(header_idx)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # FactSet's 'Price' column is the close.
    rename_map = {"price": "close"}
    df = df.rename(columns=rename_map)

    required = ["date", "open", "close"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {file_path}. Columns: {df.columns.tolist()}")

    df["date"] = pd.to_datetime(df["date"])
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["ticker"] = ticker

    if total_return:
        if TOTAL_RETURN_COLUMN not in df.columns:
            raise ValueError(
                f"--total-return requires column '{TOTAL_RETURN_COLUMN}' in {file_path}. "
                f"Columns: {df.columns.tolist()}"
            )
        tr_index = pd.to_numeric(df[TOTAL_RETURN_COLUMN], errors="coerce")
        factor = tr_index / df["close"]
        df["open"] = df["open"] * factor
        df["close"] = tr_index

    return df[["date", "ticker", "open", "close"]].dropna()

def convert_factset(input_paths: list[str], output_parquet: str, total_return: bool = False) -> None:
    """Convert FactSet files and/or directories into a single parquet file.

    Directories are scanned (non-recursively) for .xlsx/.xls/.csv files. Rows
    are de-duplicated on (date, ticker) and sorted by date and ticker.
    """
    dfs = []
    for p in input_paths:
        path = Path(p)
        if path.is_file():
            dfs.append(parse_single_factset_file(path, total_return))
        elif path.is_dir():
            for f in path.glob("*.*"):
                if f.suffix.lower() in [".xlsx", ".xls", ".csv"]:
                    dfs.append(parse_single_factset_file(f, total_return))

    if not dfs:
        raise ValueError("No data found to process.")

    final_df = pd.concat(dfs, ignore_index=True)
    final_df = final_df.drop_duplicates(subset=["date", "ticker"]).sort_values(["date", "ticker"]).reset_index(drop=True)

    Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(output_parquet, index=False, compression="snappy")
    print(f"Parquet saved to: {output_parquet} ({len(final_df)} rows, {final_df['ticker'].nunique()} tickers)")

if __name__ == "__main__":
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    unknown = flags - {"--total-return"}
    if len(paths) < 2 or unknown:
        print("Usage: python scripts/convert_factset.py [--total-return] <file_or_dir1> [file2...] <output.parquet>")
        print("  --total-return  use 'Total Return (Gross)' to adjust open/close for dividends")
        sys.exit(1)
    convert_factset(paths[:-1], paths[-1], total_return="--total-return" in flags)