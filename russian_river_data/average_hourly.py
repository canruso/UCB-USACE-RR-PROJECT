#!/usr/bin/env python3
import pandas as pd
from io import StringIO

def parse_24h_time(d_str, t_str):
    """
    Parse date string + time string, handling 24:00:00 by shifting +1 day.
    Return a pandas Timestamp or NaT if parsing fails.
    """
    d_str = str(d_str).strip()
    t_str = str(t_str).strip()
    if t_str == "24:00:00":
        base = pd.to_datetime(d_str, errors="coerce", dayfirst=False)
        if pd.isnull(base):
            return pd.NaT
        return base + pd.Timedelta(days=1)
    else:
        combo = d_str + " " + t_str
        return pd.to_datetime(combo, errors="coerce", dayfirst=False)

def hourly_to_daily_exact(infile, outfile):
    """
    Steps:
    1) Read entire file lines.
       - line 0 => "Col number..."
       - line 1 => actual CSV column names
       - line 2 => "Units..."
       - line 3 => "Type..." (we'll replace INST-VAL with PER-AVER)
       - lines 4+ => hourly data
    2) Parse line 1 as col names, handle duplicates if any.
    3) Filter lines 4+ that have correct # of commas.
    4) Find 'Date'/'Time' columns. Combine -> DateTimeIndex.
    5) Resample numeric columns (excluding 'Ordinate') => daily mean, shift index -1 day.
    6) Round means to 6 decimals, write lines 0..3 verbatim (with Type replaced),
       then daily rows with [Ordinate, Date, Time, ...].
    """

    # -- Read lines
    with open(infile, "r") as f:
        lines = f.readlines()

    if len(lines) < 4:
        raise ValueError(f"File '{infile}' has fewer than 4 lines; cannot parse.")

    # 1) We keep line0..line3
    line0 = lines[0]  # e.g. "Col number,,,1,2,3,4,5,6,7"
    line1 = lines[1]  # CSV column names
    line2 = lines[2]  # "Units..."
    line3 = lines[3]  # "Type..."
    data_lines = lines[4:]  # the actual data

    # 2) parse line1 for columns
    raw_colnames = line1.strip("\n").split(",")
    colnames = [c.strip() for c in raw_colnames]

    # handle duplicates
    seen = {}
    final_colnames = []
    for c in colnames:
        if c not in seen:
            seen[c] = 0
            final_colnames.append(c)
        else:
            seen[c] += 1
            new_c = f"{c}.{seen[c]}"
            final_colnames.append(new_c)

    expected_cols = len(final_colnames)

    # 3) filter data_lines by #commas
    data_block = []
    for ln in data_lines:
        if ln.count(",") == (expected_cols - 1):
            data_block.append(ln)

    # create DataFrame from data_block
    data_str = "".join(data_block)
    df = pd.read_csv(StringIO(data_str), header=None, names=final_colnames)

    # 4) find 'Date' + 'Time' columns (case-insensitive)
    date_col, time_col = None, None
    for c in final_colnames:
        lc = c.lower()
        if "date" in lc and date_col is None:
            date_col = c
        elif "time" in lc and time_col is None:
            time_col = c

    if not date_col or not time_col:
        raise ValueError(
            f"Could not find 'Date'/'Time' columns in {infile}.\nColumns: {final_colnames}"
        )

    # parse date/time
    dt_list = []
    for idx, row in df.iterrows():
        dtval = parse_24h_time(row[date_col], row[time_col])
        dt_list.append(dtval)

    df["__dtindex"] = dt_list
    df = df.dropna(subset=["__dtindex"])
    df.set_index("__dtindex", inplace=True)

    # 5) resample numeric => daily mean, exclude "Ordinate" from numeric
    skip = {date_col, time_col, "Ordinate"}
    numeric_candidates = [c for c in final_colnames if c not in skip]

    for c in numeric_candidates:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    daily = df[numeric_candidates].resample("D").mean()
    daily.index = daily.index - pd.Timedelta(days=1)

    # 6) build daily rows => ordinal, date=DD-MMM-YY, time=24:00:00, then numeric
    daily_rows = []
    ordinal = 1
    for day, rowvals in daily.sort_index().iterrows():
        dstr = day.strftime("%d-%b-%y")
        row_list = [ordinal, dstr, "24:00:00"]
        for c in numeric_candidates:
            # round to 6 decimals
            val = rowvals[c]
            if pd.notnull(val):
                val = round(val, 6)
            row_list.append(val)
        daily_rows.append(row_list)
        ordinal += 1

    daily_cols = ["Ordinate", "Date", "Time"] + numeric_candidates
    out_df = pd.DataFrame(daily_rows, columns=daily_cols)

    # Modify line3 => replace "INST-VAL" with "PER-AVER" (and keep rest)
    # so the "Type" line doesn't say "INST-VAL"
    line3 = line3.replace("INST-VAL", "PER-AVER")

    # 7) write output => line0..line3 EXACT (except line3 replaced), then daily rows
    with open(outfile, "w", newline="") as f:
        f.write(line0)
        f.write(line1 if line1.endswith("\n") else line1 + "\n")
        f.write(line2 if line2.endswith("\n") else line2 + "\n")
        f.write(line3 if line3.endswith("\n") else line3 + "\n")

        out_df.to_csv(f, index=False, header=False)

    print(f"** Wrote daily-averaged CSV => {outfile}")


if __name__ == "__main__":
    # Example usage
    input_files = [
        ("Calpella_hourly.csv", "Calpella_daily_averaged.csv"),
        ("Hopland_hourly.csv", "Hopland_daily_averaged.csv"),
        ("Guerneville_hourly.csv", "Guerneville_daily_averaged.csv"),
        ("WarmSprings_Inflow_hourly.csv", "WarmSprings_Inflow_daily_averaged.csv")
    ]
    for (infile, outfile) in input_files:
        print(f"** Processing {infile} => {outfile}")
        hourly_to_daily_exact(infile, outfile)