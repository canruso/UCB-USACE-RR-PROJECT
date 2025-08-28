import os
import glob

HEADER_LINE_INDEX = 1  # 0-based index for the header line to modify
SKIP_PREFIX = {"Ordinate", "Date", "Time", "Units", "Type"}

def add_prefix_to_specific_line(input_file: str, prefix: str):
    """
    Read a CSV as raw text, prefix certain columns in line HEADER_LINE_INDEX,
    and write <filename>_mts.csv without stripping other whitespace.
    """

    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_mts{ext}"

    with open(input_file, "r", newline="") as f:
        lines = f.readlines()

    # If the file doesn't have enough lines, skip
    if len(lines) <= HEADER_LINE_INDEX:
        print(f"Skipping {input_file}, it doesn't have line {HEADER_LINE_INDEX}.")
        return

    # Grab the original header line exactly (including trailing spaces, if any).
    original_line = lines[HEADER_LINE_INDEX]

    # Split by commas just enough to separate fields. We do not strip each field fully.
    # This means if there is extra spacing around commas, that might be lost upon re-join.
    # For truly exact comma spacing, you'd have to parse more carefully (e.g., via regex).
    columns = original_line.split(",")

    new_columns = []
    for col in columns:
        # 'col' includes leading/trailing spaces. Let's find the "meaningful" text inside.
        col_stripped = col.strip()
        # Decide if we skip prefixing
        if col_stripped in SKIP_PREFIX or col_stripped == "":
            # Keep this field exactly as it was
            new_columns.append(col)
        else:
            # Minimal approach: Replace only the stripped portion with prefix + stripped portion
            # so that leading spaces remain intact in 'col'.
            updated = col.replace(col_stripped, prefix + col_stripped, 1)
            new_columns.append(updated)

    # Re-join the columns with ',' but note this can remove extra spaces after commas:
    lines[HEADER_LINE_INDEX] = ",".join(new_columns)

    with open(output_file, "w", newline="") as f:
        f.writelines(lines)

    print(f"Created {output_file}, added prefix '{prefix}' to columns except {SKIP_PREFIX}.")

def main():
    csv_files = glob.glob("*.csv")
    for csv_file in csv_files:
        lower = csv_file.lower()
        if lower.endswith("daily.csv"):
            add_prefix_to_specific_line(csv_file, "daily_")
        elif lower.endswith("hourly.csv"):
            add_prefix_to_specific_line(csv_file, "hourly_")
        else:
            # Not daily or hourly, skip
            pass

if __name__ == "__main__":
    main()