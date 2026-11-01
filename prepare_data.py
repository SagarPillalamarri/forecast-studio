"""prepare_data.py
Downloads or reads manually downloaded raw files from data/raw/
and cleans each dataset into a CSV with exactly two columns: date,value.
Outputs:
  - data/ae_monthly.csv
  - data/electricity_daily.csv
"""

import os
import sys
import glob
import io
import requests
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def prepare_ae_monthly():
    """
    Prepares NHS England monthly A&E attendances (national total, England, from Aug 2010).
    Saves to data/ae_monthly.csv with columns: date,value
    """
    output_path = os.path.join(DATA_DIR, "ae_monthly.csv")
    print("\n--- [1/2] Preparing NHS England Monthly A&E Attendances ---")

    # 1. First, check if a raw file exists in data/raw/
    raw_candidates = glob.glob(os.path.join(RAW_DIR, "*AE*.*")) + \
                     glob.glob(os.path.join(RAW_DIR, "*ae*.*")) + \
                     glob.glob(os.path.join(RAW_DIR, "*attendances*.*"))

    df_cleaned = None

    if raw_candidates:
        for raw_path in raw_candidates:
            try:
                print(f"Attempting to read local raw file: {raw_path}")
                if raw_path.endswith((".xls", ".xlsx")):
                    excel_file = pd.ExcelFile(raw_path)
                    for sheet in excel_file.sheet_names:
                        temp = excel_file.parse(sheet)
                        # look for date and total columns
                        cols_lower = [str(c).lower() for c in temp.columns]
                        if any("period" in c or "date" in c or "month" in c for c in cols_lower):
                            # inspect content
                            temp.columns = [str(c).strip() for c in temp.columns]
                            for c in temp.columns:
                                if any(k in c.lower() for k in ["total attendances", "total_attendances", "attendances"]):
                                    date_c = [dc for dc in temp.columns if any(k in dc.lower() for k in ["period", "date", "month"])][0]
                                    temp['date'] = pd.to_datetime(temp[date_c], errors='coerce')
                                    temp['value'] = pd.to_numeric(temp[c], errors='coerce')
                                    valid = temp[['date', 'value']].dropna().sort_values('date')
                                    if len(valid) > 20:
                                        df_cleaned = valid
                                        print(f"Successfully parsed {len(df_cleaned)} rows from {raw_path} (sheet: {sheet})")
                                        break
                        if df_cleaned is not None:
                            break
                elif raw_path.endswith(".csv"):
                    temp = pd.read_csv(raw_path)
                    cols_lower = {c: str(c).lower().strip() for c in temp.columns}
                    # check if already date,value or similar
                    date_col = next((c for c, cl in cols_lower.items() if cl in ["date", "period", "month"]), None)
                    val_col = next((c for c, cl in cols_lower.items() if any(k in cl for k in ["total_attendances", "total attendances", "attendances", "value"])), None)
                    if date_col and val_col:
                        temp['date'] = pd.to_datetime(temp[date_col], errors='coerce')
                        temp['value'] = pd.to_numeric(temp[val_col], errors='coerce')
                        valid = temp[['date', 'value']].dropna().sort_values('date')
                        if len(valid) > 20:
                            df_cleaned = valid
                            print(f"Successfully parsed {len(df_cleaned)} rows from {raw_path}")
                            break
            except Exception as e:
                print(f"Warning: Could not parse local raw file {raw_path}: {e}")

    # 2. If no local file or failed, attempt download from OpenStat official mirror
    if df_cleaned is None:
        print("Fetching NHS England Monthly A&E Attendances dataset online...")
        try:
            url = "https://openstat.uk/dashboards_data.php?slug=nhs-ae-waiting-times"
            headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
            resp = requests.get(url, headers=headers, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                raw_df = pd.DataFrame(data)
                if "Date" in raw_df.columns and "Total_Attendances" in raw_df.columns:
                    raw_df['date'] = pd.to_datetime(raw_df['Date'] + "-01", errors='coerce')
                    raw_df['value'] = pd.to_numeric(raw_df['Total_Attendances'], errors='coerce')
                    df_cleaned = raw_df[['date', 'value']].dropna().sort_values('date').reset_index(drop=True)
                    print(f"Successfully fetched {len(df_cleaned)} months from official mirror.")
        except Exception as e:
            print(f"Mirror download encountered: {e}")

    # 3. Validation or manual instruction printing
    if df_cleaned is not None and len(df_cleaned) > 24:
        df_cleaned['date'] = df_cleaned['date'].dt.strftime('%Y-%m-01')
        df_cleaned = df_cleaned.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
        # Ensure starts from Aug 2010 if available
        aug_2010 = pd.to_datetime("2010-08-01")
        df_cleaned['dt'] = pd.to_datetime(df_cleaned['date'])
        df_cleaned = df_cleaned[df_cleaned['dt'] >= aug_2010].drop(columns=['dt'])
        df_cleaned.to_csv(output_path, index=False)
        print(f" Saved clean NHS A&E dataset to: {output_path}")
        print(f"  Rows: {len(df_cleaned)}, Range: {df_cleaned['date'].min()} to {df_cleaned['date'].max()}")
        print(f"  Columns: {list(df_cleaned.columns)}")
    else:
        print("\n" + "="*70)
        print("ERROR: Automatic fetch of NHS England Monthly A&E data failed or incomplete.")
        print("MANUAL DOWNLOAD INSTRUCTIONS:")
        print("1. Visit NHS England A&E Statistics:")
        print("   https://www.england.nhs.uk/statistics/statistical-work-areas/ae-waiting-times-and-activity/")
        print("2. Download the 'Monthly A&E Time Series' Excel workbook (.xls or .xlsx).")
        print("3. Place the file into:")
        print(f"   {RAW_DIR}/Monthly-AE-Time-Series.xls")
        print("4. Re-run: python prepare_data.py")
        print("="*70 + "\n")


def prepare_electricity_daily():
    """
    Prepares UK electricity demand from NESO (National Grid) historic demand data,
    half-hourly, aggregated to DAILY total demand. Uses at least 4 recent years (2021-2024).
    Saves to data/electricity_daily.csv with columns: date,value
    """
    output_path = os.path.join(DATA_DIR, "electricity_daily.csv")
    print("\n--- [2/2] Preparing NESO UK Electricity Daily Demand ---")

    neso_urls = {
        2021: "https://api.neso.energy/dataset/8f2fe0af-871c-488d-8bad-960426f24601/resource/18c69c42-f20d-46f0-84e9-e279045befc6/download/demanddata_2021.csv",
        2022: "https://api.neso.energy/dataset/8f2fe0af-871c-488d-8bad-960426f24601/resource/bb44a1b5-75b1-4db2-8491-257f23385006/download/demanddata_2022.csv",
        2023: "https://api.neso.energy/dataset/8f2fe0af-871c-488d-8bad-960426f24601/resource/bf5ab335-9b40-4ea4-b93a-ab4af7bce003/download/demanddata_2023.csv",
        2024: "https://api.neso.energy/dataset/8f2fe0af-871c-488d-8bad-960426f24601/resource/f6d02c0f-957b-48cb-82ee-09003f2ba759/download/demanddata_2024.csv"
    }

    daily_dfs = []

    # First check if files exist locally in data/raw
    for y in sorted(neso_urls.keys()):
        raw_year_file = os.path.join(RAW_DIR, f"demanddata_{y}.csv")
        df_year = None
        if os.path.exists(raw_year_file):
            try:
                print(f"Reading local raw file: {raw_year_file}")
                df_year = pd.read_csv(raw_year_file)
            except Exception as e:
                print(f"Failed reading {raw_year_file}: {e}")

        # If not local, download from NESO Data Portal API
        if df_year is None:
            url = neso_urls[y]
            print(f"Downloading NESO demand data for year {y} from {url}...")
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    df_year = pd.read_csv(io.StringIO(resp.text))
                    # cache raw file
                    try:
                        with open(raw_year_file, "w", encoding="utf-8") as f:
                            f.write(resp.text)
                    except Exception:
                        pass
                else:
                    print(f"HTTP {resp.status_code} when downloading {y}")
            except Exception as e:
                print(f"Download failed for year {y}: {e}")

        if df_year is not None:
            df_year.columns = [str(c).strip().upper() for c in df_year.columns]
            date_col = next((c for c in df_year.columns if "SETTLEMENT_DATE" in c or "DATE" in c), None)
            demand_col = next((c for c in df_year.columns if c in ["ND", "TSD", "ENGLAND_WALES_DEMAND"]), None)
            if date_col and demand_col:
                df_year[date_col] = pd.to_datetime(df_year[date_col], format='mixed', errors='coerce')
                df_year[demand_col] = pd.to_numeric(df_year[demand_col], errors='coerce')
                daily = df_year.groupby(df_year[date_col].dt.strftime('%Y-%m-%d'))[demand_col].sum().reset_index()
                daily.columns = ['date', 'value']
                daily_dfs.append(daily)
                print(f"  Processed {len(daily)} days for {y}")

    if daily_dfs:
        combined = pd.concat(daily_dfs).drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
        # remove any potential incomplete edge dates
        combined = combined.dropna().reset_index(drop=True)
        combined.to_csv(output_path, index=False)
        print(f" Saved clean NESO electricity dataset to: {output_path}")
        print(f"  Rows: {len(combined)}, Range: {combined['date'].min()} to {combined['date'].max()}")
        print(f"  Columns: {list(combined.columns)}")
    else:
        print("\n" + "="*70)
        print("ERROR: Automatic fetch of NESO Electricity demand data failed.")
        print("MANUAL DOWNLOAD INSTRUCTIONS:")
        print("1. Visit NESO Data Portal Historic Demand Data:")
        print("   https://www.neso.energy/data-portal/historic-demand-data")
        print("2. Download CSV files for 2021, 2022, 2023, and 2024 (e.g. demanddata_2021.csv).")
        print("3. Place them in:")
        print(f"   {RAW_DIR}/demanddata_2021.csv, demanddata_2022.csv, etc.")
        print("4. Re-run: python prepare_data.py")
        print("="*70 + "\n")


if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    prepare_ae_monthly()
    prepare_electricity_daily()
