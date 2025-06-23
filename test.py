# final_financial_report_script.py

import pandas as pd
import numpy as np
import json
import random

def build_sample_annual_df() -> pd.DataFrame:
    """
    Create a toy Apple fiscal-year payroll table that matches the
    final clarified structure.
    """
    # --- static look-ups ----------------------------------------------------
    months   = ["October 2024", "November 2024", "December 2024",
                "January 2025", "February 2025", "March 2025",
                "April 2025",  "May 2025",      "June 2025",
                "July 2025",   "August 2025",   "September 2025"]

    periods  = [f"P{i}" for i in range(1, 13)]
    quarters = (["FY Q1 2025"] * 3 + ["FY Q2 2025"] * 3 +
                ["FY Q3 2025"] * 3 + ["FY Q4 2025"] * 3)

    # Correctly map quarters to their date ranges
    report_dates_by_quarter = {
        "FY Q1 2025": "September 29, 2024 through December 28, 2024",
        "FY Q2 2025": "December 29, 2024 through March 29, 2025",
        "FY Q3 2025": "March 30, 2025 through June 28, 2025",
        "FY Q4 2025": "June 29, 2025 through September 27, 2025"
    }

    divisions = ["APPLECARE", "RETAIL", "CORPORATE"]

    # --- build one row per (division, period) -------------------------------
    rows = []
    for division in divisions:
        for idx, (month, period, qtr) in enumerate(zip(months, periods, quarters), start=1):
            rows.append({
                "PLAN_N":            87886,                # dummy plan ID
                "FISCAL_YEAR":       2025,
                "PERIOD_NM":         period,               # e.g. P1 … P12
                "REPORTING_PERIOD": f"Q{((idx-1)//3)+1} Fiscal Year 2025",
                "REPORT_QUARTER":    qtr,                  # “FY Qn 2025”
                "REPORT_DATES":      report_dates_by_quarter[qtr],
                "DIVISION":          division,
                "MONTH_YEAR_P":     f"{month} - {period}", # “October 2024 - P1”
                # random but repeatable-ish dollar figures
                "AMOUNT":            round(random.uniform(1_500_000, 6_000_000), 2)
            })

    return pd.DataFrame(rows)


def transform_finance_data(df: pd.DataFrame):
    """
    Transforms an annual financial DataFrame into monthly, quarterly, and title data structures.
    
    This version is updated to use the exact column names from the provided
    `build_sample_annual_df` function.
    """
    
    # 1. Preprocess the DataFrame
    df_processed = df.copy()
    df_processed['DIVISION'] = df_processed['DIVISION'].replace({
        'APPLECARE': 'Apple Care',
        'CORPORATE': 'Corporate'
    })
    df_processed['AMOUNT'] = pd.to_numeric(df_processed['AMOUNT'])

    # Helper function to convert a pivot table to the desired dictionary format
    def to_custom_dict(pivot_df):
        pivot_df = pivot_df.round(2)
        data_dict = {'Division': list(pivot_df.index)}
        for col in pivot_df.columns:
            data_dict[col] = [val if pd.notna(val) else None for val in pivot_df[col].tolist()]
        return data_dict

    # 2. Generate Monthly Data (data1, data2)
    monthly_pivot = df_processed.pivot_table(
        index='DIVISION',
        columns='MONTH_YEAR_P',
        values='AMOUNT',
        aggfunc='sum'
    )
    
    # Sort columns chronologically using the period number from 'MONTH_YEAR_P'
    try:
        period_numbers = {col: int(col.split(' - P')[1]) for col in monthly_pivot.columns}
    except (IndexError, ValueError):
        raise ValueError("Column 'MONTH_YEAR_P' is not in the expected 'Month Year - PX' format.")

    sorted_cols = sorted(monthly_pivot.columns, key=lambda col: period_numbers[col])
    monthly_pivot = monthly_pivot[sorted_cols]

    # Add calculated rows and apply specified division order
    division_order = ['Apple Care', 'RETAIL', 'Corporate']
    monthly_pivot = monthly_pivot.reindex(division_order)
    
    sum_row = monthly_pivot.sum(axis=0)
    monthly_pivot.loc['Apple Weekly & Biweekly'] = sum_row
    monthly_pivot.loc['Grand Total'] = sum_row
    monthly_pivot.loc['DASHED_SEPARATOR'] = np.nan
    
    final_division_order = ['Apple Care', 'RETAIL', 'Corporate', 'Apple Weekly & Biweekly', 'DASHED_SEPARATOR', 'Grand Total']
    monthly_pivot = monthly_pivot.reindex(final_division_order)

    # Split into two halves for data1 and data2
    cols_data1 = [col for col in sorted_cols if period_numbers[col] <= 6]
    cols_data2 = [col for col in sorted_cols if period_numbers[col] > 6]
    
    data1 = to_custom_dict(monthly_pivot[cols_data1])
    data2 = to_custom_dict(monthly_pivot[cols_data2])
    
    # 3. Generate Quarterly Data (data3)
    # Uses 'REPORT_QUARTER' column for grouping
    quarterly_summary = df_processed.groupby(['REPORT_QUARTER', 'DIVISION'])['AMOUNT'].sum().reset_index()
    quarterly_pivot = quarterly_summary.pivot_table(
        index='DIVISION',
        columns='REPORT_QUARTER',
        values='AMOUNT',
        aggfunc='sum'
    )

    # Sort quarter columns chronologically
    def get_q_sort_key(q_string):
        try:
            parts = q_string.split(' ')
            year = int(parts[2])
            q_num = int(parts[1][1:])
            return (year, q_num)
        except (IndexError, ValueError):
             raise ValueError("Column 'REPORT_QUARTER' is not in the expected 'FY QX Year' format.")

    sorted_q_cols = sorted(quarterly_pivot.columns, key=get_q_sort_key)
    quarterly_pivot = quarterly_pivot[sorted_q_cols]

    # Add calculated columns/rows and apply specified order
    quarterly_pivot = quarterly_pivot.reindex(division_order)
    quarterly_pivot['Grand Total'] = quarterly_pivot.sum(axis=1)
    
    sum_row_q = quarterly_pivot.sum(axis=0)
    quarterly_pivot.loc['Apple Weekly & Biweekly'] = sum_row_q
    quarterly_pivot.loc['Grand Total'] = sum_row_q
    quarterly_pivot.loc['DASHED_SEPARATOR'] = np.nan
    
    quarterly_pivot = quarterly_pivot.reindex(final_division_order)

    data3 = to_custom_dict(quarterly_pivot)

    # 4. Generate Title Data (title_data)
    try:
        # Uses 'PERIOD_NM' to reliably find the latest entry
        df_processed['PERIOD_NUM'] = df_processed['PERIOD_NM'].str.replace('P', '').astype(int)
        latest_row = df_processed.loc[df_processed['PERIOD_NUM'].idxmax()]
        # Uses 'REPORTING_PERIOD' directly for the title
        reformatted_q_name = latest_row['REPORTING_PERIOD']
        
        # Parse overall date range from 'REPORT_DATES'
        dates_df = df_processed['REPORT_DATES'].str.split(' through ', expand=True)
        start_dates = pd.to_datetime(dates_df[0], format='%B %d, %Y')
        end_dates = pd.to_datetime(dates_df[1], format='%B %d, %Y')
        overall_start_str = start_dates.min().strftime('%B %d, %Y')
        overall_end_str = end_dates.max().strftime('%B %d, %Y')

    except (AttributeError, KeyError, IndexError, TypeError) as e:
        print(f"Error generating title data: {e}")
        reformatted_q_name, overall_start_str, overall_end_str = "N/A", "N/A", "N/A"
    
    title_data = {
        'Fiscal Quarter': reformatted_q_name,
        'Start Date': overall_start_str,
        'End Date': overall_end_str,
    }

    return data1, data2, data3, title_data


# This main block executes when the script is run directly
if __name__ == "__main__":
    
    print("1. Building sample DataFrame...")
    annual_df = build_sample_annual_df()
    print(annual_df.head())
    
    print("2. Transforming DataFrame into required data structures...")
    data1, data2, data3, title_data = transform_finance_data(annual_df)
    
    print("3. Transformation complete. Printing results:")
    
    # Print the results in a readable JSON format.
    print("\n\n--- title_data ---")
    print(json.dumps(title_data, indent=2))
    
    print("\n\n--- data1 (H1 Monthly) ---")
    print(json.dumps(data1, indent=2))

    print("\n\n--- data2 (H2 Monthly) ---")
    print(json.dumps(data2, indent=2))

    print("\n\n--- data3 (Quarterly) ---")
    print(json.dumps(data3, indent=2))
