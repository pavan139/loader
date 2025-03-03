# example.py

import os
import logging
from modules.loader import load_and_audit_data
from modules.transformer import apply_advanced_filters_from_config
from modules.joiner import join_dataframes

# Configure logging to output to the console with a more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Get the directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    """Main function to demonstrate the data engineering pipeline."""
    try:
        # --- Step 1: Load and Audit Data ---
        # Use absolute paths based on the script's location
        file_path1 = os.path.join(BASE_DIR, "data/sample1.csv")
        file_path2 = os.path.join(BASE_DIR, "data/sample2.csv")
        file_type = "csv"
        audit_config = os.path.join(BASE_DIR, "config/audits/audit1.yaml")

        logging.info("=== Step 1: Loading and Auditing Data ===")
        df1 = load_and_audit_data(file_path1, file_type, audit_config)
        logging.info(f"Loaded first dataset with shape: {df1.shape}")
        
        df2 = load_and_audit_data(file_path2, file_type, audit_config)
        logging.info(f"Loaded second dataset with shape: {df2.shape}")

        # --- Step 2: Apply Transformations ---
        logging.info("=== Step 2: Applying Transformations ===")
        filter_config = os.path.join(BASE_DIR, "config/filters/filter1.yaml")
        
        df1_filtered = apply_advanced_filters_from_config(df1, filter_config)
        logging.info(f"Filtered first dataset, new shape: {df1_filtered.shape}")
        
        # --- Step 3: Join DataFrames ---
        logging.info("=== Step 3: Joining DataFrames ===")
        join_config = os.path.join(BASE_DIR, "config/join_configs/join1.yaml")
        
        joined_df = join_dataframes(df1_filtered, df2, join_config)
        logging.info(f"Joined dataset final shape: {joined_df.shape}")
        
        # Display sample of the final result
        logging.info("=== Final Dataset Sample ===")
        print(joined_df.head())
        
        # Print summary statistics
        logging.info("=== Summary Statistics ===")
        for col in joined_df.select_dtypes(include=['number']).columns:
            print(f"\nStats for '{col}':")
            print(joined_df[col].describe())
        
        return joined_df
        
    except Exception as e:
        logging.error(f"Error in data engineering pipeline: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()