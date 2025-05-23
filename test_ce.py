import pandas as pd
from xlsxwriter.workbook import Workbook
import concurrent.futures
import os
import time
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Iterator, Deque, Union # Added Deque
import io # For BytesIO
import collections # For deque
import multiprocessing # For freeze_support

# --- Script Version ---
SCRIPT_VERSION = "1.2.0"

# --- Setup Root Logger (placeholder, configured in main) ---
logger = logging.getLogger(__name__)

def setup_worker_logging(loglevel_str: str) -> None:
    """Sets up basic logging for a worker process if not already configured."""
    # Check if the root logger already has handlers (configured by parent or another call)
    # This is a simple check; more sophisticated checks might be needed in complex scenarios.
    if not logging.getLogger().hasHandlers():
        try:
            level = getattr(logging, loglevel_str.upper(), logging.INFO)
            logging.basicConfig(
                level=level,
                format='%(asctime)s - %(levelname)s - %(name)s - (%(processName)s) - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        except Exception as e:
            # Fallback if something goes wrong with logging setup in worker
            print(f"WORKER (PID {os.getpid()}) - Logging setup error: {e}")


def write_excel_worker(header_list: List[str], 
                       data_rows_list: List[List[Any]],
                       loglevel_str: str) -> bytes: # Returns bytes of the Excel file
    """
    Worker function executed by each process in the ProcessPoolExecutor.
    Creates an Excel file in an in-memory buffer and returns its bytes.
    """
    setup_worker_logging(loglevel_str) # Ensure worker logging is configured
    worker_logger = logging.getLogger(__name__ + ".worker") # Specific logger for worker context

    worker_logger.debug(f"Starting to generate Excel data in memory with {len(data_rows_list)} data rows.")
    
    io_buffer = io.BytesIO()
    try:
        workbook_options = {
            'constant_memory': True, 
            'strings_to_urls': False,
            'default_date_format': 'yyyy-mm-dd', 
            'nan_inf_to_errors': True 
        }
        # Pass io_buffer as the first argument to Workbook to write to memory
        with Workbook(io_buffer, workbook_options) as workbook:
            worksheet = workbook.add_worksheet()
            write_row_method = worksheet.write_row # Micro-optimization

            if header_list:
                write_row_method(0, 0, header_list)
            
            for i, row_data in enumerate(data_rows_list):
                write_row_method(i + 1, 0, row_data)
        
        excel_bytes = io_buffer.getvalue()
        worker_logger.debug(f"Finished generating {len(excel_bytes)} bytes of Excel data in memory.")
        return excel_bytes
    except Exception as e:
        worker_logger.error(f"!!! Error generating Excel data in memory: {e}", exc_info=True)
        raise Exception(f"Error in PID {os.getpid()} generating Excel data: {e}")
    finally:
        io_buffer.close()


def convert_large_csv_to_excel_parallel(
    csv_filepath: str, 
    ssn_column_name: str,
    output_dir: str,
    output_prefix: str,
    max_data_rows_per_file: int,
    read_chunk_rows: int, # Renamed from read_chunk_size
    num_workers: Optional[int],
    loglevel_str: str # Passed from main for worker logging setup
) -> None:
    """
    Converts a large CSV file to multiple Excel (.xlsx) files, processing in chunks
    and generating Excel file bytes in parallel using a ProcessPoolExecutor.
    The main process then writes these bytes to disk.

    IMPORTANT:
    - The input CSV file MUST be pre-sorted by the `ssn_column_name`. Failure to
      do so will result in SSN groups being split across output files if they
      span different read chunks.
    - If a single group of records for the same SSN exceeds `max_data_rows_per_file`,
      AND this group size is less than Excel's hard limit of 1,048,576 rows,
      the output file for that specific group will be larger than `max_data_rows_per_file`.
    - If a single SSN group itself exceeds Excel's hard row limit (1,048,576 rows),
      a ValueError will be raised as such a file cannot be correctly created.
    - `max_data_rows_per_file` refers to data rows. The header row adds one
      additional row to the physical Excel file.

    Args:
        csv_filepath: Path to the input CSV file.
        ssn_column_name: The exact name of the SSN column in the CSV.
        output_dir: Directory where the output Excel files will be saved.
        output_prefix: Prefix for the output Excel filenames.
        max_data_rows_per_file: Maximum number of data rows per Excel file.
        read_chunk_rows: Number of rows to read from CSV into memory at a time.
        num_workers: Number of worker processes for generating Excel file bytes.
        loglevel_str: The logging level string (e.g., "INFO", "DEBUG").
    """
    main_logger = logging.getLogger(__name__ + ".main") # Specific logger for main process
    start_time = time.time()
    excel_row_hard_limit = 1_048_576 # Excel's maximum number of rows per sheet

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            main_logger.info(f"Created output directory: {output_dir}")
        except OSError as e:
            main_logger.error(f"Failed to create output directory {output_dir}: {e}", exc_info=True)
            raise

    effective_num_workers: int
    if num_workers is None:
        cpu_c = os.cpu_count() or 1
        # Heuristic: allow more workers than CPUs for I/O overlap, but cap reasonably.
        # For very I/O bound tasks (like writing already compressed bytes returned by workers),
        # even num_workers = cpu_c might be sufficient if disk is bottleneck.
        # For CPU bound part in worker (xlsxwriter compression), cpu_c is a good start.
        effective_num_workers = min(cpu_c + 4, 32) if cpu_c else 2 
    else:
        effective_num_workers = num_workers


    main_logger.info(f"Starting parallel conversion of '{csv_filepath}' (Version: {SCRIPT_VERSION})...")
    main_logger.info(f"Parameters: SSN column='{ssn_column_name}', Max data rows/file={max_data_rows_per_file}, "
                     f"Output dir='{output_dir}', File prefix='{output_prefix}', "
                     f"Read chunk rows={read_chunk_rows}, Num workers={effective_num_workers}")
    main_logger.warning("Ensure the input CSV is pre-sorted by the SSN column for correct group processing.")
    main_logger.info("For potential performance boost with XlsxWriter, consider 'pip install XlsxWriter[c]' if C extensions are not already used.")


    file_counter: int = 0
    # Using collections.deque for potentially better memory re-use with .clear()
    rows_being_assembled_for_dispatch: Deque[List[Any]] = collections.deque()
    current_ssn_group_buffer: Deque[List[Any]] = collections.deque()
    active_ssn: Optional[str] = None
    csv_header: Optional[List[str]] = None
    total_rows_processed: int = 0
    
    # Using 'object' for SSN column to handle NAs as None/np.nan, then convert to string.
    dtype_spec: Dict[str, str] = {ssn_column_name: 'object'} 

    with concurrent.futures.ProcessPoolExecutor(max_workers=effective_num_workers) as executor:
        futures_map: Dict[concurrent.futures.Future, str] = {} # Map future to output_filepath for logging

        try:
            chunk_iterator: Iterator[pd.DataFrame] = pd.read_csv(
                csv_filepath, 
                chunksize=read_chunk_rows, 
                dtype=dtype_spec, 
                iterator=True
            )

            for chunk_df in chunk_iterator:
                if csv_header is None:
                    csv_header = chunk_df.columns.tolist()
                    if ssn_column_name not in csv_header:
                        main_logger.error(f"SSN column '{ssn_column_name}' not found in CSV header: {csv_header}")
                        raise ValueError(f"SSN column '{ssn_column_name}' not found in CSV header: {csv_header}")
                    main_logger.info(f"CSV Header identified: {csv_header}")
                
                ssn_idx: int = csv_header.index(ssn_column_name)
                current_chunk_row_count = len(chunk_df)
                total_rows_processed += current_chunk_row_count
                main_logger.debug(f"Processing chunk of {current_chunk_row_count} rows. Total rows processed so far: {total_rows_processed}")

                for row_tuple in chunk_df.itertuples(index=False, name=None): # Faster iteration
                    # Handle potential pd.NA or other nulls if 'object' dtype is used
                    ssn_val = row_tuple[ssn_idx]
                    row_ssn: str = str(ssn_val) if pd.notna(ssn_val) else "_NA_SSN_"
                    
                    row_data_as_list = list(row_tuple)

                    if active_ssn is None: 
                        active_ssn = row_ssn
                        current_ssn_group_buffer.append(row_data_as_list)
                    elif row_ssn == active_ssn: 
                        current_ssn_group_buffer.append(row_data_as_list)
                    else: 
                        # New SSN: current_ssn_group_buffer is complete for 'active_ssn'
                        if len(current_ssn_group_buffer) >= excel_row_hard_limit: # Check against 1M row Excel limit
                            main_logger.error(f"SSN group for '{active_ssn}' has {len(current_ssn_group_buffer)} rows, "
                                              f"exceeding Excel's hard limit of {excel_row_hard_limit} rows per sheet.")
                            raise ValueError(f"Single SSN group for '{active_ssn}' is too large for Excel.")

                        if rows_being_assembled_for_dispatch and \
                           (len(rows_being_assembled_for_dispatch) + len(current_ssn_group_buffer) > max_data_rows_per_file):
                            
                            file_counter += 1
                            output_filename = f"{output_prefix}_{file_counter}.xlsx"
                            output_filepath = os.path.join(output_dir, output_filename)
                            
                            main_logger.info(f"Dispatching data for '{output_filename}' ({len(rows_being_assembled_for_dispatch)} rows) to worker.")
                            # Pass a copy of the deque's contents as a list
                            future = executor.submit(write_excel_worker, csv_header, list(rows_being_assembled_for_dispatch), loglevel_str)
                            futures_map[future] = output_filepath
                            rows_being_assembled_for_dispatch.clear()

                        rows_being_assembled_for_dispatch.extend(current_ssn_group_buffer)
                        
                        active_ssn = row_ssn
                        current_ssn_group_buffer.clear()
                        current_ssn_group_buffer.append(row_data_as_list)
            
            # ---- After all chunks ----
            if current_ssn_group_buffer: # Handle the very last SSN group
                if len(current_ssn_group_buffer) >= excel_row_hard_limit:
                    main_logger.error(f"Final SSN group for '{active_ssn}' has {len(current_ssn_group_buffer)} rows, "
                                      f"exceeding Excel's hard limit.")
                    raise ValueError(f"Final SSN group for '{active_ssn}' is too large for Excel.")

                if rows_being_assembled_for_dispatch and \
                   (len(rows_being_assembled_for_dispatch) + len(current_ssn_group_buffer) > max_data_rows_per_file):
                    
                    file_counter += 1
                    output_filename = f"{output_prefix}_{file_counter}.xlsx"
                    output_filepath = os.path.join(output_dir, output_filename)
                    main_logger.info(f"Dispatching data for '{output_filename}' (final boundary) with {len(rows_being_assembled_for_dispatch)} rows.")
                    future = executor.submit(write_excel_worker, csv_header, list(rows_being_assembled_for_dispatch), loglevel_str)
                    futures_map[future] = output_filepath
                    rows_being_assembled_for_dispatch.clear()
                
                rows_being_assembled_for_dispatch.extend(current_ssn_group_buffer)
                current_ssn_group_buffer.clear() 

            if rows_being_assembled_for_dispatch: 
                file_counter += 1
                output_filename = f"{output_prefix}_{file_counter}.xlsx"
                output_filepath = os.path.join(output_dir, output_filename)
                main_logger.info(f"Dispatching final data for '{output_filename}' with {len(rows_being_assembled_for_dispatch)} rows.")
                future = executor.submit(write_excel_worker, csv_header, list(rows_being_assembled_for_dispatch), loglevel_str)
                futures_map[future] = output_filepath
                rows_being_assembled_for_dispatch.clear()

            main_logger.info(f"\nAll CSV data processed. Total rows encountered: {total_rows_processed}. "
                             f"Waiting for {len(futures_map)} Excel generation tasks to complete...")
            
            successful_writes = 0
            failed_writes = 0
            for i, future_item in enumerate(concurrent.futures.as_completed(futures_map.keys())):
                output_filepath_for_future = futures_map[future_item]
                try:
                    excel_bytes_content = future_item.result() # This will raise an exception if the worker task raised one
                    # Now write the bytes received from worker to disk in the main process
                    with open(output_filepath_for_future, 'wb') as f_disk:
                        f_disk.write(excel_bytes_content)
                    main_logger.debug(f"Successfully wrote {len(excel_bytes_content)} bytes to '{output_filepath_for_future}'.")
                    successful_writes +=1
                except Exception as exc:
                    main_logger.error(f"!!! A task for '{output_filepath_for_future}' generated an exception: {exc}", exc_info=False)
                    failed_writes += 1
                
                if (i + 1) % 10 == 0 or (i + 1) == len(futures_map):
                    main_logger.info(f"{i+1}/{len(futures_map)} generation tasks processed (Successful: {successful_writes}, Failed: {failed_writes}).")
            
            main_logger.info(f"All generation tasks finished. Successful files: {successful_writes}, Failed files: {failed_writes}.")

        except FileNotFoundError:
            main_logger.error(f"Input CSV file '{csv_filepath}' not found.", exc_info=True)
            raise 
        except ValueError as ve: 
            main_logger.error(f"ValueError during processing: {ve}", exc_info=True)
            raise
        except Exception as e:
            main_logger.error(f"An unexpected error occurred in the main processing loop: {e}", exc_info=True)
            raise
    
    end_time = time.time()
    main_logger.info(f"\nParallel conversion finished. {file_counter} Excel file(s) generation attempted.")
    main_logger.info(f"Total time taken: {end_time - start_time:.2f} seconds.")


def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description=f"CSV to Excel Converter (v{SCRIPT_VERSION})\n"
                    "Converts a large CSV file to multiple Excel (.xlsx) files in parallel, "
                    "keeping SSN groups intact.\n\n"
                    "IMPORTANT:\n"
                    "- Input CSV file MUST be pre-sorted by the SSN column.\n"
                    "- For potential XlsxWriter speed-up, consider 'pip install XlsxWriter[c]'.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help
    )
    parser.add_argument("csv_filepath", 
                        help="Path to the input CSV file.")
    parser.add_argument("ssn_column_name", 
                        help="Name of the SSN column in the CSV (case-sensitive).")
    
    default_output_dir = f"excel_output_parallel_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    parser.add_argument("--output_dir", default=default_output_dir,
                        help="Directory to save output Excel files.")
    parser.add_argument("--output_prefix", default="excel_part",
                        help="Prefix for the output Excel filenames.")
    parser.add_argument("--max_data_rows", type=int, default=9999,
                        help="Maximum number of data rows per Excel file. "
                             "A file may exceed this if a single SSN group is larger but still within Excel's overall row limit (approx 1M rows).")
    parser.add_argument("--chunk-rows", type=int, default=200000, # Renamed argument
                        help="Number of CSV rows to read into memory at a time.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel worker processes for generating Excel file data. "
                             "Default: os.cpu_count() + 4, capped at 32. Adjust based on CPU and I/O.")
    parser.add_argument("--loglevel", default="INFO", 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help="Set the logging level.")
    parser.add_argument('--version', action='version', version=f'%(prog)s {SCRIPT_VERSION}')


    args = parser.parse_args()

    # --- Configure Root Logger ---
    # This configuration will be inherited by child processes if they don't reconfigure,
    # but re-configuring in worker (as done in write_excel_worker) is safer for 'spawn' context.
    logging.basicConfig(
        level=getattr(logging, args.loglevel.upper()),
        format='%(asctime)s - %(levelname)s - %(name)s - (%(processName)s) - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # --- Run Conversion ---
    try:
        convert_large_csv_to_excel_parallel(
            csv_filepath=args.csv_filepath,
            ssn_column_name=args.ssn_column_name,
            output_dir=args.output_dir,
            output_prefix=args.output_prefix,
            max_data_rows_per_file=args.max_data_rows,
            read_chunk_rows=args.chunk_rows, # Use renamed arg
            num_workers=args.workers,
            loglevel_str=args.loglevel # Pass loglevel string to main converter
        )
    except Exception:
        # Errors are already logged by the convert function or its callees
        logging.getLogger(__name__).critical(
            "Script terminated due to an error. Please check the logs above for details."
        )
        # Consider sys.exit(1) for CI/CD or scripted environments
        # import sys
        # sys.exit(1) 

if __name__ == '__main__':
    # freeze_support() is necessary for PyInstaller/cx_Freeze on Windows
    # and good practice for multiprocessing applications.
    multiprocessing.freeze_support() 
    main()
