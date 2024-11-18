import pandas as pd
import numpy as np

# Initialize configuration variables
CSV_PATH = None
OUTPUT_PATH = None  
LABEL_VALUE = None

RANGES_STRING = "25-70,105-155,190-240,270-320,350-400,430-480,515-565,600-650,685-735,775-825"
CUT_OFF_ROW = None

def parse_ranges(ranges_str):
    """
    Parse a string of ranges into a list of tuples.
    
    Args:
        ranges_str (str): String in format "start-end,start-end,..."
    
    Returns:
        list: List of tuples [(start, end+1), ...]
    """
    ranges = []
    try:
        # Split by comma and process each range
        for range_str in ranges_str.split(','):
            # Remove whitespace and split by hyphen
            start, end = map(int, range_str.strip().split('-'))
            # Add 1 to end to make range inclusive
            ranges.append((start, end + 1))
        return ranges
    except ValueError as e:
        print(f"Error parsing ranges: {e}")
        print("Please use format: start-end,start-end (e.g., '0-50,100-150')")
        return []

def label_csv_ranges(csv_path, output_path, label_value, ranges_str, cut_off=None):
    """
    Label specific ranges in a CSV file with a given value and optionally truncate data.
    
    Args:
        csv_path (str): Path to input CSV file
        output_path (str): Path to save the labeled CSV
        label_value (int): Value to assign in the specified ranges
        ranges_str (str): String of ranges in format "start-end,start-end,..."
        cut_off (int, optional): Index at which to truncate the data. Rows >= cut_off will be removed.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        original_length = len(df)
        
        # Apply cut-off if specified
        if cut_off is not None:
            if cut_off <= 0 or cut_off >= len(df):
                print(f"Warning: Invalid cut-off point {cut_off}. Must be between 1 and {len(df)-1}")
            else:
                df = df.iloc[:cut_off]
                print(f"\nTruncated data from row {cut_off} onwards")
                print(f"Removed {original_length - len(df)} rows")
        
        # Initialize the label column with zeros
        df['label'] = 0
        
        # Parse and validate ranges
        ranges = parse_ranges(ranges_str)
        valid_ranges = []
        
        for start, end in ranges:
            if start < 0 or end > len(df) + 1:  # +1 because end is now inclusive
                print(f"Warning: Range {start}-{end-1} is out of bounds and will be skipped.")
                continue
            if start >= end:
                print(f"Warning: Invalid range {start}-{end-1} (start >= end) will be skipped.")
                continue
            valid_ranges.append((start, end))
        
        # Apply labels to specified ranges
        for start, end in valid_ranges:
            df.loc[start:end-1, 'label'] = label_value
        
        # Save the modified DataFrame
        df.to_csv(output_path, index=False)
        
        # Print statistics and ranges labeled
        total_labeled = (df['label'] == label_value).sum()
        print("\nRanges successfully labeled:")
        for start, end in valid_ranges:
            print(f"Rows {start} to {end-1} (inclusive)")
            
        print("\nLabeling Statistics:")
        print(f"Original number of rows: {original_length}")
        print(f"Final number of rows: {len(df)}")
        print(f"Labeled rows (value={label_value}): {total_labeled}")
        print(f"Default labeled rows (value=0): {len(df) - total_labeled}")
        print(f"\nFile saved successfully to: {output_path}")
        
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    print("CSV Interval Labeler")
    print(f"Processing file: {CSV_PATH}")
    print(f"Label value: {LABEL_VALUE}")
    print(f"Ranges to be labeled: {RANGES_STRING}")
    if CUT_OFF_ROW is not None:
        print(f"Data will be truncated from row {CUT_OFF_ROW} onwards")
    label_csv_ranges(CSV_PATH, OUTPUT_PATH, LABEL_VALUE, RANGES_STRING, CUT_OFF_ROW)