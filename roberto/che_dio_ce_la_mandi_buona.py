import pandas as pd
import numpy as np

def label_csv_ranges():
    # Ask for input file
    input_file = input("Enter the path to your CSV file: ")
    
    # Read the CSV file
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print("Error: File not found!")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Initialize the label column with NaN
    df['label'] = np.nan

    # Get label value from user
    label_value = input("Enter the label value for the intervals: ")

    # Get ranges from user
    print("\nEnter your ranges (format: start-end), one at a time.")
    print("Press Enter without input when done.")
    print("Example: 0-50")
    
    ranges = []
    while True:
        range_input = input("Enter range: ").strip()
        if not range_input:
            break
            
        try:
            start, end = map(int, range_input.split('-'))
            if start < 0 or end > len(df):
                print("Error: Range out of bounds. Please try again.")
                continue
            ranges.append((start, end))
        except ValueError:
            print("Invalid format. Please use start-end (e.g., 0-50)")
            continue

    # Apply labels to specified ranges
    for start, end in ranges:
        df.loc[start:end-1, 'label'] = label_value

    # Ask for output filename
    output_file = input("\nEnter the output CSV filename (e.g., labeled_data.csv): ")
    
    # Save the modified DataFrame
    try:
        df.to_csv(output_file, index=False)
        print(f"\nSuccessfully saved labeled data to {output_file}")
        
        # Print some statistics
        total_labeled = df['label'].notna().sum()
        print(f"\nStatistics:")
        print(f"Total rows: {len(df)}")
        print(f"Labeled rows: {total_labeled}")
        print(f"Unlabeled rows: {len(df) - total_labeled}")
        
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    print("CSV Interval Labeler")
    print("This script will add a label column to your CSV and assign values to specified ranges")
    label_csv_ranges()