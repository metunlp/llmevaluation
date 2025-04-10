"""
Robust Kappa calculation for DatasetX dataset.

This script calculates robust Fleiss kappa for the DatasetX dataset annotations.
"""

import os
import sys
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from robust_kappa import RobustKappa

# Paths to the annotation files
DATASETX_DIR = "datasetx"
annotator_files = [
    os.path.join(DATASETX_DIR, "datasetx_annotator1.xlsx"),
    os.path.join(DATASETX_DIR, "datasetx_annotator2.xlsx"),
    os.path.join(DATASETX_DIR, "datasetx_annotator3.xlsx")
]

# Verify all files exist
for file_path in annotator_files:
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

# Let's determine the column titles to analyze
# Since we can't directly read the Excel files here, we'll use a helper function to get them
def get_column_titles():
    # Read the first file to get column titles
    df = pd.read_excel(annotator_files[0])
    return df.columns.tolist()

column_titles = get_column_titles()
print(f"Found {len(column_titles)} columns to analyze: {column_titles}")

# Create an instance of RobustKappa
rk = RobustKappa(
    dataset_name="datasetx",
    annotator_files=annotator_files,
    column_titles=column_titles,
    n_rows=100
)

# Calculate kappa values
print("\nCalculating robust kappa values...")
kappa_values, confidence_intervals = rk.action()

# Print the results
print("\nResults:")
for column, scores in kappa_values.items():
    print(f"{column}: {scores['rk']:.4f}")
    lower, upper = confidence_intervals[column]['ci']
    print(f"  95% CI: [{lower:.4f}, {upper:.4f}]")


print("\nAnalysis completed.") 