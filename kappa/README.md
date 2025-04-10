# Robust Fleiss Kappa

## Set Environment
```bash
cd kappa
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run examples 
Ensure your data follows the same structure as the examples in the datasetx directory.

Change DATASET_DIR and annotator_files to run with your own dataset.
```bash
python examples/datasetx_robust_kappa.py
```

## Features

- Calculate Robust Kappa scores for annotator agreement analysis
- Generate confidence intervals using bootstrap methods
- Support for both total and pairwise agreement analysis
- Easily extensible with different kappa calculation methods

## Example Usage
Here's a basic example of using robust fleiss kappa calculation to analyze annotator agreement:

```python
from kappa import RobustKappa

# File paths for Excel files containing annotations from multiple annotators
files = [
    "annotator1.xlsx",
    "annotator2.xlsx",
    "annotator3.xlsx"
]

# Column names to analyze in the Excel files
columns = ["sentiment", "relevance", "toxicity"]

# Create an instance of RobustKappa
rk = RobustKappa(
    dataset_name="MyDataset",
    annotator_files=files,
    column_titles=columns,
    n_rows=100
)

# Calculate kappa values and confidence intervals
kappa_values, confidence_intervals = rk.action()

# Access the results
for column, scores in kappa_values.items():
    print(f"{column}: {scores['rk_123']:.2f}")
    lower, upper = confidence_intervals[column]['ci_123']
    print(f"  95% CI: [{lower:.2f}, {upper:.2f}]")
```

## Data Format

KappaLib expects:

- Excel files (.xlsx) containing annotator data
- Each file represents one annotator's annotations
- All files must contain the same column names
- Annotations must be convertible to binary values (0 or 1)
- Rows with NaN values are automatically filtered

## Requirements

- Python 3.8+
- pandas
- numpy
- statsmodels


