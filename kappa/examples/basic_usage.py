"""
Example usage of the KappaLib library.

This example shows how to calculate and visualize robust kappa scores
for a set of annotations across multiple categories.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add the parent directory to sys.path to allow importing the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from robust_kappa import RobustKappa

# Create a directory for our example data if it doesn't exist
os.makedirs("example_data", exist_ok=True)

# Create some example data - 5 annotators rating 100 items across 4 categories
np.random.seed(42)  # For reproducibility
n_items = 100

# Generate synthetic annotation data with varying degrees of agreement
def generate_annotations(agreement_level):
    """Generate binary annotations with controlled agreement level."""
    base = np.random.randint(0, 2, n_items)
    
    # Create annotations for 5 annotators
    annotators = [base.copy()]
    
    # Create 4 more annotators with varying levels of agreement
    for _ in range(4):
        # Flip some values based on agreement level (lower = more agreement)
        annotator = np.where(
            np.random.random(n_items) < agreement_level, 
            1 - base, 
            base
        )
        annotators.append(annotator)
    
    return annotators

# Generate data for different categories with different agreement levels
excellent_agreement = generate_annotations(0.05)  # 95% agreement
high_agreement = generate_annotations(0.15)       # 85% agreement
medium_agreement = generate_annotations(0.30)     # 70% agreement  
low_agreement = generate_annotations(0.45)        # 55% agreement

# Create DataFrames for each annotator
annotator_data = []
for i in range(5):
    data = pd.DataFrame({
        'excellent_agreement': excellent_agreement[i],
        'high_agreement': high_agreement[i],
        'medium_agreement': medium_agreement[i],
        'low_agreement': low_agreement[i]
    })
    annotator_data.append(data)

# Save to Excel files
for i, data in enumerate(annotator_data):
    data.to_excel(f"example_data/annotator{i+1}.xlsx", index=False)

print("Example data created in example_data/")

# Use the KappaLib to analyze the data
files = [f"example_data/annotator{i+1}.xlsx" for i in range(5)]

# Column names to analyze
columns = ["excellent_agreement", "high_agreement", "medium_agreement", "low_agreement"]

# Mapping from column names to display names
column_mapping = {
    "excellent_agreement": "Excellent Agreement (95%)",
    "high_agreement": "High Agreement (85%)",
    "medium_agreement": "Medium Agreement (70%)", 
    "low_agreement": "Low Agreement (55%)"
}

# Create an instance of RobustKappa
rk = RobustKappa(
    dataset_name="SyntheticData",
    annotator_files=files,
    column_titles=columns
)

# Calculate kappa values 
print("\nCalculating kappa values...")
kappa_values, confidence_intervals = rk.action()

# Print the results
print("\nResults:")
for column, scores in kappa_values.items():
    display_name = column_mapping.get(column, column)
    print(f"{display_name}: {scores['rk']:.2f}")
    lower, upper = confidence_intervals[column]['ci']
    print(f"  95% CI: [{lower:.2f}, {upper:.2f}]")

# Visualize the results (if plot method is available)
try:
    print("\nCreating visualization charts...")
    rk.plot(save_charts=True, output_dir="example_data")
    print("Visualization complete. Charts saved to example_data/")
except AttributeError:
    print("\nNote: Visualization is not available in this version of the library.")

# Calculate pairwise scores (if available in this version)
try:
    print("\nCalculating pairwise scores...")
    pairwise_scores = rk.calculate_pairwise_kappas()
    rk.plot(plot_pairwise_scores=True, save_charts=True, output_dir="example_data")
    print("Pairwise scores calculated and visualized.")
except (AttributeError, TypeError):
    print("\nNote: Pairwise score calculation is not available in this version of the library.")

print("\nExample completed.") 