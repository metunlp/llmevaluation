"""
Base classes for Kappa calculation and annotation analysis.

This module provides the foundation for analyzing inter-rater agreement
using kappa statistics, with abstractions to support various kappa calculation methods.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union, Any
from abc import ABC, abstractmethod


class ColumnInfo:
    """Container for annotation data from a single category/column."""
    
    def __init__(self, title: str, annotators: List[List[Any]]):
        """
        Initialize column information with annotations from three annotators.
        
        Args:
            title: The display title for this column (category)
            ann1: List of annotations from the first annotator
            ann2: List of annotations from the second annotator
            ann3: List of annotations from the third annotator
        """
        self.title = title
        self.annotators: List[List[Any]] = annotators

        
    def __repr__(self) -> str:
        """Return string representation of this column."""
        return f"ColumnInfo(title='{self.title}', items={len(self.ann1)})"
    
    def as_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of this column."""
        return {
            "title": self.title,
            "annotators": self.annotators,
            "num_items": len(self.annotators[0])
        }


class BaseKappa(ABC):
    """
    Abstract base class for kappa calculation methods.
    
    This class handles annotation data loading and preprocessing, and
    defines the interface for kappa calculation.
    """
    
    def __init__(self, annotator_files: List[str], column_titles: List[str], n_rows: int = None):
        """
        Initialize the kappa calculation with annotation data.
        
        Args:
            annotator_files: List of Excel file paths containing annotations
            column_titles: List of column names to analyze
            n_rows: Number of rows to analyze
        Raises:
            FileNotFoundError: If any annotator file doesn't exist
            ValueError: If files don't contain the expected columns
        """           
        self.annotator_files = annotator_files
        self.columns: List[ColumnInfo] = self._init_columns(annotator_files, column_titles, n_rows)
        
        if not self.columns:
            raise ValueError("No valid columns were found in the annotator files")

    def _init_columns(self, 
                     annotator_files: List[str], 
                     column_titles: List[str],
                     n_rows: int = None
                     ) -> List[ColumnInfo]:
        """
        Initialize column data from annotator files.
        
        Args:
            annotator_files: List of Excel file paths containing annotations
            column_titles: List of column names to analyze
            n_rows: Number of rows to analyze
            
        Returns:
            List of ColumnInfo objects containing processed annotation data
            
        Raises:
            ValueError: If a required column is missing from any file
        """
        for file_path in annotator_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Annotator file not found: {file_path}")
            
        try:
            dfs = [pd.read_excel(file) for file in annotator_files]
        except Exception as e:
            raise ValueError(f"Error reading annotator files: {str(e)}")
            
        # Validate that all required columns exist in all files
        for df in dfs:
            missing_columns = set(column_titles) - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing columns in annotator file: {missing_columns}")

        tbr_columns: List[ColumnInfo] = []
        
        for column_name in column_titles:
            # Get annotations and filter out rows where any annotator has NaN
            annotations = []
            
            n_rows = n_rows if n_rows else len(dfs[0])
            for i in range(n_rows):  # Assume all annotator files have the same number of rows
                try:
                    values = [df[column_name].iloc[i] for df in dfs]
                    if not any(pd.isna(v) for v in values):
                        annotations.append(values)
                except IndexError:
                    break  # Stop if we hit the end of any dataframe
            
            if not annotations:
                print(f"Warning: No valid annotations found for column '{column_name}'")
                continue
                

            tbr_columns.append(ColumnInfo(
                column_name,
                annotators=list(zip(*annotations))
            ))

        return tbr_columns

    @staticmethod
    def _count_0_or_1(annotations_list: List[List[Union[int, bool, str]]]) -> np.ndarray:
        """
        Convert annotations to a matrix of counts for 0/1 values.
        
        Args:
            annotations_list: List of annotation lists from different annotators
            
        Returns:
            NumPy array where each row contains counts of 0s and 1s for one item
            
        Raises:
            ValueError: If annotations can't be converted to 0/1 values
        """
        number_of_annotators = len(annotations_list)
        dict_annotations = {}

        try:
            for i, ann in enumerate(annotations_list):
                dict_annotations[f"ann{i + 1}"] = [int(j) for j in ann]
        except (ValueError, TypeError) as e:
            raise ValueError(f"Annotations must be convertible to integers (0 or 1): {str(e)}")

        df = pd.DataFrame(dict_annotations)
        sum_df = pd.DataFrame(df.sum(axis=1))
        sum_df['count_0_annotations'] = number_of_annotators - sum_df[0]
        sum_df.columns = ["count_1_annotations", 'count_0_annotations']

        return sum_df.values

    @abstractmethod
    def action(self) -> Any:
        """
        Perform kappa calculation for all columns.
        
        This abstract method must be implemented by subclasses.
        
        Returns:
            Results of kappa calculations (implementation-specific)
        """
        pass

    @abstractmethod
    def calculate_kappa(self, *args, **kwargs) -> Any:
        """
        Calculate kappa for a set of annotations.
        
        This abstract method must be implemented by subclasses.
        
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
            
        Returns:
            Kappa calculation results (implementation-specific)
        """
        pass







