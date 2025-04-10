"""
Robust Kappa calculation for inter-rater agreement analysis.

This module implements Robust Kappa calculation methods that are less sensitive
to outliers and distribution assumptions than traditional kappa approaches.
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Any
from statsmodels.stats.inter_rater import fleiss_kappa

from base_kappa import BaseKappa


class RobustKappa(BaseKappa):
    """
    Implementation of Robust Kappa calculation for inter-rater agreement.
    
    This class uses permutation and bootstrap techniques to provide more
    robust kappa estimates that are less sensitive to outliers and
    distribution assumptions.
    """
    
    def __init__(self, 
                dataset_name: str, 
                annotator_files: List[str], 
                column_titles: List[str],
                n_rows: int = None):
        """
        Initialize the RobustKappa calculator.
        
        Args:
            dataset_name: Name of the dataset (used for plot titles and file names)
            annotator_files: List of Excel file paths containing annotations
            column_titles: List of column names to analyze
        """
        super().__init__(annotator_files, column_titles, n_rows)
        self.dataset_name = dataset_name

    def action(self) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, Tuple[float, float]]]]:
        """
        Calculate robust kappa values and confidence intervals for all columns.
        
        Returns:
            Tuple containing:
                - Dictionary mapping column titles to robust kappa values
                - Dictionary mapping column titles to confidence intervals
        """
        robust_kappa_across_titles = {}
        confidence_interval_across_titles = {}

        for column in self.columns:
            kappa, conf_interval = self.calculate_kappa(column.annotators)
            
            # Store the results
            robust_kappa_across_titles[column.title] = {
                "rk": kappa,
            }

            confidence_interval_across_titles[column.title] = {
                "ci": conf_interval,
            }

        return robust_kappa_across_titles, confidence_interval_across_titles


    def calculate_kappa(self, annotations_list: List[List[Any]]) -> Tuple[float, Tuple[float, float]]:
        """
        Calculate robust kappa and confidence intervals for a set of annotations.
        
        Args:
            annotations_list: List of annotation lists from different annotators
            
        Returns:
            Tuple containing:
                - Robust kappa value
                - Confidence interval as a tuple (lower_bound, upper_bound)
                
        Raises:
            ValueError: If there are issues with the annotation data
        """
        try:
            matrix = BaseKappa._count_0_or_1(annotations_list)
            
            # Input validation
            if matrix.size == 0:
                raise ValueError("Empty annotation matrix")
                
            kappa = self._permutation_kappa(matrix, num_permutations=100)
            confidence_interval = self._bootstrap_kappa_confidence_interval(
                matrix, num_permutations=100, num_bootstrap=1000
            )
            return kappa, confidence_interval
            
        except Exception as e:
            raise ValueError(f"Error calculating kappa: {str(e)}")

    @staticmethod
    def _permutation_kappa(matrix: np.ndarray, num_permutations: int = 100) -> float:
        """
        Calculate robust Fleiss' kappa using permutation techniques.
        
        Args:
            matrix: Matrix of annotation counts (rows=items, cols=[count_1s, count_0s])
            num_permutations: Number of permutations to perform
            
        Returns:
            Robust kappa value (median of permuted kappa values)
        """
        if num_permutations < 1:
            raise ValueError("Number of permutations must be positive")
            
        kappas = []
        for _ in range(num_permutations):
            # Permute each row independently
            permuted_matrix = np.array([np.random.permutation(row) for row in matrix])
            kappa = fleiss_kappa(permuted_matrix)
            kappas.append(kappa)

        # Take the median as robust kappa
        robust_kappa = np.median(kappas)
        return robust_kappa

    @staticmethod
    def _bootstrap_kappa_confidence_interval(
            matrix: np.ndarray, 
            num_permutations: int = 100, 
            num_bootstrap: int = 1000, 
            alpha: float = 0.05
        ) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval for robust Fleiss' kappa.
        
        Args:
            matrix: Matrix of annotation counts (rows=items, cols=[count_1s, count_0s])
            num_permutations: Number of permutations for each bootstrap sample
            num_bootstrap: Number of bootstrap samples to generate
            alpha: Significance level (e.g., 0.05 for 95% confidence)
            
        Returns:
            Tuple with lower and upper bounds of the confidence interval
        """
        if not (0 < alpha < 1):
            raise ValueError("Alpha must be between 0 and 1")
            
        bootstrap_kappas = []
        n = matrix.shape[0]

        for _ in range(num_bootstrap):
            # Resample rows with replacement
            indices = [random.randint(0, n - 1) for _ in range(n)]
            resampled_matrix = matrix[indices]

            # Apply permutation-based robust kappa on the resampled data
            robust_kappa = RobustKappa._permutation_kappa(resampled_matrix, num_permutations=num_permutations)
            bootstrap_kappas.append(robust_kappa)

        # Calculate confidence interval based on bootstrap results
        lower_bound = np.percentile(bootstrap_kappas, alpha / 2 * 100)
        upper_bound = np.percentile(bootstrap_kappas, (1 - alpha / 2) * 100)
        
        return lower_bound, upper_bound

