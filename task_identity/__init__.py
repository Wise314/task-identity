"""Task-Identity: Behavioral Drift Detection for AI Systems"""

import numpy as np
from sklearn.metrics import confusion_matrix

__version__ = "0.1.0"


def calculate_task_identity(y_true_before, y_pred_before, 
                            y_true_after, y_pred_after, labels):
    """
    Calculate behavioral similarity between two time periods using confusion matrix correlation.
    
    This is the core Task-Identity algorithm: compares confusion matrices from two time periods
    using Pearson correlation to detect behavioral drift in classification models.
    
    Args:
        y_true_before (array-like): True labels from baseline period
        y_pred_before (array-like): Model predictions from baseline period
        y_true_after (array-like): True labels from current period
        y_pred_after (array-like): Model predictions from current period
        labels (array-like): Complete set of class labels
    
    Returns:
        float: Task-Identity score in range [0.0, 1.0]
            - 1.0 = identical behavior (same confusion patterns)
            - 0.0 = completely different behavior
    
    Example:
        >>> from task_identity import calculate_task_identity
        >>> task_id = calculate_task_identity(
        ...     y_true_baseline, y_pred_baseline,
        ...     y_true_current, y_pred_current,
        ...     labels=range(10)
        ... )
        >>> if task_id < 0.5:
        ...     print("Significant behavioral drift detected!")
    """
    # Input validation
    if len(y_true_before) == 0 or len(y_pred_before) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    if len(y_true_before) != len(y_pred_before):
        raise ValueError("y_true_before and y_pred_before must have same length")
    
    if len(y_true_after) != len(y_pred_after):
        raise ValueError("y_true_after and y_pred_after must have same length")
    
    # Generate confusion matrices for both time periods
    cm_before = confusion_matrix(y_true_before, y_pred_before, labels=labels)
    cm_after = confusion_matrix(y_true_after, y_pred_after, labels=labels)
    
    # Flatten confusion matrices into 1D vectors
    flat_before = cm_before.flatten()
    flat_after = cm_after.flatten()
    
    # Handle edge case: no variance means no correlation can be calculated
    if flat_before.std() == 0 or flat_after.std() == 0:
        return 0.0
    
    # Calculate Pearson correlation between confusion matrix patterns
    correlation = np.corrcoef(flat_before, flat_after)[0, 1]
    
    # Return correlation, handling NaN and ensuring non-negative result
    return max(0.0, correlation) if not np.isnan(correlation) else 0.0
