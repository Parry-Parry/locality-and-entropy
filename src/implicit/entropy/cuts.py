import numpy as np

def upper_quartile(entropy_values):
    """
    Return the query IDs with entropy values strictly above the upper quartile.

    Parameters:
        entropy_values (dict): A mapping from query_id to entropy value.
        
    Returns:
        list: The query IDs with values greater than the 75th percentile.
    """
    values = np.array(list(entropy_values.values()))
    # Using 'linear' interpolation. Adjust for your version of NumPy if needed.
    q3 = np.percentile(values, 75, method='linear')
    return [qid for qid, value in entropy_values.items() if value > q3]


def lower_quartile(entropy_values):
    """
    Return the query IDs with entropy values strictly below the lower quartile.

    Parameters:
        entropy_values (dict): A mapping from query_id to entropy value.
 
    Returns:
        list: The query IDs with values less than the 25th percentile.
    """
    values = np.array(list(entropy_values.values()))
    q1 = np.percentile(values, 25, method='linear')
    return [qid for qid, value in entropy_values.items() if value < q1]


def above_median(entropy_values):
    """
    Return the query IDs with entropy values strictly above the median.

    Parameters:
        entropy_values (dict): A mapping from query_id to entropy value.
  
    Returns:
        list: The query IDs with values greater than the median.
    """
    values = np.array(list(entropy_values.values()))
    median = np.median(values)
    return [qid for qid, value in entropy_values.items() if value > median]


def below_median(entropy_values):
    """
    Return the query IDs with entropy values strictly below the median.

    Parameters:
        entropy_values (dict): A mapping from query_id to entropy value.

    Returns:
        list: The query IDs with values less than the median.
    """
    values = np.array(list(entropy_values.values()))
    median = np.median(values)
    return [qid for qid, value in entropy_values.items() if value < median]


def outlier_quartiles(entropy_values):
    """
    Return the query IDs with entropy values considered statistical outliers.
    Outliers are defined as values lying outside the range defined by:
      lower_bound = Q1 - 1.5 * IQR and upper_bound = Q3 + 1.5 * IQR

    Parameters:
        entropy_values (dict): A mapping from query_id to entropy value.
 
    Returns:
        list: The query IDs with values outside the non-outlier range.
    """
    values = np.array(list(entropy_values.values()))
    q1 = np.percentile(values, 25, method='linear')
    q3 = np.percentile(values, 75, method='linear')
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [qid for qid, value in entropy_values.items() if value < lower_bound or value > upper_bound]


def inner_quartiles(entropy_values):
    """
    Return the query IDs with entropy values considered non-outliers.
    Non-outliers are defined as values lying within the range:
      lower_bound = Q1 - 1.5 * IQR and upper_bound = Q3 + 1.5 * IQR

    Parameters:
        entropy_values (dict): A mapping from query_id to entropy value.
 
    Returns:
        list: The query IDs with values within the non-outlier range.
    """
    values = np.array(list(entropy_values.values()))
    q1 = np.percentile(values, 25, method='linear')
    q3 = np.percentile(values, 75, method='linear')
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [qid for qid, value in entropy_values.items() if lower_bound <= value <= upper_bound]


__all__ = [
    "upper_quartile",
    "lower_quartile",
    "above_median",
    "below_median",
    "outlier_quartiles",
    "inner_quartiles"
]
