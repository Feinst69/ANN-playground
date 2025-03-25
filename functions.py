import numpy as np
from collections import Counter

# price_column is a str name of the numeric column
def iqr_range_target_filter(df, price_column):
    # Calculate the 25th and 75th percentiles using np.percentile with the updated 'method' argument
    q1 = np.percentile(df[price_column].values, 25, method="linear")
    q3 = np.percentile(df[price_column].values, 75, method="linear")

    iqr_range = q3 - q1
    # Filter DataFrame based on IQR range
    new_df = df[
        (df[price_column] > q1 - iqr_range * 1.5) &
        (df[price_column] < q3 + iqr_range * 1.5)
    ]

    # Debugging information
    print("old number of rows", df.shape[0])
    print("new number of rows", new_df.shape[0])

    return new_df


def compare_feature_lists(*lists):
    """
    # Example usage:
    list1 = ['feature1', 'feature2', 'feature3', 'feature4']
    list2 = ['feature3', 'feature4', 'feature5', 'feature6']
    list3 = ['feature2', 'feature4', 'feature7']

    result = compare_feature_lists(list1, list2, list3)
    # Returns something like:
    # {
    #     1: ['feature1', 'feature5', 'feature6', 'feature7'],
    #     2: ['feature2', 'feature3'],
    #     3: ['feature4']
    # }
    """
    # Flatten the list of lists and count occurrences of each element
    all_elements = [item for sublist in lists for item in sublist]
    element_counts = Counter(all_elements)

    # Create dictionary with count as key and list of features as value
    result = {}
    for item, count in element_counts.items():
        if count not in result:
            result[count] = []
        result[count].append(item)

    return result
