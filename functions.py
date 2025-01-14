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

    in_all, in_at_least_two, in_only_one = compare_feature_lists(list1, list2, list3)
    print("Elements in all lists:", in_all)
    print("Elements in at least two lists:", in_at_least_two)
    print("Elements in only one list:", in_only_one)

    renvoie

    Elements in all lists: ['feature4']
    Elements in at least two lists: ['feature2', 'feature3', 'feature4']
    Elements in only one list: ['feature1', 'feature5', 'feature6', 'feature7']

    """
    # Flatten the list of lists and count occurrences of each element
    all_elements = [item for sublist in lists for item in sublist]
    element_counts = Counter(all_elements)

    # Elements in all lists
    in_all_lists = [item for item, count in element_counts.items() if count == len(lists)]

    # Elements in at least two lists
    in_at_least_two_lists = [item for item, count in element_counts.items() if count >= 2]

    # Elements in only one list
    in_only_one_list = [item for item, count in element_counts.items() if count == 1]

    return in_all_lists, in_at_least_two_lists, in_only_one_list
