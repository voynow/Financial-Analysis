import pandas as pd
import numpy as np

# TODO: logic for null "volume"s


def first_last_edge_case(data):

    """
    Check for edge case of nan values in first or last position
    """

    for i in range(data.shape[1]):

        # first item in series is nan
        if np.isnan(data[0, i]):
            index = 1
            while np.isnan(data[index, i]):
                index += 1
            data[0, i] = np.copy(data[index, i])

        # last item in series is nan
        if np.isnan(data[-1, i]):
            index = 1
            while np.isnan(data[-1 * index, i]):
                index += 1
            data[-1, i] = np.copy(data[-1* index, i])

    return data


def get_subsets(nan_rows):

    """ Find column-wise sets of nan values """

    max_value = np.max(nan_rows)
    nan_sets = []
    i = 0

    # loop on i for all nan_rows in given column
    while i < (nan_rows.shape[0] - 1):

        current_set = [nan_rows[i]]

        # possible nan row values to search for
        possible_sequences = np.arange(nan_rows[i] + 1, max_value + 1)

        # indexes for searching nan_rows and possible_sequences
        subset_search_index = 0
        nan_rows_search_index = i + 1

        # loop through adjacent (by row position) nans values
        while True and len(possible_sequences) > subset_search_index:

            # if adjacent nan exists
            if possible_sequences[subset_search_index] == nan_rows[nan_rows_search_index]:

                # add to current set and increment index pointers
                current_set.append(nan_rows[nan_rows_search_index])
                subset_search_index += 1
                nan_rows_search_index += 1
                i += 1

            # break loop if no adjacent nan exists
            else:
                break

        nan_sets.append(current_set)
        i += 1

    # edge case for adding last row
    if i != nan_rows.shape[0]:
        nan_sets.append([nan_rows[-1]])

    return nan_sets


def test_subsets(subsets_tensor, nan_rows_tensor):

    """ Check conservation of all data through conversion to subsets """

    tests_passed = True

    for i in range(len(subsets_tensor)):
        items = []
        for j in range(len(subsets_tensor[i])):
            if len(subsets_tensor[i][j]) > 1:
                for k in range(len(subsets_tensor[i][j])):
                    items.append(subsets_tensor[i][j][k])
            else:
                items.append(subsets_tensor[i][j][0])

        if len(items) != nan_rows_tensor[i].shape[0]:
            print("Error: Tensors at i =", i, "have mismatched dimensions")
            tests_passed = False
        else:
            comparison_matrix = np.vstack((items, nan_rows_tensor[i])).T
            difference = np.sum(np.diff(comparison_matrix, axis=1))
            if difference:
                print("Error: Subset i =", i," is missing items from its associated nan_rows")
                tests_passed = False

    if tests_passed:
        print("1/2 Test Passed: All nan_rows preserved when converted to subsets_tensor")


def get_nan_subsets(raw_data):

    """ Find sets of nan values in each timeseries """

    subsets_tensor = []
    nan_rows_tensor = []

    # iterate through columns, find nan positions
    for i in range(raw_data.shape[1]):
        nan_rows = np.where(np.isnan(raw_data[:, i]))[0]
        nan_rows_tensor.append(nan_rows)

        # convert nan rows into sets/strings of nan values
        subsets = get_subsets(nan_rows)
        subsets_tensor.append(subsets)

    # verify conservation of data
    test_subsets(subsets_tensor, nan_rows_tensor)

    return subsets_tensor


def test_nan_fill(column, column_index, subset, first_index, last_index):
    new_distances = np.diff(column[first_index - 1 : last_index + 2])
    error_threshold = column[subset[0]] / 10e3
    difference_sum = np.sum(new_distances - new_distances[0])

    if np.absolute(difference_sum) > error_threshold:
        print("Error: Filling nan values at column =", column_index, "subset =", subset)
        return False
    else:
        return True


def nan_filler(raw_data, subsets_tensor):
    tests_passed = True
    for i in range(raw_data.shape[1]):
        subsets = subsets_tensor[i]
        column = raw_data[:, i]

        for subset in subsets:
            first_index = subset[0]
            last_index = subset[-1]
            subset_length = len(subset)

            linspace_values = np.linspace(column[first_index - 1], column[last_index + 1], subset_length + 2)
            column[first_index : last_index + 1] = linspace_values[1:-1]
            raw_data[:, i] = column

            if not test_nan_fill(column, i, subset, first_index, last_index):
                tests_passed = False

    if tests_passed:
            print("2/2 Test Passed: All nan values are now numerical")

    return raw_data


def fill_nans(df):

    print("*" * 58 + "\n\tFilling NAN values in remaining series\n" + "*" * 58, "\n")
    datetime_data = df.index

    # remove datetime column
    raw_data = df.drop([df.columns[0]], axis=1).values

    # check for first/last nan position edge case
    raw_data = first_last_edge_case(raw_data)

    # find an fill nan values
    subsets_tensor = get_nan_subsets(raw_data)
    raw_data = nan_filler(raw_data, subsets_tensor)

    # convert data back to df
    filled_df = pd.DataFrame(raw_data, columns=df.columns[1:])
    filled_df.index = datetime_data

    return filled_df
