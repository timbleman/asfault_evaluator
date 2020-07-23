import numpy as np
from scipy import stats


def list_difference_1d(a: list, b: list, function: str, normalized: bool = True):
    """ Calculates the distance between two one-dimensional lists of bins, is used to find differences in
        behaviour
        Available measures are binary difference in a bin, the absolute difference and the squared difference
        All the measures can be normalized to lie in between 0 and 1

    :param a: first list
    :param b: second list
    :param function: 'binary', 'single' or 'squared'
    :param normalized: boolean, if normalized
    :return: the calculated difference as float
    """
    assert a.__len__() == b.__len__(), "Both lists have to be of the same length!"
    # assert b.__len__() * 0.5 <= a.__len__() <= b.__len__() * 2, "Both lists have to be of similar length!"
    sum_a = sum(a)
    sum_b = sum(b)
    ratio_a_to_b = float(sum(a)) / sum(b)

    # print("ratio_a_to_b", ratio_a_to_b)
    # assert sum_a == sum_b, "Both lists have to have the same element count!"
    # TODO different sized bins may cause problems
    # fixme find a good solution, discard some roads?
    # assert sum_b * 0.25 <= sum_a <= sum_b * 4, "Both lists have to have similar element count!" + str(sum_a) + " vs " + \
    #                                          str(sum_b)

    # returns the binary difference
    def difference_bin():
        binsum = 0
        for i in range(0, a.__len__()):
            if (a[i] > 0 and b[i] <= 0) or (a[i] <= 0 and b[i] > 0):
                binsum += 1
        if normalized:
            binsum /= a.__len__()
        return binsum

    # returns the absolute difference of the bins
    # fixme normalize
    def difference_sin():
        different_sum = 0
        if normalized:
            for i in range(0, a.__len__()):
                if a[i] != b[i]:
                    different_sum += abs(a[i] - b[i] * ratio_a_to_b)
            different_sum /= sum_a + sum_b
        else:
            for i in range(0, a.__len__()):
                if a[i] != b[i]:
                    different_sum += abs(a[i] - b[i])
        return different_sum

    # returns the euclidean distance of bins
    # fixme norm has to be calculated globally and normalized
    def difference_sqrd():
        dist = 0
        a_minus_b = list(map(int.__sub__, a, b))
        if normalized:
            dist = 0.5 * (np.std(a_minus_b) ** 2) / (np.std(a) ** 2 + np.std(b) ** 2)
        else:
            dist = np.linalg.norm(a_minus_b)
        return dist

    options = {'binary': difference_bin,
               'single': difference_sin,
               'squared': difference_sqrd}
    return options.get(function)()


def bin_difference_2d(a: np.ndarray, b: np.ndarray, function: str, normalized: bool = True):
    """ Calculates the distance between two two-dimensional arrays of bins, by flattening them

    :param a: first two-dimensional array
    :param b: second two-dimensional array
    :param function: 'binary', 'single' or 'squared'
    :param normalized: boolean, if normalized
    :return: the calculated difference as float
    """
    # list() should not be necessary, but the ide warns me
    new_a = list(a.flatten('C'))
    new_b = list(b.flatten('C'))
    # print("new_a: ", new_a)
    # print("new_b: ", new_b)
    return list_difference_1d(new_a, new_b, function, normalized)


def coverage_compute_1d(a: list):
    """ Computes the coverage for one binned attribute

    :param a: one dimensional list
    :return: coverage as float
    """
    num_items_covered = sum(x > 0 for x in a)
    return num_items_covered / (a.__len__())


def coverage_compute_2d(a: np.ndarray):
    """ Computes the coverage for two binned attribute

    :param a: two dimensional array of bins
    :return: coverage as float
    """
    new_a = list(a.flatten('C'))
    return coverage_compute_1d(new_a)


def entropy_compute_1d(a: list):
    """ Calculates the entropy of a list
    :param a: list of bins
    :return: entropy as float
    """
    return stats.entropy(a)


def entropy_compute_2d(a: np.ndarray):
    """ Calculates the entropy of a 2d array
    :param a: 2d array of bins
    :return: entropy as float
    """
    new_a = list(a.flatten('C'))
    return entropy_compute_1d(new_a)


def whole_suite_statistics(dataset_dict: dict, feature: str, desired_quartile: int = 0, plot: bool = False) -> dict:
    """ Calculates common statistics on numerical feature of each road in the dataset.
    This could be a certain coverage or the length.

    :param dataset_dict: The dict that includes all roads
    :param feature: Feature to extract, has to be a numerical value
    :param desired_quartile: Desired quartile to return, optional
    :param plot: Draw a box plot and print a message
    :return: dict that includes quartiles, avg and standard deviation
    """
    import matplotlib.pyplot as plt
    all_values = []
    for key, test in dataset_dict.items():
        val = test.get(feature, None)
        assert val is not None, "The feature could not be found in the datadict"
        assert isinstance(val, (int, float)), "Value has to be a number (e.g. coverage or length)!"
        all_values.append(val)

    if plot:
        plt.boxplot(all_values)
        plt.show()

    stat_dict = {'median': np.percentile(all_values, 50),
                 'lower_quartile': np.percentile(all_values, 25),
                 'higher_quartile': np.percentile(all_values, 75),
                 'avg': np.average(all_values),
                 'std_dev': np.std(all_values)}

    if desired_quartile != 0:
        desired_quartile_val = np.percentile(all_values, desired_quartile)
        stat_dict['desired_quartile'] = desired_quartile_val

    if plot:
        print("Stats for", feature, stat_dict)
    return stat_dict
