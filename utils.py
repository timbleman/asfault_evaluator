import numpy as np
from scipy import stats
from enum import Enum
from os import path

import colorama

import evaluator_config as econf


class DicConst(Enum):
    NODES = 'nodes'
    TEST_PATH = 'test_path'
    SDL_2D = "sdl_2d"
    CUR_SDL = "curve_sdl"
    CUR_SDL_DIST = "curve_sdl_dist"
    SDL_2D_DIST = "sdl_2d_dist"


class DiffFuncConst(Enum):
    BINARY = 'binary'
    SINGLE = 'single'
    SQUARED = 'squared'


def get_root_of_test_suite(test_path: path) -> path:
    """ Finds the main parent path of a test suite

    :param test_path: os.path of some subfolder
    :return: The suites root
    """
    suite_dir_path = path.split(test_path)
    print(suite_dir_path[1])
    while not suite_dir_path[1].startswith(econf.START_OF_PARENT_DIR):
        suite_dir_path = path.split(suite_dir_path[0])
    # suite_dir_path = path.split(suite_dir_path[0])
    suite_dir_path = path.join(suite_dir_path[0], suite_dir_path[1])
    print(colorama.Fore.BLUE + "Found this parent path:", str(suite_dir_path) + colorama.Style.RESET_ALL)
    return suite_dir_path



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
            # FIXME pretty sure this should be sum_a * 2, in case on is larger than the
            # different_sum /= sum_a + sum_b
            different_sum /= sum_a * 2
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
            print(colorama.Fore.RED + "Warning: Squared normalized difference is not recommended!",
                  "The normalization does not work globally!" + colorama.Style.RESET_ALL)
            dist = np.linalg.norm(a_minus_b)
        return dist

    options = {DiffFuncConst.BINARY.value: difference_bin,
               DiffFuncConst.SINGLE.value: difference_sin,
               DiffFuncConst.SQUARED.value: difference_sqrd}
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


def whole_suite_statistics(dataset_dict: dict, feature: str, desired_percentile: int = 0, plot: bool = False) -> dict:
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
                 'min': np.min(all_values),
                 'max': np.max(all_values),
                 'avg': np.average(all_values),
                 'std_dev': np.std(all_values)}

    if desired_percentile != 0:
        desired_percentile_val = np.percentile(all_values, desired_percentile)
        stat_dict['desired_percentile'] = desired_percentile_val

    if plot:
        print("Stats for", feature, stat_dict)
    return stat_dict


def lcs(X, Y):
    """ longest common subsequence problem dynamic programming approach
    copied form here: https://www.geeksforgeeks.org/python-program-for-longest-common-subsequence/

    :param X:
    :param Y:
    :return:
    """
    # find the length of the strings
    m = len(X)
    n = len(Y)

    # declaring the array for storing the dp values
    L = [[None] * (n + 1) for i in range(m + 1)]

    """Following steps build L[m + 1][n + 1] in bottom up fashion 
    Note: L[i][j] contains length of LCS of X[0..i-1] 
    and Y[0..j-1]"""
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

                # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    return L[m][n]
    # end of function lcs


# Returns length of longest common
# substring of X[0..m-1] and Y[0..n-1]
def LCSubStr(X, Y):
    # Create a table to store lengths of
    # longest common suffixes of substrings.
    # Note that LCSuff[i][j] contains the
    # length of longest common suffix of
    # X[0...i-1] and Y[0...j-1]. The first
    # row and first column entries have no
    # logical meaning, they are used only
    # for simplicity of the program.
    m = len(X)
    n = len(Y)
    # LCSuff is the table with zero
    # value initially in each cell
    LCSuff = [[0 for k in range(n + 1)] for l in range(m + 1)]

    # To store the length of
    # longest common substring
    result = 0

    # Following steps to build
    # LCSuff[m+1][n+1] in bottom up fashion
    for i in range(m + 1):
        for j in range(n + 1):
            if (i == 0 or j == 0):
                LCSuff[i][j] = 0
            elif (X[i - 1] == Y[j - 1]):
                # seems to work for both enums and tuples
                # print(X[i - 1], "and", Y[j - 1], "match!")
                LCSuff[i][j] = LCSuff[i - 1][j - 1] + 1
                result = max(result, LCSuff[i][j])
            else:
                LCSuff[i][j] = 0
    return result
