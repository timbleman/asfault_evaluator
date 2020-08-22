import numpy as np
from scipy import stats
from enum import Enum
from os import path
from typing import List

import colorama
import math

from asfault.network import NetworkNode
#import evaluator_config as econf

START_OF_PARENT_DIR = "experiments-"
# if a segment is under this length it is considered invalid
MINIMUM_SEG_LEN = 5


# for selecting entries from the dict while avoiding typos
# stores more general information about a road and execution
class RoadDicConst(Enum):
    TEST_PATH = 'test_path'
    TEST_ID = 'test_id'
    SPEED_BINS = "speed_bins"
    STEERING_BINS = 'steering_bins'
    DISTANCE_BINS = "distance_bins"
    SPEED_STEERING_2D = "speed_steering_2d"
    OBE_2D = "obe_2d"
    EXEC_TIME = "exec_time"
    NUM_OBES = "num_obes"
    UNDER_MIN_LEN_SEGS = "under_min_len_segs"
    NODES = 'nodes'
    POLYLINE = 'polyline'
    ROAD_LEN = "road_len"


# for selecting entries from the dict while avoiding typos
# more about behavior and distances
class BehaviorDicConst(Enum):
    NUM_STATES = "num_states"
    EXEC_RESULT = "ex_result"
    AVG_CURVATURE = 'avg_curvature'
    CENTER_DIST_BINARY = "center_dist_binary"
    CENTER_DIST_SINGLE = "center_dist_single"
    STEERING_DIST_BINARY = "steering_dist_binary"
    STEERING_DIST_SINGLE = "steering_dist_single"
    BINS_STEERING_SPEED_DIST = "steering_speed_dist"

    SDL_2D = "sdl_2d"
    CUR_SDL = "curve_sdl"
    CUR_SDL_DIST = "curve_sdl_dist"
    SDL_2D_DIST = "sdl_2d_dist"
    CUR_SDL_LCS_DIST = "cur_sdl_lcs_dist"
    CUR_SDL_LCSTR_DIST = "cur_sdl_lcstr_dist"
    CUR_SDL_K_LCSTR_DIST = "cur_sdl_k_lcstr_dist"
    SDL_2D_LCS_DIST = "sdl_2d_lcs_dist"
    SDL_2D_LCSTR_DIST = "sdl_2d_lcstr_dist"
    SDL_2D_K_LCSTR_DIST = "sdl_2d_k_lcstr_dist"
    JACCARD = "jaccard"



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
    while not suite_dir_path[1].startswith(START_OF_PARENT_DIR):
        suite_dir_path = path.split(suite_dir_path[0])
    # suite_dir_path = path.split(suite_dir_path[0])
    suite_dir_path = path.join(suite_dir_path[0], suite_dir_path[1])
    print(colorama.Fore.BLUE + "Found this parent path:", str(suite_dir_path) + colorama.Style.RESET_ALL)
    return suite_dir_path


def list_matrix_measure(data_dict: dict, measure: str) -> list:
    """ puts each one to all list in a 2d list, needed for clustering algorithms

    :param data_dict: dict off all roads
    :param measure: measure like jaccard
    :return: 2d list, distance matrix
    """
    list_2d = []
    for test in data_dict.values():
        ar = test.get(measure, None)
        assert ar is not None, "The measure " + measure + " has not been found in the dict!"
        list_2d.append(list(ar.values()))
    return list_2d


def dict_of_dicts_matrix_measure(data_dict: dict, measure: str) -> dict:
    """ puts each one to all list in a 2d list, needed for clustering algorithms

    :param data_dict: dict off all roads
    :param measure: measure like jaccard
    :return: 2d list, distance matrix
    """
    dict_2d = {}
    #print("data_dict.keys", data_dict.keys())
    for key2, test in data_dict.items():
        ar = test.get(measure, None)
        assert ar is not None, "The measure " + measure + " has not been found in the dict!"
        new_dic = {}
        for key1, val in ar.items():
            new_dic[key1] = {"weight": val}
        dict_2d.update({key2: new_dic})
    #print("dict_2d.keys", dict_2d.keys())
    #print("dict_2d", dict_2d)
    return dict_2d


def dict_of_lists_matrix_measure(data_dict: dict, measure: str) -> dict:
    """ puts each one to all list in a 2d list, needed for clustering algorithms

    :param data_dict: dict off all roads
    :param measure: measure like jaccard
    :return: 2d list, distance matrix
    """
    dict_2d = {}
    for key, test in data_dict.items():
        ar = test.get(measure, None)
        assert ar is not None, "The measure " + measure + " has not been found in the dict!"
        dict_2d[key] = list(ar.values())
    return dict_2d


def list_difference_1d(a: list, b: list, function: str, normalized: bool = True, inverse: bool = True):
    """ Calculates the distance between two one-dimensional lists of bins, is used to find differences in
        behaviour
        Available measures are binary difference in a bin, the absolute difference and the squared difference
        All the measures can be normalized to lie in between 0 and 1

    :param inverse: inverses the output, only possible if normalized
    :param a: first list
    :param b: second list
    :param function: 'binary', 'single' or 'squared'
    :param normalized: boolean, if normalized
    :return: the calculated difference as float
    """
    assert a.__len__() == b.__len__(), "Both lists have to be of the same length!"
    if inverse:
        assert normalized is True, "Inversing behavior similarity is only possible if data is normalized!"
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
            if inverse:
                binsum = 1 - binsum
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
            if inverse:
                different_sum = 1 - different_sum
        else:
            for i in range(0, a.__len__()):
                if a[i] != b[i]:
                    different_sum += abs(a[i] - b[i])
        return different_sum

    # returns the euclidean distance of bins
    # fixme norm has to be calculated globally and normalized
    # fixme deprecated
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


def road_has_min_segs(nodes: List) -> bool:
    """returns whether a road has too short segments"""
    for node in nodes:
        if compute_length(node) < MINIMUM_SEG_LEN:
            return True
    return False

# copied from https://gitlab.infosun.fim.uni-passau.de/gambi/esec-fse-20/-/blob/master/code/profiles_estimator.py#L516
def compute_length(road_segment: NetworkNode):
    from asfault.network import TYPE_STRAIGHT, TYPE_L_TURN, TYPE_R_TURN
    if road_segment.roadtype == TYPE_L_TURN or road_segment.roadtype == TYPE_R_TURN:
        # https: // www.wikihow.com / Find - Arc - Length
        # Length of the segment "is" the length of the arc defined for the turn
        xc, yc, radius = compute_radius_turn(road_segment)
        angle = road_segment.angle
        return 2 * math.pi * radius * (abs(angle) / 360.0)

    if road_segment.roadtype == TYPE_STRAIGHT:
        # Apparently this might be 0, not sure why so we need to "compute" the lenght which is the value of y
        return road_segment.y_off

# copied from https://gitlab.infosun.fim.uni-passau.de/gambi/esec-fse-20/-/blob/master/code/profiles_estimator.py#L516
# Since there are some quirks in how AsFault implements turn generation using angle, pivot offset and such we adopt the
# direct strategy to compute the radius of the turn: sample three points on the turn (spine), use triangulation to find
# out where's the center of the turn is, and finally compute the radius as distance between any of the points on the
# circle and the center
def compute_radius_turn(road_segment: NetworkNode):
    from asfault.network import TYPE_STRAIGHT, TYPE_L_TURN, TYPE_R_TURN
    from shapely.geometry import Point

    if road_segment.roadtype == TYPE_STRAIGHT:
        return math.inf

    spine_coord = list(road_segment.get_spine().coords)

    # Use triangulation.
    p1 = Point(spine_coord[0])
    x1 = p1.x
    y1 = p1.y

    p2 = Point(spine_coord[-1])
    x2 = p2.x
    y2 = p2.y

    # This more or less is the middle point, not that should matters
    p3 = Point(spine_coord[int(len(spine_coord) / 2)])
    x3 = p3.x
    y3 = p3.y

    center_x = ((x1 ** 2 + y1 ** 2) * (y2 - y3) + (x2 ** 2 + y2 ** 2) * (y3 - y1) + (x3 ** 2 + y3 ** 2) * (
            y1 - y2)) / (2 * (x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2))
    center_y = ((x1 ** 2 + y1 ** 2) * (x3 - x2) + (x2 ** 2 + y2 ** 2) * (x1 - x3) + (x3 ** 2 + y3 ** 2) * (
            x2 - x1)) / (2 * (x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2))
    radius = math.sqrt((center_x - x1) ** 2 + (center_y - y1) ** 2)

    return (center_x, center_y, radius)

def extend_arrays_globally(data_dict: dict, feature: str) -> list:
    global_array = []
    for test in data_dict.values():
        local_arr = test.get(feature, None)
        assert local_arr is not None, "The feature" + feature + "has not been found in the dict!"
        global_array.extend(local_arr)

    # remove outliers
    outlier_border = 300
    #global_array[:] = [x for x in global_array if x <= outlier_border]
    print("global_array", global_array)
    return global_array


def whole_suite_statistics(dataset_dict: dict, feature: str, desired_percentile: int = 0, plot: bool = False,
                           title: str = None) -> dict:
    """ Calculates common statistics on numerical feature of each road in the dataset.
    This could be a certain coverage or the length.

    :param dataset_dict: The dict that includes all roads
    :param feature: Feature to extract, has to be a numerical value
    :param desired_quartile: Desired quartile to return, optional
    :param plot: Draw a box plot and print a message
    :return: dict that includes quartiles, avg and standard deviation
    """
    all_values = []
    for key, test in dataset_dict.items():
        val = test.get(feature, None)
        assert val is not None, "The feature could not be found in the datadict"
        assert isinstance(val, (int, float)), "Value has to be a number (e.g. coverage or length)!"
        all_values.append(val)

    return list_statistics(data_list=all_values, desired_percentile=desired_percentile, plot=plot, title=title)


def list_statistics(data_list = list, desired_percentile: int = 0, plot: bool = False,
                           title: str = None) -> dict:
    import matplotlib.pyplot as plt

    if plot:
        plt.boxplot(data_list)
        if title != None:
            plt.title(title)
        plt.show()

    stat_dict = {'median': np.percentile(data_list, 50),
                 'lower_quartile': np.percentile(data_list, 25),
                 'higher_quartile': np.percentile(data_list, 75),
                 'min': np.min(data_list),
                 'max': np.max(data_list),
                 'avg': np.average(data_list),
                 'std_dev': np.std(data_list)}

    if desired_percentile != 0:
        desired_percentile_val = np.percentile(data_list, desired_percentile)
        stat_dict['desired_percentile'] = desired_percentile_val

    if plot:
        print("Stats for", title, stat_dict)
    return stat_dict


def shape_similarity_measures_all_to_all(data_dict: dict):
    import similaritymeasures

    for name in data_dict:
        road = data_dict.get(name)
        polyline = road.get(RoadDicConst.POLYLINE.value, None)
        assert polyline is not None, "Polyline has not been added to road " + name


def lcs(X, Y, normalized: bool = True):
    """ longest common subsequence problem dynamic programming approach
    copied form here: https://www.geeksforgeeks.org/python-program-for-longest-common-subsequence/

    :param X: First string
    :param Y: Second string
    :param normalized: normalizes in the range [0, 1.0] using the length of the shorter road
    :return: lcs
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
    length = L[m][n]
    # end of function lcs
    if normalized:
        length = length / min(m, n)
        # TODO remove this if it does not break anything
        assert 0.0 <= length <= 1.0, "length " + str(length) + " is out of bounds!"
    return length


# Returns length of longest common
# substring of X[0..m-1] and Y[0..n-1]
def LCSubStr(X, Y, normalized: bool = True):
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

    if normalized:
        result = result / min(m, n)
        # TODO remove this if it does not break anything
        assert 0.0 <= result <= 1.0, "length " + str(result) + " is out of bounds!"
    return result


def k_lcstr(X, Y, k:int = 1, normalized: bool = False) -> int:
    """ Longest common substring with k mismatches
    Implementation of https://doi.org/10.1016/j.ipl.2015.03.006 algorithm
    TODO experiment with k

    :param X: First string
    :param Y: Second string
    :param k: number of mismatches
    :param normalized: normalizes in the range [0, 1.0] using the length of the shorter road
    :return: length of longest common substring with k mismatches
    """
    n = len(X)
    m = len(Y)
    length = 0
    offset_1 = 0
    offset_2 = 0
    for d in range(-m + 1, n):
        i = max(-d, 0) + d
        j = max(-d, 0)
        # using a list as a queue
        queue = []
        s = 0
        p = 0
        while p <= min(n-i, m-j) - 1:
            if X[i+p] != Y[j+p]:
                if len(queue) == k:
                    s = min(queue) + 1
                    queue.pop(0)
                queue.append(p)
            p += 1
            if (p - s) > length:
                length = p - s
                offset_1 = i + s
                offset_2 = j + s

    if normalized:
        length = length / min(m, n)
        # TODO remove this if it does not break anything
        assert 0.0 <= length <= 1.0, "length " + str(length) + " is out of bounds!"
    return length
