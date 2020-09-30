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
    SPEED_STATES = "speed_states"
    STEERING_STATES = "steering_states"
    STEERING_SPEED_STATES = "speed_steering_states"
    SPEED_BINS = "speed_bins"
    STEERING_BINS = 'steering_bins'
    STEERING_BINS_ADJUSTED = "steering_bins_non_uniform_percentile"
    DISTANCE_BINS = "distance_bins"
    SPEED_STEERING_2D = "speed_steering_2d_bins"
    OBE_2D = "obe_2d"
    EXEC_TIME = "exec_time"
    NUM_OBES = "num_obes"
    UNDER_MIN_LEN_SEGS = "under_min_len_segs"
    NODES = 'nodes'
    POLYLINE = 'polyline'
    COORD_TUPLE_REP = "coord_tuple"
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
    SPEED_DIST_BINARY = "speed_dist_binary"
    SPEED_DIST_SINGLE = "speed_dist_single"
    BINS_STEERING_SPEED_DIST = "steering_speed_dist"
    BINS_STEERING_SPEED_DIST_SINGLE = "steering_speed_dist_single"

    COORD_DTW_DIST = "coord_dtw_dist"
    COORD_EDR_DIST = "coord_edr_dist"
    COORD_ERP_DIST = "coord_erp_dist"
    COORD_FRECHET_DIST = "coord_frechet_dist"

    STEERING_DTW = "steering_dtw"
    SPEED_DTW = "speed_dtw"
    STEERING_SPEED_DTW = "steering_speed_dtw"

    SDL_2D = "sdl_2d"
    CUR_SDL = "curve_sdl"
    CUR_SDL_DIST = "curve_sdl_dist"
    SDL_2D_DIST = "sdl_2d_dist"
    CUR_SDL_LCS_DIST = "cur_sdl_lcs_dist"
    CUR_SDL_LCSTR_DIST = "cur_sdl_lcstr_dist"
    CUR_SDL_K_LCSTR_DIST = "cur_sdl_k_lcstr_dist"
    CUR_SDL_1_LCSTR_DIST = "cur_sdl_1_lcstr_dist"
    CUR_SDL_2_LCSTR_DIST = "cur_sdl_2_lcstr_dist"
    CUR_SDL_3_LCSTR_DIST = "cur_sdl_3_lcstr_dist"
    CUR_SDL_5_LCSTR_DIST = "cur_sdl_5_lcstr_dist"
    CUR_SDL_10_LCSTR_DIST = "cur_sdl_10_lcstr_dist"
    SDL_2D_LCS_DIST = "sdl_2d_lcs_dist"
    SDL_2D_LCSTR_DIST = "sdl_2d_lcstr_dist"
    SDL_2D_K_LCSTR_DIST = "sdl_2d_k_lcstr_dist"
    SDL_2D_1_LCSTR_DIST = "sdl_2d_1_lcstr_dist"
    SDL_2D_2_LCSTR_DIST = "sdl_2d_2_lcstr_dist"
    SDL_2D_3_LCSTR_DIST = "sdl_2d_3_lcstr_dist"
    SDL_2D_5_LCSTR_DIST = "sdl_2d_5_lcstr_dist"
    SDL_2D_10_LCSTR_DIST = "sdl_2d_10_lcstr_dist"
    JACCARD = "jaccard"

    ADAPT_RAND_INDEX = "adaptive_random_sampling_index"


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


def list_difference_1d_2d_bin(a, b, normalized: bool = True, inverse: bool = True):
    return list_difference_1d_2d(a=a, b=b, function='binary', normalized=normalized, inverse=inverse)


def list_difference_1d_2d_sin(a, b, normalized: bool = True, inverse: bool = True):
    return list_difference_1d_2d(a=a, b=b, function='binary', normalized=normalized, inverse=inverse)


def list_difference_1d_2d(a, b, function: str, normalized: bool = True, inverse: bool = True):
    """
    Calculates the distance between bins of two roads, is used to find differences in
    behaviour.
    Available measures are binary difference in a bin, the absolute difference and the squared difference.
    All the measures can be normalized to lie in between 0 and 1.
    Abstraction for the user not to worry about bin shape.

    :param a: bins for first road
    :param b: bins for second road
    :param function: 'binary', 'single' or 'squared' ('squared' is deprecated)
    :param normalized: boolean, if normalized, fits the shape
    :param inverse: inverses the output, only possible if normalized, then similarity not difference
    :return: the calculated difference as float
    """
    from collections import Sequence
    if isinstance(a[0], (Sequence, np.ndarray)):
        return bin_difference_2d(a=a, b=b, function=function, normalized=normalized)
    else:
        return list_difference_1d(a=a, b=b, function=function, normalized=normalized, inverse=inverse)

def list_difference_1d(a: list, b: list, function: str, normalized: bool = True, inverse: bool = True):
    """ Calculates the distance between two one-dimensional lists of bins, is used to find differences in
        behaviour
        Available measures are binary difference in a bin, the absolute difference and the squared difference
        All the measures can be normalized to lie in between 0 and 1

    :param a: first list
    :param b: second list
    :param function: 'binary', 'single' or 'squared'
    :param normalized: boolean, if normalized
    :param inverse: inverses the output, only possible if normalized
    :return: the calculated difference as float
    """
    # TODO add removal of sparsely populated bins like in the R script
    assert a.__len__() == b.__len__(), "Both lists have to be of the same length!"
    if inverse:
        assert normalized is True, "Inversing behavior similarity is only possible if data is normalized!"
    # assert b.__len__() * 0.5 <= a.__len__() <= b.__len__() * 2, "Both lists have to be of similar length!"
    sum_a = sum(a)
    sum_b = sum(b)
    ratio_a_to_b = float(sum(a)) / sum(b)


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
    def difference_sin():
        different_sum = 0
        if normalized:
            for i in range(0, a.__len__()):
                if a[i] != b[i]:
                    different_sum += abs(a[i] - b[i] * ratio_a_to_b)
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
    from asfault.network import TYPE_STRAIGHT
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
    #print("global_array", global_array)
    return global_array


def whole_suite_dist_matrix_statistic_incomplete(dataset_dict: dict, feature: str, random_selection_size: int = 0,
                                                 desired_percentile: int = 0, plot: bool = False, title: str = None) \
                                                    -> dict:
    """ Calculate a boxplot for a similarity matrix, randomly sampled to speed up the process

    :param dataset_dict: The dict that includes all roads
    :param feature: Feature to extract, has to be a numerical value
    :param random_selection_size: Number of random dist matrices that get combine
    :param desired_percentile: Desired percentile to return, optional
    :param plot: Draw a box plot and print a message
    :param title: title of the plot

    :return: dict that includes quartiles, avg and standard deviation
    """
    import random
    all_values = []
    all_keys = list(dataset_dict.keys())
    if random_selection_size == 0:
        random_selection_size = len(dataset_dict)
    keys = random.sample(all_keys, random_selection_size)

    for key in keys:
        test_dict = dataset_dict.get(key)
        dists = test_dict.get(feature, None)
        assert dists is not None, feature + " has not been added to the dict!"
        for val in dists.values():
            all_values.append(val)
            #print(val)

    return list_statistics(data_list=all_values, desired_percentile=desired_percentile, plot=plot, title=title)


def whole_suite_statistics(dataset_dict: dict, feature: str, desired_percentile: int = 0, plot: bool = False,
                           title: str = None) -> dict:

    """ Calculates common statistics on numerical feature of each road in the dataset.
    This could be a certain coverage or the length.

    :param dataset_dict: The dict that includes all roads
    :param feature: Feature to extract, has to be a numerical value
    :param desired_percentile: Desired quartile to return, optional
    :param plot: Draw a box plot and print a message
    :param title: title of the plot
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


def add_coord_tuple_representation(data_dict: dict):
    """ Turn roads saved as shapely LineStings into a list of 2d cartesian tuples
    Requires the roads to be saved as polylines
    Adds the tuples to data_dict

    :param data_dict: Dict containing all the test data
    :return: None
    """
    for road in data_dict.values():
        polyline = road.get(RoadDicConst.POLYLINE.value, None)
        assert polyline is not None, "Polyline has not been added to road!"
        coordsxy = polyline.coords.xy
        road_coords = np.zeros((len(coordsxy[0]), 2))
        road_coords[:, 0] = coordsxy[0]
        road_coords[:, 1] = coordsxy[1]

        road[RoadDicConst.COORD_TUPLE_REP.value] = road_coords


def shape_similarity_measures_all_to_all_unoptimized(data_dict: dict):
    """ Calculates different predefined shape similarity measures from https://pypi.org/project/similaritymeasures/
    Requires the roads to be saved as polylines

    :param data_dict: Dict containing all the test data
    :return: None
    """
    import similaritymeasures

    for name1 in data_dict:
        road1 = data_dict.get(name1)
        road1_coords = road1.get(RoadDicConst.COORD_TUPLE_REP.value, None)
        if road1_coords is None:
            add_coord_tuple_representation(data_dict=data_dict)
            road1_coords = road1.get(RoadDicConst.COORD_TUPLE_REP.value, None)

        # TODO more
        dicc_dtw = {}
        dicc_dtw_opti = {}
        dicc_frechet = {}
        for name2 in data_dict:
            # TODO optimize
            road2 = data_dict.get(name2)
            road2_coords = road2.get(RoadDicConst.COORD_TUPLE_REP.value, None)

            d_dtw, _ = similaritymeasures.dtw(road1_coords, road2_coords)
            dicc_dtw[name2] = d_dtw

            d_frechet = similaritymeasures.frechet_dist(road1_coords, road2_coords)
            dicc_frechet[name2] = d_frechet

        road1[BehaviorDicConst.COORD_DTW_DIST.value] = dicc_dtw
        road1[BehaviorDicConst.COORD_FRECHET_DIST.value] = dicc_frechet


def print_remaining_time(start_time, completed_operations: int, total_operations: int):
    import time
    import sys
    passed_time = time.time() - start_time
    time_per_iteration = passed_time/completed_operations
    remaining_operations = total_operations - completed_operations
    remaining_time = remaining_operations * time_per_iteration
    #print("time_per_iteration", time_per_iteration, type(time_per_iteration))
    #print("remaining_time", remaining_time, type(remaining_time))
    m, s = divmod(remaining_time, 60)
    h, m = divmod(m, 60)
    sys.stdout.write("\rRemaining loop iterations %i; remaining time %i h %i m %i s   " % (remaining_operations, h, m, s))
    sys.stdout.flush()
    #print("Remaining time", remaining_time, "second", end='\r')


def optimized_bin_borders_percentiles(all_values: list, number_of_bins: int):
    """ This is used to get better borders with equal distribution for binning.
    Includes number_of_bins + 1 elements, including the last border.
    Is used once for all steering angles with 16 bins to have a better distribution of steering bins.

    :param all_values: all values, like all angles
    :param number_of_bins: number of bins
    :return: List of borders for binning.
    """
    assert all_values, "There have to be values to compute the percentiles"
    #print("all steering values for adjusting, stats min mean max:", min(all_values), np.mean(all_values),
    #      max(all_values))
    # get the percentiles
    step_size = 100.0/number_of_bins
    percentiles = np.arange(0, 100, step_size).tolist()
    percentiles.append(100.0)
    # calculate the value of the percentiles
    percentile_res = np.percentile(a=all_values, q=percentiles)
    assert len(percentile_res) == number_of_bins + 1, "The number of computed percentiles is wrong"
    #print("percentiles", percentiles)
    #print("percentile_res", percentile_res)
    return percentile_res
