import utils

from enum import Enum
import numpy as np

import asfault
from asfault import network
from asfault.network import NetworkNode, TYPE_STRAIGHT, TYPE_L_TURN, TYPE_R_TURN
from typing import List

# for box plots
import matplotlib as mpl
import matplotlib.pyplot as plt

import colorama
import math
import evaluator_config as econf

"""has to be symmetric around zero"""
NUM_ALPHABET = 7

DEFAULT_PERCENTILE_VALUES_CUR = [-120.0, -75.0, -30.0, -1.0, 1.0, 30.0, 75.0, 120.0]


class cur(Enum):
    STRONG_LEFT = -3
    LEFT = -2
    SLIGHT_LEFT = -1
    STRAIGHT = 0
    SLIGHT_RIGHT = 1
    RIGHT = 2
    STRONG_RIGHT = 3


DEFAULT_PERCENTILE_VALUES_LEN = [10.0, 30.0, 50, 100]

NUM_LEN_ALPHABET = 4


class len_en(Enum):
    SHORT = 0
    MEDIUM = 1
    LONG = 2
    VERY_LONG = 3


class StringComparer:
    def __init__(self, data_dict: dict):
        self.data_dict = data_dict
        self.all_angles = []
        self.percentile_values = []

        # colorama.init(autoreset=True)
        if not self.data_dict:
            print(colorama.Fore.RED + "Warning, there are no roads to convert to Strings" + colorama.Style.RESET_ALL)
            # dirty hack, maybe avoid
            return

        if not econf.USE_FIXED_STRONG_BORDERS:
            self.gather_all_angles()
            self.get_curve_distribution()

        # self.gather_all_angles()
        # self.get_curve_distribution()

        # print("self.all_angles", self.all_angles)
        # self.get_const_for_angle(200)

        # for testing purposes
        # first_test_road = self.data_dict['random--la52']
        # self.nodes_to_sdl_2d(first_test_road['nodes'])
        # second_test_road = self.data_dict['random--la54']
        # import utils
        # print("lcs: ", utils.lcs(first_test_road['sdl_2d'], second_test_road['sdl_2d']))
        # print("lcstr: ", utils.LCSubStr(first_test_road['sdl_2d'], second_test_road['sdl_2d']))
        # third_test_road = self.data_dict['1-3']
        # print("first road", first_test_road, "first road", second_test_road)
        # print("first to third road", self.cur_sdl_one_to_one(first_test_road['curve_sdl'], third_test_road['curve_sdl']))
        # print(self.cur_sdl_one_to_all_unoptimized(second_test_road['curve_sdl']))
        # print("second road", second_test_road)

    def all_roads_to_curvature_sdl(self):
        """ Performs shape definition language on all roads represented as asfault nodes


        :return: None
        """
        assert self.data_dict is not None, "There have to be roads added"
        for name in self.data_dict:
            test = self.data_dict[name]
            nodes = test[utils.DicConst.NODES.value]
            assert nodes is not None, "There have to be nodes for each road"
            # save both the curve only and the 2d shape definition language representation
            self.data_dict[name][utils.DicConst.CUR_SDL.value] = self.nodes_to_curvature_sdl(nodes=nodes,
                                                                                             compress_neighbours=True)
            self.data_dict[name][utils.DicConst.SDL_2D.value] = self.nodes_to_sdl_2d(nodes=nodes)

    def nodes_to_curvature_sdl(self, nodes: List[asfault.network.NetworkNode], compress_neighbours: bool = False):
        """ curve shape definition language of a single road

        :param nodes: List of all asfault.network.NetworkNode of a road
        :param compress_neighbours: only update list if the successor is different
        :return: List with all the symbols
        """
        curve_sdl = []

        if compress_neighbours:
            for node in nodes:
                if utils.compute_length(node) >= utils.MINIMUM_SEG_LEN:
                    current_angle = self.get_const_for_angle(node.angle)
                    if current_angle != curve_sdl:
                        curve_sdl.append(current_angle)
                    # fixme distribution seems many zeros, straights get classified as slight lefts
                    # print("node.roadtype", node.roadtype, "; angle ", node.angle, "; cur type", self.get_const_for_angle(node.angle))
        else:
            for node in nodes:
                curve_sdl.append(self.get_const_for_angle(node.angle))

        too_short_segments = len(nodes) - len(curve_sdl)
        if too_short_segments > 0:
            print(str(too_short_segments) + " segments were ignored, because they were too short")

        return curve_sdl

    def all_roads_average_curvature(self, normalized: bool = True):
        def _average_curvature(road_sdl: list, normalized: bool) -> float:
            sum = 0
            for seg in road_sdl:
                sum += abs(seg.value)
            avg = sum / len(road_sdl)

            # Normalize by the maximum segment value
            if normalized:
                avg /= (NUM_ALPHABET-1)/2

            return avg

        for road in self.data_dict.values():
            avg_curve = _average_curvature(road[utils.DicConst.CUR_SDL.value], normalized)
            road[utils.DicConst.AVG_CURVATURE.value] = avg_curve

    def sdl_all_to_all_unoptimized(self):
        for name in self.data_dict:
            # TODO schau mal ob da alles passt
            distance_arr = self.compare_one_to_all_unoptimized(name, funct=self.cur_sdl_one_to_one,
                                                               representation=utils.DicConst.CUR_SDL.value)
            self.data_dict[name][utils.DicConst.CUR_SDL_DIST.value] = distance_arr

            # distance_arr = self.sdl_2d_one_to_all_unoptimized(self.data_dict[name][utils.DicConst.SDL_2D.value])
            distance_arr = self.compare_one_to_all_unoptimized(name, funct=self.sdl_2d_one_to_one,
                                                               representation=utils.DicConst.SDL_2D.value)
            self.data_dict[name][utils.DicConst.SDL_2D_DIST.value] = distance_arr

            distance_arr = self.compare_one_to_all_unoptimized(name, funct=utils.lcs,
                                                               representation=utils.DicConst.CUR_SDL.value)
            self.data_dict[name][utils.DicConst.CUR_SDL_LCS_DIST.value] = distance_arr

            distance_arr = self.compare_one_to_all_unoptimized(name, funct=utils.lcs,
                                                               representation=utils.DicConst.SDL_2D.value)
            self.data_dict[name][utils.DicConst.SDL_2D_LCS_DIST.value] = distance_arr

            distance_arr = self.compare_one_to_all_unoptimized(name, funct=utils.LCSubStr,
                                                               representation=utils.DicConst.CUR_SDL.value)
            self.data_dict[name][utils.DicConst.CUR_SDL_LCSTR_DIST.value] = distance_arr

            distance_arr = self.compare_one_to_all_unoptimized(name, funct=utils.LCSubStr,
                                                               representation=utils.DicConst.SDL_2D.value)
            self.data_dict[name][utils.DicConst.SDL_2D_LCSTR_DIST.value] = distance_arr

            distance_arr = self.compare_one_to_all_unoptimized(name, funct=utils.k_lcstr,
                                                               representation=utils.DicConst.CUR_SDL.value)
            self.data_dict[name][utils.DicConst.CUR_SDL_K_LCSTR_DIST.value] = distance_arr

            distance_arr = self.compare_one_to_all_unoptimized(name, funct=utils.k_lcstr,
                                                               representation=utils.DicConst.SDL_2D.value)
            self.data_dict[name][utils.DicConst.SDL_2D_K_LCSTR_DIST.value] = distance_arr

            distance_arr = self.compare_one_to_all_unoptimized(name, funct=self.jaccard_sdl_2d_one_to_one,
                                                               representation=utils.DicConst.SDL_2D.value)
            self.data_dict[name][utils.DicConst.JACCARD.value] = distance_arr



    # these three should be one
    def compare_one_to_all_unoptimized(self, road_name: str, funct, representation: str):
        distance_dict = {}
        road_dic = self.data_dict[road_name]
        assert road_dic is not None, "The road has not been found in the dict!"
        sdl = road_dic.get(representation, None)
        assert sdl is not None, "The sdl representation has not been found!"
        for name in self.data_dict:
            dist = funct(sdl, self.data_dict[name][representation])
            distance_dict[name] = dist
        return distance_dict

    def jaccard_sdl_2d_one_to_one(self, curve1_sdl: list, curve2_sdl: list, normalized: bool = True) -> float:
        """ Set-based jaccard index
        Only works with sdl 2d representation
        # TODO there has to be a more efficient way

        :param curve1_sdl: sdl representation of one road
        :param curve2_sdl: sdl representation of the other road
        :return: Jaccard index
        """
        shared = []
        union = []

        def _symbol_in_list(symbol: tuple, li: list) -> bool:
            result = any(elem == symbol for elem in li)
            return result

        for symbol in curve1_sdl:
            if not _symbol_in_list(symbol, union):
                union.append(symbol)
            if not _symbol_in_list(symbol, shared):
                if _symbol_in_list(symbol, curve2_sdl):
                    shared.append(symbol)
        for symbol in curve2_sdl:
            if not _symbol_in_list(symbol, union):
                union.append(symbol)

        if normalized:
            jacc = len(shared) / len(union)
        else:
            jacc = len(shared)
        return jacc

    def cur_sdl_one_to_one(self, curve1_sdl: list, curve2_sdl: list, normalized: bool = True, invert: bool = True):
        best_similarity = float('inf')
        best_startpoint = 0
        if len(curve1_sdl) < len(curve2_sdl):
            shorter_road = curve1_sdl
            longer_road = curve2_sdl
        else:
            shorter_road = curve2_sdl
            longer_road = curve1_sdl
        for start_point in range(0, len(longer_road)):
            error = self._cur_sdl_error_at_startpoint(start_point=start_point, longer_road_sdl=longer_road,
                                                      shorter_road_sdl=shorter_road)
            if error < best_similarity:
                best_similarity = error
                best_startpoint = start_point
        if normalized:
            # normalize by the length of the shorter road, as errors get summed up
            # normalize by (NUM_ALPHABET-1), as values have to range from -(NUM_ALPHABET-1)/2 to (NUM_ALPHABET-1)/2
            best_similarity = best_similarity / (len(shorter_road) * (NUM_ALPHABET - 1))

        # FIXME still experimental, might fail
        if invert:
            error = 1 - best_similarity
        assert 0 <= best_similarity <= 1, "The error " + str(error) + " is outside the range"
        return best_similarity

    def _cur_sdl_error_at_startpoint(self, start_point: int, longer_road_sdl: List[cur],
                                     shorter_road_sdl: List[cur]) -> float:
        """ calculates error element wise for the curvature shape definition language representation
        WARNING: the output is not yet normalized by length, but counting, to save unnecessary divisions
        error can be up to n * (NUM_ALPHABET-1)/2

        :param start_point:
        :param longer_road_sdl: curve sdl representation of the longer road
        :param shorter_road_sdl: curve sdl representation of the shorter road
        :return: counting error, can be up to n * (NUM_ALPHABET-1)/2
        """
        error = 0
        for i in range(0, len(shorter_road_sdl)):
            error += abs(longer_road_sdl[(start_point + i) % len(longer_road_sdl)].value - shorter_road_sdl[i].value)
        return error

    def sdl_2d_one_to_one(self, sdl_2d_1, sdl_2d_2, normalized: bool = True, invert: bool = True):

        best_similarity = float('inf')
        best_startpoint = 0
        if len(sdl_2d_1) < len(sdl_2d_2):
            shorter_road = sdl_2d_1
            longer_road = sdl_2d_2
        else:
            shorter_road = sdl_2d_2
            longer_road = sdl_2d_1
        for start_point in range(0, len(longer_road)):
            error = self._sdl_2d_error_at_startpoint(start_point=start_point, longer_road_sdl=longer_road,
                                                     shorter_road_sdl=shorter_road)
            if error < best_similarity:
                best_similarity = error
                best_startpoint = start_point
        if normalized:
            best_similarity = best_similarity / len(shorter_road)

        # FIXME still experimental, this could cause problems
        if invert:
            best_similarity = 1 - best_similarity
        assert 0 <= best_similarity <= 1, "The error " + str(best_similarity) + " is outside the range"
        return best_similarity

    def _sdl_2d_error_at_startpoint(self, start_point: int, longer_road_sdl: List,
                                    shorter_road_sdl: List) -> float:
        """ calculates error element wise for the 2d shape definition language representation
        WARNING: the output is not yet normalized by road length, but alphabet length, to save unnecessary divisions
        error can be up to n * 1.0

        :param start_point:
        :param longer_road_sdl: curve sdl representation of the longer road
        :param shorter_road_sdl: curve sdl representation of the shorter road
        :return: counting error, can be up to n * 1.0
        """
        # TODO adjust these weights
        CURVE_WEIGHT = 0.666
        LENGTH_WEIGHT = 0.333
        error = 0
        for i in range(0, len(shorter_road_sdl)):
            error_cur = 0
            error_len = 0
            error_cur += abs(longer_road_sdl[(start_point + i) % len(longer_road_sdl)][0].value
                             - shorter_road_sdl[i][0].value)
            error_len += abs(longer_road_sdl[(start_point + i) % len(longer_road_sdl)][1].value
                             - shorter_road_sdl[i][1].value)
            # normalize by the length of each alphabet
            error_cur = error_cur / (NUM_ALPHABET - 1)
            error_len = error_len / (NUM_LEN_ALPHABET - 1)
            error += (error_cur * CURVE_WEIGHT + error_len * LENGTH_WEIGHT) / (CURVE_WEIGHT + LENGTH_WEIGHT)

        return error

    def nodes_to_sdl_2d(self, nodes: List[asfault.network.NetworkNode]) -> list:
        # used to accumulate lengths of previous same curve segments
        lengths = 0
        sdl_2d = []
        current_angle = None
        next_angle = self.get_const_for_angle(nodes[0].angle)

        for i in range(0, len(nodes) - 1):
            node = nodes[i]
            lengths += node.length

            current_angle = next_angle
            next_angle = self.get_const_for_angle(nodes[i + 1].angle)
            # print("current_angle, next_angle", current_angle, ", ", next_angle)

            if next_angle != current_angle:
                length_en = self.get_const_for_length(lengths)
                # print("appending, length", length_en)
                sdl_2d.append((current_angle, length_en))
                lengths = 0

        lengths += nodes[-1].length
        last_tup = (self.get_const_for_angle(nodes[-1].angle), self.get_const_for_length(lengths))
        sdl_2d.append(last_tup)
        # print("sdl_2d", sdl_2d)

        return sdl_2d

    def get_const_for_angle(self, angle: float):
        """ returns the representation for an angle based on the percentiles
        needs self.percentile_values to be set

        :param angle: the angle
        :return: cur type
        """
        # TODO stimmt das überhaupt?
        if econf.USE_FIXED_STRONG_BORDERS:
            percentile_values_cur = DEFAULT_PERCENTILE_VALUES_CUR
        else:
            percentile_values_cur = self.percentile_values
        assert percentile_values_cur, "percentile values have to be defined"

        # start value at most negative -> most left value
        # type conversions for rounding to zero
        cur_i = -int(float(len(cur) / 2))
        # index for the curvature percentile array
        per_j = 1
        if angle <= percentile_values_cur[-1]:
            while angle > percentile_values_cur[per_j]:
                cur_i += 1
                per_j += 1
        else:
            cur_i = abs(cur_i)
        return cur(cur_i)

    def get_const_for_length(self, length: float):
        """ returns the representation for an length based on the percentiles

        :param length: the length
        :return: cur type
        """
        # TODO stimmt das überhaupt?
        percentile_values_len = DEFAULT_PERCENTILE_VALUES_LEN

        # start value at the shortest value
        len_i = 0
        # index for the curvature percentile array
        per_j = 1
        if length < percentile_values_len[-1]:
            while length > percentile_values_len[per_j]:
                len_i += 1
                per_j += 1
        else:
            len_i = NUM_LEN_ALPHABET - 1
        return len_en(len_i)

    def gather_all_angles(self):
        """ is needed to be able to get the percentiles
        has an performance overhead, should be avoided

        :return: None
        """
        for name in self.data_dict:
            test = self.data_dict[name]
            nodes = test['nodes']
            assert nodes is not None, "There have to be nodes for each road"
            for node in nodes:
                self.all_angles.append(node.angle)

    def get_curve_distribution(self):
        """ draws a box plot and self.percentile_values with bounds of all angles
        has to be called after all curves were compressed using shape definition
        has an performance overhead, should be avoided, borders fixed

        :return: None
        """
        if not self.all_angles:
            self.all_roads_to_curvature_sdl()
        # fig1, ax1 = plt.subplots()
        # ax1.set_title('Basic Plot')
        plt.boxplot(self.all_angles)
        plt.show()

        percentile_step = 100 / NUM_ALPHABET
        percentile_step_sum = 0
        self.percentile_values = []
        for i in range(0, NUM_ALPHABET + 1):
            # print("percentile_step_sum", percentile_step_sum)
            self.percentile_values.append(np.percentile(self.all_angles, percentile_step_sum))
            percentile_step_sum += percentile_step
            if percentile_step_sum > 100:
                percentile_step_sum = 100

        print("percentile values of all roads", self.percentile_values)
        median = np.percentile(self.all_angles, 50)
        print("median of all roads", median)

        # TODO remove the check and ensure a good balance
        # <= 0 <= is counter productive, maybe look at the distribution of all right curves with half of the straights
        # and all left curves with half of the straights
        assert -2 < self.percentile_values[int(np.ceil(NUM_ALPHABET / 2))] < 2 \
               and self.percentile_values[int(np.ceil(NUM_ALPHABET / 2)) - 1] <= 0 <= self.percentile_values[
                   int(np.ceil(NUM_ALPHABET / 2)) + 1], \
            "the curve distribution of the dataset is not balanced!"
