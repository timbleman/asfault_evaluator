from enum import Enum
import numpy as np

import asfault
from asfault import network
from asfault.network import NetworkNode
from typing import List

# for box plots
import matplotlib as mpl
import matplotlib.pyplot as plt

"""has to be symmetric around zero"""
NUM_ALPHABET = 7


class cur(Enum):
    STRONG_LEFT = -3
    LEFT = -2
    SLIGHT_LEFT = -1
    STRAIGHT = 0
    SLIGHT_RIGHT = 1
    RIGHT = 2
    STRONG_RIGHT = 3


class StringComparer:
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.all_angles = []
        self.percentile_values = []

        self.gather_all_angles()
        self.get_curve_distribution()

        self.all_roads_to_curvature_sdl()

        print("self.all_angles", self.all_angles)
        self.get_const_for_angle(200)

        # for testing purposes
        first_test_road = self.data_dict['1-1']
        second_test_road = self.data_dict['1-2']
        third_test_road = self.data_dict['1-3']
        print("first road", first_test_road, "first road", second_test_road)
        print("first to third road", self.cur_sdl_one_to_one(first_test_road['curve_sdl'], third_test_road['curve_sdl']))

    def all_roads_to_curvature_sdl(self):
        """ Performs shape definition language on all roads represented as asfault nodes


        :return:
        """
        assert self.data_dict is not None, "There have to be roads added"
        for name in self.data_dict:
            test = self.data_dict[name]
            # print("name, test", name, test)
            nodes = test['nodes']
            assert nodes is not None, "There have to be nodes for each road"
            # TODO add road to dict
            self.data_dict[name]['curve_sdl'] = self.nodes_to_curvature_sdl(name=name, nodes=nodes)


    def gather_all_angles(self):
        """ is needed to be able to get the percentiles
        has an performance overhead, should be avoided

        :param nodes:
        :return:
        """
        for name in self.data_dict:
            test = self.data_dict[name]
            nodes = test['nodes']
            assert nodes is not None, "There have to be nodes for each road"
            for node in nodes:
                self.all_angles.append(node.angle)

    def nodes_to_curvature_sdl(self, name: str, nodes: List[asfault.network.NetworkNode]):
        """ shape definition language of a single road

        :return:
        """
        curve_sdl = []
        # self.all_angles = []
        jo = 0
        for node in nodes:
            curve_sdl.append(self.get_const_for_angle(node.angle))
            # fixme distribution seems many zeros, straights get classified as slight lefts
            print("node.roadtype", node.roadtype, "; angle ", node.angle, "; cur type",
                  self.get_const_for_angle(node.angle))
            if jo == 0:
                print("print node to dict ", node.to_dict(node))
                jo = 1
        print("curve_sdl", curve_sdl)
        return curve_sdl

    def cur_sdl_one_to_one(self, curve1_sdl, curve2_sdl):
        best_similarity = float('inf')
        best_startpoint = 0
        if len(curve1_sdl) < len(curve2_sdl):
            shorter_road = curve1_sdl
            longer_road = curve2_sdl
        else:
            shorter_road = curve2_sdl
            longer_road = curve1_sdl
        for start_point in range(0, len(longer_road)):
            error = 0
            for i in range(0, len(shorter_road)):
                error += abs(longer_road[(start_point + i) % len(longer_road)].value - shorter_road[i].value)
            if error < best_similarity:
                best_similarity = error
                best_startpoint = start_point
        return best_similarity

    def get_const_for_angle(self, angle: float):
        """ returns the representation for an angle based on the percentiles
        needs self.percentile_values to be set

        :param angle: the angle
        :return: cur type
        """
        assert self.percentile_values, "percentile values have to be defined"

        # start value at most negative -> most left value
        # type conversions for rounding to zero
        cur_i = -int(float(len(cur) / 2))
        # index for the curvature percentile array
        per_j = 1
        if angle <= self.percentile_values[-1]:
            while angle > self.percentile_values[per_j]:
                cur_i += 1
                per_j += 1
        else:
            cur_i = abs(cur_i)
        return cur(cur_i)

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
            print("percentile_step_sum", percentile_step_sum)
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
