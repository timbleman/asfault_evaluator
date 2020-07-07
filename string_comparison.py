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

        self.all_roads_to_curvature_sdl()

        self.get_curve_distribution()
        self.get_const_for_angle(200)

    def all_roads_to_curvature_sdl(self):
        """ Performs shape definition language on all roads represented as asfault nodes


        :return:
        """
        assert self.data_dict is not None, "There have to be roads added"
        for name in self.data_dict:
            test = self.data_dict[name]
            #print("name, test", name, test)
            nodes = test['nodes']
            assert nodes is not None, "There have to be nodes for each road"
            self.nodes_to_curvature_sdl(name=name, nodes=nodes)

    def nodes_to_curvature_sdl(self, name: str, nodes: List[asfault.network.NetworkNode]):
        """ shape definition language of a single road

        :return:
        """
        curve_sdl = []
        self.all_angles = []
        jo = 0
        for node in nodes:
            print(node.angle)
            self.all_angles.append(node.angle)
            if jo == 0:
                print("print node to dict ", node.to_dict(node))
                jo = 1

    def get_const_for_angle(self, angle: float):
        """ returns the representation for an angle based on the percentiles
        needs self.percentile_values to be set

        :param angle: the angle
        :return: cur type
        """
        assert self.percentile_values, "percentile values have to be defined"

        # type conversions for rounding to zero
        cur_i = int(float(len(cur)/2))
        per_j = 1
        print("cur_i", cur_i)
        if angle <= self.percentile_values[-1]:
            while angle > self.percentile_values[per_j]:
                cur_i += 1
                per_j += 1
        else:
            cur_i = abs(cur_i)
        return cur(cur_i)

    def get_curve_distribution(self):
        """ draws a box plot and self.percentile_values with bounds of all angles

        :return: None
        """
        if not self.all_angles:
            self.all_roads_to_curvature_sdl()
        #fig1, ax1 = plt.subplots()
        #ax1.set_title('Basic Plot')
        plt.boxplot(self.all_angles)
        plt.show()

        num_alphabet = 7
        percentile_step = 100/num_alphabet
        percentile_step_sum = 0
        self.percentile_values = []
        for i in range(0, num_alphabet+1):
            print("percentile_step_sum", percentile_step_sum)
            self.percentile_values.append(np.percentile(self.all_angles, percentile_step_sum))
            percentile_step_sum += percentile_step
            if percentile_step_sum > 100:
                percentile_step_sum = 100

        print("percentile values of all roads", self.percentile_values)
        median = np.percentile(self.all_angles, 50)
        print("median of all roads", median)

        strongest_left = np.percentile(self.all_angles, 14)
        medium_left = np.percentile(self.all_angles, 28)
        smallest_left = np.percentile(self.all_angles, 42)
        strongest_right = np.percentile(self.all_angles, 86)
        medium_right = np.percentile(self.all_angles, 82)
        smallest_right = np.percentile(self.all_angles, 58)
