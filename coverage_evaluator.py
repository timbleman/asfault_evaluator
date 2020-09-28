from os import listdir, path
from os.path import isfile, join

import logging as l

import numpy as np
import json
from asfault.tests import RoadTest, TestExecution, CarState

from scipy import stats
import colorama
from pathlib import Path

from code_obe_evaluator import OBEEvaluator, OBE

import evaluator_config as econf

import utils
from utils import RoadDicConst
from utils import BehaviorDicConst

# TODO OBE Coverage

NUM_BINS = 16
NUM_OBE_BINS = 10
STEERING_RANGE = (-1, 1)
SPEED_RANGE = (0, 85)
# fixme 180 instead of 360 ?
ANGLE_RANGE = (-360, 360)

# 17 dividers for 16 bins, calculated by utils.optimized_bin_borders_percentiles()
# These values are computed using the DriverAI set
# first value -0.68029414 has been replaced by -1.0, The value exists as the car drives on the right side of the road
adjusted_steering_borders = [-1.0, -0.08968573, -0.05965406, -0.04873885, -0.04021686, -0.03441211,
                             -0.03070541, -0.02555755, -0.00635312, 0.01986001, 0.02915875, 0.03418135,
                             0.04203678, 0.04906175, 0.05933272, 0.08456601, 1.0]


def get_set_name(set_path: Path) -> str:
    """ tries to find a unique name for each executed set
    combines the first part of the parent folders name and the last number
    Requires a notation like Bill Bosshards

    :param set_path: Path to the folder where outputs and executions are stored
    :return: name to identify the set
    """
    name_of_parent = str(set_path.parts[-2])
    # Add first part of parents name
    set_name = name_of_parent[0:econf.first_chars_of_experiments_subfolder]

    # Add the last number
    path_index = -1
    while name_of_parent[path_index - 1].isdigit():
        path_index -= 1
    set_name += name_of_parent[path_index:]

    print("set name ", set_name, "for", str(set_path))
    return set_name


def setup_logging(log_level):
    level = l.INFO
    if log_level == "DEBUG":
        level = l.DEBUG
    elif log_level == "WARNING":
        level = l.WARNING
    elif log_level == "ERROR":
        level = l.ERROR

    term_handler = l.StreamHandler()
    l.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                  level=level, handlers=[term_handler])


def get_obes_dict(executions):
    obe_evaluator = OBEEvaluator(executions)

    obe_data = list()
    for global_id, obe in enumerate(obe_evaluator.obes):
        obe_dict = OBE.to_dict(obe)
        # Extend obe information with additional features
        obe_dict['global_id'] = global_id
        obe_data.append(obe_dict)
    # print("obe data ", obe_data)
    return obe_data


class CoverageEvaluator:
    def __init__(self, set_path: Path):
        env_directory = str(set_path)
        print()
        l.info("Start evaluation of OBEs from %s", env_directory)
        print(colorama.Fore.BLUE + "Start evaluation of OBEs from %s", str(env_directory), colorama.Style.RESET_ALL)

        executions, executions_dict = self._get_execution_list_and_dict(env_directory)

        set_name = get_set_name(set_path)

        # TODO refactor, remove these
        '''from here on the old constructor'''
        self.speed_arr = []
        self.steering_arr = []
        self.distance_arr = []
        self.obe_speed_arr = []
        self.obe_angle_arr = []

        # do not activate this unless you need all steering angles in a list
        self.collect_all_angles = False
        self.all_angles = []

        # needed for infinite speeds
        self.broken_speed_tests = []

        self.suite_bins = {}
        self.global_name = set_name

        executions = list(executions_dict.values())

        obe_dict = get_obes_dict(executions)

        for name in executions_dict:
            self._fill_bins(name, executions_dict[name], obe_dict)

    def _get_execution_list_and_dict(self, env_directory):
        # Load the configuration from the given env and store in the "global" asfault configurations ev, ex
        from asfault.app import read_environment
        read_environment(env_directory)
        from asfault.config import rg as asfault_environment

        # Read all the execution data of this experiment
        l.info("Reading execution data from %s", str(asfault_environment.get_execs_path()))

        executions = list()
        executions_dict = {}

        for test_file_name in [f for f in listdir(asfault_environment.get_execs_path()) if
                               isfile(join(asfault_environment.get_execs_path(), f))]:
            test_file = path.join(asfault_environment.get_execs_path(), test_file_name)

            # Load test object from file
            with open(test_file, 'r') as in_file:
                test_dict = json.loads(in_file.read())

            the_test = RoadTest.from_dict(test_dict)

            executions.append(the_test.execution)
            executions_dict[test_file_name] = the_test.execution

        return (executions, executions_dict)

    def _fill_bins(self, test_file_name: str, execution, obe_dict):
        """ fills the bins for a single execution

        :param test_file_name: name of a single test execution
        :param execution: execution of a test
        :param obe_dict: dictionary containing OBEs
        :return: None
        """
        from asfault.config import rg as asfault_environment

        self.speed_arr = []
        self.steering_arr = []
        self.distance_arr = []
        self.obe_speed_arr = []
        self.obe_angle_arr = []
        broken_speed = False
        num_states = len(execution.states)
        for state in execution.states:
            state_dict = CarState.to_dict(state)
            speed_kph = np.linalg.norm([state.vel_x, state.vel_y]) * 3.6
            if speed_kph >= 500:
                if econf.rm_broken_speed_roads:
                    print("Broken speed detected")
                    broken_speed = True
                    self.speed_arr.append(speed_kph)
                    self.steering_arr.append(state_dict['steering'])
                else:
                    print("Broken speed detected, state not added")
            else:
                self.speed_arr.append(speed_kph)
                self.steering_arr.append(state_dict['steering'])
            self.distance_arr.append(state.get_centre_distance())

        # do not do this always
        if self.collect_all_angles:
            print("steering arr stats for adjusting, min mean max:", min(self.steering_arr), np.mean(self.steering_arr),
                  max(self.steering_arr))
            self.all_angles.extend(self.steering_arr)

        obe_list = [d for d in obe_dict if d['test_id'] == execution.test.test_id]
        for obe in obe_list:
            obe_speed = obe['speed']
            self.obe_speed_arr.append(obe_speed)
            obe_angle = (obe['road_angle'] - obe['heading_angle']) % 360
            self.obe_angle_arr.append(obe_angle)

        road_nodes = execution.test.get_path()
        road_polyline = execution.test.get_path_polyline()
        test_path = path.join(asfault_environment.get_execs_path(),
                              test_file_name)  # test_file_name #""# execution.test
        exec_time = execution.end_time - execution.start_time
        if broken_speed:
            self.broken_speed_tests.append(test_path)
        # .testid instead of whole execution object?
        bins = {RoadDicConst.TEST_ID.value: execution.test.test_id,
                RoadDicConst.TEST_PATH.value: test_path,
                RoadDicConst.SPEED_BINS.value: self.get_speed_bins(),
                RoadDicConst.STEERING_BINS.value: self.get_steering_bins(),
                RoadDicConst.STEERING_BINS_ADJUSTED.value: self.get_steering_bins_adjusted(),
                RoadDicConst.DISTANCE_BINS.value: self.get_distance_bins((0, 20)),  # TODO is (0, 20) a good range?
                RoadDicConst.SPEED_STEERING_2D.value: self.get_speed_steering_2d(),
                RoadDicConst.OBE_2D.value: self.get_obe_speed_angle_bins(),
                RoadDicConst.NODES.value: road_nodes,
                RoadDicConst.UNDER_MIN_LEN_SEGS.value: utils.road_has_min_segs(road_nodes),
                RoadDicConst.EXEC_TIME.value: exec_time,
                RoadDicConst.NUM_OBES.value: len(obe_list),
                RoadDicConst.POLYLINE.value: road_polyline,
                RoadDicConst.ROAD_LEN.value: road_polyline.length,
                BehaviorDicConst.NUM_STATES.value: num_states,
                BehaviorDicConst.EXEC_RESULT.value: execution.result}

        # print("bins: ", bins)
        road_name = self.global_name + str(execution.test.test_id)
        # check if road is included, may need renaming if the road is dífferent
        if road_name in self.suite_bins:
            print("road is already included in the dict!")
        else:
            self.suite_bins[road_name] = bins

    def get_all_bins(self):
        return self.suite_bins

    def get_broken_speed_tests(self) -> list:
        return self.broken_speed_tests

    def get_bins(self, data: list, bounds):
        """ Returns a list of equal width bins for an array, the bins are non-binary but counting.

        :param data: List of data points
        :param bounds: Tuple from start to end (start, end), values outside get discard
        :return: List of bins
        """
        bins, bin_edges, binnum = stats.binned_statistic(data, data, bins=NUM_BINS, range=bounds, statistic='count')
        return bins

    def get_non_uniform_bins(self, data: list, bounds, dividers: list):
        """ Returns a list of non-equal width bins for an array, the bins are non-binary but counting.

        :param data: List of data points
        :param bounds: Tuple from start to end (start, end), values outside get discard
        :param dividers: Points at which to cut, has to be number_of_bins + 1
        :return: List of bins
        """
        bins, bin_edges, binnum = stats.binned_statistic(data, data, bins=dividers, range=bounds, statistic='count')
        return bins

    def get_steering_bins(self):
        return self.get_bins(self.steering_arr, STEERING_RANGE)

    def get_steering_bins_adjusted(self):
        return self.get_non_uniform_bins(data=self.steering_arr, bounds=STEERING_RANGE,
                                         dividers=adjusted_steering_borders)

    def get_speed_bins(self):
        # TODO change the bounds, find a good compromise, maybe dynamically
        return self.get_bins(self.speed_arr, SPEED_RANGE)

    def get_distance_bins(self, bounds):
        return self.get_bins(self.distance_arr, bounds)

    def get_speed_steering_2d(self):
        """ Returns a two dimensional array of bins with the steering input as x-axis and the speed as y-axis

        :return: histogram as a two-dimensional array
        """
        histogram, steering_edges, speed_edges = np.histogram2d(self.steering_arr, self.speed_arr, bins=NUM_BINS,
                                                                range=(STEERING_RANGE, SPEED_RANGE), normed=False)
        return histogram

    def get_obe_speed_angle_bins(self):
        histogram, speed_edges, angle_edges = np.histogram2d(self.obe_speed_arr, self.obe_angle_arr, bins=NUM_OBE_BINS,
                                                             range=(SPEED_RANGE, ANGLE_RANGE), normed=False)
        return histogram
