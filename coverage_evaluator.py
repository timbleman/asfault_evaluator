from os import listdir, path
from os.path import isfile, join

import sys
import matplotlib.pyplot as plt

import logging as l

import csv

import numpy as np
import numpy.linalg as la
import json
from asfault.tests import RoadTest, TestExecution, CarState
import math

from scipy import stats

from shapely.geometry import LineString

from pathlib import Path

from code_obe_evaluator import OBEEvaluator, OBE

# TODO OBE Coverage

NUM_BINS = 16
STEERING_RANGE = (-1, 1)
SPEED_RANGE = (0, 100)
# fixme 180 instead of 360
ANGLE_RANGE = (-360, 360)

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
    print("obe data ", obe_data)
    return obe_data

# TODO create callable method that returns results of a execution

def cov_evaluate_set(set_path: Path):
    # Local import to main
    import os
    import csv

    setup_logging(l.INFO)
    # ENV DIR
    #   Contains CONFIGURATION (TO LOAD)
    #   Contains EXEC folder to look for test executions
    env_directory = str(set_path)# sys.argv[1]

    l.info("Start evaluation of OBEs from %s", env_directory)

    # Load the configuration from the given env and store in the "global" asfault configurations ev, ex
    from asfault.app import read_environment
    read_environment(env_directory)

    from asfault.config import rg as asfault_environment
    from asfault.config import ev as asfault_evolution_config
    # from asfault.config import ex as asfault_execution_config

    # Read all the execution data of this experiment
    l.info("Reading execution data from %s", str(asfault_environment.get_execs_path()))

    executions = list()

    for test_file_name in [f for f in listdir(asfault_environment.get_execs_path()) if isfile(join(asfault_environment.get_execs_path(), f))]:

        test_file = path.join(asfault_environment.get_execs_path(), test_file_name)

        # Load test object from file
        with open(test_file , 'r') as in_file:
            test_dict = json.loads(in_file.read())

        the_test = RoadTest.from_dict(test_dict)
        executions.append(the_test.execution)

    print("executions ", executions)

    # TODO find meaningful global name
    set_name = ""
    path_index = -1
    while str(set_path)[path_index-1].isdigit():
        path_index -= 1
    set_name = str(set_path)[path_index:]
    print("set name ", set_name, "for", str(set_path))
    cov_evaluator = CoverageEvaluator(executions, "1-")
    all_bins_of_a_folder = cov_evaluator.get_all_bins()
    # print("all_bins_of_a_folder ", all_bins_of_a_folder)

    return all_bins_of_a_folder

    # obe_dict = get_obes_dict(executions)


def main():
    # Local import to main
    import os
    import csv
    str_path = sys.argv[1]
    cov_evaluate_set(Path(str_path))
    """
    setup_logging(l.INFO)
    # ENV DIR
    #   Contains CONFIGURATION (TO LOAD)
    #   Contains EXEC folder to look for test executions
    env_directory = sys.argv[1]

    l.info("Start evaluation of OBEs from %s", env_directory)

    # Load the configuration from the given env and store in the "global" asfault configurations ev, ex
    from asfault.app import read_environment
    read_environment(env_directory)

    from asfault.config import rg as asfault_environment
    from asfault.config import ev as asfault_evolution_config
    # from asfault.config import ex as asfault_execution_config

    # Read all the execution data of this experiment
    l.info("Reading execution data from %s", str(asfault_environment.get_execs_path()))

    executions = list()

    for test_file_name in [f for f in listdir(asfault_environment.get_execs_path()) if isfile(join(asfault_environment.get_execs_path(), f))]:

        test_file = path.join(asfault_environment.get_execs_path(), test_file_name)

        # Load test object from file
        with open(test_file , 'r') as in_file:
            test_dict = json.loads(in_file.read())

        the_test = RoadTest.from_dict(test_dict)
        executions.append(the_test.execution)

    print("executions ", executions)

    voc_evaluator = CoverageEvaluator(executions)
    """
    '''
    # Instantiate the OBE Evaluator
    obe_evaluator = OBEEvaluator(executions)

    obe_data = list()
    for global_id, obe in enumerate(obe_evaluator.obes):
        obe_dict = OBE.to_dict(obe)
        # Extend obe information with additional features
        obe_dict['global_id'] = global_id

        l.info("\tPlotting OBE %i %s", global_id, obe.test.test_id)
        # Return the id of the figures to chose which one to save to pdf
        obe_plot_id, polar_plot_id, vector_plot_id = obe_evaluator.plot_obe(obe, theta_bins, speed_bins)
        # Load the figure
        plt.figure(obe_plot_id)
        # Store it file
        obe_plot_file = os.path.abspath(path.join(asfault_environment.get_plots_path(), ''.join([str(global_id).zfill(3), '_', 'obe', '.png'])))
        obe_dict['obe_plot_file'] = obe_plot_file
        plt.savefig(obe_plot_file)

        # Load the  next figure
        plt.figure(polar_plot_id)
        # Store it
        polar_plot_file = os.path.abspath(
            path.join(asfault_environment.get_plots_path(), ''.join([str(global_id).zfill(3), '_', 'polar', '.png'])))
        obe_dict['polar_plot_file'] = polar_plot_file
        plt.savefig(polar_plot_file)

        obe_data.append(obe_dict)
    '''
    '''
    html_index_file= path.join(os.path.dirname(os.path.abspath(env_directory)), 'index.html')
    l.info("Generate HTML report %s", html_index_file)
    html_index = generate_html_index(obe_data)
    with open(html_index_file, 'w') as out:
        out.write(html_index )


    csv_file = path.join(os.path.dirname(os.path.abspath(env_directory)), '.obes')
    l.info("Generate CSV file %s", csv_file)
    # Taken from: https://www.tutorialspoint.com/How-to-save-a-Python-Dictionary-to-CSV-file
    # 'obe_id' is the unique id of the obe
    csv_columns = ['global_id', 'test_id', 'obe_id', 'speed', 'heading_angle', 'road_angle']
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns, extrasaction='ignore')
            writer.writeheader()
            for data in obe_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")
    '''


class CoverageEvaluator:
    def __init__(self, executions, global_name):
        self.speed_arr = []
        self.steering_arr = []
        self.distance_arr = []
        speed_steering_2d = []
        self.obe_speed_arr = []
        self.obe_angle_arr = []

        self.suite_bins = {}
        self.global_name = global_name

        obe_dict = get_obes_dict(executions)

        for execution in executions:
            """
            # TODO what happens if there are multiple obes?
            obe_list = [d for d in obe_dict if d['test_id'] == execution.test.test_id]
            for obe in obe_list:
                obe_speed = obe['speed']
                self.obe_speed_arr.append(obe_speed)
                obe_angle = (obe['road_angle'] - obe['heading_angle']) % 360
                self.obe_angle_arr.append(obe_angle)
            print("single obe ", obe_list)
            """
            self._fill_bins(execution, obe_dict)
            '''
            obes = self._extract_obes_from_test(execution)

            self._fill_bins(execution)
            #
            self.obe_speed.extend([obe.get_speed() for obe in obes])
            # Angles must be given in radiants
            self.theta.extend([obe.get_heading_angle() for obe in obes])
            self.obes.extend(obes)

            # This is the same for each execution !
            self.bounds = execution.test.network.bounds
            '''

    def _fill_bins(self, execution, obe_dict):
        """ fills the bins for a single execution

        :param execution:
        :param obe_dict:
        :return:
        """
        # fixme bins are adding
        self.speed_arr = []
        self.steering_arr = []
        self.distance_arr = []
        speed_steering_2d = []
        self.obe_speed_arr = []
        self.obe_angle_arr = []
        for state in execution.states:
            state_dict = CarState.to_dict(state)
            self.speed_arr.append(np.linalg.norm([state.vel_x, state.vel_y]) * 3.6)
            self.steering_arr.append(state_dict['steering'])
            self.distance_arr.append(state.get_centre_distance())

            # TODO what happens if there are multiple obes?
        obe_list = [d for d in obe_dict if d['test_id'] == execution.test.test_id]
        for obe in obe_list:
            obe_speed = obe['speed']
            self.obe_speed_arr.append(obe_speed)
            obe_angle = (obe['road_angle'] - obe['heading_angle']) % 360
            self.obe_angle_arr.append(obe_angle)
        print("single obe list ", obe_list)
        # print("arrays for each feature: ", speed_arr, steering_arr, distance_arr)

        # .testid instead of whole execution object?
        bins = {'test_id': execution.test.test_id, 'speed_bins': self.get_speed_bins(), 'steering_bins': self.get_steering_bins(),
                "distance_bins": self.get_distance_bins((0, 100)), "speed_steering_2d": self.get_speed_steering_2d(),
                "obe_2d": self.get_obe_speed_angle_bins()}

        #print("bins: ", bins)
        road_name = self.global_name + str(execution.test.test_id)
        # check if road is included, may need renaming if the road is dífferent
        if road_name in self.suite_bins:
            print("road is already included in the dict!")
        else:
            self.suite_bins[road_name] = bins

    def get_all_bins(self):
        return self.suite_bins

    def get_bins(self, data: list, bounds):
        """ Returns a list of bins for an array, the bins are non-binary but counting

        :param data: List of data points
        :param bounds: Tuple from start to end (start, end), values outside get discard
        :return: List of bins
        """
        bins, bin_edges, binnum = stats.binned_statistic(data, data, bins=NUM_BINS, range=bounds, statistic='count')
        return bins

    def get_steering_bins(self):
        return self.get_bins(self.steering_arr, STEERING_RANGE)

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
        histogram, speed_edges, angle_edges = np.histogram2d(self.obe_speed_arr, self.obe_angle_arr, bins=NUM_BINS,
                                                                range=(SPEED_RANGE, ANGLE_RANGE), normed=False)
        return histogram


if __name__ == "__main__":
    main()
