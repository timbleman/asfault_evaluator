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

NUM_BINS = 16
STEERING_RANGE = (-1, 1)
SPEED_RANGE = (0, 100)
ANGLE_RANGE = (-np.pi, np.pi)

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


def main():
    # Local import to main
    import os
    import csv

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
    def __init__(self, executions):
        self.speed_arr = []
        self.steering_arr = []
        self.distance_arr = []

        for execution in executions:
            self._fill_bins(execution)
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

    def _fill_bins(self, execution):
        for state in execution.states:
            state_dict = CarState.to_dict(state)
            self.speed_arr.append(np.linalg.norm([state.vel_x, state.vel_y]) * 3.6)
            self.steering_arr.append(state_dict['steering'])
            self.distance_arr.append(state.get_centre_distance())

        # print("arrays for each feature: ", speed_arr, steering_arr, distance_arr)

        # testid instead of whole execution object?
        bins = {'test': execution.test, 'speed_bins': self.get_speed_bins(), 'steering_bins': self.get_steering_bins(),
                "distance_bins": self.get_distance_bins((0, 100)), "speed_steering_2d": []}
        print("bins: ", bins)

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


if __name__ == "__main__":
    main()
