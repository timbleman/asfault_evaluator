from typing import Dict
import numpy as np

import coverage_evaluator
import utils

NUM_BINS = coverage_evaluator.NUM_BINS


# TODO write csvs


class SuiteBehaviourComputer:
    def __init__(self, t_dict: Dict, start: int = 0, end: int = 0):
        self.test_dict = t_dict
        self.coverage_dict = {}
        self.start = start
        self.end = end

    def calculate_suite_coverage_1d(self, feature: str):
        """ Calculates the 1d coverage of a selected feature across the whole test suite
            The coverage is added to the test suite dictionary

        :param feature: the feature name across the coverage is calculated
        :return: Coverage across the suite
        """
        global_bins = [0] * NUM_BINS
        for test in self.test_dict.values():
            print("test ", test)
            cov_col = test.get(feature)
            assert cov_col is not None, "The bin " + feature + " has not been added or spelling is incorrect"
            global_bins = np.add(global_bins, cov_col)
        cov = utils.coverage_compute_1d(global_bins)
        self.coverage_dict["whole_suite_" + feature + "_coverage"] = cov
        entr = utils.entropy_compute_1d(global_bins)
        self.coverage_dict["whole_suite_" + feature + "_entropy"] = entr
        print("The suite covered ", self.coverage_dict["whole_suite_" + feature + "_coverage"], "% of", feature)
        return cov

    def calculate_suite_2d_coverage(self, feature: str):
        """ Calculates the 2d coverage of steering across the whole test suite
            The coverage is added to the test suite dictionary

        :return: Coverage across the suite
        """
        global_2d_cov = np.zeros((NUM_BINS, NUM_BINS))
        for test in self.test_dict.values():
            cov_col = test.get(feature)
            assert cov_col is not None, "The bin " + feature + " has not been added or spelling is incorrect"
            global_2d_cov = np.add(global_2d_cov, cov_col)
        cov = utils.coverage_compute_2d(global_2d_cov)
        self.coverage_dict["whole_suite_" + feature + "_coverage"] = cov
        self.coverage_dict["whole_suite_" + feature + "_entropy"] = utils.entropy_compute_2d(global_2d_cov)
        print("The suite covered ", self.coverage_dict["whole_suite_" + feature + "_coverage"], "% of", feature)
        return cov

    def road_compare_1d(self, road_to_compare: str, measure: str, function: str = 'binary'):
        """ compares the coverage of a single-dimensional feature of a road to all others in the suite

        :param road_to_compare: the baseline road which is compared to all others
        :param measure: the feature which is compare, has to be present for each road in the suite dict
        :return: the road similarities
        """
        road_similarities = {}
        main_bin = self.test_dict.get(road_to_compare).get(measure)
        assert main_bin is not None, "The bin " + measure + " has not been added or spelling is incorrect"
        # print(main_bin)
        for test in self.test_dict.values():
            compared_road_name = test['test_id']
            road_similarities[compared_road_name] = utils.list_difference_1d(main_bin,
                                                                             test.get(measure),
                                                                             function=function, normalized=True)
        self.test_dict.get(road_to_compare)[road_to_compare + '_' + function + '_' + measure] = road_similarities
        return road_similarities

    def road_compare_2d(self, road_to_compare: str, measure: str):
        """ compares the coverage of a two-dimensional feature of a road to all others in the suite

        :param road_to_compare: the baseline road which is compared to all others
        :param measure: the feature which is compare, has to be present for each road in the suite dict
        :return: None
        """
        road_similarities = {}
        main_bin = self.test_dict.get(road_to_compare).get(measure)
        assert main_bin is not None, "The bin " + measure + " has not been added or spelling is incorrect"
        # print(main_bin)
        for test in self.test_dict.values():
            compared_road_name = test['test_id']
            road_similarities[compared_road_name] = utils.bin_difference_2d(main_bin,
                                                                            test.get(measure),
                                                                            function='binary', normalized=True)
        self.test_dict.get(road_to_compare)[road_to_compare + ' ' + measure] = road_similarities
        return road_similarities

    def get_test_dict(self):
        return self.test_dict
