from typing import Dict
import numpy as np

import coverage_evaluator
import utils

NUM_BINS = coverage_evaluator.NUM_BINS



class SuiteBehaviourComputer:
    def __init__(self, t_dict: Dict, start: int = 0, end: int = 0):
        self.test_dict = t_dict
        self.coverage_dict = {}
        self.start = start
        self.end = end

    def calculate_suite_coverage_1d(self, feature: str, add_for_each: bool = True):
        """ Calculates the 1d coverage of a selected feature across the whole test suite
            The coverage is added to the test suite dictionary

        :param feature: the feature name across the coverage is calculated
        :param add_for_each: adds the single coverage to each road
        :return: Coverage across the suite
        """
        global_bins = [0] * NUM_BINS
        for key, test in self.test_dict.items():
            # print("test ", test)
            cov_col = test.get(feature, None)
            assert cov_col is not None, "The bin " + feature + " has not been added or spelling is incorrect"
            # the if should be avoided, add coverage to each road
            if add_for_each:
                single_coverage = utils.coverage_compute_1d(cov_col)
                test[feature + "_cov"] = single_coverage
            global_bins = np.add(global_bins, cov_col)
        cov = utils.coverage_compute_1d(global_bins)
        self.coverage_dict["whole_suite_" + feature + "_coverage"] = cov
        entr = utils.entropy_compute_1d(global_bins)
        self.coverage_dict["whole_suite_" + feature + "_entropy"] = entr
        print("The suite covered ", self.coverage_dict["whole_suite_" + feature + "_coverage"], "% of", feature)
        return cov

    def calculate_suite_2d_coverage(self, feature: str, add_for_each: bool = True):
        """ Calculates the 2d coverage of steering across the whole test suite
            The coverage is added to the test suite dictionary

        :param feature: the feature name across the coverage is calculated
        :param add_for_each: adds the single coverage to each road
        :return: Coverage across the suite
        """
        global_2d_cov = np.zeros((NUM_BINS, NUM_BINS))
        for key, test in self.test_dict.items():
            cov_col = test.get(feature, None)
            assert cov_col is not None, "The bin " + feature + " has not been added or spelling is incorrect"
            # the if should be avoided, add coverage to each road
            if add_for_each:
                single_coverage = utils.coverage_compute_1d(cov_col)
                test[feature + "_cov"] = single_coverage
            global_2d_cov = np.add(global_2d_cov, cov_col)
        cov = utils.coverage_compute_2d(global_2d_cov)
        self.coverage_dict["whole_suite_" + feature + "_coverage"] = cov
        self.coverage_dict["whole_suite_" + feature + "_entropy"] = utils.entropy_compute_2d(global_2d_cov)
        print("The suite covered ", self.coverage_dict["whole_suite_" + feature + "_coverage"], "% of", feature)
        return cov

    def calculate_whole_suite_sum(self, feature: str):
        """ Extracts all values of a feature and adds them up, has to be numeric

        :param feature: the feature name across the coverage is calculated
        :return: sum over the feature
        """
        sum1 = 0
        for key, test in self.test_dict.items():
            cov_col = test.get(feature, None)
            assert cov_col is not None, "The bin " + feature + " has not been added or spelling is incorrect"
            sum1 += cov_col
        return sum1

    def calculate_whole_suite_time(self):
        """ Extracts all execution times and adds them up

        :return: sum of execution times
        """
        all_keys = list(self.test_dict.keys())
        first_test = self.test_dict.get(all_keys[0])
        sum = first_test.get(utils.RoadDicConst.EXEC_TIME.value)
        for i in range(1, len(all_keys)):
            test = self.test_dict.get(all_keys[i])
            t = test.get(utils.RoadDicConst.EXEC_TIME.value)
            sum += t
        return sum

    def behaviour_all_to_all(self):
        for name in self.test_dict:
            # TODO schau mal ob da alles passt
            distance_arr = self.behavior_compare_1d(name, measure=utils.RoadDicConst.DISTANCE_BINS.value,
                                                               function='binary')
            self.test_dict[name][utils.DicConst.CENTER_DIST_BINARY.value] = distance_arr

            distance_arr = self.behavior_compare_1d(name, measure=utils.RoadDicConst.DISTANCE_BINS.value,
                                                    function='single')
            self.test_dict[name][utils.DicConst.CENTER_DIST_SINGLE.value] = distance_arr

            distance_arr = self.behavior_compare_2d(name, measure=utils.RoadDicConst.SPEED_STEERING_2D.value)
            self.test_dict[name][utils.DicConst.BINS_STEERING_SPEED_DIST.value] = distance_arr


    def behavior_compare_1d(self, road_to_compare: str, measure: str, function: str = 'binary'):
        """ compares the coverage of a single-dimensional feature of a road to all others in the suite

        :param road_to_compare: the baseline road which is compared to all others
        :param measure: the feature which is compare, has to be present for each road in the suite dict
        :return: the road similarities
        """
        road_similarities = {}
        # TODO do this everywhere, more pythonic
        main_bin = self.test_dict.get(road_to_compare, None).get(measure, None)
        assert main_bin is not None, "The bin " + measure + " has not been added or spelling is incorrect"
        # print(main_bin)
        """
        for test in self.test_dict.values():
            compared_road_name = test['test_id']
            road_similarities[compared_road_name] = utils.list_difference_1d(main_bin,
                                                                             test.get(measure),
                                                                             function=function, normalized=True)
        """
        for name in self.test_dict:
            test_to_compare = self.test_dict[name]
            road_similarities[name] = utils.list_difference_1d(main_bin,
                                                               test_to_compare.get(measure),
                                                               function=function, normalized=True)
        self.test_dict.get(road_to_compare)[road_to_compare + '_' + function + '_' + measure] = road_similarities
        return road_similarities

    def behavior_compare_2d(self, road_to_compare: str, measure: str):
        """ compares the coverage of a two-dimensional feature of a road to all others in the suite

        :param road_to_compare: the baseline road which is compared to all others
        :param measure: the feature which is compare, has to be present for each road in the suite dict
        :return: None
        """
        road_similarities = {}
        main_bin = self.test_dict.get(road_to_compare, None).get(measure, None)
        assert main_bin is not None, "The bin " + measure + " has not been added or spelling is incorrect"
        # print(main_bin)
        """
        for test in self.test_dict.values():
            compared_road_name = test['test_id']
            road_similarities[compared_road_name] = utils.bin_difference_2d(main_bin,
                                                                            test.get(measure),
                                                                            function='binary', normalized=True)
        """
        for name in self.test_dict:
            test_to_compare = self.test_dict[name]
            road_similarities[name] = utils.bin_difference_2d(main_bin,
                                                               test_to_compare.get(measure),
                                                               function='binary', normalized=True)
        self.test_dict.get(road_to_compare)[road_to_compare + '_' + measure] = road_similarities
        return road_similarities

    def get_test_dict(self):
        return self.test_dict
