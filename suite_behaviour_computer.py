from typing import Dict
import numpy as np
import time

import colorama

import coverage_evaluator
import evaluator_config as econf
import utils
from utils import RoadDicConst
from utils import BehaviorDicConst




class SuiteBehaviourComputer:
    def __init__(self, t_dict: Dict, start: int = 0, end: int = 0):
        self.test_dict = t_dict
        self.coverage_dict = {}
        self.start = start
        self.end = end

        # what measures to compute on the output bins
        # adjust these to add new measure computed on different bins
        self.output_metrics_config = [
            {'bins': RoadDicConst.DISTANCE_BINS.value, 'fun': utils.list_difference_1d_2d_bin, 'out_name': BehaviorDicConst.CENTER_DIST_BINARY.value},
            {'bins': RoadDicConst.DISTANCE_BINS.value, 'fun': utils.list_difference_1d_2d_sin, 'out_name': BehaviorDicConst.CENTER_DIST_SINGLE.value},
            {'bins': RoadDicConst.STEERING_BINS.value, 'fun': utils.list_difference_1d_2d_bin, 'out_name': BehaviorDicConst.STEERING_DIST_BINARY.value},
            {'bins': RoadDicConst.STEERING_BINS.value, 'fun': utils.list_difference_1d_2d_sin, 'out_name': BehaviorDicConst.STEERING_DIST_SINGLE.value},
            {'bins': RoadDicConst.STEERING_BINS_ADJUSTED.value, 'fun': utils.list_difference_1d_2d_bin, 'out_name': BehaviorDicConst.STEERING_ADJUSTED_DIST_BINARY.value},
            {'bins': RoadDicConst.STEERING_BINS_ADJUSTED.value, 'fun': utils.list_difference_1d_2d_sin, 'out_name': BehaviorDicConst.STEERING_ADJUSTED_DIST_SINGLE.value},
            {'bins': RoadDicConst.SPEED_BINS.value, 'fun': utils.list_difference_1d_2d_bin, 'out_name': BehaviorDicConst.SPEED_DIST_BINARY.value},
            {'bins': RoadDicConst.SPEED_BINS.value, 'fun': utils.list_difference_1d_2d_sin, 'out_name': BehaviorDicConst.SPEED_DIST_SINGLE.value},
            {'bins': RoadDicConst.SPEED_STEERING_2D.value, 'fun': utils.list_difference_1d_2d_bin, 'out_name': BehaviorDicConst.BINS_STEERING_SPEED_DIST.value},
            {'bins': RoadDicConst.SPEED_STEERING_2D.value, 'fun': utils.list_difference_1d_2d_sin, 'out_name': BehaviorDicConst.BINS_STEERING_SPEED_DIST_SINGLE.value},
            {'bins': RoadDicConst.SPEED_STEERING_2D_ADJ.value, 'fun': utils.list_difference_1d_2d_bin, 'out_name': BehaviorDicConst.BINS_STEERING_SPEED_DIST_ADJUSTED.value},
            {'bins': RoadDicConst.SPEED_STEERING_2D_ADJ.value, 'fun': utils.list_difference_1d_2d_sin, 'out_name': BehaviorDicConst.BINS_STEERING_SPEED_DIST_ADJUSTED_SINGLE.value},
            {'bins': RoadDicConst.STEERING_STATES.value, 'fun': self.dtw_compare, 'out_name': BehaviorDicConst.STEERING_DTW.value},
            {'bins': RoadDicConst.SPEED_STATES.value, 'fun': self.dtw_compare, 'out_name': BehaviorDicConst.SPEED_DTW.value},
            {'bins': RoadDicConst.STEERING_SPEED_STATES.value, 'fun': self.dtw_compare, 'out_name': BehaviorDicConst.STEERING_SPEED_DTW.value}]

        self.output_metrics_to_compute = list(filter(lambda metr: metr['out_name'] in econf.output_metrics_to_analyse, self.output_metrics_config))
        print("self.output_metrics_to_compute", self.output_metrics_to_compute)

    def calculate_suite_coverage_1d(self, feature: str, add_for_each: bool = True):
        """ Calculates the 1d coverage of a selected feature across the whole test suite
            The coverage is added to the test suite dictionary

        :param feature: the feature name across the coverage is calculated
        :param add_for_each: adds the single coverage to each road
        :return: Coverage across the suite
        """
        # determine the number of bins dynamically
        arbitrary_test = next(iter(self.test_dict.values()))
        bins = arbitrary_test.get(feature, None)
        assert bins is not None, "No bins were added"
        num_bins = len(bins)
        global_bins = [0] * num_bins

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
        # determine the number of bins dynamically, bins need to be square
        arbitrary_test = next(iter(self.test_dict.values()))
        bins = arbitrary_test.get(feature, None)
        assert bins is not None, "No bins were added"
        num_bins = len(bins)
        global_2d_cov = np.zeros((num_bins, num_bins))

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
        sum = first_test.get(RoadDicConst.EXEC_TIME.value)
        for i in range(1, len(all_keys)):
            test = self.test_dict.get(all_keys[i])
            t = test.get(RoadDicConst.EXEC_TIME.value)
            sum += t
        return sum

    def behaviour_all_to_all(self):
        #output_metrics_to_compute_names =
        yanda = list(map(lambda mtr: mtr['out_name'], self.output_metrics_to_compute))
        print(colorama.Fore.BLUE + "Computing difference in the outputs " + str(yanda) + colorama.Style.RESET_ALL)
        start_time_loop = time.time()
        current_ops = 0
        total_ops = len(self.test_dict)
        print("In total", total_ops*total_ops, "comparison passes and", total_ops,
              "loop iterations will have to be completed for output behavior.")
        for name in self.test_dict:
            for out_met in self.output_metrics_to_compute:
                #print("bins, fun, name", out_met['bins'], out_met['fun'], out_met['out_name'])
                #distance_arr = self.behavior_compare_1d_2d(name, measure=out_met['bins'],
                #                                        function=out_met['fun'])
                #self.test_dict[name][out_met['out_name']] = distance_arr

                distance_arr = self.compare_one_to_all_unoptimized(road_name=name, funct=out_met['fun'],
                                                                  representation=out_met['bins'])
                self.test_dict[name][out_met['out_name']] = distance_arr


                #distance_arr = self.compare_one_to_all_unoptimized(road_name=name, funct=self.dtw_compare,
                #                                                  representation=RoadDicConst.STEERING_STATES.value)
                #self.test_dict[name][BehaviorDicConst.STEERING_DTW.value] = distance_arr
            """
            distance_arr = self.compare_one_to_all_unoptimized(road_name=name, funct=self.dtw_compare,
                                                               representation=RoadDicConst.SPEED_STATES.value)
            self.test_dict[name][BehaviorDicConst.SPEED_DTW.value] = distance_arr

            distance_arr = self.compare_one_to_all_unoptimized(road_name=name, funct=self.dtw_compare,
                                                               representation=RoadDicConst.STEERING_SPEED_STATES.value)
            self.test_dict[name][BehaviorDicConst.STEERING_SPEED_DTW.value] = distance_arr
            """

            current_ops += 1
            utils.print_remaining_time(start_time_loop, current_ops, total_ops)



    def dtw_compare(self, test_1_states: list, test_2_states: list):
        """ The dtw function requires at least two dimensional inputs, for that 1d bins need a second axis that
        represents time.

        :param test_1_states: States for test one, at least 2 dimensions
        :param test_2_states: States for test two, at least 2 dimensions
        :return: Value of dtw
        """
        import similaritymeasures
        d_dtw, _ = similaritymeasures.dtw(test_1_states, test_2_states)
        return d_dtw

    def compare_one_to_all_unoptimized(self, road_name: str, funct, representation: str):
        """ # TODO this should teplace behavior_compare_1d_2d

        :param road_name:
        :param funct:
        :param representation:
        :return:
        """
        distance_dict = {}
        road_dic = self.test_dict[road_name]
        assert road_dic is not None, "The road has not been found in the dict!"
        rep = road_dic.get(representation, None)
        assert rep is not None, "The representation has not been found!"
        for name in self.test_dict:
            dist = funct(rep, self.test_dict[name][representation])
            distance_dict[name] = dist
        return distance_dict

    def behavior_compare_1d_2d(self, road_to_compare: str, measure: str, function: str = 'binary'):
        """ compares the coverage of a single-dimensional feature of a road to all others in the suite

        :param road_to_compare: the baseline road which is compared to all others
        :param measure: the feature which is compare, has to be present for each road in the suite dict
        :param function: "binary" (coverage) or "single" (counting) comparison
        :return: the road similarities
        """
        road_similarities = {}
        # TODO do this everywhere, more pythonic
        main_bin = self.test_dict.get(road_to_compare, None).get(measure, None)
        assert main_bin is not None, "The bin " + measure + " has not been added or spelling is incorrect"

        for name in self.test_dict:
            test_to_compare = self.test_dict[name]
            road_similarities[name] = utils.list_difference_1d_2d(main_bin,
                                                               test_to_compare.get(measure),
                                                               function=function, normalized=True)
        self.test_dict.get(road_to_compare)[road_to_compare + '_' + function + '_' + measure] = road_similarities
        return road_similarities

    def get_test_dict(self):
        return self.test_dict
