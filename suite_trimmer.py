import os
from pathlib import Path
from typing import List

import operator
import colorama

import utils
from utils import RoadDicConst


class SuiteTrimmer:
    def __init__(self, data_dict: dict, base_path):
        """
        :param data_dict: dict that includes road-dicts with bins and multiple features
        :param base_path: base path, used to write the info file
        """
        self.data_dict = data_dict
        self.base_path = base_path

    def get_random_percentage_unworthy(self, percentage: int, block_size: int = 10) -> List:
        """ Remove a certain number of tests that exceed the treshold.
        If block_size = 10, percentage is only accurate to groups of ten
        block_size = 10 ensures good mixing

        :param percentage: 0 to 100, how many should be left. May not be exactly accurate based on blocksize.
        :param block_size: Interval in which will be removed.
        :return: paths to remove.
        """
        assert 0 <= percentage <= 100, "Percentage is out of range!"
        assert 5 < block_size, "block_size is out of range!"

        unworthy_paths = []
        key_list = list(self.data_dict.keys())
        threshold = block_size * (percentage/100)
        print("threshold", threshold)

        for i in range(0, len(key_list)):
            if i % block_size >= threshold:
                road = self.data_dict.get(key_list[i], None)
                path = road.get(RoadDicConst.TEST_PATH.value, None)
                assert path is not None, "There has been no path added for " + str(key_list[i])
                unworthy_paths.append(path)

        return unworthy_paths

    def get_unworthy_paths(self, feature: str, op: operator, threshold: float) -> List:
        """ Returns a list of paths that do meet the criteria
        Only works on single numeric features! (e.g. "num_states")

        :param feature: Used for deciding whether to keep or delete the test, has to be numeric (e.g. "num_states")
        :param op: Operator to decide what to remove (e.g. operator.le)
        :param threshold: Numeric threshold
        :return: List of paths
        """
        unworthy_paths = []
        for test in self.data_dict.values():
            val = test.get(feature, None)
            assert val is not None, "The feature has not been added or spelling is incorrect"
            if op(val, threshold):
                unworthy_paths.append(test[RoadDicConst.TEST_PATH.value])

        return unworthy_paths

    def trim_dataset(self, feature: str, op: operator, threshold: float, force: bool = False):
        """ Trimms the dataset using a single numeric feature and using a threshold and a comparison function

        :param feature: Used for deciding whether to keep or delete the test, has to be numeric (e.g. "num_states")
        :param op: Operator to decide what to remove (e.g. operator.le)
        :param threshold: Numeric threshold
        :param force: Force remove or wait for user input
        :return: None
        """
        print("Using", threshold, "as a threshold to remove tests")
        unworthy_paths = self.get_unworthy_paths(feature, op, threshold)

        # create a readable reason why it has been removed and write file
        reason = feature + " " + str(op) + " " + str(threshold)
        self.trim_dataset_list(unworthy_paths=unworthy_paths, description=reason, force=force)

    def trim_dataset_list(self, unworthy_paths: list, description: str, force: bool = False) -> bool:
        """ Trimms the dataset given a list of paths to remove

        :param unworthy_paths: List of paths or names of tests to remove
        :param description: description for removal of these tests, will be written in the parent dict
        :param force: Force remove or wait for user input
        :return: Bool if suite has been trimmed
        """
        print(colorama.Fore.RED + "A total of", len(unworthy_paths), "tests will be removed from the set:",
              unworthy_paths, colorama.Style.RESET_ALL)
        # ask for user input to confirm if force is disabled
        if not force:
            print(colorama.Fore.RED + "Sure to continue? y or n" + colorama.Style.RESET_ALL)
            answer = str(input())
            if answer != 'y' and answer != 'Y':
                print(colorama.Fore.BLUE + "Aborted test deletion" + colorama.Style.RESET_ALL)
                return False

        # remove all paths that fit the criteria
        for p in unworthy_paths:
            try:
                os.remove(path=p)
            except OSError as error:
                print(error)
                print(colorama.Fore.RED + "Could not remove" + str(p) + colorama.Style.RESET_ALL)
                return False

        print(colorama.Fore.BLUE + "Successfully removed the tests!" + colorama.Style.RESET_ALL)

        self._write_info(removed_paths=unworthy_paths, reason=description)
        return True

    def _write_info(self, removed_paths: list, reason: str):
        """ Writes a file that gives Details what has been removed and why

        :param removed_paths: List of paths or names of removed tests
        :param reason: reason for removal of these tests
        :return: None
        """
        info_doc_path = os.path.join(self.base_path, 'suite_info.txt')
        # check whether the file already exists, create new or append
        if not os.path.exists(info_doc_path):
            f = open(info_doc_path, "x")
            f.write("Deleted following tests:\n")
        else:
            f = open(info_doc_path, "a")
        f.write(reason + "\n")
        for p in removed_paths:
            f.write(str(p) + "\n")
        f.close()

    def trim_dataset_percentile(self, feature: str, op: operator, threshold_percentile: int, force: bool = False):
        """ Trimms the dataset not using a fixed numeric threshold, but percentage

        :param feature: Used for deciding whether to keep or delete the test, has to be numeric (e.g. "num_states")
        :param op: Operator to decide what to remove (e.g. operator.le)
        :param threshold_percentile: Percentile, has to be in range (0, 100)
        :param force: Force remove or wait for user input
        :return: None
        """
        assert 0 <= threshold_percentile <= 100, "Threshold percentile out of range!"

        # get a numeric threshold based on the percentile
        stat_dict = utils.whole_suite_statistics(self.data_dict, feature=feature,
                                                 desired_percentile=threshold_percentile)
        thres = stat_dict['desired_percentile']

        self.trim_dataset(feature=feature, op=op, threshold=thres, force=force)
