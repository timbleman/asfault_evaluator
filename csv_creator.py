import csv
import os
from os import path
from typing import List

import utils

import colorama
import evaluator_config as econf

class CSVCreator:
    def __init__(self, data_dict, root_path):
        self.root_path = root_path
        self.data_dict = data_dict

    def write_all_two_roads_dists(self, road_1_name: str, measures: List[str]) -> None:
        """ calles write_two_roads_dists for each possible neighbor for road_1
        names have to be exactly like in the dict

        :param road_1_name: the name of the first road
        :param measures: a list of measures to be written
        :return: None
        """
        assert measures, "There have to be measures declared"
        road_dict = self.data_dict.get(road_1_name, None)
        assert road_dict is not None, "The road " + road_1_name + " has not been found in the dict!"

        keys_of_all = self.data_dict.keys()
        print("keys_of_all", keys_of_all)

        for road2 in keys_of_all:
            self.write_two_roads_dists(road_1_name=road_1_name, road_2_name=road2, measures=measures)

    def write_two_roads_dists(self, road_1_name: str, road_2_name: str, measures: List[str]) -> None:
        """ writes csv files for two roads to compare
        creates a new sub folder in every tests output folder
        names have to be exactly like in the dict

        :param road_1_name: the name of the first road
        :param road_2_name: the name of the second road to compare to
        :param measures: a list of measures to be written
        :return: None
        """
        assert measures, "There have to be measures declared"
        road_dict = self.data_dict.get(road_1_name, None)
        assert road_dict is not None, "The road " + road_1_name + " has not been found in the dict!"
        first_measure = road_dict.get(measures[0], None)
        assert first_measure is not None, "The measure " + measures[0] + " has not been found for the road " + road_1_name
        csv_columns = ['measure', road_2_name]

        test_path = road_dict.get('test_path', None)
        assert test_path is not None, "There is no path for road " + road_1_name

        def _get_dict_to_write(index: int):
            """ local function to get a dict with the names and values of a comparison
            there is a unnecessary declaration of a new array, the alternative would be to insert and remove an element

            :param index: index of the selected measure
            :return: dict with measure name and values
            """
            assert index < len(measures), "The index is too big"
            dicc = {'measure': measures[index]}
            extracted_dict = road_dict.get(measures[index], None)
            assert extracted_dict is not None, "The measure " + measures[index] + " has not been found for the road " \
                                               + road_1_name
            # TODO shape checking
            dicc[road_2_name] = extracted_dict[road_2_name]
            return dicc

        # two splits to get to the \output\ directory
        # Create folder
        test_dir_path = path.split(test_path)
        test_dir_path = path.split(test_dir_path[0])
        # Conditional if folder does not exist
        folder_path = path.join(test_dir_path[0], road_1_name + "_csvs")
        if not path.isdir(folder_path):
            os.mkdir(folder_path)
        csv_file = path.join(folder_path, road_1_name + "_vs_" + road_2_name + ".csv")
        # TODO l.info("Generate CSV file %s", csv_file)
        try:
            with open(csv_file, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns, extrasaction='ignore')
                writer.writeheader()
                for i in range(0, len(measures)):
                    dic = _get_dict_to_write(i)
                    writer.writerow(dic)
            # print(colorama.Fore.GREEN + " wrote csv_file to ", csv_file + colorama.Style.RESET_ALL)
        except IOError:
            print("I/O error")

    def write_all_to_all_dist_matrix(self, measure: str, notes: str = ""):
        """ Writes 2d distance matrices (like jaccard) as csv

        :param measure: Name of the metric
        :param notes: add notes to file name (for example the alphabet size)
        :return: None
        """
        assert measure, "There has to be a measure declared"

        csv_columns = list(self.data_dict.keys())
        csv_columns.insert(0, "road")
        #print("csv_columns", csv_columns)

        first_road_dict = self.data_dict.get(csv_columns[1], None)
        assert first_road_dict is not None, "There has to be at least one road!"
        first_path = first_road_dict.get(utils.RoadDicConst.TEST_PATH.value, None)
        assert os.path.exists(first_path), "The path has not been set correctly!"

        # get the root of the dataset
        root_folder = utils.get_root_of_test_suite(test_path=first_path)

        def _get_dict_to_write(test_name: str, csv_columns: list) -> dict:
            road_dict = self.data_dict.get(test_name)
            dicc = {"road": test_name}
            values = road_dict.get(measure, None)
            assert values is not None, str(measure) + "has not been found for" + test_name
            dicc.update(values)
            return dicc

        csv_file = path.join(root_folder, measure + notes + '.csv')

        self.generic_all_tests_csv_writing(measure, csv_columns, _get_dict_to_write, csv_file)


    def write_all_tests_one_value(self, measure: str = utils.RoadDicConst.NUM_OBES.value):
        """ Writes a csv containing a single value (like num_obe) for each test

        :param measure: Name of the metric
        :return: None
        """
        root_folder = self.root_path
        csv_columns = ["test", measure]

        def _get_dict_to_write(test_name: str, csv_columns: list) -> dict:
            measure = csv_columns[1]
            test = self.data_dict.get(test_name, None)
            assert test is not None, "The road has not been found in the dict!"
            val = test.get(measure, None)
            assert val is not None, "The " + measure + "has not been found for " + test_name + "!"

            dicc = {"test": test_name, measure: val}
            return dicc

        csv_file = path.join(root_folder, "for_each_" + measure + '.csv')

        self.generic_all_tests_csv_writing(measure, csv_columns, _get_dict_to_write, csv_file)


    def write_whole_suite_1d_coverages(self, measure: str):
        """ For writing 1d counting bins (like steering_bins) for each test.

        :param measure: Name of the bins
        :return: None
        """
        root_folder = self.root_path

        first_test = self.data_dict.get(list(self.data_dict.keys())[0])
        first_bins = first_test.get(measure, None)
        assert first_bins is not None, measure + " has not been found in the dict!"
        num_bins = len(first_bins)
        csv_columns = [str(el) for el in range(0, num_bins)]
        csv_columns.insert(0, 'names')

        def _get_dict_to_write(test_name: str, csv_columns: list) -> dict:
            test = self.data_dict.get(test_name, None)
            assert test is not None, "The road has not been found in the dict!"
            val_1d = test.get(measure, None)
            assert val_1d is not None, "The " + measure + "has not been found for " + test_name + "!"

            dicc = dict((zip(csv_columns[1:], val_1d)))
            dicc[csv_columns[0]] = test_name
            return dicc

        csv_file = path.join(root_folder, measure + '.csv')

        self.generic_all_tests_csv_writing(measure, csv_columns, _get_dict_to_write, csv_file)

    def write_whole_suite_2d_coverages(self, measure: str):
        """ For writing 2d counting bins (like steering_speed_matrix) for each test.
        Flattens the 2d matrix.

        :param measure: Name of the bins
        :return: None
        """
        root_folder = self.root_path

        first_test = self.data_dict.get(list(self.data_dict.keys())[0])
        first_bins = first_test.get(measure, None)
        assert first_bins is not None, measure + " has not been found in the dict!"
        num_bins_outer = len(first_bins)
        num_bins_inner = len(first_bins[0])
        csv_columns = [str(el) for el in range(0, num_bins_outer * num_bins_inner)]
        csv_columns.insert(0, 'names')

        def _get_dict_to_write(test_name: str, csv_columns: list) -> dict:
            test = self.data_dict.get(test_name, None)
            assert test is not None, "The road has not been found in the dict!"
            val_2d = test.get(measure, None)
            assert val_2d is not None, "The " + measure + "has not been found for " + test_name + "!"

            flattened_2d = val_2d.flatten()
            dicc = dict((zip(csv_columns[1:], flattened_2d)))
            dicc[csv_columns[0]] = test_name
            return dicc

        csv_file = path.join(root_folder, measure + '.csv')

        self.generic_all_tests_csv_writing(measure, csv_columns, _get_dict_to_write, csv_file)

    def generic_all_tests_csv_writing(self, measure: str, csv_columns: list, _get_dict_to_write_func,
                                      csv_file: path):
        """ Generic all csv writing for arbitratry get_dict funtions

        :param measure: measure to select from the dict with all tests
        :param csv_columns: the names of the csv columns
        :param _get_dict_to_write_func: the function that computes the dict
        :param csv_file: the path to the file
        :return: None
        """
        # TODO refactor them all
        try:
            with open(csv_file, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns, extrasaction='ignore')
                writer.writeheader()
                for key in self.data_dict.keys():
                    dic = _get_dict_to_write_func(test_name=key, csv_columns=csv_columns)
                    writer.writerow(dic)
                print(colorama.Fore.GREEN + " wrote " + measure + " csv_file to", csv_file + colorama.Style.RESET_ALL)

        except IOError:
            print("I/O error")


    def write_whole_suite_multiple_values(self, file_name: str, name_value_tuple_list: List[tuple],
                                          first_row_name: str = 'coverage_measure'):
        """

        :param file_name: Name of the file in the root dict
        :param name_value_tuple_list: List of tuples with (name of measure, value of measure)
        :return: None
        """
        root_folder = self.root_path

        csv_columns = [first_row_name, 'value']

        csv_file = path.join(root_folder, file_name + '.csv')

        try:
            with open(csv_file, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns, extrasaction='ignore')
                writer.writeheader()
                for tup in name_value_tuple_list:
                    dic = {first_row_name: tup[0], 'value': tup[1]}
                    writer.writerow(dic)
                print(colorama.Fore.GREEN + " wrote " + file_name + " csv_file to ", csv_file + colorama.Style.RESET_ALL)
        except IOError:
            print("I/O error")


    def write_single_road_to_all_dists(self, road_name: str, measures: List[str]):
        assert measures, "There have to be measures declared"
        road_dict = self.data_dict.get(road_name, None)
        assert road_dict is not None, "The road " + road_name + " has not been found in the dict!"
        first_measure = road_dict.get(measures[0], None)
        assert first_measure is not None, "The measure " + measures[0] + " has not been found for the road " + road_name
        csv_columns = ['measure']
        csv_columns.extend(list(first_measure.keys()))

        test_path = road_dict.get('test_path', None)
        assert test_path is not None, "There is no path for road " + road_name

        def _get_dict_to_write(index: int):
            """ local function to get a dict with the names and values of a comparison
            there is a unnecessary declaration of a new array, the alternative would be to insert and remove an element

            :param index: index of the selected measure
            :return: dict with measure name and values
            """
            assert index < len(measures), "The index is too big"
            dicc = {'measure': measures[index]}
            extracted_dict = road_dict.get(measures[index], None)
            assert extracted_dict is not None, "The measure " + measures[index] + " has not been found for the road " \
                                               + road_name
            # TODO shape checking
            dicc.update(extracted_dict)
            return dicc

        # two splits to get to the \output\ directory
        # TODO maybe decide for a different location?
        test_dir_path = path.split(test_path)
        test_dir_path = path.split(test_dir_path[0])
        csv_file = path.join(test_dir_path[0], road_name + '.csv')
        # csv_file = "csv_test.csv"
        # TODO l.info("Generate CSV file %s", csv_file)
        try:
            with open(csv_file, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns, extrasaction='ignore')
                writer.writeheader()
                """dic = {'measure': measures[0]}
                dic.update(first_measure)
                print("{'measure': measures[0]}.update(first_measure)", dic)"""
                for i in range(0, len(measures)):
                    dic = _get_dict_to_write(i)
                    writer.writerow(dic)
            print(colorama.Fore.GREEN + "wrote csv_file to ", csv_file + colorama.Style.RESET_ALL)
        except IOError:
            print("I/O error")
