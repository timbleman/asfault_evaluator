import csv
import os
from os import path
from typing import List

import colorama


class CSVCreator:
    def __init__(self, data_dict):
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
            # print(colorama.Fore.GREEN + "wrote csv_file to", csv_file + colorama.Style.RESET_ALL)
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
            print(colorama.Fore.GREEN + "wrote csv_file to", csv_file + colorama.Style.RESET_ALL)
        except IOError:
            print("I/O error")


"""
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
"""
