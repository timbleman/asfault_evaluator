import csv
from os import path
from typing import List


class CSVCreator:
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.write_single_road_dists(road_name="1-2", measures=['curve_sdl_dist', '1-2_binary_steering_bins'])

    def write_single_road_dists(self, road_name: str, measures: List[str]):
        print("data_dict list", list(self.data_dict.keys()))
        assert measures, "There have to be measures declared"
        road_dict = self.data_dict.get(road_name, None)
        assert road_dict is not None, "The road " + road_name + " has not been found in the dict!"
        first_measure = road_dict.get(measures[0], None)
        assert first_measure is not None, "The measure " + measures[0] + " has not been found for the road " + road_name
        csv_columns = ['measure']
        csv_columns.extend(list(first_measure.keys()))
        print("csv_columns", csv_columns)

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
            print("wrote csv_file to", csv_file)
        except IOError:
            print("I/O error")
        # TODO do this everywhere this is more pythonic
        # main_bin = self.test_dict.get(road_to_compare, None).get(measure, None)


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
