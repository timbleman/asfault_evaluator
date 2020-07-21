from asfault.tests import RoadTest, TestExecution
from os import listdir, path
from os.path import isfile, join
from typing import List
# replace os.path
from pathlib import Path

import coverage_evaluator
from string_comparison import StringComparer
from suite_behaviour_computer import SuiteBehaviourComputer
from csv_creator import CSVCreator

# ssh -i /root/.ssh/id_rsa.pub ubuntu@160.85.252.213  # bill test suite pc

def _configure_asfault() -> None:
    from asfault.config import init_configuration, load_configuration
    from tempfile import TemporaryDirectory
    temp_dir = TemporaryDirectory(prefix="testGenerator")
    init_configuration(temp_dir.name)
    load_configuration(temp_dir.name)

def get_all_paths(parent_dir: Path) -> List[Path]:
    """ Return all paths of test executions inside a folder using Bill Bosshards notation
    TODO filtering empty folders

    :param parent_dir: Path to the parent folder (e.g. experiments-driver-ai or experiments-beamng-ai)
    :return: List of all sub-sub-paths
    """

    # TODO find an appropriate length to match
    length_to_match = 10

    all_paths = []
    for child in parent_dir.iterdir():
        # windows Path management fucks this up
        name_of_test_folder = str(child.absolute().parts[-1])
        start_of_test_folder = "." + name_of_test_folder[0:length_to_match-1]
        for grand_child in child.iterdir():
            name_of_cfg_output_folder = str(grand_child.absolute().parts[-1])
            #print(start_of_test_folder, "vs", name_of_cfg_output_folder)
            if start_of_test_folder == name_of_cfg_output_folder[0:length_to_match]:
                #print("", child, "and", grand_child, "matched!")
                all_paths.append(grand_child)
    print("All paths to run: ", all_paths)
    return all_paths


def main():
    # Local import to main
    import os
    import csv
    # ce = CoverageEvaluator()
    parent_dir = Path(r"C:\Users\fraun\experiments-driver-ai")
    all_paths = get_all_paths(parent_dir)

    # FIXME the folder structure seems broken sometimes
    # FIXME there is an defective road in C:\Users\fraun\experiments-driver-ai\one-plus-one--lanedist--driver-ai--small--no-repair--with-restart--2\.one-plus-one-EA--lanedist--ext--small--no-repair--with-restart--env
    all_paths.pop(2)


    # # merging two dicts
    # global_dict.update(local_dict)

    # r"C:\Users\fraun\AppData\Local\Packages\CANONI~1.UBU\LOCALS~1\rootfs\home\TIMBLE~1\ASFAUL~1\ASFAUL~1\EXPERI~1\EXPERI~1\RANDOM~1\RANDOM~1"
    # r"C:\Users\fraun\AppData\Local\Packages\CANONI~1.UBU\LOCALS~1\rootfs\home\TIMBLE~1\ASFAUL~1\ASFAUL~1\EXPERI~1\EXPERI~1\RADF79~1\RANDOM~1"
    #env_directory = Path(r"C:\Users\fraun\AppData\Local\Packages\CANONI~1.UBU\LOCALS~1\rootfs\home\TIMBLE~1\ASFAUL~1\ASFAUL~1\EXPERI~1\EXPERI~1\RANDOM~1\RANDOM~1")
    #env_directory = Path(r"C:\Users\fraun\experiments-driver-ai\one-plus-one--lanedist--driver-ai--small--no-repair--with-restart--6\.one-plus-one-EA--lanedist--ext--small--no-repair--with-restart--env")

    env_directory = Path(r"C:\Users\fraun\experiments-driver-ai\random--lanedist--driver-ai--small--no-repair--with-restart--2\.random--lanedist--ext--small--no-repair--with-restart--env")
    data_bins_dict = coverage_evaluator.cov_evaluate_set(env_directory)

    """
    # commented for testing purposes
    data_bins_dict = {}
    for env_directory in all_paths:
        print("Start evaluation of OBEs from %s", env_directory)
        # TODO check wheter identifier already exists in dict
        data_bins_dict.update(coverage_evaluator.cov_evaluate_set(env_directory))
    #print("partial_bins ", partial_bins)
    """

    #sbh = SuiteBehaviourComputer(partial_bins)
    #print("speed coverage", sbh.calculate_suite_coverage_1d('speed_bins'))
    #print("obe coverage", sbh.calculate_suite_2d_coverage('obe_2d'))
    #print("road compare 1d", sbh.behavior_compare_1d("1-2", 'steering_bins'))
    #print("road compare 2d", sbh.behavior_compare_2d("1-2", 'speed_steering_2d'))

    # unnecessary, pass by reference
    #partial_bins = sbh.get_test_dict()

    print("all roads ", data_bins_dict.keys())

    str_comparer = StringComparer(data_dict=data_bins_dict)

    csv_creator = CSVCreator(data_dict=data_bins_dict)
    #csv_creator.write_single_road_dists(road_name="1-2", measures=['curve_sdl_dist', '1-2_binary_steering_bins'])


if __name__ == "__main__":
    main()
