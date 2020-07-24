from asfault.tests import RoadTest, TestExecution
from os import listdir, path
from os.path import isfile, join
from typing import List
# replace os.path
from pathlib import Path

import coverage_evaluator
import utils
from string_comparison import StringComparer
from suite_behaviour_computer import SuiteBehaviourComputer
from csv_creator import CSVCreator
from suite_trimmer import SuiteTrimmer

import colorama

# ssh -i /root/.ssh/id_rsa.pub ubuntu@160.85.252.213  # bill test suite pc

# TODO do this everywhere this is more pythonic
# main_bin = self.test_dict.get(road_to_compare, None).get(measure, None)

# this path is broken
# "C:\Users\fraun\experiments-driver-ai\one-plus-one--lanedist--driver-ai--small--no-repair--with-restart--2\.one-plus-one-EA--lanedist--ext--small--no-repair--with-restart--env\output\execs\test_0028.json"

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
        # skip all files in the parent folder
        if path.isfile(child):
            print("Lol", child, "is a file")
            continue
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
    #parent_dir = Path(r"C:\Users\fraun\experiments-driver-ai")
    parent_dir = Path(r"C:\Users\fraun\experiments-driver-ai-long-execution")
    all_paths = get_all_paths(parent_dir)

    # FIXME the folder structure seems broken sometimes
    # FIXME there is an defective road in C:\Users\fraun\experiments-driver-ai\one-plus-one--lanedist--driver-ai--small--no-repair--with-restart--2\.one-plus-one-EA--lanedist--ext--small--no-repair--with-restart--env
    print(all_paths.pop(2))


    #env_directory = Path(r"C:\Users\fraun\experiments-driver-ai-trimmed\random--lanedist--driver-ai--small--no-repair--with-restart--2\.random--lanedist--ext--small--no-repair--with-restart--env")
    #data_bins_dict = coverage_evaluator.cov_evaluate_set(env_directory)


    # commented for testing purposes
    data_bins_dict = {}
    for env_directory in all_paths:
        print("Start evaluation of OBEs from %s", env_directory)
        # TODO check whether identifier already exists in dict
        data_bins_dict.update(coverage_evaluator.cov_evaluate_set(env_directory))
    #print("partial_bins ", partial_bins)


    sbh = SuiteBehaviourComputer(data_bins_dict)
    print("speed coverage", sbh.calculate_suite_coverage_1d('speed_bins'))
    print("obe coverage", sbh.calculate_suite_2d_coverage('obe_2d'))
    print("road compare 1d", sbh.behavior_compare_1d("random--la22", 'steering_bins'))
    #print("road compare 2d", sbh.behavior_compare_2d("random--la22", 'speed_steering_2d'))

    # unnecessary, pass by reference
    #partial_bins = sbh.get_test_dict()

    str_comparer = StringComparer(data_dict=data_bins_dict)

    csv_creator = CSVCreator(data_dict=data_bins_dict)
    csv_creator.write_two_roads_dists(road_1_name="random--la22", road_2_name="random--la23", measures=['curve_sdl_dist', 'random--la22_binary_steering_bins'])
    csv_creator.write_all_two_roads_dists(road_1_name="random--la22", measures=['curve_sdl_dist', 'random--la22_binary_steering_bins'])
    #csv_creator.write_single_road_dists(road_name="1-2", measures=['curve_sdl_dist', '1-2_binary_steering_bins'])

    utils.whole_suite_statistics(dataset_dict=data_bins_dict, feature="num_states", plot=True)

    print(colorama.Fore.GREEN + "Collected a total of", len(data_bins_dict), "roads!" + colorama.Style.RESET_ALL)
    names_of_all = list(data_bins_dict.keys())
    print("all roads ", names_of_all)
    print(colorama.Fore.GREEN + "Computed following measures for each road", data_bins_dict[names_of_all[0]].keys(), "" + colorama.Style.RESET_ALL)
    #print("all roads ", data_bins_dict)

    suite_trimmer = SuiteTrimmer(data_dict=data_bins_dict, base_path=parent_dir)
    import operator
    #print("unworthy paths:", suite_trimmer.get_unworthy_paths(feature="num_states", op=operator.le, threshold=300))
    #suite_trimmer.trim_dataset(feature="num_states", op=operator.le, threshold=300)
    suite_trimmer.trim_dataset_percentile(feature="num_states", op=operator.le, threshold_percentile=60)


if __name__ == "__main__":
    main()
