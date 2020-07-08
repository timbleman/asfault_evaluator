from asfault.tests import RoadTest, TestExecution
from os import listdir, path
from os.path import isfile, join
# replace os.path
from pathlib import Path

import coverage_evaluator
from string_comparison import StringComparer
from suite_behaviour_computer import SuiteBehaviourComputer

# ssh -i /root/.ssh/id_rsa.pub ubuntu@160.85.252.213  # bill test suite pc

def _configure_asfault() -> None:
    from asfault.config import init_configuration, load_configuration
    from tempfile import TemporaryDirectory
    temp_dir = TemporaryDirectory(prefix="testGenerator")
    init_configuration(temp_dir.name)
    load_configuration(temp_dir.name)


def main():
    # Local import to main
    import os
    import csv
    # ce = CoverageEvaluator()
    #parent_dir = Path(r"C:\Users\fraun\AppData\Local\Packages\CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc\LocalState\rootfs\home\timbleman\asfault_dataset\asfault-experiments\experiments-driver-ai\experiments-driver-ai")
    #for child in parent_dir.iterdir():
    #    # windows Path management fucks this up
    #    print(child.absolute())
    # # merging two dicts
    # global_dict.update(local_dict)

    # todo getting road coords: self.state_before_obe.test.get_path_polyline().coords

    # r"C:\Users\fraun\AppData\Local\Packages\CANONI~1.UBU\LOCALS~1\rootfs\home\TIMBLE~1\ASFAUL~1\ASFAUL~1\EXPERI~1\EXPERI~1\RANDOM~1\RANDOM~1"
    # r"C:\Users\fraun\AppData\Local\Packages\CANONI~1.UBU\LOCALS~1\rootfs\home\TIMBLE~1\ASFAUL~1\ASFAUL~1\EXPERI~1\EXPERI~1\RADF79~1\RANDOM~1"
    env_directory = Path(r"C:\Users\fraun\AppData\Local\Packages\CANONI~1.UBU\LOCALS~1\rootfs\home\TIMBLE~1\ASFAUL~1\ASFAUL~1\EXPERI~1\EXPERI~1\RANDOM~1\RANDOM~1")
    partial_bins = coverage_evaluator.cov_evaluate_set(env_directory)
    print("Start evaluation of OBEs from %s", env_directory)
    #print("partial_bins ", partial_bins)

    sbh = SuiteBehaviourComputer(partial_bins)
    #print("speed coverage", sbh.calculate_suite_coverage_1d('speed_bins'))
    #print("obe coverage", sbh.calculate_suite_2d_coverage('obe_2d'))
    print("road compare 1d", sbh.road_compare_1d("1-2", 'steering_bins'))
    print("road compare 2d", sbh.road_compare_2d("1-2", 'speed_steering_2d'))

    # unnecessary, pass by reference
    partial_bins = sbh.get_test_dict()

    #print("partial_bins ", partial_bins)

    str_comparer = StringComparer(data_dict=partial_bins)


if __name__ == "__main__":
    main()
