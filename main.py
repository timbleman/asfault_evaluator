from asfault.tests import RoadTest, TestExecution
from os import listdir, path
from os.path import isfile, join
# replace os.path
from pathlib import Path

import coverage_evaluator

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

    # r"C:\Users\fraun\AppData\Local\Packages\CANONI~1.UBU\LOCALS~1\rootfs\home\TIMBLE~1\ASFAUL~1\ASFAUL~1\EXPERI~1\EXPERI~1\RANDOM~1\RANDOM~1"
    # r"C:\Users\fraun\AppData\Local\Packages\CANONI~1.UBU\LOCALS~1\rootfs\home\TIMBLE~1\ASFAUL~1\ASFAUL~1\EXPERI~1\EXPERI~1\RADF79~1\RANDOM~1"
    env_directory = Path(r"C:\Users\fraun\AppData\Local\Packages\CANONI~1.UBU\LOCALS~1\rootfs\home\TIMBLE~1\ASFAUL~1\ASFAUL~1\EXPERI~1\EXPERI~1\RANDOM~1\RANDOM~1")
    coverage_evaluator.cov_evaluate_set(env_directory)
    print("Start evaluation of OBEs from %s", env_directory)


if __name__ == "__main__":
    main()
