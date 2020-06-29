from asfault.tests import RoadTest, TestExecution
from os import listdir, path
from os.path import isfile, join


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

    env_directory = path("")
    print("Start evaluation of OBEs from %s", env_directory)