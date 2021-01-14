from typing import List
import evaluator_config as econf
import random
import utils
import colorama
from pathlib import Path


from utils import BehaviorDicConst

class AdaptiveRandSampler:
    def __init__(self, data_dict: dict):
        self.data_dict = data_dict
        # List of strings containing names of the roads
        self.population = []

    # TODO sample of n with threshold
    def sample_of_n(self, measure: str, n: int, func, first_test: str = None, force_first: bool = True,
                    size_candidate_list: int = 10):
        """ Simulates adaptive random sampling like in https://doi.org/10.1007/978-3-540-30502-6_23.
        Draws a number of candidates and adds the best one based on a function func to the population.

        :param measure: Name of the distance measure for each road
        :param n: The size of the final population
        :param func: Function to select the best candidate
        :param size_candidate_list: The size of each candidates list
        :return: None
        """
        all_keys = list(self.data_dict.keys())
        assert all_keys is not None, "The data dict is empty!"
        # create an initial population with one random road
        # set seed or None
        random.seed = econf.SEED_ADAPTIVE_RANDOM
        if first_test is not None:
            if force_first:
                assert first_test in all_keys, \
                    colorama.Fore.RED + "The selected startpoint is not included in the set!" + colorama.Style.RESET_ALL
            first_one = first_test
            all_keys.remove(first_one)
        else:
            first_one = all_keys.pop(random.randrange(0, len(all_keys)))
        print(colorama.Fore.CYAN + "Picked " + first_one + " to be first in the population!" + colorama.Style.RESET_ALL)
        self.population = [first_one]
        # index instead of m in order to be easier to adapt in the future
        test_index = 0
        self.data_dict[first_one][BehaviorDicConst.SAMPLING_INDEX.value] = test_index

        assert n < len(all_keys) - size_candidate_list

        for m in range(0, n-1):
            candidates = []
            # get a sample of indices not containing any duplicates for selecting candidates
            indices = random.sample(range(0, len(all_keys)), 10)
            for i in indices:
                candidates.append(all_keys[i])

            best_candidate = func(candidates, measure)

            all_keys.pop(all_keys.index(best_candidate))
            self.population.append(best_candidate)

            test_index += 1
            self.data_dict[best_candidate][BehaviorDicConst.SAMPLING_INDEX.value] = test_index

        self.add_value_for_undefineds(measure=BehaviorDicConst.SAMPLING_INDEX.value, default_value=-1)


    def add_value_for_undefineds(self, measure: str, default_value=-1):
        """ Sets undefined attribute of dict entries to a default. Used for the adaptive random sampling order.

        :param measure: Name for the attribute
        :param default_value: Value to set
        :return: None
        """
        for key, test in self.data_dict.items():
            val = test.get(measure, None)
            if val is None:
                self.data_dict[key][measure] = default_value

    def get_unworthy_paths(self) -> List:
        """ Returns os.paths for each road that is not in the population and shall be removed.

        :return: List of os.path
        """
        assert self.population, "There is no population!"
        unworthy_paths = []
        for key, test in self.data_dict.items():
            if key not in self.population:
                test_path = test.get(utils.RoadDicConst.TEST_PATH.value)
                assert test_path is not None, "Path not found for " + key + "!"
                unworthy_paths.append(test_path)

        return unworthy_paths

    def pick_smallest_max_similarity(self, candidate_list: List[str], measure: str) -> str:
        """ Heavily inspired by https://doi.org/10.1007/978-3-540-30502-6_23.
        Picks the candidate with the lowest maximum distance and returns its key.

        :param candidate_list: List of candidate keys to select from
        :param measure: Name of the distance measure for each road
        :return: Key of the best candidate
        """
        lowest_similarity = float('inf')
        best_candidate = None
        for candidate in candidate_list:
            candidate_dic = self.data_dict.get(candidate, None)
            assert candidate_dic is not None, candidate + " has not been found!"
            candidate_dists = candidate_dic.get(measure, None)
            assert candidate_dists is not None, measure + " has not been added to " + candidate + "!"

            # min value
            max_candidate_similarity = -1.0
            # find the maximum similarity from the candidate to one item from the population
            for specimen in self.population:
                distance_to_candidate = candidate_dists.get(specimen, None)
                assert distance_to_candidate is not None, "No distance between " + candidate + " and " + specimen + " found"
                max_candidate_similarity = max(max_candidate_similarity, distance_to_candidate)

            if max_candidate_similarity < lowest_similarity:
                lowest_similarity = max_candidate_similarity
                best_candidate = candidate

        assert best_candidate is not None
        return best_candidate

    def pick_highest_min_similarity(self, candidate_list: List[str], measure: str) -> str:
        """ Heavily inspired by https://doi.org/10.1007/978-3-540-30502-6_23.
        Picks the candidate with the highest minimum distance and returns its key.

        :param candidate_list: List of candidate keys to select from
        :param measure: Name of the distance measure for each road
        :return: Key of the best candidate
        """
        highest_similarity = -1
        best_candidate = None
        for candidate in candidate_list:
            candidate_dic = self.data_dict.get(candidate, None)
            assert candidate_dic is not None, candidate + " has not been found!"
            candidate_dists = candidate_dic.get(measure, None)
            assert candidate_dists is not None, measure + " has not been added to " + candidate + "!"

            # max value
            min_candidate_similarity = float('inf')
            # find the maximum similarity from the candidate to one item from the population
            for specimen in self.population:
                distance_to_candidate = candidate_dists.get(specimen, None)
                assert distance_to_candidate is not None, "No distance between " + candidate + " and " + specimen + " found"
                min_candidate_similarity = min(min_candidate_similarity, distance_to_candidate)

            if min_candidate_similarity > highest_similarity:
                highest_similarity = min_candidate_similarity
                best_candidate = candidate

        assert best_candidate is not None
        return best_candidate

    def pick_lowest_sum_similarity(self, candidate_list: List[str], measure: str) -> str:
        """ Inspired by https://doi.org/10.1007/978-3-540-30502-6_23.
        Picks the candidate with the lowest sum of distances to all others in the population and returns its key.

        :param candidate_list: List of candidate keys to select from
        :param measure: Name of the distance measure for each road
        :return: Key of the best candidate
        """
        lowest_similarity = float('inf')
        best_candidate = None
        for candidate in candidate_list:
            candidate_dic = self.data_dict.get(candidate, None)
            assert candidate_dic is not None, candidate + " has not been found!"
            candidate_dists = candidate_dic.get(measure, None)
            assert candidate_dists is not None, measure + " has not been added to " + candidate + "!"

            # min value
            sum_similarity = -1.0
            # find the maximum similarity from the candidate to one item from the population
            for specimen in self.population:
                distance_to_candidate = candidate_dists.get(specimen, None)
                assert distance_to_candidate is not None, "No distance between " + candidate + " and " + specimen + " found"
                sum_similarity += distance_to_candidate

            if sum_similarity < lowest_similarity:
                lowest_similarity = sum_similarity
                best_candidate = candidate

        assert best_candidate is not None
        return best_candidate


def prepare_folders_for_sampling(parent_path: Path, configs: list, destination_path: Path):
    """ Creates subfolders in destination path that combine the name of the parent_path and a config.

    :param parent_path: Path to the suite including all tests
    :param configs: List of strings that describe the configurations
    :param destination_path: Path to to store all the subsets
    :return: List of folder names
    """
    assert parent_path.exists(), "The parent suite to copy the files from does not exist!"
    import shutil

    print(colorama.Fore.CYAN, "Creating multiple folders for smaller subsets", colorama.Style.RESET_ALL)

    if not destination_path.exists():
        destination_path.mkdir()

    parent_name = parent_path.name # maybe .parts instead of .name
    folder_names = []

    for cfg in configs:
        folder_name = str(parent_name) + "-" + cfg
        folder_path = destination_path.joinpath(folder_name)
        if folder_path.exists():
            print("Removing old folders of the subset")
            shutil.rmtree(folder_path)
        folder_path.mkdir()

        for fiofo in parent_path.iterdir():
            if fiofo.is_dir():
                fo_name = fiofo.name
                dst_folder = folder_path.joinpath(fo_name)
                shutil.copytree(src=fiofo, dst=dst_folder)
        folder_names.append(folder_name)

    return folder_names


def apdaptive_rand_sample_multiple_subsets(start_points: list, parent_path: Path, destination_path: Path):
    """ Creates multiple subfolders in destination path that have logical names and include the full test suite
    as in parent_path.

    :param start_points: start points for adaptive random sampling, needed for names
    :param parent_path: Path to the suite including all tests
    :param destination_path: Path to to store all the subsets
    :return: [{'folder':, 'diversity':, 'start_point':}], used for adaptive random sampling
    """
    print(colorama.Fore.GREEN, "Creating folder names for new test suites", colorama.Style.RESET_ALL)
    # TODO assert that the start points are part of the parent suite
    # create appropriate descriptors to folder names
    configs = []
    for stp in start_points:
        configs.append("highdiv_" + stp)
        configs.append("lowdiv_" + stp)

    # create the folders with appropriate names
    folder_names = prepare_folders_for_sampling(parent_path, configs, destination_path)
    print(colorama.Fore.GREEN, "Creating subfolders", folder_names, "in", destination_path, colorama.Style.RESET_ALL)
    print(colorama.Fore.GREEN, "Copying contents of", parent_path, colorama.Style.RESET_ALL)
    # create list of [{'folder':, 'diversity':, 'start_point':}]
    runner_list = []
    for fldr in folder_names:
        start_point = fldr.split('_')[-1]
        if 'highdiv' in fldr:
            hiorlo = 'high'
        elif 'lowdiv' in fldr:
            hiorlo = 'low'
        else:
            raise ValueError("highdiv or lowdiv not included in the folder name!")
        dicc = {'folder': fldr,
                'diversity': hiorlo,
                'startpoint': start_point}
        runner_list.append(dicc)
    return runner_list


def mirror_subsets_only_results(subsets_destination_path: Path):
    """ Copies all the files (in this case .csvs of experiments) from a set of subsets.
    Automatically creates a folder that shares the name with the actual one but adds "_only_results".

    :param subsets_destination_path:
    :return:
    """
    assert subsets_destination_path.exists(), "The path for the subsets has to be created"
    import shutil

    print(colorama.Fore.CYAN + "Trying to move files for each subset in " + str(subsets_destination_path)
            + colorama.Style.RESET_ALL)

    # create and if necessary delete folder for only the results
    last_part = subsets_destination_path.parts[-1] + "_only_results"
    only_res_path = subsets_destination_path.parent.joinpath(last_part)
    if only_res_path.is_dir():
        shutil.rmtree(only_res_path)
        print("Removed old only results folder.")
    only_res_path.mkdir()

    # iterate through all subsets
    for subset in subsets_destination_path.iterdir():
        if subset.is_dir():
            # iterate through all folders and replicate them
            folder_repl = list(subset.parts)
            folder_repl[-2] = last_part
            outer = Path(*folder_repl)
            outer.mkdir()
            for el in subset.iterdir():
                if el.is_file():
                    dest_list = list(el.parts)
                    dest_list[-3] = last_part
                    dest = Path(*dest_list)
                    shutil.copy(src=el, dst=dest)
