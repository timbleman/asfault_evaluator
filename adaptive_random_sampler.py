from typing import List
import evaluator_config as econf
import random
import utils
import colorama

from utils import BehaviorDicConst

class AdaptiveRandSampler:
    def __init__(self, data_dict: dict):
        self.data_dict = data_dict
        # List of strings containing names of the roads
        self.population = []

    # TODO sample of n with threshold
    # TODO select one start road
    # TODO high and low diversity function
    def sample_of_n(self, measure: str, n: int, func, first_test: str = None, size_candidate_list: int = 10):
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
        if first_test is not None and first_test in all_keys:
            first_one = first_test
            all_keys.remove(first_one)
        else:
            first_one = all_keys.pop(random.randrange(0, len(all_keys)))
        print(colorama.Fore.BLUE + "Picked " + first_one + " to be first in the population!" + colorama.Style.RESET_ALL)
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
            #print("candidates: ", candidates)

            best_candidate = func(candidates, measure)

            all_keys.pop(all_keys.index(best_candidate))
            self.population.append(best_candidate)

            test_index += 1
            self.data_dict[best_candidate][BehaviorDicConst.SAMPLING_INDEX.value] = test_index

        self.add_value_for_undefineds(measure=BehaviorDicConst.SAMPLING_INDEX.value, default_value=-1)


    def add_value_for_undefineds(self, measure: str, default_value=-1):
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
