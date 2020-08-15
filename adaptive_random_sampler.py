from typing import List
import evaluator_config as econf
import random
import utils


class AdaptiveRandSampler:
    def __init__(self, data_dict: dict):
        self.data_dict = data_dict
        # List of strings containing names of the roads
        self.population = []

    def sample_of_n(self, measure: str, n: int, func, size_candidate_list: int = 10):
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
        first_one = all_keys.pop(random.randrange(0, len(all_keys)))
        self.population = [first_one]

        assert n < len(all_keys) - size_candidate_list

        for m in range(0, n-1):
            candidates = []
            # get a sample of indices not containing any duplicates for selecting candidates
            indices = random.sample(range(0, len(all_keys)), 10)
            for i in indices:
                candidates.append(all_keys[i])
            print("candidates: ", candidates)

            best_candidate = func(candidates, measure)

            all_keys.pop(all_keys.index(best_candidate))
            self.population.append(best_candidate)

        print(self.population)

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
