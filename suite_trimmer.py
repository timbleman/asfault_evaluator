import operator

class SuiteTrimmer:
    def __init__(self, data_dict: dict, base_path):
        self.data_dict = data_dict
        self.base_path = base_path

    def get_unworthy_paths(self, feature: str, op: operator, threshold: float) -> list:
        unworthy_paths = []
        for test in self.data_dict.values():
            val = test.get(feature, None)
            assert val is not None, "The feature has not been added or spelling is incorrect"
            if op(val, threshold):
                unworthy_paths.append(test['test_path'])

        return unworthy_paths
