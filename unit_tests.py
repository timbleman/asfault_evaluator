import unittest
from unittest.mock import MagicMock

from asfault.network import NetworkNode, TYPE_R_TURN, TYPE_L_TURN, TYPE_STRAIGHT

import utils

import string_comparison


def mocked_compute_length(road_segment: NetworkNode):
    return road_segment.length

MAX_ANG = string_comparison.DEFAULT_PERCENTILE_VALUES_CUR[-1]


class TestCurveSDL(unittest.TestCase):

    def setUp(self) -> None:
        from asfault.network import NetworkNode, TYPE_R_TURN, TYPE_L_TURN, TYPE_STRAIGHT

        string_comparison.NUM_ALPHABET = 7
        string_comparison.DEFAULT_PERCENTILE_VALUES_CUR = [-120.0, -75.0, -30.0, -1.0, 1.0, 30.0, 75.0, 120.0]
        #string_comparison.DEFAULT_PERCENTILE_VALUES_LEN =

        node0 = NetworkNode(key=0, roadtype=None, seg_id=0)
        #@patch()

        node0 = MagicMock()
        node0.angle = 0
        node0.length = 10
        node0.y_off = 10
        node0.roadtype = TYPE_STRAIGHT

        node1 = MagicMock()
        node1.angle = 3
        node1.length = 15
        node1.roadtype = TYPE_R_TURN

        node2 = MagicMock()
        node2.angle = -120
        node2.length = 25
        node2.roadtype = TYPE_L_TURN

        node3 = MagicMock()
        node3.angle = -120
        node3.length = 15
        node3.roadtype = TYPE_L_TURN

        node4 = MagicMock()
        node4.angle = 0
        node4.length = 20
        node4.y_off = 10
        node4.roadtype = TYPE_STRAIGHT

        self.nodes0_list = [node0, node1, node2, node3, node4]
        road0_dict = {'nodes': self.nodes0_list}
        data_dict = {'0': road0_dict}

        self.str_comparer = string_comparison.StringComparer(data_dict=data_dict)

    def test_get_const_for_straight_angle(self):
        const = self.str_comparer.get_const_for_angle(0)
        #print("const", const)
        self.assertEqual(string_comparison.cur.STRAIGHT, const)

    def test_get_const_for_strong_right_angle(self):
        const = self.str_comparer.get_const_for_angle(MAX_ANG)
        max_val = (string_comparison.NUM_ALPHABET - 1) / 2
        # max_val for strong right, name may depend on size
        self.assertEqual(string_comparison.cur(max_val), const)

    def test_get_const_for_right_bounds(self):
        const = self.str_comparer.get_const_for_angle(MAX_ANG + 50)
        max_val = (string_comparison.NUM_ALPHABET - 1) / 2
        # max_val for strong right, name may depend on size
        self.assertEqual(string_comparison.cur(max_val), const)

    def test_get_const_for_strong_left_bounds(self):
        const = self.str_comparer.get_const_for_angle(-MAX_ANG - 50)
        max_val = (string_comparison.NUM_ALPHABET - 1) / 2
        # -max_val for strong left, name may depend on size
        self.assertEqual(string_comparison.cur(-max_val), const)

    def test_get_const_for_slight_right_angle(self):
        const = self.str_comparer.get_const_for_angle(2)
        max_val = (string_comparison.NUM_ALPHABET - 1) / 2
        # 1 for Slight right, name may depend on size
        self.assertEqual(string_comparison.cur(1), const)

    def test_get_const_for_strong_left_angle(self):
        const = self.str_comparer.get_const_for_angle(-MAX_ANG)
        max_val = (string_comparison.NUM_ALPHABET - 1) / 2
        # -max_val for strong left, name may depend on size
        self.assertEqual(string_comparison.cur(-max_val), const)

    def test_nodes_to_curvature_sdl(self):
        with unittest.mock.patch.object(self.str_comparer, "_compute_length", new=mocked_compute_length):
            #self.str_comparer.nodes_to_curvature_sdl(self.nodes0_list)
            curve_sdl = []
            for node in self.nodes0_list:
                curve_sdl.append(self.str_comparer.get_const_for_angle(node.angle))
            result = self.str_comparer.nodes_to_curvature_sdl(self.nodes0_list)
            self.assertEqual(curve_sdl, result, "The sdl compression is off")

    def test_get_const_for_length_0_negative(self):
        const_0 = self.str_comparer.get_const_for_length(0)
        const_neg = self.str_comparer.get_const_for_length(-1)
        self.assertEqual(const_0, string_comparison.len_en(0), "get_const_for_length does not handle 0")
        self.assertEqual(const_neg, string_comparison.len_en(0), "get_const_for_length does not handle negative values")

    def test_get_const_for_length_too_big(self):
        max_val_percentile = string_comparison.DEFAULT_PERCENTILE_VALUES_LEN[-1]
        const = self.str_comparer.get_const_for_length(max_val_percentile + 10)
        self.assertEqual(string_comparison.len_en(string_comparison.NUM_LEN_ALPHABET - 1), const)

    def test_get_const_for_length_border(self):
        max_val_percentile = string_comparison.DEFAULT_PERCENTILE_VALUES_LEN[-1]
        const = self.str_comparer.get_const_for_length(max_val_percentile)
        self.assertEqual(string_comparison.len_en(string_comparison.NUM_LEN_ALPHABET - 1), const)

if __name__ == '__main__':
    unittest.main()