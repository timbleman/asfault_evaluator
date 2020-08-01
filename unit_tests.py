import unittest
from unittest.mock import MagicMock

from asfault.network import NetworkNode, TYPE_R_TURN, TYPE_L_TURN, TYPE_STRAIGHT

import utils
import coverage_evaluator

import string_comparison


def mocked_compute_length(road_segment: NetworkNode):
    return road_segment.length


MAX_ANG = string_comparison.DEFAULT_PERCENTILE_VALUES_CUR[-1]


class TestUtils(unittest.TestCase):

    def setUp(self) -> None:
        self.zero_bins = [0] * 16
        self.first_bins = [0] * 16
        self.first_bins[0] = 10
        self.first_bins[1] = 10
        self.first_bins[2] = 10

        self.first_bins_halved = [0] * 16
        self.first_bins_halved[0] = 5
        self.first_bins_halved[1] = 5
        self.first_bins_halved[2] = 5

        self.second_bins = [0] * 16
        self.second_bins[2] = 10
        self.second_bins[3] = 10
        self.second_bins[4] = 10

    def test_list_difference_1d_binary_not_normalized(self):
        diff = utils.list_difference_1d(a=self.first_bins, b=self.second_bins, function=utils.DiffFuncConst.BINARY.value,
                                 normalized=False)
        self.assertEqual(4, diff)

    def test_list_difference_1d_binary_normalized(self):
        diff = utils.list_difference_1d(a=self.first_bins, b=self.second_bins, function=utils.DiffFuncConst.BINARY.value,
                                 normalized=True)

        expected = (4/coverage_evaluator.NUM_BINS)
        self.assertEqual(expected, diff)

    def test_list_difference_1d_single_not_normalized(self):
        diff = utils.list_difference_1d(a=self.first_bins, b=self.second_bins, function=utils.DiffFuncConst.SINGLE.value,
                                 normalized=False)
        # 10 (val of list entry) * 4 (number of different bins)
        expected = 10 * 4
        self.assertEqual(expected, diff)

    def test_list_difference_1d_single_normalized(self):
        diff = utils.list_difference_1d(a=self.first_bins, b=self.second_bins, function=utils.DiffFuncConst.SINGLE.value,
                                 normalized=True)
        # (10 (val of list entry) * 4 (number of different bins)) / 60 (number of values in total)
        expected = (10 * 4) / 60
        self.assertEqual(expected, diff)

    def test_list_difference_1d_single_not_normalized_unequal(self):
        diff = utils.list_difference_1d(a=self.first_bins_halved, b=self.second_bins,
                                        function=utils.DiffFuncConst.SINGLE.value, normalized=False)
        # 5 (val of list entry) * 2 (number of different bins) + 10 * 2 + 5
        expected = 5 * 2 + 10 * 2 + 5
        self.assertEqual(expected, diff, "Unequal element count causes problems")

    def test_list_difference_1d_single_normalized_unequal(self):
        diff = utils.list_difference_1d(a=self.first_bins_halved, b=self.second_bins,
                                        function=utils.DiffFuncConst.SINGLE.value, normalized=True)
        # 5 (val of list entry) * 2 (number of different bins) + 10 * 2 + 5
        expected = 2/3
        self.assertEqual(expected, diff, "Unequal element count causes problems")

    def test_k_lcstr_not_normalized(self):
        # should detect "fuel"
        str1 = "qwerfueltzuio"
        str2 = "yxfxelnnmkl"
        result = utils.k_lcstr(str1, str2, normalized=False)
        self.assertEqual(4, result)

    def test_k_lcstr_normalized(self):
        # should detect "fuel"
        str1 = "qwerfueltzuio"
        str2 = "yxfxelnnmkl"
        result = utils.k_lcstr(str1, str2, normalized=True)
        expected = 4 / len(str2)
        self.assertEqual(expected, result)


class TestCurveSDL(unittest.TestCase):

    def setUp(self) -> None:
        from asfault.network import NetworkNode, TYPE_R_TURN, TYPE_L_TURN, TYPE_STRAIGHT

        string_comparison.NUM_ALPHABET = 7
        string_comparison.DEFAULT_PERCENTILE_VALUES_CUR = [-120.0, -75.0, -30.0, -1.0, 1.0, 30.0, 75.0, 120.0]
        # string_comparison.DEFAULT_PERCENTILE_VALUES_LEN =

        node0 = NetworkNode(key=0, roadtype=None, seg_id=0)
        # @patch()

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

        node35 = MagicMock()
        node35.angle = -3
        node35.length = 15
        node35.roadtype = TYPE_L_TURN

        node4 = MagicMock()
        node4.angle = 0
        node4.length = 20
        node4.y_off = 10
        node4.roadtype = TYPE_STRAIGHT

        # do not change these, this arrangement is needed for some tests
        self.nodes0_list = [node0, node1, node2, node3, node4]
        self.nodes1_list = [node0, node1, node2, node35, node4]
        road0_dict = {utils.DicConst.NODES.value: self.nodes0_list}
        road1_dict = {utils.DicConst.NODES.value: self.nodes1_list}
        data_dict = {'0': road0_dict, "1": road1_dict}

        self.str_comparer = string_comparison.StringComparer(data_dict=data_dict)

    def test_get_const_for_straight_angle(self):
        const = self.str_comparer.get_const_for_angle(0)
        # print("const", const)
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
            # self.str_comparer.nodes_to_curvature_sdl(self.nodes0_list)
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

    def test_cur_sdl_one_to_one(self):
        # TODO
        sdl_road_1 = self.str_comparer.nodes_to_curvature_sdl(self.nodes0_list)
        sdl_road_2 = self.str_comparer.nodes_to_curvature_sdl(self.nodes1_list)
        result = self.str_comparer.cur_sdl_one_to_one(sdl_road_1, sdl_road_2, normalized=True, invert=False)
        print(result)

    def test_cur_sdl_one_to_one_max_difference(self):
        max_cur = string_comparison.DEFAULT_PERCENTILE_VALUES_CUR[-1]
        node_SL = MagicMock()
        node_SL.angle = -max_cur
        node_SL.length = 15
        node_SL.roadtype = TYPE_L_TURN

        node_SR = MagicMock()
        node_SR.angle = max_cur
        node_SR.length = 75
        node_SR.roadtype = TYPE_R_TURN

        node_alt_list_0 = [node_SL, node_SR, node_SL, node_SR]
        node_alt_list_1 = [node_SR, node_SL, node_SR, node_SL]

        sdl_road_1 = self.str_comparer.nodes_to_curvature_sdl(node_alt_list_0)
        sdl_road_2 = self.str_comparer.nodes_to_curvature_sdl(node_alt_list_1)
        result = self.str_comparer.cur_sdl_one_to_one(sdl_road_1, sdl_road_2, normalized=True, invert=False)
        self.assertAlmostEqual(0, result, msg="Translation should not affect the result!")

    def test__cur_sdl_error_at_startpoint(self):
        max_cur = string_comparison.DEFAULT_PERCENTILE_VALUES_CUR[-1]
        node_SL = MagicMock()
        node_SL.angle = -max_cur
        node_SL.length = 15
        node_SL.roadtype = TYPE_L_TURN

        node_SR = MagicMock()
        node_SR.angle = max_cur
        node_SR.length = 75
        node_SR.roadtype = TYPE_R_TURN

        node_alt_list_0 = [node_SL, node_SR, node_SL, node_SR]
        node_alt_list_1 = [node_SR, node_SL, node_SR, node_SL]

        sdl_road_0 = self.str_comparer.nodes_to_curvature_sdl(node_alt_list_0)
        sdl_road_1 = self.str_comparer.nodes_to_curvature_sdl(node_alt_list_1)
        result = self.str_comparer._cur_sdl_error_at_startpoint(start_point=0, longer_road_sdl=sdl_road_0,
                                                                shorter_road_sdl=sdl_road_1)
        #
        expected_error = (string_comparison.NUM_ALPHABET - 1) * len(sdl_road_1)
        self.assertAlmostEqual(expected_error, result, msg="The error should be maxed!")

    def test__sdl_2d_error_at_startpoint(self):
        max_cur = string_comparison.DEFAULT_PERCENTILE_VALUES_CUR[-1]
        node_SL = MagicMock()
        node_SL.angle = -max_cur
        node_SL.length = string_comparison.DEFAULT_PERCENTILE_VALUES_LEN[0]
        node_SL.roadtype = TYPE_L_TURN

        node_SR = MagicMock()
        node_SR.angle = max_cur
        node_SR.length = string_comparison.DEFAULT_PERCENTILE_VALUES_LEN[-1] + 50
        node_SR.roadtype = TYPE_R_TURN

        node_alt_list_0 = [node_SL, node_SR, node_SL, node_SR]
        node_alt_list_1 = [node_SR, node_SL, node_SR, node_SL]

        sdl_road_0 = self.str_comparer.nodes_to_sdl_2d(node_alt_list_0)
        sdl_road_1 = self.str_comparer.nodes_to_sdl_2d(node_alt_list_1)
        result = self.str_comparer._sdl_2d_error_at_startpoint(start_point=0, longer_road_sdl=sdl_road_0,
                                                               shorter_road_sdl=sdl_road_1)
        # errors are summed up --> 4
        expected_error = 1.0 * len(sdl_road_1)
        self.assertAlmostEqual(expected_error, result, msg="The error should be maxed, check the weights!")

    def test_lcs_curve_sdl_not_normalized(self):
        result = utils.lcs(self.nodes0_list, self.nodes1_list, normalized=False)
        self.assertEqual(4, result)

    def test_lcs_curve_sdl_normalized(self):
        result = utils.lcs(self.nodes0_list, self.nodes1_list, normalized=True)
        expected = 4/len(self.nodes0_list)
        self.assertAlmostEqual(expected, result)

    def test_lcstr_curve_sdl_not_normalized(self):
        result = utils.LCSubStr(self.nodes0_list, self.nodes1_list, normalized=False)
        self.assertEqual(3, result)

    def test_lcstr_curve_sdl_normalized(self):
        result = utils.LCSubStr(self.nodes0_list, self.nodes1_list, normalized=True)
        expected = 3/len(self.nodes0_list)
        self.assertEqual(expected, result)

    def test_all_roads_to_curvature_sdl(self):
        with unittest.mock.patch.object(self.str_comparer, "_compute_length", new=mocked_compute_length):
            self.str_comparer.all_roads_to_curvature_sdl()
            road_dict = list(self.str_comparer.data_dict.values())[0]
            self.assertTrue(utils.DicConst.SDL_2D.value in road_dict, "two dimensional sdl has not been added")
            self.assertTrue(utils.DicConst.CUR_SDL.value in road_dict, "one dimensional sdl has not been added")

    def test_sdl_all_to_all_execution(self):
        with unittest.mock.patch.object(self.str_comparer, "_compute_length", new=mocked_compute_length):
            self.str_comparer.all_roads_to_curvature_sdl()
            self.str_comparer.sdl_all_to_all_unoptimized()
            road_dict = list(self.str_comparer.data_dict.values())[0]
            self.assertTrue(utils.DicConst.SDL_2D_DIST.value in road_dict,
                            "two dimensional sdl dist has not been added")
            self.assertTrue(utils.DicConst.CUR_SDL_DIST.value in road_dict,
                            "one dimensional sdl dist has not been added")
            sdl_error = self.str_comparer.data_dict["0"].get(utils.DicConst.CUR_SDL_DIST.value)
            self.assertAlmostEqual(0, sdl_error["0"], msg="Ones own curvature sdl error has to be 0")
            sdl_2d_error = self.str_comparer.data_dict["0"].get(utils.DicConst.SDL_2D_DIST.value)
            self.assertAlmostEqual(0, sdl_error["0"], msg="Ones own 2d sdl error has to be 0")

    def test_nodes_to_sdl_2d_segment_compression(self):
        with unittest.mock.patch.object(self.str_comparer, "_compute_length", new=mocked_compute_length):
            max_len = string_comparison.DEFAULT_PERCENTILE_VALUES_LEN[-1]

            # create a node that has a little over the quarter length of the last segment
            node_quarter_len = MagicMock()
            node_quarter_len.angle = 20
            node_quarter_len.length = max_len/4 + 1
            node_quarter_len.roadtype = TYPE_R_TURN

            print(max_len, max_len/4 + 1)

            node_list = [node_quarter_len, self.nodes0_list[-2], self.nodes0_list[-1], node_quarter_len,
                         node_quarter_len, node_quarter_len, node_quarter_len]

            # expects the quarters to be combined into one long segment
            result = self.str_comparer.nodes_to_sdl_2d(nodes=node_list)
            expected = string_comparison.len_en(string_comparison.NUM_LEN_ALPHABET-1)
            self.assertEqual(expected, result[-1][1])
            # a reduction is expected
            self.assertEqual(4, len(result))

    def test_get_const_for_length_min_len(self):
        with unittest.mock.patch.object(self.str_comparer, "_compute_length", new=mocked_compute_length):
            min_len = string_comparison.DEFAULT_PERCENTILE_VALUES_LEN[0]

            # expects the shortest symbol
            result = self.str_comparer.get_const_for_length(min_len)
            expected = string_comparison.len_en(0)
            self.assertEqual(expected, result)



if __name__ == '__main__':
    unittest.main()
