from os import path
from typing import List
# replace os.path
from pathlib import Path
import time

import coverage_evaluator
import utils
from utils import BehaviorDicConst
from utils import RoadDicConst
import evaluator_config as econf
from string_comparison import StringComparer
from suite_behaviour_computer import SuiteBehaviourComputer
from csv_creator import CSVCreator
from suite_trimmer import SuiteTrimmer
from clusterer import Clusterer
from adaptive_random_sampler import AdaptiveRandSampler

import colorama

# ssh -i /root/.ssh/id_rsa.pub ubuntu@160.85.252.213  # bill test suite pc

# TODO do this everywhere this is more pythonic
# main_bin = self.test_dict.get(road_to_compare, None).get(measure, None)

# these paths are broken
# "C:\Users\fraun\exp-ba\experiments-driver-ai\one-plus-one--lanedist--driver-ai--small--no-repair--with-restart--2\.one-plus-one-EA--lanedist--ext--small--no-repair--with-restart--env\output\execs\test_0029.json"
# "C:\Users\fraun\exp-ba\experiments-driver-ai\one-plus-one--lanedist--driver-ai--small--no-repair--with-restart--2\.one-plus-one-EA--lanedist--ext--small--no-repair--with-restart--env\output\execs\test_0042.json"

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
            print("Ignoring", child, ", it is a file")
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


def old_main():
    #parent_dir = Path(r"C:\Users\fraun\experiments-driver-ai")
    """!!IMPORTANT: THE PARENT DIRECTOR HAS TO START WITH "experiments-"!!"""
    # regular, sets, invalid tests removed, including OBE tests
    # "C:\Users\fraun\exp-ba\experiments-driver-ai-wo-minlen-wo-infspeed"
    parent_dir = Path(r"C:\Users\fraun\exp-ba\experiments-beamng-ai-wo-minlen-wo-infspeed")
    #parent_dir = Path(r"C:\Users\fraun\exp-ba\experiments-driver-ai-wo-minlen-wo-infspeed")
    # "C:\Users\fraun\exp-ba\experiments-driver-ai-test"
    #parent_dir = Path(r"C:\Users\fraun\exp-ba\experiments-beamng-ai-wo-15-4-low-div")
    #parent_dir = Path(r"C:\Users\fraun\exp-ba\experiments-beamng-ai-no-obe-wo-minlen-wo-infspeed")
    # for creating diversity sets
    #parent_dir = Path(r"C:\Users\fraun\exp-ba\div_bng\experiments-beamng-ai-wo-ml-wo-is-lowdiv-obe-11")
    all_paths = get_all_paths(parent_dir)

    # FIXME the folder structure seems broken sometimes
    # FIXME there is an defective road in C:\Users\fraun\experiments-driver-ai\one-plus-one--lanedist--driver-ai--small--no-repair--with-restart--2\.one-plus-one-EA--lanedist--ext--small--no-repair--with-restart--env
    #print(all_paths.pop(2))
    broken_tests = []
    # These have to be adjusted for adpative random sampling
    RECOMPUTE_AFTER_REMOVAL = False
    ADAPTIVE_RAND_SAMPLE = False
    compute = True
    while compute:
        compute = False

        start_gathering = time.time()
        QUICK = False
        if QUICK:
            env_directory = Path(r"C:\Users\fraun\exp-ba\experiments-driver-ai-test\random--lanedist--driver-ai--small--no-repair--with-restart--4\.random--lanedist--ext--small--no-repair--with-restart--env")
            parent_dir = Path(r"C:\Users\fraun\exp-ba\experiments-driver-ai-test")
            cov_eval = coverage_evaluator.CoverageEvaluator(set_path=env_directory)
            data_bins_dict = cov_eval.get_all_bins()
            broken_tests.extend(cov_eval.get_broken_speed_tests())
        else:
            # commented for testing purposes
            data_bins_dict = {}
            for env_directory in all_paths:
                print("Start evaluation of OBEs from %s", env_directory)
                # TODO check whether identifier already exists in dict
                cov_eval = coverage_evaluator.CoverageEvaluator(set_path=env_directory)
                data_bins_dict.update(cov_eval.get_all_bins())
                broken_tests.extend(cov_eval.get_broken_speed_tests())
            print(len(broken_tests), "broken_tests have to be ignored because of broken speeds", broken_tests)
            end_gathering = time.time()
            print(end_gathering - start_gathering, "seconds to gather the data")


        start_suite_behaviour = time.time()
        sbh = SuiteBehaviourComputer(data_bins_dict)
        coverage_tuple_list = []
        for measure in econf.coverages_1d_to_analyse:
            cov_value = sbh.calculate_suite_coverage_1d(feature=measure, add_for_each=False)
            coverage_tuple_list.append((measure, cov_value))
            print(str(measure) + " coverage", cov_value)
            if econf.CLEANUP_BINS:
                m_cl = str(measure) + RoadDicConst.BIN_CLEANUP.value
                cov_value = sbh.calculate_suite_coverage_1d(feature=m_cl, add_for_each=False)
                coverage_tuple_list.append((m_cl, cov_value))
                print(str(m_cl) + " coverage", cov_value)
        for measure in econf.coverages_2d_to_analyse:
            cov_value = sbh.calculate_suite_2d_coverage(feature=measure, add_for_each=False)
            coverage_tuple_list.append((measure, cov_value))
            print(str(measure) + " coverage", cov_value)
            if econf.CLEANUP_BINS:
                m_cl = str(measure) + RoadDicConst.BIN_CLEANUP.value
                cov_value = sbh.calculate_suite_2d_coverage(feature=m_cl, add_for_each=False)
                coverage_tuple_list.append((m_cl, cov_value))
                print(str(m_cl) + " coverage", cov_value)
        print("coverage_tuple_list", coverage_tuple_list)
        #print("road compare 1d", sbh.behavior_compare_1d("random--la22", 'steering_bins'))
        #print("road compare 2d", sbh.behavior_compare_2d("random--la22", 'speed_steering_2d'))
        end_suite_behaviour = time.time()
        print(end_suite_behaviour - start_suite_behaviour, "seconds to compute the test behavior")

        from road_visualizer.visualize_centerline import visualize_centerline
        road0_lstr = list(data_bins_dict.values())[0].get(RoadDicConst.POLYLINE.value)
        #visualize_centerline(road0_lstr, road_width=10)   # fixme this breaks box plots
        #print(list(data_bins_dict.values())[0].get(RoadDicConst.POLYLINE.value).coords.xy[0])  #TODO remove
        #print(list(data_bins_dict.values())[0].get(RoadDicConst.POLYLINE.value).coords.xy[1])  #TODO remove

        other_data_tuples_list = []
        total_time = sbh.calculate_whole_suite_time()
        other_data_tuples_list.append(("total_time", total_time))
        print("total_time", total_time)
        num_obes = sbh.calculate_whole_suite_sum(feature=RoadDicConst.NUM_OBES.value)
        other_data_tuples_list.append(("num_obes", num_obes))
        print("num_obes", num_obes)

        sbh.behaviour_all_to_all()
        # unnecessary, pass by reference
        #partial_bins = sbh.get_test_dict()


        start_str_trans = time.time()
        str_comparer = StringComparer(data_dict=data_bins_dict)
        str_comparer.all_roads_to_curvature_sdl()
        str_comparer.print_ang_len_for_road('random--la52')
        str_comparer.print_ang_len_for_road('random--la52', use_fnc=False)
        end_str_trans = time.time()
        print(end_str_trans - start_str_trans, "seconds to compute the string translation")

        start_str_comp = time.time()
        str_comparer.sdl_all_to_all_unoptimized()
        end_str_comp = time.time()
        print(end_str_comp - start_str_comp, "seconds to compute the string distances")
        #str_comparer.all_roads_average_curvature()

        # only do this if all steering angles get collected in the coverage evaluator
        # utils.optimized_bin_borders_percentiles(cov_eval.all_angles, 16)

        #utils.whole_suite_dist_matrix_statistic_incomplete(data_bins_dict, feature=utils.BehaviorDicConst.JACCARD.value,
        #                                                   title="Jaccard box plots", plot=True)

        SHAPE_METRICS = False
        if SHAPE_METRICS:
            # predefined shape based metrics
            start_predefined = time.time()
            utils.add_coord_tuple_representation(data_dict=data_bins_dict)
            utils.align_shape_of_roads(data_dict=data_bins_dict)
            #print(list(data_bins_dict.values())[0].get(RoadDicConst.COORD_TUPLE_REP.value))
            utils.shape_similarity_measures_all_to_all_unoptimized(data_dict=data_bins_dict)
            end_predefined = time.time()
            print(end_predefined - start_predefined, "seconds to compute the similaritymeasures distances")

        start_csv = time.time()
        WRITE_CSV = True
        if WRITE_CSV:
            print()
            print(colorama.Fore.GREEN + "Writing csvs" + colorama.Style.RESET_ALL)
            csv_creator = CSVCreator(data_dict=data_bins_dict, root_path=parent_dir)
            csv_creator.write_whole_suite_multiple_values("whole_suite_coverages", coverage_tuple_list)
            csv_creator.write_whole_suite_multiple_values("other_numerics", other_data_tuples_list, first_row_name="measures")

            for metr in econf.string_metrics_to_analyse:
                descr = str_comparer.get_configuration_description()
                csv_creator.write_all_to_all_dist_matrix(metr, notes=descr)

            for metr in econf.output_metrics_to_analyse:
                csv_creator.write_all_to_all_dist_matrix(measure=metr)

            if SHAPE_METRICS:
                csv_creator.write_all_to_all_dist_matrix(measure=BehaviorDicConst.COORD_DTW_DIST.value)
                csv_creator.write_all_to_all_dist_matrix(measure=BehaviorDicConst.COORD_FRECHET_DIST.value)

            csv_creator.write_all_tests_one_value(measure=RoadDicConst.NUM_OBES.value)
            csv_creator.write_all_tests_one_value(measure=BehaviorDicConst.NUM_STATES.value)
            #csv_creator.write_all_to_all_dist_matrix(measure=BehaviorDicConst.CUR_SDL_LCS_DIST.value)
            #csv_creator.write_all_to_all_dist_matrix(measure=BehaviorDicConst.CUR_SDL_LCSTR_DIST.value)
            #csv_creator.write_two_roads_dists(road_1_name="random--la22", road_2_name="random--la23", measures=['curve_sdl_dist', 'random--la22_binary_steering_bins'])
            #csv_creator.write_all_two_roads_dists(road_1_name="random--la22", measures=['curve_sdl_dist', 'random--la22_binary_steering_bins'])
            #csv_creator.write_single_road_dists(road_name="1-2", measures=['curve_sdl_dist', '1-2_binary_steering_bins'])

            for cov_metr in econf.coverages_1d_to_analyse:
                csv_creator.write_whole_suite_1d_coverages(cov_metr)
            # check ob das noch stimmt
            #print("spst 2d", data_bins_dict['random--la11'][RoadDicConst.SPEED_STEERING_2D.value])
            #print("spst 2d adj", data_bins_dict['random--la11'][RoadDicConst.SPEED_STEERING_2D_ADJ.value])
            for cov_metr in econf.coverages_2d_to_analyse:
                csv_creator.write_whole_suite_2d_coverages(cov_metr)

        end_csv = time.time()
        print(end_csv - start_csv, "seconds to write the csvs")

        #print(list(data_bins_dict.values())[2]["curve_sdl"])

        #utils.whole_suite_statistics(dataset_dict=data_bins_dict, feature="num_states", plot=True)

        print(colorama.Fore.GREEN + "Collected a total of", len(data_bins_dict), "roads!" + colorama.Style.RESET_ALL)
        names_of_all = list(data_bins_dict.keys())
        print("all roads ", names_of_all)
        print(colorama.Fore.GREEN + "Computed following measures for each road", data_bins_dict[names_of_all[0]].keys(), "" + colorama.Style.RESET_ALL)
        #print("all roads ", data_bins_dict)

        suite_trimmer = SuiteTrimmer(data_dict=data_bins_dict, base_path=parent_dir)

        if ADAPTIVE_RAND_SAMPLE:
            sampler = AdaptiveRandSampler(data_dict=data_bins_dict) # mini suite: "random--la53", regular: "random--la311"
            # bng obe start: rand--la11/111/617, non obe start: "random--la311"
            # drvr obe start: , non obe start:
            # highest min similarity --> low diversity (? think when ur awake)
            #sampler.sample_of_n(measure=BehaviorDicConst.JACCARD.value, n=30, first_test="random--la311", func=sampler.pick_highest_min_similarity)
            # smallest max similarity --> high diversity (? think when ur awake)
            sampler.sample_of_n(measure=BehaviorDicConst.JACCARD.value, n=30, first_test="random--la11",
                                func=sampler.pick_highest_min_similarity)
            unworthy_paths = sampler.get_unworthy_paths()
            rem = suite_trimmer.trim_dataset_list(unworthy_paths=unworthy_paths, description="diversity suite")
            compute = True
            ADAPTIVE_RAND_SAMPLE = False
            if WRITE_CSV and rem:
                csv_creator.write_all_tests_one_value(BehaviorDicConst.SAMPLING_INDEX.value)


        clusterer = Clusterer(data_dict=data_bins_dict)
        #clusterer.perform_optics(measure=BehaviorDicConst.JACCARD.value)
        #clusterer.networkx_plot_measure(measure=BehaviorDicConst.BINS_STEERING_SPEED_DIST.value, draw_edges=True)

        #print("data_bins_dict['random--la52']", data_bins_dict['random--la52'])
        #print("data_bins_dict['random--la54']['speed_steering_2d']", data_bins_dict['random--la54']['speed_steering_2d'])

        # halving the suite size
        #unworthy_paths = suite_trimmer.get_random_percentage_unworthy(percentage=50)
        #suite_trimmer.trim_dataset_list(unworthy_paths=unworthy_paths, description="halving the suite size")
        import operator
        # remove broken tests
        #suite_trimmer.trim_dataset_list(unworthy_paths=broken_tests, description="Broken tests with infinite speed removed")
        #suite_trimmer.trim_dataset(feature=RoadDicConst.UNDER_MIN_LEN_SEGS.value, op=operator.eq, threshold=True)

        #print("unworthy paths:", suite_trimmer.get_unworthy_paths(feature="num_states", op=operator.le, threshold=300))
        #suite_trimmer.trim_dataset(feature="num_states", op=operator.le, threshold=300)
        # remove all tests with obes
        #suite_trimmer.trim_dataset(feature=utils.RoadDicConst.NUM_OBES.value, op=operator.ge, threshold=0.9)
        #suite_trimmer.trim_dataset_percentile(feature="num_states", op=operator.le, threshold_percentile=2)

def adaptive_random_sampling_multiple_subsets(parent_folder=r"C:\Users\fraun\exp-ba\div_bng"):
    # this should be automatically created
    # name, high or low, startpoint
    # the files have to be filled accordingly, this should be automated and copying files from a source
    subsets = [
        {'path': r"experiments-beamng-ai-wo-ml-wo-is-highdiv-obe-11", 'diversity': "high", 'startpoint': "random--la11"},  # OBE
        {'path': r"experiments-beamng-ai-wo-ml-wo-is-highdiv-obe-111", 'diversity': "high", 'startpoint': "random--la111"},
        {'path': r"experiments-beamng-ai-wo-ml-wo-is-highdiv-obe-617", 'diversity': "high", 'startpoint': "random--la617"},
        {'path': r"experiments-beamng-ai-wo-ml-wo-is-highdiv-noobe-311", 'diversity': "high", 'startpoint': "random--la311"},  # nonOBE
        {'path': r"experiments-beamng-ai-wo-ml-wo-is-highdiv-noobe-222", 'diversity': "high", 'startpoint': "random--la222"},
        {'path': r"experiments-beamng-ai-wo-ml-wo-is-highdiv-noobe-711", 'diversity': "high", 'startpoint': "random--la711"},
        {'path': r"experiments-beamng-ai-wo-ml-wo-is-lowdiv-obe-11", 'diversity': "low", 'startpoint': "random--la11"},  # OBE
        {'path': r"experiments-beamng-ai-wo-ml-wo-is-lowdiv-obe-111", 'diversity': "low", 'startpoint': "random--la111"},
        {'path': r"experiments-beamng-ai-wo-ml-wo-is-lowdiv-obe-617", 'diversity': "low", 'startpoint': "random--la617"},
        {'path': r"experiments-beamng-ai-wo-ml-wo-is-lowdiv-noobe-311", 'diversity': "low", 'startpoint': "random--la311"},  # nonOBE
        {'path': r"experiments-beamng-ai-wo-ml-wo-is-lowdiv-noobe-222", 'diversity': "low", 'startpoint': "random--la222"},
        {'path': r"experiments-beamng-ai-wo-ml-wo-is-lowdiv-noobe-711", 'diversity': "low", 'startpoint': "random--la711"}
    ]
    for seti in subsets:
        spath = path.join(parent_folder, seti['path'])
        adaptive_random_sample_oneset(spath, seti['startpoint'], seti['diversity'])

def adaptive_random_sample_oneset(parent_dir, start_point, diversity: str):
    """ Perfoms adaptive random sampling on one subset, forces removal

    :param parent_dir:
    :param start_point:
    :param diversity: "high" or "low" for now
    :return:
    """
    """!!IMPORTANT: THE PARENT DIRECTOR HAS TO START WITH "experiments-"!!"""

    all_paths = get_all_paths(Path(parent_dir))

    # FIXME the folder structure seems broken sometimes
    # FIXME there is an defective road in C:\Users\fraun\experiments-driver-ai\one-plus-one--lanedist--driver-ai--small--no-repair--with-restart--2\.one-plus-one-EA--lanedist--ext--small--no-repair--with-restart--env
    #print(all_paths.pop(2))
    broken_tests = []
    # These have to be adjusted for adpative random sampling
    RECOMPUTE_AFTER_REMOVAL = True
    ADAPTIVE_RAND_SAMPLE = True
    compute = True
    while compute:
        compute = False

        start_gathering = time.time()
        # quick not possible for adaptive random sample
        # commented for testing purposes
        data_bins_dict = {}
        for env_directory in all_paths:
            print("Start evaluation of OBEs from %s", env_directory)
            # TODO check whether identifier already exists in dict
            cov_eval = coverage_evaluator.CoverageEvaluator(set_path=env_directory)
            data_bins_dict.update(cov_eval.get_all_bins())
            broken_tests.extend(cov_eval.get_broken_speed_tests())
        print(len(broken_tests), "broken_tests have to be ignored because of broken speeds", broken_tests)
        end_gathering = time.time()
        print(end_gathering - start_gathering, "seconds to gather the data")


        start_suite_behaviour = time.time()
        sbh = SuiteBehaviourComputer(data_bins_dict)
        coverage_tuple_list = []
        for measure in econf.coverages_1d_to_analyse:
            cov_value = sbh.calculate_suite_coverage_1d(feature=measure, add_for_each=False)
            coverage_tuple_list.append((measure, cov_value))
            print(str(measure) + " coverage", cov_value)
            if econf.CLEANUP_BINS:
                m_cl = str(measure) + RoadDicConst.BIN_CLEANUP.value
                cov_value = sbh.calculate_suite_coverage_1d(feature=m_cl, add_for_each=False)
                coverage_tuple_list.append((m_cl, cov_value))
                print(str(m_cl) + " coverage", cov_value)
        for measure in econf.coverages_2d_to_analyse:
            cov_value = sbh.calculate_suite_2d_coverage(feature=measure, add_for_each=False)
            coverage_tuple_list.append((measure, cov_value))
            print(str(measure) + " coverage", cov_value)
            if econf.CLEANUP_BINS:
                m_cl = str(measure) + RoadDicConst.BIN_CLEANUP.value
                cov_value = sbh.calculate_suite_2d_coverage(feature=m_cl, add_for_each=False)
                coverage_tuple_list.append((m_cl, cov_value))
                print(str(m_cl) + " coverage", cov_value)
        print("coverage_tuple_list", coverage_tuple_list)

        end_suite_behaviour = time.time()
        print(end_suite_behaviour - start_suite_behaviour, "seconds to compute the test behavior")


        other_data_tuples_list = []
        total_time = sbh.calculate_whole_suite_time()
        other_data_tuples_list.append(("total_time", total_time))
        print("total_time", total_time)
        num_obes = sbh.calculate_whole_suite_sum(feature=RoadDicConst.NUM_OBES.value)
        other_data_tuples_list.append(("num_obes", num_obes))
        print("num_obes", num_obes)

        sbh.behaviour_all_to_all()

        start_str_trans = time.time()
        str_comparer = StringComparer(data_dict=data_bins_dict)
        str_comparer.all_roads_to_curvature_sdl()
        str_comparer.print_ang_len_for_road('random--la52')
        str_comparer.print_ang_len_for_road('random--la52', use_fnc=False)
        end_str_trans = time.time()
        print(end_str_trans - start_str_trans, "seconds to compute the string translation")

        start_str_comp = time.time()
        str_comparer.sdl_all_to_all_unoptimized()
        end_str_comp = time.time()
        print(end_str_comp - start_str_comp, "seconds to compute the string distances")
        #str_comparer.all_roads_average_curvature()

        # only do this if all steering angles get collected in the coverage evaluator
        # utils.optimized_bin_borders_percentiles(cov_eval.all_angles, 16)

        SHAPE_METRICS = False
        if SHAPE_METRICS:
            # predefined shape based metrics
            start_predefined = time.time()
            utils.add_coord_tuple_representation(data_dict=data_bins_dict)
            utils.align_shape_of_roads(data_dict=data_bins_dict)
            #print(list(data_bins_dict.values())[0].get(RoadDicConst.COORD_TUPLE_REP.value))
            utils.shape_similarity_measures_all_to_all_unoptimized(data_dict=data_bins_dict)
            end_predefined = time.time()
            print(end_predefined - start_predefined, "seconds to compute the similaritymeasures distances")

        start_csv = time.time()
        WRITE_CSV = True
        if WRITE_CSV:
            print()
            print(colorama.Fore.GREEN + "Writing csvs" + colorama.Style.RESET_ALL)
            csv_creator = CSVCreator(data_dict=data_bins_dict, root_path=parent_dir)
            csv_creator.write_whole_suite_multiple_values("whole_suite_coverages", coverage_tuple_list)
            csv_creator.write_whole_suite_multiple_values("other_numerics", other_data_tuples_list, first_row_name="measures")

            for metr in econf.string_metrics_to_analyse:
                descr = str_comparer.get_configuration_description()
                csv_creator.write_all_to_all_dist_matrix(metr, notes=descr)

            for metr in econf.output_metrics_to_analyse:
                csv_creator.write_all_to_all_dist_matrix(measure=metr)

            if SHAPE_METRICS:
                csv_creator.write_all_to_all_dist_matrix(measure=BehaviorDicConst.COORD_DTW_DIST.value)
                csv_creator.write_all_to_all_dist_matrix(measure=BehaviorDicConst.COORD_FRECHET_DIST.value)

            csv_creator.write_all_tests_one_value(measure=RoadDicConst.NUM_OBES.value)
            csv_creator.write_all_tests_one_value(measure=BehaviorDicConst.NUM_STATES.value)

            for cov_metr in econf.coverages_1d_to_analyse:
                csv_creator.write_whole_suite_1d_coverages(cov_metr)

            for cov_metr in econf.coverages_2d_to_analyse:
                csv_creator.write_whole_suite_2d_coverages(cov_metr)

        end_csv = time.time()
        print(end_csv - start_csv, "seconds to write the csvs")


        print(colorama.Fore.GREEN + "Collected a total of", len(data_bins_dict), "roads!" + colorama.Style.RESET_ALL)
        names_of_all = list(data_bins_dict.keys())
        print("all roads ", names_of_all)
        print(colorama.Fore.GREEN + "Computed following measures for each road", data_bins_dict[names_of_all[0]].keys(), "" + colorama.Style.RESET_ALL)
        #print("all roads ", data_bins_dict)

        suite_trimmer = SuiteTrimmer(data_dict=data_bins_dict, base_path=parent_dir)

        if ADAPTIVE_RAND_SAMPLE:
            sampler = AdaptiveRandSampler(data_dict=data_bins_dict) # mini suite: "random--la53", regular: "random--la311"
            # bng obe start: rand--la11/111/617, non obe start: "random--la311"
            # drvr obe start: , non obe start:
            if diversity == "low":
                # highest min similarity --> low diversity (? think when ur awake)
                sampler.sample_of_n(measure=BehaviorDicConst.JACCARD.value, n=30, first_test=start_point,
                                    func=sampler.pick_highest_min_similarity)
            elif diversity == "high":
                # smallest max similarity --> high diversity (? think when ur awake)
                sampler.sample_of_n(measure=BehaviorDicConst.JACCARD.value, n=30, first_test=start_point,
                                    func=sampler.pick_highest_min_similarity)
            unworthy_paths = sampler.get_unworthy_paths()
            # force to not wait on user input
            rem = suite_trimmer.trim_dataset_list(unworthy_paths=unworthy_paths, description="diversity suite", force=True)
            compute = True
            ADAPTIVE_RAND_SAMPLE = False
            if WRITE_CSV and rem:
                csv_creator.write_all_tests_one_value(BehaviorDicConst.SAMPLING_INDEX.value)



if __name__ == "__main__":
    #old_main()
    adaptive_random_sampling_multiple_subsets()
