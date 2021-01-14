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

import click
import colorama

# ssh -i /root/.ssh/id_rsa.pub ubuntu@160.85.252.213  # bill test suite pc

# TODO do this everywhere this is more pythonic
# main_bin = self.test_dict.get(road_to_compare, None).get(measure, None)

# these paths are broken
# "C:\Users\fraun\exp-ba\experiments-driver-ai\one-plus-one--lanedist--driver-ai--small--no-repair--with-restart--2\.one-plus-one-EA--lanedist--ext--small--no-repair--with-restart--env\output\execs\test_0029.json"
# "C:\Users\fraun\exp-ba\experiments-driver-ai\one-plus-one--lanedist--driver-ai--small--no-repair--with-restart--2\.one-plus-one-EA--lanedist--ext--small--no-repair--with-restart--env\output\execs\test_0042.json"

upper_dir_p = Path(econf.upper_dir)

def _configure_asfault() -> None:
    # asfault setup
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


def old_main(suite: str = "bng", wo_obe: bool = False, remove: str = None):
    """
    Use this main for single suites or to remove obe tests

    :param suite: "bng" or "drvr" or a path, select from two predefined suites
    :param wo_obe: use the set excluding all OBEs
    :param remove: "obe" or "nonobe", removes and recomputes
    :return:
    """
    #parent_dir = Path(r"C:\Users\fraun\experiments-driver-ai")
    """!!IMPORTANT: THE PARENT DIRECTOR HAS TO START WITH "experiments-"!!"""
    # regular, sets, invalid tests removed, including OBE tests
    if suite == "bng":
        if wo_obe:
            parent_dir = upper_dir_p.joinpath(r"experiments-beamng-ai-no-obe-wo-minlen-wo-infspeed")
        else:
            parent_dir = upper_dir_p.joinpath(r"experiments-beamng-ai-wo-minlen-wo-infspeed")
    elif suite == "drvr":
        if wo_obe:
            parent_dir = upper_dir_p.joinpath(r"experiments-driver-ai-no-obe-wo-minlen-wo-infspeed")
        else:
            parent_dir = upper_dir_p.joinpath(r"experiments-driver-ai-wo-minlen-wo-infspeed")
    else:
        parent_dir = Path(suite)
    assert path.exists(parent_dir), "The predefined suite paths do not exist or the supplied path is incorrect!"
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
        # set to True to run only a short subset for testing purposes
        QUICK = False
        if QUICK:
            env_directory = upper_dir_p.joinpath(r"experiments-driver-ai-test\random--lanedist--driver-ai--small--no-repair--with-restart--4\.random--lanedist--ext--small--no-repair--with-restart--env")
            parent_dir = upper_dir_p.joinpath(r"experiments-driver-ai-test")
            parent_dir = utils.get_root_of_test_suite(env_directory)
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


        # Compute in econf selected coverage metrics
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

        end_suite_behaviour = time.time()
        print(end_suite_behaviour - start_suite_behaviour, "seconds to compute the test behavior")

        from road_visualizer.visualize_centerline import visualize_centerline
        road0_lstr = list(data_bins_dict.values())[0].get(RoadDicConst.POLYLINE.value)
        #visualize_centerline(road0_lstr, road_width=10)   # fixme this breaks box plots

        other_data_tuples_list = []
        total_time = sbh.calculate_whole_suite_time()
        other_data_tuples_list.append(("total_time", total_time))
        print("total_time of the whole suite", total_time)
        num_obes = sbh.calculate_whole_suite_sum(feature=RoadDicConst.NUM_OBES.value)
        other_data_tuples_list.append(("num_obes", num_obes))
        print("num_obes", num_obes)

        sbh.behaviour_all_to_all()
        # unnecessary, pass by reference
        #partial_bins = sbh.get_test_dict()


        start_str_trans = time.time()
        str_comparer = StringComparer(data_dict=data_bins_dict)
        str_comparer.all_roads_to_sdl()
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

            for numeric_val in econf.numeric_vals_to_write:
                csv_creator.write_all_tests_one_value(numeric_val)
            #csv_creator.write_all_tests_one_value(measure=RoadDicConst.NUM_OBES.value)
            #csv_creator.write_all_tests_one_value(measure=BehaviorDicConst.NUM_STATES.value)
            #csv_creator.write_all_to_all_dist_matrix(measure=BehaviorDicConst.CUR_SDL_LCS_DIST.value)
            #csv_creator.write_all_to_all_dist_matrix(measure=BehaviorDicConst.CUR_SDL_LCSTR_DIST.value)
            #csv_creator.write_two_roads_dists(road_1_name="random--la22", road_2_name="random--la23", measures=['curve_sdl_dist', 'random--la22_binary_steering_bins'])
            #csv_creator.write_all_two_roads_dists(road_1_name="random--la22", measures=['curve_sdl_dist', 'random--la22_binary_steering_bins'])
            #csv_creator.write_single_road_dists(road_name="1-2", measures=['curve_sdl_dist', '1-2_binary_steering_bins'])

            for cov_metr in econf.coverages_1d_to_analyse:
                csv_creator.write_whole_suite_1d_coverages(cov_metr)
            for cov_metr in econf.coverages_2d_to_analyse:
                csv_creator.write_whole_suite_2d_coverages(cov_metr)

        end_csv = time.time()
        print(end_csv - start_csv, "seconds to write the csvs")

        #utils.whole_suite_statistics(dataset_dict=data_bins_dict, feature="num_states", plot=True)

        print(colorama.Fore.GREEN + "Collected a total of", len(data_bins_dict), "roads!" + colorama.Style.RESET_ALL)
        names_of_all = list(data_bins_dict.keys())
        print("all roads ", names_of_all)
        print(colorama.Fore.GREEN + "Computed following measures for each road", data_bins_dict[names_of_all[0]].keys(), "" + colorama.Style.RESET_ALL)

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


        # Some advanced and future work stuff
        # uncomment to use OPTICS clustering or networkx scatterplots
        #clusterer = Clusterer(data_dict=data_bins_dict)
        #clusterer.perform_optics(measure=BehaviorDicConst.JACCARD.value)
        #clusterer.networkx_plot_measure(measure=BehaviorDicConst.JACCARD.value, draw_edges=False, draw_graphweights=False)

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

def adaptive_random_sampling_multiple_subsets(bng_or_drvr: str, num_per_configuration):
    if bng_or_drvr == "bng":
        destination_folder = upper_dir_p.joinpath(r"div_bng5")
        parent_folder = upper_dir_p.joinpath(r"experiments-beamng-ai-wo-minlen-wo-infspeed")
        obe_start_points = [r"random--la11", r"random--la111", r"random--la617", r"random--la219", r"random--la811"]
        non_obe_start_points = [r"random--la311", r"random--la222", r"random--la711", r"random--la84", r"random--la918"]
    elif bng_or_drvr == "drvr":
        destination_folder = upper_dir_p.joinpath(r"div_drvr5")
        parent_folder = upper_dir_p.joinpath(r"experiments-driver-ai-wo-minlen-wo-infspeed")
        obe_start_points = [r"random--la219", r"random--la318", r"random--la520", r"random--la438", r"random--la712"]
        non_obe_start_points = [r"one-plus-o212", r"random--la42", r"random--la68", r"random--la94", r"random--la1010"]
    else:
        raise ValueError("Select between bng and drvr!")

    start_points = obe_start_points[0:num_per_configuration]
    start_points.extend(non_obe_start_points[0:num_per_configuration])
    # this should be automatically created
    # name, high or low, startpoint
    # the files have to be filled accordingly, this should be automated and copying files from a source
    #start_points = ["random--la11", "random--la111", "random--la617", "random--la311", "random--la222", "random--la711"]
    import adaptive_random_sampler
    subsets = adaptive_random_sampler.apdaptive_rand_sample_multiple_subsets(start_points=start_points,
                                                                             destination_path=destination_folder,
                                                                             parent_path=parent_folder)

    print("subsets", subsets)
    # perform adaptive random sampling
    for seti in subsets:
        spath = path.join(destination_folder, seti['folder'])
        adaptive_random_sample_oneset(spath, seti['startpoint'], seti['diversity'], entirely_random=False)


def random_sampling_multiple_subsets(bng_or_drvr, num_subsets):
    if bng_or_drvr == "bng":
        destination_folder = upper_dir_p.joinpath(r"div_bng5")
        parent_folder = upper_dir_p.joinpath(r"experiments-beamng-ai-wo-minlen-wo-infspeed")
    elif bng_or_drvr == "drvr":
        destination_folder = upper_dir_p.joinpath(r"div_drvr5")
        parent_folder = upper_dir_p.joinpath(r"experiments-driver-ai-wo-minlen-wo-infspeed")
    else:
        raise ValueError("Select between bng and drvr!")

    # write list of configs for each subset containing "-random0" and further
    cnfgs = []
    for i in range(0, num_subsets):
        cnfgs.append("-random" + str(i))
    import adaptive_random_sampler
    subsets = adaptive_random_sampler.prepare_folders_for_sampling(configs=cnfgs, parent_path=parent_folder,
                                                                   destination_path=destination_folder)
    for seti in subsets:
        spath = path.join(destination_folder, seti)
        adaptive_random_sample_oneset(spath, None, "", entirely_random=True)


def adaptive_random_sample_oneset(parent_dir, start_point, diversity: str, entirely_random: False):
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


        # Compute in econf selected coverage metrics
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

        end_suite_behaviour = time.time()
        print(end_suite_behaviour - start_suite_behaviour, "seconds to compute the test behavior")


        other_data_tuples_list = []
        total_time = sbh.calculate_whole_suite_time()
        other_data_tuples_list.append(("total_time of the whole suite", total_time))
        print("total_time", total_time)
        num_obes = sbh.calculate_whole_suite_sum(feature=RoadDicConst.NUM_OBES.value)
        other_data_tuples_list.append(("num_obes", num_obes))
        print("num_obes", num_obes)

        sbh.behaviour_all_to_all()

        start_str_trans = time.time()
        str_comparer = StringComparer(data_dict=data_bins_dict)
        str_comparer.all_roads_to_sdl()
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

        suite_trimmer = SuiteTrimmer(data_dict=data_bins_dict, base_path=parent_dir)
        # perform adaptive random sampling
        if ADAPTIVE_RAND_SAMPLE and not entirely_random:
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

        # perform random sampling
        if ADAPTIVE_RAND_SAMPLE and entirely_random:
            unworthy_paths= suite_trimmer.get_random_number_unworthy(remaining_number=30)
            rem = suite_trimmer.trim_dataset_list(unworthy_paths=unworthy_paths, description="random sampling", force=True)
            compute = True
            ADAPTIVE_RAND_SAMPLE = False


@click.group()
def modes():
  pass

@click.command(help="Moves csv files of (adaptive) random sampling subsets.")
@click.option('--parentPath', is_flag=False, default="None", help="Parent path of sampled subsets to copy csvs from")
def copy(parentpath):
    if not Path(parentpath).exists():
        click.echo("The path does not exist. Check whether the folder names are right.")
        return
    import adaptive_random_sampler
    # copy only the .csv, this should only be done after both subset classes are created to move all
    adaptive_random_sampler.mirror_subsets_only_results(Path(parentpath))

@click.command(help="Computing distances and writing out. Creates subsets if needed.")
@click.option('--suite', is_flag=False, help="Select between 'bng' and 'drvr' set.")
@click.option('--woOBE', is_flag=True, help="Collect data for full non OBE suites.")
@click.option('--rs', is_flag=True, help="Use random sampling.")
@click.option('--ars', is_flag=True, help="Use adaptive random sampling.")
@click.option('--subsetnum', required=False, is_flag=False, default="5", help="Number of subsets.")
def run(suite, rs, ars, woobe, subsetnum):
    valid = True
    def check_valid_subset_num(selection, max_num=5):
        # if (isinstance(selection, int)):  # works only for type int
        if (selection.isdigit()):
            if not 0 < int(selection) <= 5:
                click.echo("Only subset sizes between 1 and 5 are possible. " +
                           "Add more startpoints in main.py to solve this.")
                return False
        else:
            click.echo("The number of subsets has to be positive int.")
            return False
        return True

    # ensure user only chooses between bng and drvr
    if suite:
        if suite == "bng" or suite == "drvr":
            click.echo("{0}".format(suite) + " suite selected.")
        else:
            click.echo("Only bng and drvr are valid suite names.")
            valid = False

    # check whether only random or adaptive random sampling is selected
    if rs and not ars:
        valid = valid and check_valid_subset_num(subsetnum)
    if ars and not rs:
        valid = valid and check_valid_subset_num(subsetnum)
    if ars and rs:
        click.echo("Cannot run random sampling and adaptive random sampling simultaneously!")
        valid = False

    if woobe and (rs or ars):
        click.echo("(Adaptive) random sampling does not work without OBE tests.")
        valid = False

    if valid:
        if rs:
            click.echo("Creating subsets using random sampling.")
            random_sampling_multiple_subsets(bng_or_drvr=suite, num_subsets=int(subsetnum))
        elif ars:
            click.echo(
                "Creating subsets using adaptive random sampling. OBE and non-OBE startpoints, as well as high " +
                "and low diversity.")
            adaptive_random_sampling_multiple_subsets(bng_or_drvr=suite, num_per_configuration=int(subsetnum))
        else:
            click.echo("Using the " + "{0}".format(suite) + " suite, without OBE tests: " + "{0}".format(woobe))
            old_main(suite, wo_obe=woobe)

modes.add_command(copy)
modes.add_command(run)

if __name__ == "__main__":
    colorama.init()
    # Uncomment these if you do not want to use the command-line interface
    #old_main("drvr", wo_obe=False)
    #adaptive_random_sampling_multiple_subsets('drvr', 5)
    #random_sampling_multiple_subsets(bng_or_drvr="drvr", num_subsets=5)
    #import adaptive_random_sampler
    # copy only the .csv, this should only be done after both subset classes are created to move all
    #adaptive_random_sampler.mirror_subsets_only_results(upper_dir_p.joinpath(r"div_drvr5"))
    #adaptive_random_sampler.mirror_subsets_only_results(upper_dir_p.joinpath(r"div_bng"))
    #adaptive_random_sampler.prepare_folders_for_sampling(parent_path=upper_dir_p.joinpath(r"experiments-beamng-ai-wo-minlen-wo-infspeed"),
    #                                                     configs=["1", "2", "3"],
    #                                                     destination_path=upper_dir_p.joinpath(r"div_test"))
    #to_sample = adaptive_random_sampler.apdaptive_rand_sample_multiple_subsets(['1', '2', '3'],
    #                                                               parent_path=upper_dir_p.joinpath(r"experiments-beamng-ai-wo-minlen-wo-infspeed"),
    #                                                               destination_path=upper_dir_p.joinpath(r"div_test"))
    #print(to_sample)

    # Use command-line interface
    import click
    modes()