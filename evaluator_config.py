import utils
from utils import RoadDicConst, BehaviorDicConst

upper_dir = r"C:\Users\fraun\exp-ba"

# detemines how many chars of a subfolder name to look at for a unique name
first_chars_of_experiments_subfolder = 10

# save obe or all road plots, requires a lot of disk io, disk space and is slow
OBE_WRITE = False
ALL_ROADS_WRITE = False

# False if you want a custom sized angle alphabet
# Has to be True for unit tests, otherwise borders are off
USE_FIXED_STRONG_BORDERS = True
# 28 (7 ang 4 len), 44 (11 ang 4 len), or 60 (15 ang 4 len), defaults to 44
ALPHABET_SIZE = 28

# Use a seed for adaptive random sampling if it should be deterministic
SEED_ADAPTIVE_RANDOM = 1234

# remove tests with broken speed, otherwise these segments get ignored
rm_broken_speed_roads = True

# some configuration has been moved to utils because cross imports created problems
# also add bins where sparsely covered bins are removed
CLEANUP_BINS = True
coverages_1d_to_analyse = [utils.RoadDicConst.SPEED_BINS.value,
                           utils.RoadDicConst.STEERING_BINS.value,
                           utils.RoadDicConst.STEERING_BINS_ADJUSTED.value,
                           utils.RoadDicConst.DISTANCE_BINS.value]
coverages_2d_to_analyse = [utils.RoadDicConst.SPEED_STEERING_2D.value,
                           utils.RoadDicConst.SPEED_STEERING_2D_ADJ.value,
                           utils.RoadDicConst.OBE_2D.value]

# what output metrics should be computed and written out
# to reconfigure these metrics change output_metrics_config in suite_behaviour_computer.py
output_metrics_to_analyse = [BehaviorDicConst.CENTER_DIST_BINARY.value,
                             BehaviorDicConst.CENTER_DIST_SINGLE.value,
                             BehaviorDicConst.STEERING_DIST_BINARY.value,
                             BehaviorDicConst.STEERING_DIST_SINGLE.value,
                             BehaviorDicConst.STEERING_ADJUSTED_DIST_BINARY.value,
                             BehaviorDicConst.STEERING_ADJUSTED_DIST_SINGLE.value,
                             BehaviorDicConst.SPEED_DIST_BINARY.value,
                             BehaviorDicConst.SPEED_DIST_SINGLE.value,
                             BehaviorDicConst.BINS_STEERING_SPEED_DIST.value,
                             BehaviorDicConst.BINS_STEERING_SPEED_DIST_SINGLE.value,
                             BehaviorDicConst.BINS_STEERING_SPEED_DIST_ADJUSTED.value,
                             BehaviorDicConst.BINS_STEERING_SPEED_DIST_ADJUSTED_SINGLE.value,
                             #BehaviorDicConst.STEERING_DTW.value,
                             #BehaviorDicConst.SPEED_DTW.value,
                             #BehaviorDicConst.STEERING_SPEED_DTW.value
                             ]

string_metrics_to_analyse = [BehaviorDicConst.CUR_SDL_DIST.value,
                             BehaviorDicConst.SDL_2D_DIST.value,
                             #BehaviorDicConst.CUR_SDL_LCS_DIST.value,
                             #BehaviorDicConst.SDL_2D_LCS_DIST.value,
                             #BehaviorDicConst.CUR_SDL_LCSTR_DIST.value,
                             #BehaviorDicConst.SDL_2D_LCSTR_DIST.value,
                             #BehaviorDicConst.CUR_SDL_1_LCSTR_DIST.value,
                             #BehaviorDicConst.SDL_2D_1_LCSTR_DIST.value,
                             #BehaviorDicConst.CUR_SDL_3_LCSTR_DIST.value,
                             #BehaviorDicConst.SDL_2D_3_LCSTR_DIST.value,
                             #BehaviorDicConst.CUR_SDL_5_LCSTR_DIST.value,
                             #BehaviorDicConst.SDL_2D_5_LCSTR_DIST.value,
                             BehaviorDicConst.JACCARD.value
                             ]

numeric_vals_to_write = [RoadDicConst.NUM_OBES.value,
                         BehaviorDicConst.NUM_STATES.value,
                         RoadDicConst.ROAD_LEN.value,
                         #RoadDicConst.EXEC_TIME.value
                         ]
