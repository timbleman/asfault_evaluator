import utils
from utils import RoadDicConst, BehaviorDicConst

# detemines how many chars of a subfolder name to look at for a unique name
first_chars_of_experiments_subfolder = 10

# save obe or all road plots, requires a lot of disk io, disk space and is slow
OBE_WRITE = False
ALL_ROADS_WRITE = False

# False if you want a custom sized angle alphabet
# Has to be True for unit tests, otherwise borders are off
USE_FIXED_STRONG_BORDERS = True

# Use a seed for adaptive random sampling if it should be deterministic
SEED_ADAPTIVE_RANDOM = 1234

# remove tests with broken speed, otherwise these segments get ignored
rm_broken_speed_roads = True

# some configration has been moved to utils because cross imports created problems

coverages_1d_to_analyse = [utils.RoadDicConst.SPEED_BINS.value,
                           utils.RoadDicConst.STEERING_BINS.value,
                           utils.RoadDicConst.DISTANCE_BINS.value]
coverages_2d_to_analyse = [utils.RoadDicConst.SPEED_STEERING_2D.value,
                           utils.RoadDicConst.OBE_2D.value]

output_metrics_to_analyse = [BehaviorDicConst.CENTER_DIST_BINARY.value,
                             BehaviorDicConst.CENTER_DIST_SINGLE.value,
                             BehaviorDicConst.STEERING_DIST_BINARY.value,
                             BehaviorDicConst.STEERING_DIST_SINGLE.value,
                             BehaviorDicConst.SPEED_DIST_BINARY.value,
                             BehaviorDicConst.SPEED_DIST_SINGLE.value,
                             BehaviorDicConst.BINS_STEERING_SPEED_DIST.value,
                             BehaviorDicConst.BINS_STEERING_SPEED_DIST_SINGLE.value]
