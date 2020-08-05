import utils

# detemines how many chars of a subfolder name to look at for a unique name
first_chars_of_experiments_subfolder = 10

USE_FIXED_STRONG_BORDERS = True

# remove tests with broken speed, otherwise these segments get ignored
rm_broken_speed_roads = True

# some configration has been moved to utils because cross imports created problems

coverages_1d_to_analyse = [utils.RoadDicConst.SPEED_BINS.value,
                        utils.RoadDicConst.STEERING_BINS.value,
                        utils.RoadDicConst.DISTANCE_BINS.value]
coverages_2d_to_analyse = [utils.RoadDicConst.SPEED_STEERING_2D.value,
                        utils.RoadDicConst.OBE_2D.value]
