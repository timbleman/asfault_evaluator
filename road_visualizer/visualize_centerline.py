from road_visualizer.road_visualizer import RoadVisualizer
from pathlib import Path
from shapely.geometry import LineString
from math import floor
import os

def visualize_centerline(coords_lstr: LineString, road_width: float, path: Path = None):
    # fixme breaks box plots
    z_val = 0.01

    def _calculate_parallel_coords(coords_lstr: LineString, offset: float) \
            -> LineString:
        original_line = coords_lstr
        try:
            offset_line = original_line.parallel_offset(offset)
            coords = offset_line.coords.xy
        except (NotImplementedError, Exception):
            return None
        # NOTE The parallel LineString may have a different number of points than initially given
        num_coords = len(coords[0])
        #z_vals = repeat(0.01, num_coords)
        #marking_widths = repeat(line_width, num_coords)
        #return list(zip(coords[0], coords[1], z_vals, marking_widths))
        return offset_line

    left_line = _calculate_parallel_coords(coords_lstr, offset=road_width/2)
    right_line = _calculate_parallel_coords(coords_lstr, offset=-road_width/2)
    coords_l = left_line.coords.xy
    coords_m = coords_lstr.coords.xy
    coords_r = right_line.coords.xy
    len_m = len(coords_m[0])
    left_to_middle = float(len(coords_l[0])) / len_m
    right_to_middle = float(len(coords_r[0])) / len_m

    """ # plotting the whole LineStrings works flawlessly 
    import matplotlib.pyplot as plt
    plt.plot(coords_l[0], coords_l[1])
    plt.plot(coords_m[0], coords_m[1])
    plt.plot(coords_r[0], coords_r[1])
    plt.ylabel("road")
    plt.show()
    """

    #print("lenghts", len(coords_lstr.coords.xy[0]), len(left_line.coords.xy[0]), len(right_line.coords.xy[0]))
    #print("coords", coords_lstr.coords.xy)

    # debugging, own matplot
    xlistl, ylistl = [], []
    xlistm, ylistm = [], []
    xlistr, ylistr = [], []

    # initialize coords list for i = 0
    x_l, y_l = coords_l[0][0], coords_l[1][0]
    x_m, y_m = coords_m[0][0], coords_m[1][0]
    x_r, y_r = coords_r[0][0], coords_r[1][0]
    dicc = {'right': [x_r, y_r, z_val], 'left': [x_l, y_l, z_val], 'middle': [x_m, y_m, z_val]}
    coords_list = [dicc]

    xlistl.append(x_l)
    ylistl.append(y_l)
    xlistm.append(x_m)
    ylistm.append(y_m)
    xlistr.append(x_r)
    ylistr.append(y_r)

    for i in range(1, len_m-1):
        index_l = floor(left_to_middle*i)
        index_r = floor(right_to_middle*i)

        x_l, y_l = coords_l[0][index_l], coords_l[1][index_l]

        x_m, y_m = coords_m[0][i], coords_m[1][i]

        x_r, y_r = coords_r[0][index_r], coords_r[1][index_r]

        xlistl.append(x_l)
        ylistl.append(y_l)
        xlistm.append(x_m)
        ylistm.append(y_m)
        xlistr.append(x_r)
        ylistr.append(y_r)

        dicc = {'right': [x_r, y_r, z_val], 'left': [x_l, y_l, z_val], 'middle': [x_m, y_m, z_val]}
        coords_list.append(dicc)

    # append end
    x_l, y_l = coords_l[0][-1], coords_l[1][-1]
    x_m, y_m = coords_m[0][-1], coords_m[1][-1]
    x_r, y_r = coords_r[0][-1], coords_r[1][-1]
    dicc = {'right': [x_r, y_r, z_val], 'left': [x_l, y_l, z_val], 'middle': [x_m, y_m, z_val]}
    coords_list.append(dicc)

    xlistl.append(x_l)
    ylistl.append(y_l)
    xlistm.append(x_m)
    ylistm.append(y_m)
    xlistr.append(x_r)
    ylistr.append(y_r)

    import matplotlib.pyplot as plt
    plt.plot(xlistl, ylistl)
    plt.plot(xlistm, ylistm)
    plt.plot(xlistr, ylistr)
    plt.ylabel("trimmed list")
    plt.show()

    #print("coords_list", coords_list)

    rv = RoadVisualizer(coords_list)
    rv.plot()

    html_file=os.path.join('.', 'testplot.html')

    rv.store_to(html_file)
