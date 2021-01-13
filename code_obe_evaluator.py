# This code has been sourced from here: https://gitlab.infosun.fim.uni-passau.de/gambi/esec-fse-20/-/blob/master/code/obe-evaluator.py#L315
#
# Estimate Output Coverage and Entropy of the OBE caused by our tests. They goal is to assess the
# effectiveness of the test suites in terms of what type of OBE they generate.
#
# Possibly we can estimate the "diversity" of the generated tests
#
#   TODO: Assignm criticality values or scores to OBEs:
#    - criticality increases with speed (which indirectly tells how far the car went out the road)
#    - criticality increases with angle (which indirectly tells how badly the car went out the road)
#    - THIS IS NOT A METRIC OF COVERAGE, this SHOULD BE A WAY TO UNDERSTAND HOW MANY DIFFERENT OBES ARE ACTUALLY GENERATED
#       BY COMPARING APPROACHES. WE NEED SET THEORY ?
#
#
from os import listdir, path
from os.path import isfile, join

import sys
import matplotlib.pyplot as plt

import logging as l

import csv

import numpy as np
import numpy.linalg as la
import json
from asfault.tests import RoadTest, TestExecution, CarState
import math

from shapely.geometry import LineString

import evaluator_config as econf

# Mostly a conf parameter ?
# EPS = 1.2e-16
EPS = 1e-10


# GLOBAL Variables defining the bins/grid system to evaluate OBEs
max_theta = 360
theta_step = 10
theta_bins = [math.radians(angle) for angle in range(0, max_theta+theta_step, theta_step)]

# This corresponds to the speed limit + 10
max_obe_speed = 100
obe_speed_step = 10
speed_bins = [speed for speed in range(0, max_obe_speed+obe_speed_step, obe_speed_step)]


from abc import ABC, abstractmethod

class AbstractCriticalityEstimator(ABC):

    def __init__(self, obe):
        super().__init__()
        self.obe = obe

    @abstractmethod
    def compute_criticality(self):
        pass


class SimpleCriticalityEstimator(AbstractCriticalityEstimator):

    def __init__(self, obe, theta_bins, speed_bins, heading_angle_weight, speed_weight):
        super().__init__(obe)

        # Define the relative importance of each dimension
        self.speed_weight = speed_weight
        self.heading_angle_weight = heading_angle_weight

        # Define boundaries for normalization (min/max)
        self.theta_bins = theta_bins
        self.speed_bins = speed_bins

    def compute_criticality(self):
        # Criticality does not distinguish "left/right" heading angles so we need to report everything to 0 - 180 range
        angle = self.obe.get_heading_angle()[0] # ndarray ?
        if angle > math.radians(180):
            angle = math.radians(360) - angle

        # Normalize values speed and heading angle
        norm_deg = (angle - self.theta_bins[0]) / (self.theta_bins[-1] - self.theta_bins[0])
        # Note that here also relatively slow speeds (i.e., 20 KM/H) might be extremely critical, and criticality might not be linear
        # https://www.wired.com/2011/04/crashing-into-wall/
        # https://www.science.org.au/curious/technology-future/physics-speeding-cars
        # IMPACT ON A LARGE OBJECT
        # If, instead of hitting a pedestrian, the car hits a tree, a brick wall, or some other heavy object, then the car’s energy of motion
        #   (kinetic energy) is all dissipated when the car body is bent and smashed.
        #   Since the kinetic energy (E) is given by E=(1/2) mass*(speed*speed)
        # it increases as the square of the impact velocity.
        #
        # So we can say that criticality is QUADRATIC in SPEED.
        #
        norm_speed = (self.obe.get_speed() - self.speed_bins[0]) / (self.speed_bins[-1] - self.speed_bins[0])

        # Compute the criticality as weighted sum - SIMPLIFIED MODEL
        return self.speed_weight * norm_speed + self.heading_angle_weight * norm_deg


class OBE:
    """ Compute and store data about OBEs?"""

    BORDER = 10

    # THIS IS WRONG AND SHALL BE RENAMED TO to_simple_dict or something
    @staticmethod
    def to_dict(obe):
        ret = dict()
        ret['test_id'] = obe.test.test_id
        ret['obe_id'] = obe.id
        ret['speed'] = obe.get_speed()
        ret['heading_angle'] = obe.get_heading_angle()
        # 0 straing, -X left, +X right
        # We do not consider length of the road at the moment
        ret['road_angle'] = obe.get_road_angle()

        return ret

    @staticmethod
    def from_dict(obe_dict):
        # for each recorded obe we create the OBE element
        # TODO Do we really need the state_after_obe ?!
        return OBE(obe_dict['test'], obe_dict['obe_id'], obe_dict['state_before_obe'], obe_dict['obe_states'], obe_dict['segment_before_obe'])

    # This might be computed using a factory
    def __init__(self, test, id, state_before_obe, obe_states, segment_before_obe):
        """ States (tests.CarState objects) contain observation of position and velocity of the car @ OBE"""
        self.test = test
        self.id = id
        self.state_before_obe = state_before_obe
        self.obe_states = obe_states
        self.segment_before_obe = segment_before_obe

    def get_road_angle(self):
        from asfault.network import TYPE_STRAIGHT, TYPE_R_TURN, TYPE_L_TURN
        if self.segment_before_obe.roadtype == TYPE_STRAIGHT:
            return 0
        elif self.segment_before_obe.roadtype == TYPE_R_TURN:
            return math.fabs(self.segment_before_obe.angle)
        elif self.segment_before_obe.roadtype == TYPE_L_TURN:
            return - math.fabs(self.segment_before_obe.angle)
        else:
            return math.nan

    def get_velocity_at_obe(self):
        return self.obe_states[0].vel_x, self.obe_states[0].vel_y

    def get_velocity_before_obe(self):
        return self.state_before_obe.vel_x, self.state_before_obe.vel_y

    def get_speed(self):
        """Return the avg speed of obe in KM/H (3.6)"""
        return np.mean([self.state_before_obe.get_speed(), self.obe_states[0].get_speed()])*3.6

    def get_bounding_box(self):

        xs = [self.state_before_obe.pos_x, self.state_before_obe.pos_x + self.state_before_obe.vel_x]
              # self.state_after_obe.pos_x, self.state_after_obe.pos_x + self.state_after_obe.vel_x]

        ys = [self.state_before_obe.pos_y, self.state_before_obe.pos_y + self.state_before_obe.vel_y]
              # self.state_after_obe.pos_y, self.state_after_obe.pos_y + self.state_after_obe.vel_y]

        for state in self.obe_states:
            xs.extend([state.pos_x, state.pos_x + state.vel_x])
            ys.extend([state.pos_y, state.pos_y + state.vel_y])

        min_x = min(xs) - self.BORDER
        min_y = min(ys) - self.BORDER
        max_x = max(xs) + self.BORDER
        max_y = max(ys) + self.BORDER

        return [min_x, min_y, max_x, max_y]

    def dot(self, a, b):
        return np.sum(a * b, axis=-1)

    def mag(self, a):
        return np.sqrt(np.sum(a * a, axis=-1))

    def angle(self, a, b):
        cosab = self.dot(a, b) / (self.mag(a) * self.mag(b))  # cosine of angle between vectors
        angle = np.arccos(cosab)  # what you currently have (absolute angle)
        b_t = b[:, [1, 0]] * [1, -1]  # perpendicular of b
        # Note this is the opposite sign than S.O. question
        is_cc = self.dot(a, b_t) > 0
        # invert the angles for counter-clockwise rotations
        angle[is_cc] = 2 * np.pi - angle[is_cc]
        return angle

    def _pairs(self, lst):
        """ THere shoulf be something in default libs."""
        for i in range(1, len(lst)):
            yield lst[i - 1], lst[i]


    def get_heading_angle(self):
        """ Return the heading angle of the car before it went OBE """
        # https://stackoverflow.com/questions/56710732/how-to-efficiently-calculate-full-2pi-angles-between-three-vectors-of-2d-point
        # https://stackoverflow.com/questions/14066933/direct-way-of-computing-clockwise-angle-between-2-vectors
        # https://newtonexcelbach.com/2014/03/01/the-angle-between-two-vectors-python-version/
        # dot = x1*x2 + y1*y2      # dot product between [x1, y1] and [x2, y2]
        # det = x1*y2 - y1*x2      # determinant
        # angle = atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)

        # Obtain the right vectors from the various points
        A = np.array([[self.state_before_obe.pos_x , self.state_before_obe.pos_y]])
        B = np.array([[self.obe_states[0].pos_x , self.obe_states[0].pos_y]])

        AB = B - A
        AB = AB

        proj = self.state_before_obe.get_path_projection()

        # This might be useful to add to asfault directly
        # Process the path two points at the time. The path define the road direction
        # https://gis.stackexchange.com/questions/84512/get-the-vertices-on-a-linestring-either-side-of-a-point
        # plt.show()
        # x, y = self.state_before_obe.test.get_path_polyline().coords.xy
        # plt.plot(x, y, color="yellow")
        # plt.plot(proj.x, proj.y, marker="o", color="blue")

        road_direction = None
        for pair in self._pairs(list(self.state_before_obe.test.get_path_polyline().coords)):
            # Check whether this pair contains the projection which by definition lies on the path
            # print("Checking if ", pair[0], pair[1], "contains", proj)
            # x, y = LineString([pair[0], pair[1]]).coords.xy
            # plt.plot(x, y, color="red")
            # plt.plot(proj.x, proj.y, marker="o")
            # This does not work despite
            segment = LineString([pair[0], pair[1]])
            if segment.contains(proj) or segment.buffer(EPS).contains(proj):
                road_direction = np.array([pair[0]]) - np.array([pair[1]])
                break

        P = np.array([[proj.x, proj.y]])

        # TODO How to decide this w.r.t. direction of the road?
        # Direction of the road might be given by the PATH of the test
        N = np.array([[-(P[0][1] - A[0][1]), (P[0][0] - A[0][0])]])
        N1 = np.array([[(P[0][1] - A[0][1]), -(P[0][0] - A[0][0])]])

        if road_direction is not None:
            if self.dot(road_direction, N) > 0:
                N = N1
        else:
            l.warning("CANNOT FIND THE ROAD DIRECTION FOR %s", str(self.state_before_obe))


        # Vector defined by the last point before OBE and the first point of OBE
        # angle, is_cc = self.angle(N, AB)
        # print(angle, is_cc)

        angle = self.angle(AB, N)

        return angle[0]

    def plot_position_on(self,ax):
        ax.plot(self.state_before_obe.pos_x, self.state_before_obe.pos_y, marker="o", color='blue')
        for state in self.obe_states:
            ax.plot(state.pos_x, state.pos_y, marker="x", color='red')
        # ax.plot(self.state_after_obe.pos_x, self.state_after_obe.pos_y, marker="o", color='blue')

    def plot_velocity_on(self, ax):
        # Plot velocity
        ax.plot(
            [self.state_before_obe.pos_x, self.state_before_obe.pos_x + self.state_before_obe.vel_x],
            [self.state_before_obe.pos_y, self.state_before_obe.pos_y + self.state_before_obe.vel_y],
            color='blue')

        for state in self.obe_states:
            ax.plot(
                [state.pos_x, state.pos_x + state.vel_x],
                [state.pos_y, state.pos_y + state.vel_y],
                color='red')

        # ax.plot(
        #     [self.state_after_obe.pos_x, self.state_after_obe.pos_x + self.state_after_obe.vel_x],
        #     [self.state_after_obe.pos_y, self.state_after_obe.pos_y + self.state_after_obe.vel_y],
        #     color='blue')

    def plot_debug_data_on(self, ax):
        # from get_heading_angle
        proj = self.state_before_obe.get_path_projection()
        ax.plot(proj.x, proj.y, marker="s", color='black')
        # Norm vector defining the direction of the road
        # (-dy, dx)
        v21 = [-(proj.y - self.state_before_obe.pos_y), (proj.x - self.state_before_obe.pos_x)]
        # (dy, -dx)
        v22 = [(proj.y - self.state_before_obe.pos_y), -(proj.x - self.state_before_obe.pos_x)]
        # Scale the norm by a factor of 10
        scale = 10
        ax.plot([proj.x, proj.x + v21[0]*scale],
                [proj.y, proj.y + v21[1]*scale], color='black')
        ax.plot([proj.x, proj.x + v22[0]*scale],
                [proj.y, proj.y + v22[1]*scale], color='black')


class OBEEvaluator:

    def __init__(self, executions):
        self.obes = []
        self.obe_speed = []
        self.theta = []

        for execution in executions:
            obes = self._extract_obes_from_test(execution)

            self._fill_bins(execution)
            #
            self.obe_speed.extend([obe.get_speed() for obe in obes])
            # Angles must be given in radiants
            self.theta.extend([obe.get_heading_angle() for obe in obes])
            self.obes.extend(obes)

            # This is the same for each execution !
            self.bounds = execution.test.network.bounds

        # has to be activated in econf
        # writes road plot in the output/plots/ folder
        if econf.ALL_ROADS_WRITE:
            from asfault.plotter import TestPlotter
            from asfault.config import rg as asfault_environment
            for fi_execution in executions:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                fi_name = fi_execution.test.test_id
                # get the paths
                output_path = path.split(asfault_environment.get_execs_path())[0]
                plots_path = path.join(output_path, 'plots')
                road_plot_file = path.join(plots_path, str(fi_name) + ".png")
                asfault_plotter = TestPlotter(ax, "Test of a road", fi_execution.test.network.bounds)
                asfault_plotter.plot_test(fi_execution.test)
                # plt.show()
                plt.savefig(road_plot_file)

    def _fill_bins(self, execution):
        # TODO weg?
        speed_arr = []
        steering_arr = []
        distance_arr = []

        for state in execution.states:
            state_dict = CarState.to_dict(state)
            speed_arr.append(np.linalg.norm([state.vel_x, state.vel_y]) * 3.6)
            steering_arr.append(state_dict['steering'])
            distance_arr.append(state.get_centre_distance())

        #print("arrays for each feature: ", speed_arr, steering_arr, distance_arr)

        #bins = {'test_id': execution.test, 'speed_bins': [], 'steering_bins': [], "distance_bins": [],
        #        "speed_steering_2d": []}
        #print("bins: ", bins)


    def _extract_obes_from_test(self, execution):
        """ This one is inspired by the TestExecution"""

        obes = []
        obe_id = 0

        is_oob = False
        last_state = None
        current_obe = None
        # Ensure that we record OBEs also if we did not observe the end of it, like we do when we stop the search !

        obe_data = list()

        for state in execution.states:

            is_off_track = execution.off_track(state)

            if is_off_track:


                if not is_oob:
                    is_oob = True
                    obe_id += 1
                    current_obe = dict()
                    obe_data.append(current_obe)
                    current_obe['test'] = execution.test
                    current_obe['obe_id'] = obe_id
                    current_obe['obe_states'] = [state]
                    current_obe['state_before_obe'] = last_state
                    current_obe['segment_before_obe'] = last_state.get_segment()
                else:
                    current_obe['obe_states'].append(state)
            else:
                is_oob = False
                current_obe = None

            last_state = state

        for obe_dict in obe_data:
            obes.append(OBE.from_dict(obe_dict))

        if execution.oobs != len(obes):
            l.error("OBE Count disagree for test %s expected %d != actual %d",
                    execution.test.test_id, execution.oobs, len(obes))

        return obes

    def _plot_obe_road_segments(self, ax, obe):
        from asfault.plotter import TestPlotter
        # Get min/max x,y for setting the limits
        # Plot position
        bbox = obe.get_bounding_box()

        asfault_plotter = TestPlotter(ax, "OBE", self.bounds)
        asfault_plotter.plot_test(obe.test)

        ax.set_xlim(bbox[0], bbox[2])
        ax.set_ylim(bbox[1], bbox[3])

    def plot_obe(self, obe, theta_bins, speed_bins):
        # Plot roads and center around OBE points (visualize all the OBE)
        fig1 = plt.figure('obe', figsize=(6, 6))
        # Ensures old figure is cleared out
        fig1.clear()

        ax1 = plt.subplot()
        ax1.set_title(" ".join(["Angle", str("{:.2f}".format(math.degrees(obe.get_heading_angle()))),
                                "Speed", str("{:.2f}".format(obe.get_speed()))]))

        self._plot_obe_road_segments(ax1, obe)
        #
        obe.plot_position_on(ax1)
        obe.plot_velocity_on(ax1)
        obe.plot_debug_data_on(ax1)

        # Plot the polar graph
        fig2 = plt.figure('polar', figsize=(3, 3))
        # Ensures old figure is cleared out
        fig2.clear()

        ax2 = plt.subplot(projection='polar')

        ax2.scatter(obe.get_heading_angle(), obe.get_speed(), marker="o", color='black')
        self._set_polar_axes(ax2, theta_bins, speed_bins)

        # Plot the vectors (from Origin)
        A = np.array([[obe.state_before_obe.pos_x, obe.state_before_obe.pos_y]])
        B = np.array([[obe.obe_states[0].pos_x, obe.obe_states[0].pos_y]])
        AB = B - A

        X = [0]
        Y = [0]
        U = [AB[0][0]]
        V = [AB[0][1]]

        # Plot the NORM
        proj = obe.state_before_obe.get_path_projection()
        v21 = [-(proj.y - obe.state_before_obe.pos_y), (proj.x - obe.state_before_obe.pos_x)]

        X.append(0)
        Y.append(0)
        # SCALE THE NORM
        scale=10
        U.append(v21[0]*scale)
        V.append(v21[1]*scale)

        # Plot the vectors to debug if angle is right
        fig3 = plt.figure('vector', figsize=(3, 3))
        # Ensures old figure is cleared out
        fig3.clear()

        ax3 = plt.subplot()

        ax3.quiver(X, Y, U, V, color=['red', 'green'], angles='xy', scale_units='xy', scale=1)
        max_lim = max( max(U), max(V))
        ax3.set_xlim(-max_lim, max_lim)
        ax3.set_ylim(-max_lim, max_lim)
        ax3.set_aspect('equal')

        return 'obe', 'polar', 'vector'

    def plot_obes_distribution(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        ax.scatter(self.theta, self.obe_speed, marker="o", color='black')
        ax.set_title("OBEs distribution", va='bottom')
        # TODO If we do not explictly consider the direction of the road we cannot achieve more than 90-deg of OBE
        ax.set_thetamin(0)
        ax.set_thetamax(90)

        return fig

    def _adjust_lightness(self, color, amount=0.5):
        import matplotlib.colors as mc
        import colorsys
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

    def get_cell_color(self, cell_value):
        if cell_value > 0:
            return self._adjust_lightness('green', 1)
        else:
            return 'red'

    def _set_polar_axes(self, ax, theta_bins, speed_bins):
        # Set the grid according to the binning
        #
        # Check that theta_bins[0] == 0 and theta_bins[-1] ==360
        #
        ax.set_thetamin(math.degrees(theta_bins[0]))
        ax.set_thetamax(math.degrees(theta_bins[-1]))

        # TODO Use labels from -90/-180  to  +90/180
        # Show only a subset of labels
        ax.set_yticks(speed_bins[0::2])
        ax.set_xticks(theta_bins[0:-1:4])

        # Zero is in the middle of the graph
        ax.set_theta_zero_location("N")

    def plot_obe_coverage(self, theta_bins, speed_bins):
        coverage = np.zeros((len(theta_bins), len(speed_bins)))
        total_cells = (len(theta_bins) - 1) * (len(speed_bins) - 1)

        covered_cells = 0
        for obe_index in range(len(self.obes)): # Theta and obe_speed have the same lenght
            # Digitalize return the index of the bin, bins start from 1 not from 0
            # Sometimes it returns an array, sometimes an index, sometimes an array with one element only...
            i = (np.digitize(self.theta[obe_index], theta_bins, right=False) - 1)[0]
            j = np.digitize(self.obe_speed[obe_index], speed_bins, right=False) - 1

            # Mark
            if coverage[i][j] == 0:
                covered_cells += 1

            coverage[i][j] += 1

        # TOTAL COVERAGE
        global_coverage = (covered_cells / total_cells) * 100.0

        fig = plt.figure('coverage')
        fig.clear()
        ax = fig.add_subplot(111, projection='polar')

        self._set_polar_axes(ax, theta_bins, speed_bins)

        # https://stackoverflow.com/questions/10837296/shade-cells-in-polar-plot-with-matplotlib
        # The figure is created with bars, we draw longer bars before shorter ones
        # The bar must go at the top of the cell/interval not the bottom !
        for i in reversed(range(len(theta_bins)-1)):
             for j in reversed(range(len(speed_bins)-1)):
                 # print(i, ',' ,j, '=', coverage[i][j], math.degrees(theta_bins[i]), math.degrees(theta_bins[i+1]), '-', speed_bins[j], speed_bins[j+1])
                 color = self.get_cell_color(coverage[i][j])

                 # make color
                 ax.bar((theta_bins[i]+theta_bins[i+1])/2,
                        speed_bins[j+1], width=math.radians(theta_step), color=color,
                        edgecolor='black', zorder=-1)

        # This plots each single OBE - Why those should be "inside" the cells?
        ax.scatter(self.theta, self.obe_speed, marker="o", s=3, color='white', zorder=1)
        ax.set_title(' '.join(['OBEs',
                               'Coverage',
                               ''.join([str(covered_cells),'/', str(total_cells)]),
                               ''.join(['(', "{:.2f}".format(global_coverage), '%', ')'])]),
                     va='bottom')



        # Store to file
        return covered_cells, total_cells, 'coverage'




def _configure_asfault() -> None:
    from asfault.config import init_configuration, load_configuration
    from tempfile import TemporaryDirectory
    temp_dir = TemporaryDirectory(prefix="testGenerator")
    init_configuration(temp_dir.name)
    load_configuration(temp_dir.name)



def setup_logging(log_level):
    level = l.INFO
    if log_level == "DEBUG":
        level = l.DEBUG
    elif log_level == "WARNING":
        level = l.WARNING
    elif log_level == "ERROR":
        level = l.ERROR

    term_handler = l.StreamHandler()
    l.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                  level=level, handlers=[term_handler])

def generate_html_index(obe_data):
    from jinja2 import Template
    # TODO Fix ME
    html_index_template = Template("""
    <hmlt>
        <body>
            <table>
                {% for obe_dict in obe_data %}
                <tr>
                    <td>
                        <img src="{{ obe_dict['obe_plot_file'] }}" height="400" width="400">
                    </td>
                    <td>
                        <img src="{{ obe_dict['polar_plot_file'] }}" height="400" width="400">
                    </td>
                </tr>
                {% endfor %}
            </table>
        </body>
    </html>
    """)
    return html_index_template.render(obe_data=obe_data)



def main():
    # Local import to main
    import os
    import csv

    setup_logging(l.INFO)
    # ENV DIR
    #   Contains CONFIGURATION (TO LOAD)
    #   Contains EXEC folder to look for test executions
    env_directory = sys.argv[1]

    l.info("Start evaluation of OBEs from %s", env_directory)

    # Load the configuration from the given env and store in the "global" asfault configurations ev, ex
    from asfault.app import read_environment
    read_environment(env_directory)

    from asfault.config import rg as asfault_environment
    from asfault.config import ev as asfault_evolution_config
    # from asfault.config import ex as asfault_execution_config

    # Read all the execution data of this experiment
    l.info("Reading execution data from %s", str(asfault_environment.get_execs_path()))

    executions = list()

    for test_file_name in [f for f in listdir(asfault_environment.get_execs_path()) if isfile(join(asfault_environment.get_execs_path(), f))]:

        test_file = path.join(asfault_environment.get_execs_path(), test_file_name)

        # Load test object from file
        with open(test_file , 'r') as in_file:
            test_dict = json.loads(in_file.read())

        the_test = RoadTest.from_dict(test_dict)
        executions.append(the_test.execution)

    # Instantiate the OBE Evaluator
    obe_evaluator = OBEEvaluator(executions)

    obe_data = list()
    # TODO commented the file generation to save on space
    for global_id, obe in enumerate(obe_evaluator.obes):
        obe_dict = OBE.to_dict(obe)
        # Extend obe information with additional features
        obe_dict['global_id'] = global_id

        l.info("\tPlotting OBE %i %s", global_id, obe.test.test_id)
        # Return the id of the figures to chose which one to save to pdf
        obe_plot_id, polar_plot_id, vector_plot_id = obe_evaluator.plot_obe(obe, theta_bins, speed_bins)
        # Load the figure
        plt.figure(obe_plot_id)
        # Store it file
        obe_plot_file = os.path.abspath(path.join(asfault_environment.get_plots_path(), ''.join([str(global_id).zfill(3), '_', 'obe', '.png'])))
        obe_dict['obe_plot_file'] = obe_plot_file
        if econf.OBE_WRITE:
            plt.savefig(obe_plot_file)

        # Load the  next figure
        plt.figure(polar_plot_id)
        # Store it
        polar_plot_file = os.path.abspath(
            path.join(asfault_environment.get_plots_path(), ''.join([str(global_id).zfill(3), '_', 'polar', '.png'])))
        obe_dict['polar_plot_file'] = polar_plot_file
        if econf.OBE_WRITE:
            plt.savefig(polar_plot_file)

        obe_data.append(obe_dict)
    print("obe data ", obe_data)


    html_index_file= path.join(os.path.dirname(os.path.abspath(env_directory)), 'index.html')
    l.info("Generate HTML report %s", html_index_file)
    html_index = generate_html_index(obe_data)
    with open(html_index_file, 'w') as out:
        out.write(html_index )


    csv_file = path.join(os.path.dirname(os.path.abspath(env_directory)), '.obes')
    l.info("Generate CSV file %s", csv_file)
    # Taken from: https://www.tutorialspoint.com/How-to-save-a-Python-Dictionary-to-CSV-file
    # 'obe_id' is the unique id of the obe
    csv_columns = ['global_id', 'test_id', 'obe_id', 'speed', 'heading_angle', 'road_angle']
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns, extrasaction='ignore')
            writer.writeheader()
            for data in obe_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")


def old_main():
    # THIS MAIN METHOD ASSUMES THE FORMATE GENERATED BY SINGLE-OBJECTIVE BUT WE ARE WORKING WITH ASFAULT OUTPUTS
    setup_logging(l.INFO)

    # TODO This is wrong. We need to load the actual configuration used by AsFault
    # Ensures that the (default) conf is there


    test_path = sys.argv[1] # "./test/data/test-suite"
    result_path = sys.argv[2] # "./test/data/results"

    _configure_asfault()


    l.info("Start OBEs evaluation for:\n%s\n%s", test_path, result_path)

    test_files = [f for f in listdir(test_path) if isfile(join(test_path, f))]

    executions = []

    errors = False

    for test_file in test_files:
        # test_file == test-00.json
        # execution_file='result-test-00.json'
        t = path.join(test_path, test_file)
        e = path.join(result_path, "".join(['result-', test_file]))

        # Load test object from file
        with open(t, 'r') as in_file:
            test_dict = json.loads(in_file.read())

        the_test = RoadTest.from_dict(test_dict)

        # If the test contains already the execution data - common if we use lanedist-old-asfault we use it, otherwise
        #   we read it from the exec_dict file

        if the_test.execution:
            l.debug("EXECUTION DATA ALREADY INSIDE TEST")
            execution = the_test.execution
        else:
            with open(e, 'r') as in_file:
                execution_dict = json.loads(in_file.read())

            try:
                TestExecution.verify(test_dict, execution_dict)
            except AssertionError as error:
                # Plot the error message:
                l.error("Test %s and execution %s might not be compatible", t, e)
                # Plot the entire execution

                execution = TestExecution.from_dict(the_test, execution_dict)
                fig = plt.figure()
                ax = fig.add_subplot(111)
                asfault_plotter = TestPlotter(ax, "OBE", execution.test.network.bounds)
                asfault_plotter.plot_test(execution.test)
                asfault_plotter.plot_car_trace(execution.states)

                # Plot the offending states
                if error.args[0] == "Observed test start is too far away from expected test start":
                    from shapely.geometry import Point
                    test_start = Point(test_dict['start'][0], test_dict['start'][1])
                    test_start_buffer_x, test_start_buffer_y = test_start.buffer(10).exterior.xy
                    observed_start = Point(execution_dict['states'][0]['pos_x'], execution_dict['states'][0]['pos_y'])
                    plt.plot(test_start_buffer_x, test_start_buffer_x, color = "red")
                    plt.plot(test_start.x, test_start.y, color="red", marker="x")
                    plt.plot(observed_start.x, observed_start.y, color="blue", marker="o")

                elif error.args[0] == "Observed test end is too far away from expected test end":
                    from shapely.geometry import Point
                    test_goal = Point(test_dict['goal'][0], test_dict['goal'][1])
                    test_goal_buffer_x, test_goal_buffer_y = test_goal.buffer(10).exterior.xy
                    observed_goal = Point(execution_dict['states'][-1]['pos_x'], execution_dict['states'][-1]['pos_y'])

                    plt.plot(test_goal_buffer_x, test_goal_buffer_y, color="green")
                    plt.plot(test_goal.x, test_goal.y, color="green", marker="x")
                    plt.plot(observed_goal.x, observed_goal.y, color="blue", marker="o")


                error_plot_file = path.join(result_path, ''.join(['error_plot', str(test_dict['test_id']), '.pdf']))
                l.error("Check error plot %s", error_plot_file)
                plt.savefig(error_plot_file)
                errors = True

            if errors:
                sys.exit(-1)

            execution = TestExecution.from_dict(the_test, execution_dict)

        # Check whether the execution is valid (starting point of test must be the first state)
        executions.append(execution)

    obe_evaluator = OBEEvaluator(executions)

    # Store detailed values of criticality, angle, speed for OBEs. Maybe also segment at which the car went OBE?

    # Collect all the criticalities and other details about and plot

    obe_data = list()

    for id, obe in enumerate(obe_evaluator.obes):

        # Same importance of speed and angle
        criticality_estimator = SimpleCriticalityEstimator(obe, theta_bins, speed_bins, 1.0, 1.0)
        obe_criticality = criticality_estimator.compute_criticality()
        del criticality_estimator

        # Speed is five times more important
        criticality_estimator = SimpleCriticalityEstimator(obe, theta_bins, speed_bins, 1.0, 5.0)
        speed_more_critical_obe_criticality = criticality_estimator.compute_criticality()
        del criticality_estimator

        l.info("Plotting OBE %i %s", id, obe.test.test_id)

        # Return the id of the figures to chose which one to save to pdf
        obe_plot_id, polar_plot_id, vector_plot_id = obe_evaluator.plot_obe(obe, theta_bins, speed_bins)
        # Plot
        # Load the figure
        plt.figure(obe_plot_id)
        # Store it
        plt.savefig(path.join(result_path, ''.join([str(id).zfill(3), '_', 'obe','.pdf'])))
        # Load the figure
        plt.figure(polar_plot_id)
        # Store it
        plt.savefig(path.join(result_path, ''.join([str(id).zfill(3), '_', 'polar', '.pdf'])))
        # Load the figure
        plt.figure(vector_plot_id)
        # Store it
        plt.savefig(path.join(result_path, ''.join([str(id).zfill(3), '_', 'vector', '.pdf'])))

        obe_plot_location = path.join(result_path, ''.join([str(id).zfill(3), '_', 'obe','.pdf']))

        obe_data.append([id, obe.test.test_id, obe.get_speed(),
                        obe.get_heading_angle()[0], # ndarray for whatever reason...
                         obe_plot_location, obe_criticality,
                         speed_more_critical_obe_criticality])

    f = open(path.join(result_path, 'obes.csv'), 'w')

    with f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["OBE_ID", "Test_ID", "Speed", "Heading Angle", "Plot", "Criticality", "Speed Criticality"])

        # Write data
        for row in obe_data:
            writer.writerow(row)


    l.debug("Computing OBE Distribution/Coverage")

    # fig = obe_evaluator.plot_obes_distribution(executions)
    covered_cells, total_cells, coverage_plot_id = obe_evaluator.plot_obe_coverage(theta_bins, speed_bins)
    global_coverage = covered_cells / total_cells

    # Load the figure
    plt.figure(coverage_plot_id)
    # Store it
    plt.savefig(path.join(result_path, 'obe_coverage.pdf'))

    # Save to .coverage result file
    coverage_file = path.join(result_path, '.coverage')
    with open(coverage_file, 'w') as out_file:
        out_file.write(",".join([str(covered_cells), str(total_cells), str(global_coverage)]))
        out_file.write('\n')
        #
        out_file.flush()

if __name__ == "__main__":
    main()

