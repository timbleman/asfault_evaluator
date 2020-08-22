#
# Test File for Obe-Evaluator
#
""" # staged for removal
from os import listdir, path
import matplotlib.pyplot as plt
import json
from asfault.tests import RoadTest, TestExecution

import code_obe_evaluator

def test():
    # turn protected _configure_asfault() again
    code_obe_evaluator.configure_asfault()

    # test_path = "/Users/gambi/esec-fse-20/data/beamng-ai-curvature-large/001/test-suite" #"./test/data/test-suite"
    # result_path = "/Users/gambi/esec-fse-20/data/beamng-ai-curvature-large/001/results/beamng" #"./test/data/resul
    # ts"
    # test_path = "/Users/gambi/esec-fse-20/data/loiacono-curvature-huge/002/test-suite"
    # result_path = "/Users/gambi/esec-fse-20/data/loiacono-curvature-huge/002/results/deepdrive"

    test_path = "/Users/gambi/esec-fse-20/data/loiacono-speed-large/003/test-suite"
    result_path = "/Users/gambi/esec-fse-20/data/loiacono-speed-large/003/results/beamng"

    # test_path = "/Users/gambi/Dropbox/AsFault/tests/WEIRD-TEST-FROM-loiacono-speed-large-003"
    # result_path = "/Users/gambi/Dropbox/AsFault/Results/BeamNG-WEIRD-TEST-FROM-loiacono-speed-large-003/"

    #

    executions = []
    # test_files = [f for f in listdir(test_path) if isfile(join(test_path, f))]
    test_files = ['test-00.json']

    for test_file in test_files:
        t = path.join(test_path, test_file)
        e = path.join(result_path, "".join(['result-', test_file]))

        # Load test object from file
        with open(t, 'r') as in_file:
            test_dict = json.loads(in_file.read())

        the_test = RoadTest.from_dict(test_dict)

        with open(e, 'r') as in_file:
            execution_dict = json.loads(in_file.read())


        #TestExecution.verify(test_dict, execution_dict)

        execution = TestExecution.from_dict(the_test, execution_dict)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        asfault_plotter = TestPlotter(ax, "OBE", execution.test.network.bounds)
        asfault_plotter.plot_test(execution.test)
        asfault_plotter.plot_car_trace(execution.states)

        fig.clear()



        executions.append(execution)

    # Extract obes from the executions
    obe_evaluator = code_obe_evaluator.OBEEvaluator(executions)

    # obe_evaluator.plot_obes_distribution()
    obe_evaluator.plot_obe_coverage()
    #
    # Plot an OBE
    for obe in obe_evaluator.obes:
        obe_evaluator.plot_obe(obe)

if __name__ == "__main__":
    #main()
     test()

"""