# asfault_evaluator

Preliminaries:
    Autonomous driving agents are usually tested using simulated roads. 
    This project aims to investigate correlations between similarity of road shapes and test outcome as well as driving behavior.
    This code extracts road geometry and car output from asfault executions datasets. 
   
-------------- 
Prerequisites:
--------------
    - Windows 10
    - Python 3.6
    - BeamNG and beamngpy, see https://github.com/BeamNG/BeamNGpy
    - AsFault, see https://github.com/alessiogambi/AsFault
    
------------
Installation
------------
    - It is easiest to clone the repository in C:/asfault_evaluator. 
        AsFault execution datasets and folders are included in /suites/. 
        If these are placed in another location, the "upper_dir" in evaluator_config.py has to be changed.
    - AsFault: An AsFault folder with a virtual environment is included in /asfault2/. 
        If the environment does not work, create a new one using the provided requirements. 
        It is important, that the environment is included in the /asfault2/ folder, 
        otherwise usage as a library does not work. AsFault itself has not been modified.
    
-----
Usage
-----
This Python implementation is used to create data that is written into csv-files and then analyzed using R.
Basic functions to replicate thesis experiments are simplified by a basic command line interface. This includes metric computation for the used sets and (adaptive) random sampling of subsets.
The interface has been created via click and help can be accessed by :code:`python main.py --help`. The virtual environment inside the AsFault folder has to be activated.
For my thesis, I used two different datasets created using the drivers BeamNG.AI and DriverAI. These can be evaluated using :code:`python main.py run --suite=drvr --woOBE`. The suites are either bng or drvr, --woOBE only runs successful tests. This step needs to be repeated for different alphabet configurations, as explained below.
Multiple adaptive random sampling (--ars) and random sampling (--rs) subsets are created using :code:`python main.py run --suite=drvr --ars --subsetnum=5`. For ARS, there are four configurations, either with OBE test as start point and with high and low diversity. The number of subsets per configuration defaults to the maximum of 5. To increase this, more start points would need to be added to main.py.
25 different subsets were used in my thesis. To transfer only the created csvs more easily, :code:`python main.py copy --parentPath="C:\asfault_ev_test\suites\div_drvr5"` can be used. 

Additional configuration is possible in evalutor_config.py. Important settings are:
    1. :code:`upper_dir = r"C:\Users\fraun\exp-ba"` This is the path where all the suites are located in.
    2. :code:`ALPHABET_SIZE = 28` This varies the number of angle symbols. Changing the length alphabet is possible in string_comparison.py.
    3. :code:`coverages_1d_to_analyse`, :code:`coverages_2d_to_analyse`, :code:`output_metrics_to_analyse`, :code:`string_metrics_to_analyse`, :code:`numeric_vals_to_write`. By commenting entries of these lists it can be selected which metrics to compute and write out.
    
Longest common substring with k mismatches requires adjustment in sdl_all_to_all_unoptimized in string_comparison.py. These results do not play a big role in the thesis.
Documentation is provided for almost all functions.

To analyse generated data using the R scripts, the csvs have to be reachable by R.

---------------
Troubleshooting
---------------
    - Datasets fail to load, pathlib errors: Windows only supports paths of length 260 by default. 
        As the execution datasets have very long names, it is suggested to not place those in 
        a complex folder structure, but instead as close to C:/ as possible.
    - Shapely refuses to install on Windows: Download an unofficial binary here: 
        https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely. 
        Make sure to select the right version of Python.

