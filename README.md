# asfault_evaluator

This code extracts road geometry and agent behaviour from an asfault executions dataset.
string_comparison.py turns the road segments into shape definition language and computes all distances. The distance measures (sliding window, longest common substring, longest common subsequence, longest common substring with k mismatches, jaccard) are defined in string_comparison.py and utils.py.
utils.py also includes the function for comparing the behaviour bins, counting and binary difference can be used.

Currently there is no user or command line interface, the behaviour has to be adjusted in main.py: The path to parent directory has to be set and the functions that should be performed have to be added.

This is still work in progress, code may be changed and comments will be added!
