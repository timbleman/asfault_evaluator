from sklearn.cluster import OPTICS, compute_optics_graph
import suite_behaviour_computer
import utils

class Clusterer:
    def __init__(self, data_dict: dict):
        self.data_dict = data_dict

    def perform_optics(self, measure: str):
        dist_matrix = utils.list_matrix_measure(data_dict=self.data_dict, measure=measure)

        assert len(dist_matrix) == len(dist_matrix[0]), "The distance matrix has to be a square!"
        print("dist_matrix", dist_matrix)
        clust = OPTICS(eps=0.3, metric='precomputed')
        clust.fit(dist_matrix)
        print("clust.ordering_", clust.ordering_)
        print("clust.xi", clust.xi)
        print("clust.labels_", clust.labels_)
        print("clust.cluster_hierarchy_", clust.cluster_hierarchy_)
        print("clust.core_distances_", clust.core_distances_)

        import matplotlib as plt
        #ordering, _, _, _ = compute_optics_graph(X=dist_matrix, metric="precomputed")
        #print("ordering", ordering)

    def networkx_plot_measure(self, measure: str, draw_node_names = True, draw_graphweights: bool = True, draw_edges: bool = False):
        """ Plots all specimens of the main dict. A measure is selected for the edge weights.
        Graphvis tries to get the optimal arrangement based on weights, hard to do for 2d space.
        Use dot, neato and fdp seem to maximize distances and just create a circular arrangement.

        :param measure: distance measure for the edges
        :return: None
        """
        import networkx as nx
        from networkx.drawing.nx_agraph import graphviz_layout
        import matplotlib.pyplot as plt

        dist_matrix_dict = utils.dict_of_dicts_matrix_measure(data_dict=self.data_dict, measure=measure)
        print("dist_matrix_dict", dist_matrix_dict)
        G = nx.from_dict_of_dicts(d=dist_matrix_dict)
        G.graph['edges']={'arrowsize':'4.0'}
        # neato, fdp seem to mazimize --> circle shape
        # dot works best, however, 2d representation is difficult
        pos = graphviz_layout(G, prog="fdp")
        if draw_edges:
            nx.draw_networkx(G, pos=pos)
        else:
            nx.draw_networkx_nodes(G, pos=pos)
        if draw_graphweights:
            nx.draw_networkx_edge_labels(G, pos=pos)
        if draw_node_names:
            nx.draw_networkx_labels(G, pos)
        plt.tight_layout()
        # adjust the [x, y] values for fitting all the text
        axes = plt.gca()
        #axes.set_xlim([-60, 800])
        # use these to turn off axis for plotting
        #plt.xticks([])
        #plt.yticks([])
        plt.show()

    def networkx_plot_measure_spring_equal_length_edges(self, measure: str):
        import networkx as nx
        import matplotlib.pyplot as plt

        dist_matrix_dict = utils.dict_of_dicts_matrix_measure(data_dict=self.data_dict, measure=measure)
        print("dist_matrix_dict", dist_matrix_dict)
        G = nx.from_dict_of_dicts(d=dist_matrix_dict)
        G.graph['edges']={'arrowsize':'4.0'}
        pos = nx.spring_layout(G)
        nx.draw(G, pos)
        plt.show()

    def networkx_plot_measure_strange_list(self, measure: str):
        import networkx as nx
        import matplotlib.pyplot as plt

        dist_matrix_dict = utils.dict_of_lists_matrix_measure(data_dict=self.data_dict, measure=measure)
        print("dist_matrix_dict", dist_matrix_dict)
        G = nx.from_dict_of_lists(d=dist_matrix_dict)
        G.graph['edges']={'arrowsize':'4.0'}
        nx.draw(G, with_labels = True)
        plt.show()
