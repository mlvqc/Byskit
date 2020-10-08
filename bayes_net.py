import numpy as np


class Vertex:

    def __init__(self, node):
        # Identifier of the node
        self.node = node

        # Nodes that are parents of this
        self.parents = {}

        # Nodes that are children of this
        self.children = {}

    def add_child(self, node, probabilities):
        assert sum(np.array(probabilities)) == 1

        self.children[node] = probabilities

    def get_direct_children(self):
        return self.children.keys()

    def get_all_children(self):
        # recursively get_direct_children
        pass

    def get_child_probabilities(self, node):
        return self.children[node]


class Bayesnet:

    def __init__(self):
        self.vertex_dictionary = {}
        self.number_vertices = 0

    def __iter__(self):
