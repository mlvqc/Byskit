import numpy as np


class Vertex:

    def __init__(self, id, probabilities, child=False):
        # Identifier of the node
        self.id = id

        # Nodes that are parents of this
        self.parents = []

        # Nodes that are children of this
        self.children = []

        if child:
            assert not np.any(np.sum(probabilities, axis=1) - 1)

        self.probabilities = probabilities

    def __str__(self):
        return self.id

    def add_child(self, child):
        self.children.extend(child)

    def add_parent(self, parent):
        self.parents.extend(parent)

    def get_direct_children(self):
        return self.children

    def get_all_children(self):
        # recursively get_direct_children
        pass


class Bayesnet:

    def __init__(self):
        self.vertex_dictionary = {}
        self.number_vertices = 0

    def __iter__(self):
        return iter(self.vertex_dictionary.values())

    def __call__(self, id):
        return self.vertex_dictionary[id]

    def add_root(self, name, probabilities):
        new_root = Vertex(name, probabilities)
        self.vertex_dictionary[name] = new_root

    def get_vertex(self, node):
        if node in self.vertex_dictionary:
            return self.vertex_dictionary[node]
        else:
            return None

    def add_child(self, id, parents, probability_table):
        child = Vertex(id, probability_table, child=True)

        child.add_parent(parents)

        self.vertex_dictionary[id] = child
        self.number_vertices += 1

        for parent in parents:
            self.vertex_dictionary[parent].add_child(child.id)


if __name__ == '__main__':
    net = Bayesnet()

    net.add_root('A', [0.2, 0.8])
    net.add_root('B', [0.3, 0.7])

    net.add_child('C', ['A', 'B'], np.array([[0.15, 0.3, 0.4, 0.1], [0.85, 0.7, 0.6, 0.9]]).T)

    print(net('C').parents)
    print(net('A').probabilities[0])

    print("conditional prob of child: ", net('C').probabilities[0, 0])
