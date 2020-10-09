import numpy as np
import matplotlib.pyplot as plt
from qiskit import *

# First princinple for two parent nodes and one child
class byskit():
    def __init__(self, backend, parents, child):
        self.backend = backend
        self.parents = parents
        self.child = child
        self.n = int(np.shape(parents)[0]/2)
        self.n_child = np.shape(child)[1]
        self.ctrl = QuantumRegister(self.n, 'ctrl')
        self.anc = QuantumRegister(self.n - 1, 'anc')
        self.tgt = QuantumRegister(self.n_child, 'tgt')
        self.circ = QuantumCircuit(self.ctrl, self.anc, self.tgt)

        self.parent_init()
        self.child_init()

    def parent_init(self):
        for i in range(self.n):
            theta = self.calc_theta(self.parents[2*i], self.parents[2*i+1])
            self.circ.ry(theta, i)

        self.circ.barrier()

    def child_init(self):
        self.a = np.arange(0, 2 ** self.n)
        self.gates = []
        for i in self.a:
            s = str(np.binary_repr(i, width=self.n))
            self.gates.append(s)

        for i in range(2**self.n):
            self.xgate(self.gates[i])
            for j in range(self.n_child):
                theta = self.calc_theta(self.child[2 * i + 1,j], self.child[2 * i,j])
                self.cn_ry(theta,j)
            self.xgate(self.gates[i])
            self.circ.barrier()

    def xgate(self,gate):
        for index, item in enumerate(gate):
            if int(item) == 0:
                self.circ.x(index)

    #RY gates
    def cn_ry(self,theta,target):
        # compute
        self.circ.ccx(self.ctrl[0], self.ctrl[1], self.anc[0])
        for i in range(2, self.n):
            self.circ.ccx(self.ctrl[i], self.anc[i - 2], self.anc[i - 1])

        # copy
        self.circ.cry(theta,self.anc[self.n - 2], self.tgt[target])

        # uncompute
        for i in range(self.n - 1, 1, -1):
            self.circ.ccx(self.ctrl[i], self.anc[i - 2], self.anc[i - 1])
        self.circ.ccx(self.ctrl[0], self.ctrl[1], self.anc[0])


    def calc_theta(self,p1,p0):
        return 2 * np.arctan(np.sqrt((p1)/(p0)))

    def plot(self):
        self.circ.draw(output='mpl')
        plt.show()

    def execute_circ(self):
        self.circ.measure_all()
        results = execute(self.circ, self.backend, shots=4321)
        return results

def gen_random_weights(n_parent,n_child):
    p = np.random.rand(n_parent)
    parents = []
    for i in p:
        parents.append(i)
        parents.append(1 - i)
    parents = np.array(parents)

    child = np.random.rand(2 ** (n_parent + 1), n_child)
    for i in range(n_child):
        for j in range(2 ** (n_parent)):
            child[2 * j + 1, i] = 1 - child[2 * j, i]

    return parents, child

if __name__=='__main__':
    from qiskit import IBMQ

    IBMQ.load_account()
    # provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
    provider = IBMQ.get_provider(hub='ibm-q-oxford', group='on-boarding', project='on-boarding-proj')
    from qiskit import BasicAer
    backend = BasicAer.get_backend('unitary_simulator')

    n_parent = 2
    n_child = 3

    parents,children = gen_random_weights(n_parent,n_child)
    b = byskit(backend,parents,children)
    b.plot()
