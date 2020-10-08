import numpy as np
import matplotlib.pyplot as plt
from qiskit import *


class hardcode_figure5:

    def __init__(self, provider, backend, parents, child):



# First princinple for two parent nodes and one child
class byskit():
    def __init__(self, provider, backend, parents, child):
        self.provider = provider
        self.backend = backend
        self.parents = parents
        self.child = child
        self.nqubits = np.shape(self.parents)[0]+1
        self.c = QuantumCircuit(self.nqubits)

        for index, item in enumerate(self.parents):
            theta = self.calc_theta(item[0],item[1])
            self.c.ry(theta,index)

        self.c.barrier()
        self.c.x(0)
        self.c.x(1)

        for index, item in enumerate(self.child):
            theta = self.calc_theta(item[0],item[1])
            self.c.cry(theta,[0,1],2)


        self.c.draw(output='mpl')
        plt.show()

        self.n = 6
        self.ctrl = QuantumRegister(self.n, 'ctrl')
        self.anc = QuantumRegister(self.n - 1, 'anc')
        self.tgt = QuantumRegister(1, 'tgt')
        theta = 1
        self.circ = QuantumCircuit(self.ctrl, self.anc, self.tgt)
        self.cn_ry(theta)
        self.circ.draw(output='mpl')
        plt.show()

    #RY gates
    def cn_ry(self,theta):
        # compute
        self.circ.ccx(self.ctrl[0], self.ctrl[1], self.anc[0])
        for i in range(2, self.n):
            self.circ.ccx(self.ctrl[i], self.anc[i - 2], self.anc[i - 1])

        # copy
        self.circ.cry(theta,self.anc[self.n - 2], self.tgt[0])

        # uncompute
        for i in range(self.n - 1, 1, -1):
            self.circ.ccx(self.ctrl[i], self.anc[i - 2], self.anc[i - 1])
        self.circ.ccx(self.ctrl[0], self.ctrl[1], self.anc[0])


    def calc_theta(self,p1,p0):
        return 2 * np.arctan(np.sqrt((p1)/(p0)))


if __name__=='__main__':
    from jupyterthemes import jtplot

    jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False)

    from qiskit.tools.jupyter import *
    from qiskit import IBMQ

    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
    # provider = IBMQ.get_provider(hub='ibm-q-oxford', group='on-boarding', project='on-boarding-proj')
    from qiskit import BasicAer
    backend = BasicAer.get_backend('unitary_simulator')
    a0 = 0.2
    a1 = 0.8
    b0 = 0.3
    b1 = 0.7
    c000 = 0.15
    c001 = 0.3
    c010 = 0.4
    c011 = 0.1
    c100 = 0.85
    c101 = 0.7
    c110 = 0.6
    c111 = 0.9

    parent = [[a0,a1],[b0,b1]]
    child1 = [[c000,c100],[c001,c101],[c010,c110],[c011,c111]]


    b = byskit(provider,backend,parent,child1)

    print(np.shape(b.parents))
    print(np.shape(b.child))