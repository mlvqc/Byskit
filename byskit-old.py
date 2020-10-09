import numpy as np
import matplotlib.pyplot as plt
from qiskit import *

# First princinple for two parent nodes and one child
class byskit():
    def __init__(self, provider, backend, n, parents, child):
        self.provider = provider
        self.backend = backend
        self.parents = parents
        self.child = child
        self.n = n
        self.ctrl = QuantumRegister(self.n, 'ctrl')
        self.anc = QuantumRegister(self.n - 1, 'anc')
        self.tgt = QuantumRegister(1, 'tgt')
        self.circ = QuantumCircuit(self.ctrl, self.anc, self.tgt)

        self.parent_init()
        self.child_init()

        self.circ.draw(output='mpl')
        plt.show()

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
            theta = self.calc_theta(self.child[2*i+1], self.child[2*i])
            self.xgate(self.gates[i])
            self.cn_ry(theta)
            self.xgate(self.gates[i])
            self.circ.barrier()

    def xgate(self,gate):
        for index, item in enumerate(gate):
            if int(item) == 0:
                self.circ.x(index)

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


#if __name__=='__main__':
from jupyterthemes import jtplot

jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False)

from qiskit.tools.jupyter import *
from qiskit import IBMQ

IBMQ.load_account()
# provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
provider = IBMQ.get_provider(hub='ibm-q-oxford', group='on-boarding', project='on-boarding-proj')
from qiskit import BasicAer
backend = BasicAer.get_backend('unitary_simulator')

n = 3
parents = np.random.rand(n*2)
child = np.random.rand(2**(n+1))

b = byskit(provider,backend,n,parents,child)