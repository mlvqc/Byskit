import numpy as np
import matplotlib.pyplot as plt
from qiskit import *
from qiskit.aqua.algorithms import Grover


# First princinple for two parent nodes and one child
class byskit():
    def __init__(self, backend, parents, child, evd = None):
        self.backend = backend
        self.parents = parents
        self.child = child
        self.n = int(np.shape(parents)[0]/2)
        self.n_child = np.shape(child)[1]
        self.ctrl = QuantumRegister(self.n, 'ctrl')
        self.anc = QuantumRegister(self.n - 1, 'anc')
        self.tgt = QuantumRegister(self.n_child, 'tgt')
        if evd != None:
            self.oracle = QuantumRegister(evd,'oracle')
            self.circ = QuantumCircuit(self.ctrl, self.anc, self.tgt, self.oracle)
        else:
            self.circ = QuantumCircuit(self.ctrl, self.anc, self.tgt)

        #self.c_ctrl = ClassicalRegister(self.n, 'c_ctrl')
        #self.c_tgt = ClassicalRegister(self.n_child, 'c_tgt')

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

    def rejection_sampling(self, evidence, shots=1000, amplitude_amplification=False):
        # Run job many times to get multiple samples
        samples_list = []
        self.n_samples = shots

        if amplitude_amplification==True:
            self.amplitude_amplification(evidence)

        self.circ.measure_all()

        #self.circ.measure((self.ctrl, self.tgt),(self.c_ctrl, self.c_tgt))

        for i in range(self.n_samples):
            job = execute(self.circ, backend=self.backend, shots=1)
            result = list(job.result().get_counts(self.circ).keys())[0]
            accept = True
            for e in evidence:
                if result[evidence[e]['n']]==evidence[e]['state']:
                    pass
                else:
                    accept=False
            if accept == True:
                #print('Accepted result ', result)
                samples_list.append(result)

        print()
        print(self.n_samples, 'samples drawn:', len(samples_list), 'samples accepted,', self.n_samples - len(samples_list),
              'samples rejected.')
        print('Percentage of samples rejected: ', 100 * (1 - (len(samples_list) / self.n_samples)), '%')

        return samples_list


    def evaluate(self, samples_list, observations):
        p_o = 0
        for sample in samples_list:
            accept = True
            for o in observations:
                if sample[observations[o]['n']] == observations[o]['state']:
                    pass
                else:
                    accept = False
            if accept == True:
                #print('Observation true given evidence')
                p_o += 1
        p_o /= len(samples_list)

        print('Probabilty of observations given evidence = ', p_o)

        return p_o

    def amplitude_amplification(self,evidence):
        self.state_preparation = self.circ
        self.oracle = QuantumCircuit(self.ctrl, self.anc, self.tgt)
        for index, e in enumerate(evidence):
            if evidence[e]['state'] == '1':
                self.oracle.z([evidence[e]['n']])

        self.grover_op = Grover(self.oracle, state_preparation=self.state_preparation)
        self.grover_op.draw()

    def oracle(self):
        pass

    def u_gate(self):
        pass

def gen_random_weights(n_parent,n_child):
    np.random.seed(0)
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
    #provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
    provider = IBMQ.get_provider(hub='ibm-q-oxford', group='on-boarding', project='on-boarding-proj')
    from qiskit import Aer #BasicAer
    #backend = BasicAer.get_backend('unitary_simulator')
    backend = Aer.get_backend('qasm_simulator')

    n_parent = 3
    n_child = 3

    parents, children = gen_random_weights(n_parent, n_child)
    b = byskit(backend, parents, children)
    b.plot()

    evidence = {
        'one':{
            'n':1,
            'state':'1'
        }
    }
    #b.rejection_sampling(evidence,amplitude_amplification=True)
    sample_list = b.rejection_sampling(evidence)

    observations = {
        'three':{
            'n':2,
            'state':'0'
        }
    }

    prob = b.evaluate(sample_list, observations)