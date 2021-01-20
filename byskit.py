import numpy as np
import matplotlib.pyplot as plt
from qiskit import *
from qiskit.aqua.algorithms import Grover


# First princinple for two parent nodes and one child
class byskit():
    def __init__(self, backend,network, loaded_net, evd = None):
        self.backend = backend
        self.network = network
        self.net_keys = [key for key in self.network]
        self.loaded_net = loaded_net
        self.reg = {}

        self.create_circ()

        self.root_init()

        child_index = np.array([0,0])
        parent_index = np.array([0, 0])
        for index in range(len(self.net_keys)-1):
            parent_key = self.net_keys[index]
            child_key = self.net_keys[index+1]
            if parent_key != 'root':
                parent_index = np.array([parent_index[1], parent_index[1] + self.network[self.net_keys[index + 1]]])
            child_index = np.array([child_index[1],child_index[1]+self.network[self.net_keys[index+1]]])
            self.child_init(parent_key,parent_index,child_key,child_index)

    def create_circ(self):
        self.n_anc = 0
        self.n_tgt = 0
        for key in self.network:
            if key == 'root':
                n = self.network['root']
                self.reg['cntrl'] = QuantumRegister(self.network['root'], 'cntrl')

            else:
                self.n_anc = max(n-1,self.n_anc)
                self.n_tgt += self.network[key]
                n = self.network[key]
        self.reg['anc'] = QuantumRegister(self.n_anc,'anc')
        self.reg['tgt'] = QuantumRegister(self.n_tgt, 'tgt')

        self.circ = QuantumCircuit(self.reg['cntrl'],self.reg['anc'],self.reg['tgt'])


    def root_init(self):
        for i in range(self.network['root']):
            theta = self.calc_theta(self.loaded_net['root'][2*i], self.loaded_net['root'][2*i+1])
            self.circ.ry(theta, i)

        self.circ.barrier()

    def child_init(self,parent_key,parent_index,child_key,child_index):
        parent_index = parent_index[0]
        child_index = child_index[0]
        self.a = np.arange(0, 2 ** self.network[parent_key])
        self.gates = []
        for i in self.a:
            s = str(np.binary_repr(i, width=self.network[parent_key]))
            self.gates.append(s)

        for i in range(2**self.network[parent_key]):
            self.xgate(self.gates[i],parent_index)
            for j in range(self.network[child_key]):
                theta = self.calc_theta(self.loaded_net[child_key][2 * i + 1,j], self.loaded_net[child_key][2 * i,j])
                self.cn_ry(theta,j,parent_key,parent_index,child_key,child_index)
            self.xgate(self.gates[i],parent_index)
            self.circ.barrier()

    def xgate(self,gate,parent_index):
        for index, item in enumerate(gate):
            if int(item) == 0:
                self.circ.x(index+parent_index)

    #RY gates
    def cn_ry(self,theta,target,parent_key,parent_index,child_key,child_index):
        # compute
        if parent_key == 'root':
            self.circ.ccx(self.reg['cntrl'][0], self.reg['cntrl'][1], self.reg['anc'][0])
            for i in range(2, self.network[parent_key]):
                self.circ.ccx(self.reg['cntrl'][i], self.reg['anc'][i - 2], self.reg['anc'][i - 1])

            # copy
            self.circ.cry(theta,self.reg['anc'][self.network[parent_key] - 2], self.reg['tgt'][target])

            # uncompute
            for i in range(self.network[parent_key] - 1, 1, -1):
                self.circ.ccx(self.reg['cntrl'][i], self.reg['anc'][i - 2], self.reg['anc'][i - 1])
            self.circ.ccx(self.reg['cntrl'][0], self.reg['cntrl'][1], self.reg['anc'][0])

        else:
            self.circ.ccx(self.reg['tgt'][parent_index+0], self.reg['tgt'][parent_index+1], self.reg['anc'][0])
            for i in range(2, self.network[parent_key]):
                self.circ.ccx(self.reg['tgt'][parent_index+i], self.reg['anc'][i - 2], self.reg['anc'][i - 1])

            # copy
            self.circ.cry(theta,self.reg['anc'][self.network[parent_key] - 2], self.reg['tgt'][child_index+target])

            # uncompute
            for i in range(self.network[parent_key] - 1, 1, -1):
                self.circ.ccx(self.reg['tgt'][parent_index+i], self.reg['anc'][i - 2], self.reg['anc'][i - 1])
            self.circ.ccx(self.reg['tgt'][parent_index+0], self.reg['tgt'][parent_index+1], self.reg['anc'][0])

    def calc_theta(self,p1,p0):
        return 2 * np.arctan(np.sqrt((p1)/(p0)))

    def plot(self):
        self.circ.draw(output='mpl')
        plt.show()

    def execute_circ(self):
        self.circ.measure_all()
        results = execute(self.circ, self.backend, shots=4321)
        return results

    def rejection_sampling(self, evidence, shots=5000, amplitude_amplification=False):
        # Run job many times to get multiple samples
        samples_list = []
        self.n_samples = shots

        if amplitude_amplification==True:
            self.amplitude_amplification(evidence)

        self.circ.measure_all()

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

def gen_random_net(network):
    np.random.seed(0)
    loaded_net = {}
    for key in network:
        if key == 'root':
            n_parent = network[key]
            p = np.random.rand(n_parent)
            parents = []
            for i in p:
                parents.append(i)
                parents.append(1 - i)
            loaded_net[key] = np.array(parents)

        else:
            n_child = network[key]
            child = np.random.rand(2 ** (n_parent + 1), n_child)

            for i in range(n_child):
                for j in range(2 ** (n_parent)):
                    child[2 * j + 1, i] = 1 - child[2 * j, i]

            loaded_net[key] = child

            n_parent = n_child

    return loaded_net


if __name__=='__main__':
    from qiskit import IBMQ
    IBMQ.load_account()
    #provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
    provider = IBMQ.get_provider(hub='ibm-q-oxford', group='on-boarding', project='on-boarding-proj')
    from qiskit import Aer #BasicAer
    #backend = BasicAer.get_backend('unitary_simulator')
    backend = Aer.get_backend('qasm_simulator')

    #network = {'root':2,'child-1':3,'child-2':3,'child-3':2}
    network = {'root':2,'child-1':3,'child-2':3}

    loaded_net = gen_random_net(network)
    b = byskit(backend, network, loaded_net)
    b.plot()

    evidence = {
        'one':{
            'n':1,
            'state':'1'
        },
        'two':{
            'n':5,
            'state':'0'
        }
    }
    #b.rejection_sampling(evidence,amplitude_amplification=True)
    sample_list = b.rejection_sampling(evidence, shots=1000,amplitude_amplification=False)

    observations = {
        'one':{
            'n':2,
            'state':'0'
        },
        'two': {
            'n': 4,
            'state': '1'
        }
    }

    prob = b.evaluate(sample_list, observations)


from qiskit import IBMQ
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')

from qiskit import Aer
backend = Aer.get_backend('qasm_simulator')

network = {'root':2,'child-1':3,'child-2':3}
loaded_net = gen_random_net(network)
b = byskit(backend, network, loaded_net)
b.plot()

evidence = {
    'one':{
        'n':1,
        'state':'1'
    },
    'two':{
        'n':5,
        'state':'0'
    }
}

sample_list = b.rejection_sampling(evidence, shots=1000, amplitude_amplification=True)

observations = {
    'one':{
        'n':2,
        'state':'0'
    },
    'two': {
        'n': 4,
        'state': '1'
    }
}

prob = b.evaluate(sample_list, observations)