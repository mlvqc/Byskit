import numpy as np
import matplotlib.pyplot as plt
from qiskit import *
import matplotlib.pyplot as plt


n = 5  # must be >= 2

ctrl = QuantumRegister(n, 'ctrl')
anc = QuantumRegister(n-1, 'anc')
tgt = QuantumRegister(1, 'tgt')

circ = QuantumCircuit(ctrl, anc, tgt)

# compute
circ.ccx(ctrl[0], ctrl[1], anc[0])
for i in range(2, n):
    circ.ccx(ctrl[i], anc[i-2], anc[i-1])

# copy
circ.cx(anc[n-2], tgt[0])

# uncompute
for i in range(n-1, 1, -1):
    circ.ccx(ctrl[i], anc[i-2], anc[i-1])
circ.ccx(ctrl[0], ctrl[1], anc[0])

circ.draw(output='mpl')
plt.show()

'''ctrl = QuantumRegister(n, 'ctrl')
anc = QuantumRegister(n-1, 'anc')
tgt = QuantumRegister(1, 'tgt')

circ = QuantumCircuit(ctrl, anc, tgt)

circ.mct(ctrl, tgt[0], anc, mode='basic-dirty-ancilla')
circ.draw(output='mpl')
plt.show()
'''
