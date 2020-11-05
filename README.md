# Byskit

A compiler for automatically turning simple classical Bayesian networks into quantum circuits to run on Qiskit. 

Development team: [Sebastian Orbell](https://www.linkedin.com/in/sebastian-orbell-57541b193/), [Joseph Hickie](https://www.linkedin.com/in/joseph-hickie/) and [Brandon Severin](https://brandonseverin.carrd.co/) from the [Natalia Ares Group](https://www.natalia-ares.com/) at the University of Oxford.

Requirements: [Anaconda Python](https://www.anaconda.com/products/individual), [Qiskit](https://qiskit.org/)

See more: https://sebastianorbell.github.io/byskit.html

## Summary

> "Would you risk it for a quantum biscuit?"

Byskit plants an initial step towards taking full advantage of the amplitude amplification when running discrete Bayesian networks on a quantum computer. It provides the ability to easily translate a simple classical discrete Bayesian network to its corresponding quantum circuit. Previously these circuits were drawn manually. By ‘simple’ we mean the network has two generations i.e. one set of parents and one set of children. Future iterations could configure quantum circuits for arbitrarily complex Bayesian networks.

To have a play download the code and checkout the "Test notebook.ipynb" Jupyter Notebook.

## Background 

To quickly get everyone up to speed we will first explain what a Bayesian network is. 

### Classical Bayesian Networks 

A Bayesian network is a probabilistic model that graphically represents random variables and the conditional dependencies between them. The model is based on a directed acyclic graph with nodes representing random variables and the directed edges between them representing their conditional dependencies. These dependencies are modelled with conditional probability distributions or tables. 

These probability distributions can either be specified manually or learned using data. Once they are populated, we can use the network to make inferences of different types. For instance, we can make predictions about outcomes based on a set of inputs, or we can make diagnoses where we infer what inputs may have led to an observed outcome. These inferences can be exact or approximate. 

The main graph in this slide is simple, with a child and two roots. However, Bayesian networks can be arbitrarily complex - see the example to the right. Performing exact inference on bayesian networks is sharp-P hard, meaning they become classically intractable as they grow very large. We can use approximate inference to avoid this, but this is still NP hard generally. Approximate inference uses other techniques such as Markov Chain Monte Carlo sampling or rejection sampling methods to speed up inference. 

### Quantum Implementation

By implementing a bayesian network on quantum hardware, it is possible to obtain a square root speedup of rejection sampling of a Bayesian network. This is based on the principle of amplitude amplification, which is the generalisation of Grover’s algorithm, to facilitate a quantum version of rejection sampling. Quantum rejection sampling has been implemented and demonstrated on toy problems such as stock price prediction and bankruptcy protection. 

The probability of each node’s outcome is mapped to a probability amplitude in a qubit by preparing each one in a specified state. Where in the graph representation we would propagate the probabilities through each layer, in the quantum hardware implementation we use qubit gates to change the states of control qubits and subsequently read out the measurement qubit to take a sample. 

## Algorithm Workflow

First we start out with the drawing of the classical network. Then to generate the quantum implementation of the network Byskit: 

1. Maps each node to a qubit (this one to one mapping is easy if each node has only two states). 
2. Maps the marginal/ conditional probabilities of each node to the probability amplitudes associated with qubit states.
3. Composes $$C^{n}R_{Y}\theta$$ rotation gates, using ancilla qubits, to achieve the required probability amplitudes.

## Use Cases

We keep the use cases of any Bayesian network but obtain the benefit of the heft of the quantum system. This enables us to run Bayesian networks with large numbers of nodes and take advantage of the computational speed up provided by quantum computers. Here we have highlighted some typical issues that may fall into this category. Computational drug design, a specific example would be looking at the probabilities of particular protein mutations in antibodies not being rejected and attacked by the human immune system while also being effective at binding with the target pathogen. Diagnostics and decision automation, this could be biological or physical when the feature sets are large. 

## Looking Forward

Byskit is at the moment a simple implementation. Bayesian networks can be much more complex in the number of parents and children in addition to how the generations link with each other. Before we can go and tackle the world’s hardest problems we will still need to add the functionality of being able to compile more complex Bayesian networks. We will then interface Byskit with popular Bayesian network libraries such as tensorflow-probability. 

