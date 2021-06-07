CSE293: Chisel Neural Net
=======================

Contributors: Reese Levine, Chris Liu

Project Summary
======================
This project implements a neural net in hardware using Chisel. The goal of this project is to allow users to define a neural
network in a compositional way. Layers can be stacked sequentially to create arbitrary neural networks. Users interact with
the net through the `NeuralNet` module, which implements a state machine for training and inference.

Parameterization
=====================
Users can specify many parameters. The overall neural net can be parameterized by the input/output sizes, the width and precision of data,
the number of training epochs, and the maximum number of training samples it needs to be able to hold. Additionally, the neural net
takes as parameters a sequence of layers, which will perform the actual computation. Each layer can be parameterized by its own
input/output sizes, and fully connected layers can be parameterized by their learning rate.

State Machine Summary
======================
The neural net starts in the ready state. Two boolean inputs control the next state; predict transitions to the predicting/inference state,
while train transfers the neural net to the training state. In order to transfer states, a decoupled parameter
named `numSamples` must also be valid, which tells the net how many samples to train/predict on.

### Training State
In the training state, the neural net first goes through `numSamples` cycles to write training and validation data to local
memory. This way, the net can perform multiple rounds of training without the user having to re-input data. After
loading all data, the net translates back and forth between forward and backpropagation states until all training
epochs have concluded. The number of cycles this takes varies depending on the number of layers in the net. After
training is complete, the net transfers back to the ready state.

### Predicting state
In the predicting state, the neural net takes in one sample at a time, performs forward propagation, and returns the predicted
value to the user using the decoupled `result` output. The `result` only stays valid for one cycle, at which point the net performs
prediction on the next sample, or returns to the ready state if there are no more samples.


Project Tasks
====================
- Initial project setup: done
- Fully connected layer initialization and forward propagation (and initial tests): done
- Fully connected layer backpropagation: done 
- Neural net, with parameterized layers: done
- Activation layer: done
- Convolutional layer: Future Work

Possible optimizations
===================
- Hardware sharing across subsequent layers
- More cycles to compute matrix/vector operations


