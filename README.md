# Recurrent Neural Network with LSTM Cells, in pure Python
A vanilla implementation of a Recurrent Neural Network (RNN) with Long-Short-Term-Memory cells, without using any ML libraries.

## Background

These networks are particularly good for learning long-term dependencies within data, and can be applied to a variety of problems including language modelling, translation and speech recognition.

An LSTM cell has 4 gates, based on the following formulas:

<img src="https://latex.codecogs.com/svg.latex?\large&space;\begin{align*}&space;input^t&space;&=&space;\sigma(W_ix^t&space;&plus;&space;U_ih^{t-1})&space;\\&space;forget^t&space;&=&space;\sigma(W_fx^t&space;&plus;&space;U_fh^{t-1})&space;\\&space;output^t&space;&=&space;\sigma(W_ox^t&space;&plus;&space;U_oh^{t-1})&space;\\&space;g^t&space;&=&space;\tanh(W_gx^t&space;&plus;&space;U_gh^{t-1})&space;\\&space;\end{align*}" title="\large \begin{align*} i^t &= \sigma(W_ix^t + U_ih^{t-1}) \\ f^t &= \sigma(W_fx^t + U_fh^{t-1}) \\ o^t &= \sigma(W_ox^t + U_oh^{t-1}) \\ g^t &= \tanh(W_gx^t + U_gh^{t-1}) \\ \end{align*}" />

Each gate has it's own set of paramaters to learn, which makes training vanilla implementations (such as this one) expensive.

These are collected into a single cell state value:

<img src="https://latex.codecogs.com/svg.latex?\large&space;\begin{align*}&space;cell^t&space;=&space;input^t&space;\odot&space;g^t&space;&plus;&space;forget^t&space;\odot&space;cell^{t-1}&space;\end{align*}" title="\large \begin{align*} cell^t = input^t \odot g^t + forget^t \odot cell^{t-1} \end{align*}" />

This is then given to a hidden state, as a normal RNN cell would: LSTM cells can effectively be treated no differently to any other cell within the network.

## Training and Initialisation

To initialise the network, create an instance of the class by calling the constructor with the arguments:

```python
rnn = new LSTM_RNN(lr, in_dim, h_dim, out_dim)
```

Where `lr` is the learning rate; `in_dim` is the dimension of the input layer; `h_dim` is the dimension of the hidden layer and `out_dim` is the dimension of the output layer. These should correspond to your training data.

The training data should be encoded as integers, and given as two lists: a list of inputs and a corresponding one of targets. The RNN can then be trained by calling the function:

```python
rnn.train(iterations, inputs, targets)
```
