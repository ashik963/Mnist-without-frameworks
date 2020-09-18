from random import seed
from random import random
import numpy as np
import argparse
from numpy import genfromtxt

parser = argparse.ArgumentParser()

# hyperparameters setting
parser.add_argument('train',  type=str,help='training image set' )
parser.add_argument('trainlabel', type=str, help='training label set')
parser.add_argument('test',  type=str,help='Testing image set' )
parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
parser.add_argument('--epochs', type=int, default=150,help='number of epochs to train')
parser.add_argument('--n_x', type=int, default=784, help='number of inputs')
parser.add_argument('--n_h1', type=int, default=200,help='number of hidden units 1')
parser.add_argument('--n_h2', type=int, default=100,help='number of hidden units 2')
parser.add_argument('--beta', type=float, default=0.9,help='parameter for momentum')
parser.add_argument('--batch_size', type=int,default=64, help='input batch size')

def sigmoid(z):
    """
    sigmoid activation function.

    inputs: z
    outputs: sigmoid(z)
    """
    s = 1. / (1. + np.exp(-z))
    return s

## cross entropy loss
def compute_loss(Y, Y_hat):
    """
    compute loss function
    """
    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1./m) * L_sum

    return L

def feed_forward(X, params):
    """
    feed forward network: 2 - layer neural net

    inputs:
        params: dictionay a dictionary contains all the weights and biases

    return:
        cache: dictionay a dictionary contains all the fully connected units and activations
    """
    cache = {}

    # Z1 = W1.dot(x) + b1
    cache["Z1"] = np.matmul(params["W1"], X) + params["b1"]

    # A1 = sigmoid(Z1)
    cache["A1"] = sigmoid(cache["Z1"])
    
    # Z2 = W2.dot(x) + b2
    cache["Z2"] = np.matmul(params["W2"], cache["A1"]) + params["b2"]

    # A2 = sigmoid(Z2)
    cache["A2"] = sigmoid(cache["Z2"])

    
    # Z2 = W2.dot(A1) + b2
    cache["Z3"] = np.matmul(params["W3"], cache["A2"]) + params["b3"]

    # A2 = softmax(Z2)
    cache["A3"] = np.exp(cache["Z3"]) / np.sum(np.exp(cache["Z3"]), axis=0)

    return cache

def back_propagate(X, Y, params, cache, m_batch):
    """
    back propagation

    inputs:
        params: dictionay a dictionary contains all the weights and biases
        cache: dictionay a dictionary contains all the fully connected units and activations

    return:
        grads: dictionay a dictionary contains the gradients of corresponding weights and biases
    """
    # error at last layer
    dZ3 = cache["A3"] - Y

    # gradients at last layer (Py2 need 1. to transform to float)
    dW3 = (1. / m_batch) * np.matmul(dZ3, cache["A2"].T)
    db3 = (1. / m_batch) * np.sum(dZ3, axis=1, keepdims=True)
    
    # back propgate through first layer
    dA2 = np.matmul(params["W3"].T, dZ3)
    dZ2 = dA2 * sigmoid(cache["Z2"]) * (1 - sigmoid(cache["Z2"]))

    # gradients at first layer (Py2 need 1. to transform to float)
    dW2 = (1. / m_batch) * np.matmul(dZ2,  cache["A1"].T)
    db2 = (1. / m_batch) * np.sum(dZ2, axis=1, keepdims=True)
    
    # back propgate through first layer
    dA1 = np.matmul(params["W2"].T, dZ2)
    dZ1 = dA1 * sigmoid(cache["Z1"]) * (1 - sigmoid(cache["Z1"]))

    # gradients at first layer (Py2 need 1. to transform to float)
    dW1 = (1. / m_batch) * np.matmul(dZ1, X.T)
    db1 = (1. / m_batch) * np.sum(dZ1, axis=1, keepdims=True)
    
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2,"dW3": dW3, "db3": db3}

    return grads

def _main_(opt):
    train_data=genfromtxt(opt.train, delimiter=',')
    train_label=genfromtxt(opt.trainlabel, delimiter=',')
    test_data=genfromtxt(opt.test, delimiter=',')
    x_train=np.float32(train_data)
    x_test=np.float32(test_data)
    y_train = np.int32(train_label).reshape(-1, 1)
    # one-hot encoding
    digits = 10
    examples = y_train.shape[0]
    y = y_train.reshape(1, examples)
    Y_new = np.eye(digits)[y.astype('int32')]
    Y_new = Y_new.T.reshape(digits, examples)
    # training set
    X_train, X_test = x_train.T, x_test.T
    Y_train = Y_new
    # initialization
    params = {"W1": np.random.randn(opt.n_h1, opt.n_x) * np.sqrt(1. / opt.n_x),
          "b1": np.zeros((opt.n_h1, 1)) * np.sqrt(1. / opt.n_x),
          "W2": np.random.randn(opt.n_h2, opt.n_h1) * np.sqrt(1. / opt.n_h1),
          "b2": np.zeros((opt.n_h2, 1)) * np.sqrt(1. / opt.n_h1),
          "W3": np.random.randn(digits, opt.n_h2) * np.sqrt(1. / opt.n_h2),
          "b3": np.zeros((digits, 1)) * np.sqrt(1. / opt.n_h2)}
    
    
    for i in range(opt.epochs):
        seed(1234)
        # shuffle training set
        permutation = np.random.permutation(X_train.shape[1])
        X_train_shuffled = X_train[:, permutation]
        Y_train_shuffled = Y_train[:, permutation]

        for j in range(opt.batch_size):

            # get mini-batch
            begin = j * opt.batch_size
            end = min(begin + opt.batch_size, X_train.shape[1] - 1)
            X = X_train_shuffled[:, begin:end]
            Y = Y_train_shuffled[:, begin:end]
            m_batch = end - begin

            # forward and backward
            cache = feed_forward(X, params)
            grads = back_propagate(X, Y, params, cache, m_batch)

            # with momentum (optional)
            dW1 = (opt.beta * grads["dW1"] + (1. - opt.beta) * grads["dW1"])
            db1 = (opt.beta * grads["db1"] + (1. - opt.beta) * grads["db1"])
            dW2 = (opt.beta * grads["dW2"] + (1. - opt.beta) * grads["dW2"])
            db2 = (opt.beta * grads["db2"] + (1. - opt.beta) * grads["db2"])
            dW3 = (opt.beta * grads["dW3"] + (1. - opt.beta) * grads["dW3"])
            db3 = (opt.beta * grads["db3"] + (1. - opt.beta) * grads["db3"])

            # gradient descent
            params["W1"] = params["W1"] - opt.lr * dW1
            params["b1"] = params["b1"] - opt.lr * db1
            params["W2"] = params["W2"] - opt.lr * dW2
            params["b2"] = params["b2"] - opt.lr * db2
            params["W3"] = params["W3"] - opt.lr * dW3
            params["b3"] = params["b3"] - opt.lr * db3

        # forward pass on training set
        cache = feed_forward(X_train, params)
        train_loss = compute_loss(Y_train, cache["A3"])
        print("Epoch {}: training loss = {}".format(i + 1, train_loss))
    cache = feed_forward(X_test, params)
    np.savetxt('test_predictions.csv',np.argmax(cache["A3"],axis=0).reshape(-1,1), delimiter=",",fmt='% 4d')

if __name__ == '__main__':
    opt = parser.parse_args()
    _main_(opt)