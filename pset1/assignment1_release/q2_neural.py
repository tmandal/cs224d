import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    # data : N x Dx
    # W1 : Dx x H
    # b1 : 1 x H
    # W2 : H x Dy
    # b2 : 1 x Dy
    #labels : N x Dy
    N = data.shape[0]
    Z1 = data.dot(W1) + b1  # Z1 : N x H
    Hout = sigmoid(Z1)      # Hout : N x H
    Z2 = Hout.dot(W2) + b2  # Z2 : N x Dy
    Y = softmax(Z2)         # Y : N x Dy
    cost_vec = - labels * np.log(Y) # N x Dy
    cost = np.sum(cost_vec)
    ### END YOUR CODE
    
    ### YOUR CODE HERE: backward propagation
    gradZ2 = Y - labels    # N x Dy
 
    gradb2 = gradZ2     # N x Dy

    # [h1 h2 ...hH] . [w11 w12 .. w1Dy]
    #                 [w21 w22 .. w2Dy]
    #                 [...............]
    #                 [wH1 wH2 .. wHDy]
    # = [H(:).W(:,1) H(:).W(:,2) ... H(:).W(:,Dy)]
    # = [h1*w11 + h2*w21 + ... h1*w12 + h2*w22 + ... ... h1*w1Dy + h2*w2Dy + ...]
    gradW2 = np.zeros((N, H, Dy))  # N x H x Dy
    for i in xrange(N):
        gradW2[i] = np.dot(Hout[i, :].reshape(-1, 1), gradZ2[i, :].reshape(1, -1))

    gradHout = gradZ2.dot(W2.T) # N x H

    gradZ1 = sigmoid_grad(Hout) * gradHout    # N x H

    gradb1 = gradZ1

    gradW1 = np.zeros((N, Dx, H))  # N x Dx x H
    for i in xrange(N):
        gradW1[i] = np.dot(data[i, :].reshape(-1, 1), gradZ1[i, :].reshape(1, -1))

    gradW1 = np.sum(gradW1, axis=0)
    gradb1 = np.sum(gradb1, axis=0)
    gradW2 = np.sum(gradW2, axis=0)
    gradb2 = np.sum(gradb2, axis=0)
    ### END YOUR CODE
    
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), 
        gradW2.flatten(), gradb2.flatten()))
    
    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using 
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
        dimensions), params)

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
