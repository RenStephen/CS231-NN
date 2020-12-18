import numpy as np
from random import shuffle
# from past.builtins import xrange
import sys

if sys.version_info >= (3, 0):
    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))


def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  # np.zeros(W.shape)?????

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  classes=W.shape[1]
  num_images=X.shape[0]

  # want to loop through the test images
  for i in range(num_images):

    # every image in X looks like this 0:[x,x,x,x...3072.....x,x,x,x,x] get the shape of this image with the weights
    # in doing so we are getting the value of this weight matrix for this image, what we're predicting
    # this is oging to look like a [1,C] matrix with floats in each index
    scores=X[i].dot(W)

    # for overflow, so we dont get huge numbers
    scores -= scores.max()

    # this will help determine our probabilities
    denominator = np.sum(np.exp(scores))

    # this gets what this image is supposed to be 
    correct_class = y[i]
    numerator=np.exp(scores[correct_class])

    # adds ths final softmax value to the loss value
    loss += -np.log( numerator /denominator )

    # how the fuck do i calculate the gradient?
    dW[:, correct_class] += (-1) * (denominator - numerator) / denominator * X[i]
    for j in range(classes):
        # pass correct class gradient
        if j == correct_class:
            continue
        # for incorrect classes
        dW[:, j] += np.exp(scores[j]) / denominator * X[i]
      
  # at this ppoint we will have got our total loss and we need to regularize it
  rW=reg*np.sum(W*W)
  loss=loss/num_images+rW
  dW /= num_images
  dW += 2 * reg * W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass


  # loss function W=DxC X=NxD scores = XW = (NxD)(DxC)=NxC
  N=X.shape[0]
  scores=X.dot(W)
  scores-=scores.max()
  scores=np.exp(scores)
  score_sums=np.sum(scores,axis=1)

  correct_scores=[]
  for i in range(N):
    correct_scores.append(scores[i,y[i]])

  correct_scores=np.array(correct_scores)
  loss=-np.sum(np.log(correct_scores/score_sums))/N+reg*np.sum(W*W)

  # now we need to calculate the gradient on the softmax loss function
  

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

