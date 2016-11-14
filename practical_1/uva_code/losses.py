import numpy as np
"""
This module implements various losses for the network.
You should fill in code into indicated sections. 
"""

def HingeLoss(x, y):
  """
  Computes hinge loss and gradient of the loss with the respect to the input for multiclass SVM.

  Args:
    x: Input data.
    y: Labels of data. 

  Returns:
    loss: Scalar hinge loss.
    dx: Gradient of the loss with the respect to the input x.

  """
  ########################################################################################
  # TODO:                                                                                #
  # Compute hinge loss on input x and y and store it in loss variable. Compute gradient  #
  # of the loss with respect to the input and store it in dx variable.                   #
  ########################################################################################
  loss = 1 - x - y
  loss = np.maximum(loss, np.zeros(loss.shape))

  xt = np.asarray([x[k,y[k]] for k in range(x.shape[0])])
  xt_mat = np.zeros(x.shape)
  v = np.zeros(x.shape)
  dx = np.zeros(x.shape)

  for idx, val in enumerate(y):
    xt_mat[idx, val] = 1
    v[idx, val] = np.sum(xt[idx, :])
  dx = np.sum(-y * (loss > 0), 1)
  ########################################################################################
  #                              END OF YOUR CODE                                        #
  ########################################################################################

  return loss, dx

def CrossEntropyLoss(x, y):
  """
  Computes cross entropy loss and gradient with the respect to the input.

  Args:
    x: Input data.
    y: Labels of data. 

  Returns:
    loss: Scalar cross entropy loss.
    dx: Gradient of the loss with the respect to the input x.
  
  """
  ########################################################################################
  # TODO:                                                                                #
  # Compute cross entropy loss on input x and y and store it in loss variable. Compute   #
  # gradient of the loss with respect to the input and store it in dx.                   #
  ########################################################################################
  
  xt = np.asarray([x[k,y[k]] for k in range(x.shape[0])])
  
  dx = np.zeros(x.shape)
  for idx, val in enumerate(y):
    dx[idx, val] = -1
  loss = - np.mean(np.log(xt))
  ########################################################################################
  #                              END OF YOUR CODE                                        #
  ########################################################################################

  return loss, dx


def SoftMaxLoss(x, y):
  """
  Computes the loss and gradient with the respect to the input for softmax classfier.

  Args:
    x: Input data.
    y: Labels of data. 

  Returns:
    loss: Scalar softmax loss.
    dx: Gradient of the loss with the respect to the input x.

  """
  ########################################################################################
  # TODO:                                                                                #
  # Compute softmax loss on input x and y and store it in loss variable. Compute gradient#
  # of the loss with respect to the input and store it in dx variable.                   #
  ########################################################################################
  xexp = np.exp(x)
  sum_exp = np.sum(xexp, 1)
  log_sum_exp = np.log(sum_exp)
  xt = np.asarray([x[k,y[k]] for k in range(x.shape[0])])
  log_softmax_t = xt - log_sum_exp
  loss = - np.mean(log_softmax_t)
  sum_exp_stack = np.stack([sum_exp for k in range(x.shape[1])], axis=1)
  softmax = xexp / sum_exp_stack
  dx = softmax
  for idx, val in enumerate(y):
    dx[idx, val] -= 1
  dx = dx / x.shape[0]
  ########################################################################################
  #                              END OF YOUR CODE                                        #
  ########################################################################################

  return loss, dx

