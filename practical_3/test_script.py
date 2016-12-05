import numpy as np
import cifar10_utils

cifar10 = cifar10_utils.get_cifar10('./cifar10/cifar-10-batches-py')
y = cifar10.test.labels
f_out = open('./logs/cifar10/test_labels', 'w+')
np.save(f_out, y)
f_out.close()
