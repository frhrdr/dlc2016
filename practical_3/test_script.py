import numpy as np
import cifar10_siamese_utils

ctest = cifar10_siamese_utils.get_cifar10('./cifar10/cifar-10-batches-py').test
d = cifar10_siamese_utils.create_dataset(source_data=ctest, num_tuples=3,
                                         batch_size=10, fraction_same=0.2)
for u in d[0]:
    print u.shape

