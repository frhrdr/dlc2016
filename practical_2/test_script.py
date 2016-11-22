import tensorflow as tf

print(tf.constant(True) == True)
print(tf.constant(True) == tf.constant(True))
print(tf.equal(tf.constant(True), tf.constant(True)))
