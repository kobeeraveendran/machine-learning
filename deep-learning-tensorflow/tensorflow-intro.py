import tensorflow as tf 

# to get rid of the CPU-related AVX2 instruction warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# to check if tensorflow is using GPU acceleration as it should
# output should be what device is being used (in my case, a GPU)
sess = tf.Session(config = tf.ConfigProto(log_device_placement = True))
sess.close()

x1 = tf.constant(5)
x2 = tf.constant(6)

# output is a tensor for both methods
# not as efficient
# result = x1 * x2

# faster version
result = tf.multiply(x1, x2)

# opening and closing tf sessions
sess = tf.Session()
print(sess.run(result))
sess.close()

# same as above, but automatically closes session after execution
with tf.Session() as sess:
    output = sess.run(result)
    print(output)
