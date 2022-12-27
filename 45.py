import tensorflow as tf

if tf.test.gpu_device_name():
    print("Default GPU device: {}".format(tf.test.gpu_device_name()))
else:
    print("We have tensorflow CPU")