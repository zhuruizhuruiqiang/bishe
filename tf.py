import tensorflow as tf
import numpy as np

a=np.linspace(0,119,120,True)
# b=np.array([
#     [[[1., 2., 3., 4.],
#      [1., 2., 3., 4.]],
#
#      [[1., 2., 3., 4.],
#      [1., 2., 3., 4.]],
#
#      [[1., 2., 3., 4.],
#      [1., 2., 3., 4.]]],
#
#      [[[1., 2., 3., 4.],
#      [1., 2., 3., 4.]],
#
#      [[1., 2., 3., 4.],
#      [1., 2., 3., 4.]],
#
#      [[1., 2., 3., 4.],
#      [1., 2., 3., 4.]]],
#
#      [[[1., 2., 3., 4.],
#      [1., 2., 3., 4.]],
#
#      [[1., 2., 3., 4.],
#      [1., 2., 3., 4.]],
#
#      [[1., 2., 3., 4.],
#      [1., 2., 3., 4.]]]])

b=np.array([
    [[[1., 1., 1., 1.],
     [10., 10., 10., 10.]],

     [[2., 2., 2., 2.],
     [11., 11., 11., 11.]],

     [[3., 3., 3., 3.],
     [12., 12., 12., 12.]]],

     [[[4., 4., 4., 4.],
     [13., 13., 13., 13.]],

     [[5., 5., 5., 5.],
     [14., 14., 14., 14.]],

     [[6., 6., 6., 6.],
     [15., 15., 15., 15.]]],

     [[[7., 7., 7., 7.],
     [16., 16., 16., 16.]],

     [[8., 8., 8., 8.],
     [17., 17., 17., 17.]],

     [[9., 9., 9., 9.],
     [18., 18., 18., 18.]]]])

input=tf.reshape(np.array(a,np.float32),[3,5,4,2])
input=tf.ones([3,5,4,2])
filter=tf.constant(b.astype(np.float32))
op=tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
init=tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    print('input')
    print(input.eval())
    print('filter')
    print(filter.eval()[:,:,:,0])
    print('result')
    result = sess.run(op)
    print(result[0,:,:,0])
    print(result[0, :, :, 1])
    print(result[0, :, :, 2])
    print(result[0, :, :, 3])
