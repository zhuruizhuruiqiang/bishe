import tensorflow as tf
import numpy as np
import os
import pickle
import time
import sys


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.9, 'Keep probability for training dropout.')
flags.DEFINE_string('data_dir', 'E:/bishe/cifar-100-python/data', 'Directory for storing data')
flags.DEFINE_string('summaries_dir', 'E:/bishe/mnist_logs', 'Summaries directory')

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

def load_data(dataset_path):
    def gene_label(x):
        l=len(x)
        labelout=np.zeros([l,100], dtype='int32')
        for i in range(l):
            labelout[i][x[i]]=1
        return labelout

    dic = {}
    fo = open(os.path.join(dataset_path, 'train'), 'rb')
    dic.update(pickle.load(fo, encoding='ISO-8859-1'))
    train_data = dic['data']
    train_label = gene_label(dic['fine_labels'])
    #tra_label = dic['coarse_labels']
    fo = open(os.path.join(dataset_path, 'test'), 'rb')
    dic.update(pickle.load(fo, encoding='ISO-8859-1'))
    fo.close()
    test_data = dic['data'][0:5000]
    test_label = gene_label(dic['fine_labels'][0:5000])
    #te_label = dic['coarse_labels'][0:5000]
    # 分成训练集、验证集、测试集，大小如下
    valid_data = dic['data'][5000:10000]
    valid_label = gene_label(dic['fine_labels'][5000:10000])
    #val_label = dic['coarse_labels'][5000:10000]
    rval = [(train_data, train_label), (valid_data, valid_label),(test_data, test_label)]
    return rval

# 保存训练参数的函数
def save_params(param1, param2, param3, param4, param5, param6, param7, param8):
    write_file = open('params.pkl', 'wb')
    pickle.dump(param1, write_file, -1)
    pickle.dump(param2, write_file, -1)
    pickle.dump(param3, write_file, -1)
    pickle.dump(param4, write_file, -1)
    pickle.dump(param5, write_file, -1)
    pickle.dump(param6, write_file, -1)
    pickle.dump(param7, write_file, -1)
    pickle.dump(param8, write_file, -1)
    write_file.close()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, mean=0.0,stddev=0.001)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.001, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def mean_pool_2x2(x):
  return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def evaluate_pictures(n_epochs=200,batch_size=100,dataset='E:/bishe/cifar-100-python'):


    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # 计算各数据集的batch个数
    n_train_batches = train_set_x.shape[0]
    n_train_batches = int(n_train_batches / batch_size)
    print("... building the model")

    x = tf.placeholder(tf.float32, shape=[None, 3072])
    y = tf.placeholder(tf.float32, shape=[None, 100])
    keep_prob = tf.placeholder(tf.float32)

    x_image = tf.transpose(tf.reshape(x, [-1, 3, 32, 32]),perm=[0,2,3,1])##
    W_conv1 = weight_variable([5, 5, 3, 64])
    b_conv1 = bias_variable([64])
    h_pool1 = max_pool_2x2(tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1))

    input1=tf.nn.local_response_normalization(h_pool1)
    W_conv2 = weight_variable([5, 5, 64, 128])
    b_conv2 = bias_variable([128])
    h_pool2 = mean_pool_2x2(tf.nn.relu(conv2d(input1, W_conv2) + b_conv2))

    input2 = tf.nn.local_response_normalization(h_pool2)
    W_conv3 = weight_variable([5, 5, 128, 500])
    b_conv3 = bias_variable([500])
    h_pool3 = mean_pool_2x2(tf.nn.relu(conv2d(input2, W_conv3) + b_conv3))

    input3 = tf.nn.local_response_normalization(h_pool3)
    h_fc1_drop = tf.nn.dropout(input3, keep_prob)
    W_fc1 = weight_variable([4 * 4 * 500, 100])
    b_fc1 = bias_variable([100])
    y_conv = tf.matmul(tf.reshape(h_fc1_drop, [-1, 4 * 4 * 500]), W_fc1) + b_fc1

    # h_fc1 = tf.nn.relu(tf.matmul(tf.reshape(h_pool3, [-1, 4 * 4 * 500]), W_fc1) + b_fc1)
    # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    #
    # W_fc2 = weight_variable([20000, 100])
    # b_fc2 = bias_variable([100])
    # y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
    train_step = tf.train.AdadeltaOptimizer(0.1).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    tf.summary.scalar('accuracy', accuracy)
    # variable_summaries(W_conv1)
    # variable_summaries(b_conv1)
    # variable_summaries(W_conv2)
    # variable_summaries(b_conv2)
    # variable_summaries(W_conv3)
    # variable_summaries(b_conv3)
    # variable_summaries(W_fc1)
    # variable_summaries(b_fc1)
    # variable_summaries(W_fc2)
    # variable_summaries(b_fc2)

    # tf.summary.histogram('W_conv1', W_conv1)
    # tf.summary.histogram('b_conv1', b_conv1)
    # tf.summary.histogram('W_conv2', W_conv2)
    # tf.summary.histogram('b_conv2', b_conv2)
    # tf.summary.histogram('W_conv3', W_conv3)
    # tf.summary.histogram('b_conv3', b_conv3)
    # tf.summary.histogram('W_fc1', W_fc1)
    # tf.summary.histogram('b_fc1', b_fc1)
    # tf.summary.histogram('W_fc2', W_fc2)
    # tf.summary.histogram('b_fc2', b_fc2)

    # tf.summary.histogram('h_pool1', h_pool1)
    # tf.summary.histogram('h_pool1', h_pool2)
    # tf.summary.histogram('h_pool1', h_pool3)
    # tf.summary.histogram('h_pool1', h_fc1)
    tf.summary.histogram('h_pool1', y_conv)

    sess=tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    best_validation_acc = np.inf
    epoch = 0
    done_looping = False

    print("... training")
    summary_op=tf.summary.merge_all()
    summary_writer=tf.summary.FileWriter(FLAGS.summaries_dir,graph_def=sess.graph_def)
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index
            _,acc=sess.run([train_step, accuracy],feed_dict={x: train_set_x[minibatch_index * batch_size: (minibatch_index + 1) * batch_size],
                y: train_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size], keep_prob: 0.2})
            print('epoch %i, step %d,minibatch %i/%i, train acc %g' %
                  (epoch, iter, minibatch_index + 1, n_train_batches,acc))
            if (iter + 1) % 100 == 0:
                # compute zero-one loss on validation set
                validation_acc = accuracy.eval(feed_dict={x: valid_set_x, y: valid_set_y, keep_prob: 0.2})
                print('                         validation acc %g' %(validation_acc ))
                # test it on the test set
                summary_str, acc = sess.run([summary_op, accuracy],
                                            feed_dict={x: test_set_x, y: test_set_y, keep_prob: 0.2})
                summary_writer.add_summary(summary_str, iter)
                print('                         test acc %g' % (acc))
                # if we got the best validation score until now
                if validation_acc > best_validation_acc:
                    # save best validation score and iteration number
                    best_validation_acc = validation_acc
                    save_params(W_conv1, b_conv1, W_conv2, b_conv2,W_fc1,b_fc1)  # 保存参数


    print('Optimization complete.')
    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: test_set_x, y: test_set_y, keep_prob: 1.0}))
    # print >> sys.stderr, ('The code for file ' +
    #                       os.path.split(__file__)[1] +
    #                       ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    evaluate_pictures()