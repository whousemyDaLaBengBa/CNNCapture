import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import readdata
import numpy as np
import tensorflow as tf

IMGHIGHT = 20
IMGWIDTH = 20

'''
X,Y = readdata.get_next_train_branch(1024)
X = np.array(X)
Y = np.array(Y)
'''


#构造神经元
X = tf.placeholder(tf.float32, [None, IMGHIGHT*IMGWIDTH])#None表示行不定,列为......
Y = tf.placeholder(tf.float32, [None, 1])


def crack_img_neuron():
    w = tf.Variable(tf.random_normal([IMGHIGHT*IMGWIDTH, 1]), dtype = tf.float32)
    b = tf.Variable(tf.random_normal([1, 1]), dtype = tf.float32)

    out_x = tf.add(tf.matmul(X, w), b)
    return out_x

def get_acc(Y_p,Y):
    tot = len(Y_p)
    right = 0.0
    for k in range(tot):
        if (Y_p[k] > 0.5 and Y[k] > 0.5):
            right = right + 1.0
        elif (Y_p[k] < 0.5 and Y[k] < 0.5):
            right = right + 1.0

    acc = right/(tot*1.0)
    return str(acc)
    


def train_model():
    out_x = crack_img_neuron()
    Y_p = tf.nn.sigmoid(out_x)

    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=out_x, labels=Y)
    loss = tf.reduce_mean(loss)
    op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)


    x_test,y_test = readdata.get_next_train_branch(1024)
    x_test = np.array(x_test) / 255.0
    y_test = np.array(y_test)

    saver = tf.train.Saver()


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0

        while(True):
            step = step + 1
            _,loss_p,Y_pp = sess.run([op, loss, Y_p], feed_dict={X: x_test, Y: y_test})

            #print(loss_p)
            #print(Y_pp)
            #break
            if step % 200 == 0:
                print(str(step) + '  ' + str(loss_p))
                if step % 1000 == 0:
                    print(str(step) + '  ' + get_acc(Y_pp, y_test))
                if step % 2000 == 0:
                    saver.save(sess, './model/crack_capcha.model', global_step=step)

train_model()

        


        



