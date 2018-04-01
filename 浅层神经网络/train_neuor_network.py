import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import readdata
import numpy as np
import tensorflow as tf

IMGHIGHT = 20
IMGWIDTH = 20
LAYER1_NUM = 3
LAYER2_NUM = 3

'''
X,Y = readdata.get_next_train_branch(1024)
X = np.array(X)
Y = np.array(Y)
'''


#构造神经元
X = tf.placeholder(tf.float32, [None, IMGHIGHT*IMGWIDTH])#None表示行不定,列为......
Y = tf.placeholder(tf.float32, [None, 1])


#构造神经网络
def crack_img_neuronnetwork():
    #第一层其实关键是做矩阵乘法，看清楚矩阵的维度即可，3表示3个神经元
    w_1 = tf.Variable(tf.random_normal([IMGHIGHT*IMGWIDTH, LAYER1_NUM]), dtype = tf.float32)
    b_1 = tf.Variable(tf.random_normal([1, LAYER1_NUM]), dtype = tf.float32)
    x_1 = tf.add(tf.matmul(X, w_1), b_1)
    y_1 = tf.nn.relu(x_1)


    #第二层,隐藏层
    w_2 = tf.Variable(tf.random_normal([LAYER1_NUM, LAYER2_NUM]), dtype = tf.float32)
    b_2 = tf.Variable(tf.random_normal([1, LAYER2_NUM]), dtype = tf.float32)
    x_2 = tf.add(tf.matmul(y_1, w_2), b_2)
    y_2 = tf.nn.relu(x_2)

    #输出层
    w_3 = tf.Variable(tf.random_normal([LAYER2_NUM, 1]), dtype = tf.float32)
    b_3 = tf.Variable(tf.random_normal([1, 1]), dtype = tf.float32)
    x_3 = tf.add(tf.matmul(y_2, w_3), b_3)
    return x_3

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

    #op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)  前5k-1w次遇到了训练瓶颈，精度在99.5%于是决定调整rate
    op = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(loss) #在调整后仍不见改善，应该是达到最优解了+

    x_test,y_test = readdata.get_next_train_branch(1024)
    x_test = np.array(x_test) / 255.0
    y_test = np.array(y_test)

    saver = tf.train.Saver()


    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())   前1w次训练
        saver.restore(sess, tf.train.latest_checkpoint('./model/')) #前1w-前1.5w次训练
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




#train_model()

        


        



