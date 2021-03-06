
# coding: utf-8

# In[19]:


import tensorflow as tf
import numpy as np

xy=np.loadtxt('data-04-zoo.csv', delimiter=',',dtype=np.float32)
x_data=xy[:,0:-1]
y_data=xy[:,[-1]]

nb_classes=7 # because Values of y_data is 0~6
X=tf.placeholder(tf.float32, [None, 16])
Y=tf.placeholder(tf.int32, [None, 1]) # 0~6!

Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot_done = tf.reshape(Y_one_hot, [-1,nb_classes])

W=tf.Variable(tf.random_normal([16,nb_classes]), name='weight')
b=tf.Variable(tf.random_normal([nb_classes]), name='bias')

#logits=tf.matmul(X,W)+b
hypothesis=tf.nn.softmax(tf.matmul(X,W)+b) ## hypothesis=tf.nn.softmax(logits)

#Cross entropy cost
cost_i=tf.nn.softmax_cross_entropy_with_logits(logits=tf.matmul(X,W)+b, labels=Y_one_hot_done)
cost=tf.reduce_mean(cost_i)
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction=tf.argmax(hypothesis,axis=1)
correct_prediction=tf.equal(prediction, tf.argmax(Y_one_hot_done,axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
        if(step%100==0):
            loss, acc= sess.run([cost,accuracy], feed_dict={X:x_data, Y:y_data})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step,loss,acc))
          ##  AA,BB=sess.run([Y_one_hot,Y_one_hot_done], feed_dict={X:x_data, Y:y_data})
          ##  print("\nbefore reshaping ", AA,"\nAfter ", BB)
    
    #Let's see if we can predict
    pred = sess.run(prediction, feed_dict={X: x_data})
    for p,y in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))
    
    
    
    
    
    
    

