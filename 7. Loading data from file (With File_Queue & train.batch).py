
# coding: utf-8

# In[28]:


import tensorflow as tf
filename_queue = tf.train.string_input_producer(
['data-01-test-score.csv'], shuffle=False, name='filfename_queue')

reader=tf.TextLineReader()
key, value=reader.read(filename_queue)

#Default values, in case of empty columns. Also specifies the type of the decoded result.
record_defaults=[[0.],[0.],[0.],[0.]]
xy=tf.decode_csv(value, record_defaults=record_defaults)

#Collect batches of csv in
train_x_batch, train_y_batch= tf.train.batch([xy[0:-1], xy[-1:]], batch_size=8)

#placeholders for a tensor that will be always fed.
X=tf.placeholder(tf.float32, shape=[None,3])
Y=tf.placeholder(tf.float32, shape=[None,1])

W=tf.Variable(tf.random_normal([3,1]), name='weight')
b=tf.Variable(tf.random_normal([1]), name='bias')

#Hypothesis
hypothesis=tf.matmul(X,W)+b

#cost function
cost=tf.reduce_mean(tf.square(hypothesis-Y))

#Minimize
optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train=optimizer.minimize(cost)

#Launch the graph in a session.
sess=tf.Session()
#Init gl_vars
sess.run(tf.global_variables_initializer())

#Start populating the filename queue.
coord=tf.train.Coordinator()
threads=tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})
    if(step%10==0):
        print(step, "Cost: ", cost_val, "\nPrediction\n", hy_val)
        
coord.request_stop()
coord.join(threads)

