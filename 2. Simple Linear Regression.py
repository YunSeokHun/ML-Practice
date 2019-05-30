
# coding: utf-8

# In[7]:


import tensorflow as tf
#X and Y data
x_train=[1,2,3,4,5]
y_train=[43,60,80,109,122]

W=tf.Variable(tf.random_normal([1]), name="weight")
b=tf.Variable(tf.random_normal([1]), name="bias")


#Hypothesis is Wx+b
hypothesis=x_train*W+b

#cost function
cost=tf.reduce_mean(tf.square(hypothesis-y_train))

#Minimize
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
train=optimizer.minimize(cost)

#Launch the graph in a session
sess=tf.Session()
#Init gl_vars in the graph
sess.run(tf.global_variables_initializer())
#Fit the line
for step in range(10001):
    sess.run(train)
    if step%50==0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))


# In[9]:


import tensorflow as tf
W=tf.Variable(tf.random_normal([1]), name="weight")
b=tf.Variable(tf.random_normal([1]), name="bias")
X=tf.placeholder(tf.float32,shape=[None])
Y=tf.placeholder(tf.float32,shape=[None])

#Hypothesis: Y=Wx+b
hypothesis=W*X+b
#cost function
cost=tf.reduce_mean(tf.square(hypothesis-Y))
#Minimize
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
train=optimizer.minimize(cost)

#Launch the graph in a session
sess=tf.Session()
#Init gl_vars in the graph
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, W_val, b_val,_ = sess.run([cost,W,b,train],feed_dict={X: [1,2,3,4,5],Y: [2.1,3.1,4.1,5.1,6.1]})
    if step%20==0:
        print(step,cost_val,W_val,b_val)

