
# coding: utf-8

# In[2]:


import tensorflow as tf
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
X=[1,2,3]
Y=[1,2,3]

W=tf.placeholder(tf.float32)
# our hypothesis for linear model is W*X
hypothesis=W*X

#cost function
cost=tf.reduce_mean(tf.square(hypothesis-Y))

#Launch the graph in a session
sess=tf.Session()
#Init the gl_vars
sess.run(tf.global_variables_initializer())

#Variables for plotting cost function
W_val=[]
cost_val=[]
for i in range(-30, 50):
    feed_W=i*0.1
    curr_cost,curr_W=sess.run([cost,W], feed_dict={W: feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)

#Showing the cost function (Plotting!)
plt.plot(W_val, cost_val)
plt.show()
    


# In[5]:


import tensorflow as tf
x_data=[1,2,3]
y_data=[1,2,3]

W=tf.Variable(tf.random_normal([1]), name='weight')
X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

#Hypothesis is X*W
hypothesis=W*X
#cost function
cost=tf.reduce_sum(tf.square(hypothesis-Y))

#Minimizing cost (W -= Learning_rate * gradiant(derivatives))
#This part can be done generally and simply by (train.GradientDescentOptimizer(learning_rate=?))& minimize
learning_rate=0.1
gradient=tf.reduce_mean((W*X-Y)*X)
descent=W-learning_rate*gradient
update=W.assign(descent)

#Launch the graph in a session.
sess=tf.Session()
#init gl_vars
sess.run(tf.global_variables_initializer())
for step in range(121):
    sess.run(update, feed_dict={X: x_data, Y:y_data})
    print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))


# In[9]:


import tensorflow as tf
X=[1,2,3]
Y=[1,2,3]
#Set wrong model weights
W=tf.Variable(5.0)
#Hypothesis: Y=W*X
hypothesis=W*X
#cost function
cost=tf.reduce_mean(tf.square(hypothesis-Y))
#Minimizing Generally
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1)
train=optimizer.minimize(cost)

#Launch the graph in a session
sess=tf.Session()
#init gl_vars
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run(W))
    sess.run(train)

