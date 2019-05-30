
# coding: utf-8

# In[1]:


import tensorflow as tf
X=[1,2,3]
Y=[1,2,3]
#Set wrong model weights
W=tf.Variable(5.0,tf.float32)
#Hypothesis is Y=W*X
hypothesis=W*X
#Manual gradient
gradient=tf.reduce_mean((W*X-Y)*X)*2
#Cost function
cost=tf.reduce_mean(tf.square(hypothesis-Y))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)

#Get gradients and Apply gradients
gvs=optimizer.compute_gradients(cost)
apply_gradients=optimizer.apply_gradients(gvs)

#Launch the graph in a session.
sess=tf.Session()
#init G_vars
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run([gradient, W, gvs]))
    sess.run(apply_gradients)

    
    

