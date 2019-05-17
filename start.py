
# coding: utf-8

# In[1]:


2*3


# In[2]:


print(1)


# In[3]:


import tensorflow as tf
hello=tf.constant("Hello, tf!!")
sess=tf.Session()
print(sess.run(hello))

