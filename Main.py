#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import keras
import pandas as pd
import os
import json
from tqdm import tqdm 
import matplotlib.pyplot as plt
from VQA_blocks import *
from Data_loader import *


# In[2]:


embed_size = 300
q_len = 26
height = 224
width = 224
lstm_units = 256
attention_dim = 512
num_output = 1000

batch_size = 32
lr = 0.001


# In[3]:


loss, image_inp, question_inp, true_label , attention_layers , pred_label =   SAN(height ,
                                                                                width, 
                                                                                q_len,
                                                                                embed_size, 
                                                                                lstm_units, 
                                                                                attention_dim, 
                                                                                num_output, 
                                                                                batch_size
                                                                                )

print("Model Loaded")
print("Input image dimensions : ",image_inp.shape)
print("Input question embedding dimension : ", question_inp.shape)
print("True Answers Dimensions : " , true_label.shape)
print("Attention matrix dimensions : " ,attention_layers[0].shape)


# In[65]:


data_gen = load_data(batch_size)


# In[66]:


image , question , answer , ids = next(data_gen)
print(ids)


# In[6]:


train_op = tf.train.AdamOptimizer(lr).minimize(loss)
saver = tf.train.Saver()


# In[7]:


epochs = 1
total_size = 388158
loss_data = []


# In[8]:


with tf.Session() as sess:
    
    #saver.restore(sess, "Trained Model/model.ckpt")
    
    sess.run(tf.global_variables_initializer())
    
    for i in range(epochs):
        
        num_iters = int(total_size/batch_size)
        
        for iters in tqdm(range(num_iters)):
            
            image,question,answer,ids = next(data_gen)
            
            _ , loss_out = sess.run([train_op,loss] , feed_dict = {image_inp:image , question_inp:question , true_label : answer} )
            
            loss_data.append(loss_out)
        
        save_path = saver.save(sess, "Trained Model/model.ckpt")
        print("Model Saved")
        np.save('Loss/'+str(i)+'.npy' , np.array(loss_data))
        break


# In[81]:


imgs_train = json.load(open('Raw_Data/vqa_raw_train.json' , 'r'))
ans_map = json.load(open('train.json' , 'r'))


image , question , answer , ids = next(data_gen)
questions_str = []

for i in tqdm(range(ids.shape[0])):
    for j in range(len(imgs_train)):        
        if imgs_train[j]['ques_id'] == ids[i]:
            questions_str.append(imgs_train[j]['question'])

print("Questions Loaded")
    
with tf.Session() as sess:

        saver.restore(sess, "Trained Model/model.ckpt")
        ans = sess.run(pred_label , feed_dict = {image_inp:image , question_inp:question } )
        pred_answer = np.argmax(ans , axis = 1)


def get_pred(pred_answer,questions_str):
    num = np.random.randint(0,batch_size , 5)
    
    for i in range(num.shape[0]):
        print(num[i],"Image : ")
        plt.imshow(image[num[i]])
        plt.show()
        print("Question : " , questions_str[num[i]])
        print("Answer : " , ans_map[str(pred_answer[num[i]])])


# In[87]:


get_pred(pred_answer,questions_str)


# In[ ]:




