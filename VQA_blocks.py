#!/usr/bin/env python
# coding: utf-8

# In[12]:


import tensorflow as tf
import numpy as np
import keras
import pandas as pd
import os
import matplotlib.pyplot as plt


dropout_rate = 0.4

# In[2]:


def image_layer(input_tenor):
    
    with tf.variable_scope("image"):
    
	    base_model = tf.keras.applications.VGG16(input_tensor=input_tenor, include_top=False,weights='imagenet')
	   
	    base_model.trainable = False
		    
	    x = base_model.layers[-2].output
	    
	    x = tf.reshape(x , [-1,x.shape[2]*x.shape[1] , x.shape[3]])
	    
	    x = tf.layers.dense(x,1024)

	    return x


# In[3]:


#image = tf.placeholder(tf.float32 , [batch_size,224,224,3])
#x = image_layer(image)
#print(x.shape)


# In[4]:


def question_layer(embed_size ,embed_len , num_units , q_len , quest , batch_size ):
    
    rnn = tf.nn.rnn_cell
    
    lstm1 = rnn.BasicLSTMCell(num_units)
    lstm_drop1 = rnn.DropoutWrapper(lstm1, output_keep_prob = 1 - 0.8)
    
    lstm2 = rnn.BasicLSTMCell(num_units)
    lstm_drop2 = rnn.DropoutWrapper(lstm2, output_keep_prob = 1 - 0.8)
    
    final = rnn.MultiRNNCell([lstm_drop1,lstm_drop2])
    
    
    state = final.zero_state(batch_size, tf.float32)
    loss = 0.0
    
    with tf.variable_scope("embed" , reuse=False):
        
        for i in range(q_len):
            
                if i==0:
                    ques_emb_linear = tf.zeros([batch_size, embed_size])
            
                else:
                    tf.get_variable_scope().reuse_variables()
                    ques_emb_linear = quest[:,i-1]

                # LSTM based question model
                ques_emb_drop = tf.nn.dropout(ques_emb_linear, 1-dropout_rate)
                ques_emb = tf.tanh(ques_emb_drop)

                output, state = final(ques_emb, state)
        
    question_emb = tf.reshape(tf.transpose(state, [2, 1, 0, 3]), [batch_size, -1])
    
    return question_emb
# In[5]:


#quest = tf.placeholder(tf.int32 , [batch_size ,q_len])
#temp = question_layer(512 , 256 , quest , batch_size)
#print(temp.shape)


# In[6]:


def attention(image_tensor , question_tensor , out_dim  , dropout):
    
    img = tf.nn.tanh(tf.layers.dense(image_tensor , out_dim))
    
    ques = tf.nn.tanh(tf.layers.dense(question_tensor , out_dim))
    
    ques = tf.expand_dims(ques , axis = -2)
    
    IQ = tf.nn.tanh(img + ques)
     
    if dropout:
        IQ = tf.nn.dropout(IQ , 0.5)
    
    temp = tf.layers.dense(IQ , 1)
    
    temp = tf.reshape(temp , [-1,temp.shape[1]])
    
    p = tf.nn.softmax(temp)
    
    p_exp = tf.expand_dims(p , axis = -1)
    
    att_layer = tf.reduce_sum(p_exp * image_tensor , axis = 1)
    
    final_out = att_layer + question_tensor
        
    return p , final_out


# In[7]:


#att = attention(x , temp ,  512 , True)


# In[8]:


def SAN(img_h , img_w , q_len , embed_size , lstm_units , attention_dim , num_output , batch_size ):
    
    image = tf.placeholder(tf.float32 , [batch_size,img_h,img_w,3])
    quest = tf.placeholder(tf.float32 , [batch_size ,q_len , embed_size])
    label = tf.placeholder(tf.int32, [batch_size,]) 
    
    image_embed = image_layer(image)
    
    ques_embed = question_layer(embed_size ,attention_dim , lstm_units , q_len ,  quest , batch_size )
    
    att_l1 , att = attention( image_embed , ques_embed ,  attention_dim , True)
    
    att_l2 , att = attention( image_embed , att  , attention_dim , True)
    
    att = tf.nn.dropout(att , dropout_rate)
    
    att = tf.layers.dense(att , num_output)
	
    print(att.shape , label.shape	)	
    
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels= label , logits=att)
    
    loss = tf.reduce_mean(loss)
    
    att = tf.nn.softmax(att)
    
    print(att.shape)
    
    attention_layers = [att_l1 , att_l2]
    
    return loss , image , quest , label , attention_layers , att

