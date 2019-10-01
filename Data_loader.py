import json
import h5py
import numpy as np
import copy
from random import shuffle, seed
import sys
import os.path
import argparse
import glob
import numpy as np
import scipy.io
import pdb
import string
import h5py
import nltk
from nltk.tokenize import word_tokenize
import gensim
import json
import re
import cv2
import matplotlib.pyplot as plt


def extract_feat(doc):
    feat = []
    for word in doc:
        try:
            feat.append(model_w2v[word])
        except:
            pass
    return feat

def tokenize(sentence):
    return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) if i!='' and i!=' ' and i!='\n'];

def prepro_question(imgs, method):
  
    # preprocess all the question
    print('example processed tokens:')
    for i,img in enumerate(imgs):
        s = img['question']
        if method == 'nltk':
            txt = word_tokenize(str(s).lower())
        else:
            txt = tokenize(s)
        img['processed_tokens'] = txt
        if i < 10: print(txt)
        if i % 1000 == 0:
            sys.stdout.write("processing %d/%d (%.2f%% done)   \r" %  (i, len(img), i*100.0/len(imgs)) )
            sys.stdout.flush()   
    return imgs

def get_top_answers(imgs, num_ans):
    counts = {}
    for img in imgs:
        ans = img['ans'] 
        counts[ans] = counts.get(ans, 0) + 1

    cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
    #print('top answer and their counts:') 
    #print('\n'.join(map(str,cw[:20])))
    
    vocab = []
    for i in range(num_ans):
        vocab.append(cw[i][1])
    return vocab[:num_ans]

def filter_question(imgs, atoi):
    new_imgs = []
    for i, img in enumerate(imgs):
        if atoi.get(img['ans'],len(atoi)+1) != len(atoi)+1:
            new_imgs.append(img)

    print('question number reduce from %d to %d '%(len(imgs), len(new_imgs)))
    return new_imgs


imgs_train = json.load(open('Raw_Data/vqa_raw_train.json' , 'r'))
seed(125)
shuffle(imgs_train)
num_ans = 1000

top_ans = get_top_answers(imgs_train, num_ans )
atoi = {w:i for i,w in enumerate(top_ans)}
itoa = {i:w for i,w in enumerate(top_ans)}

feat_dim = 300
imgs_data_train = json.load(open('vqa_final_train.json' , 'r'))
seed(125)
shuffle(imgs_data_train)

num_ans = 1000


method = 'nltk'
max_length = 26
dir_path = "Final_Data/QA/"
N = len(imgs_data_train)

# In[6]:


def load_data(batch):
    
    start = 0
    end = batch
    
    while True:
        
        #print(start,end)
        
        images = []
        questions = []
        answers = []
        ids = []
        #arrs = np.random.randint(0,len(imgs_data_train),batch)
        #data = [imgs_data_train[i] for i in arrs]
        
        data = imgs_data_train[start:end]
        
        start = end
        end = end + batch
        
        if end > len(imgs_data_train):
            start = 0
            end = batch
        
        for i,img in enumerate(data):
        
            img_path = img['img_path']  
            question_id = img['ques_id']

            label_arrays = np.zeros((1, max_length, feat_dim), dtype='float32')
            
            with h5py.File(os.path.join(dir_path,str(question_id) + '.h5'),'r') as hf:
                question = hf['.']['ques_train'].value
                answer = hf['.']['answers'].value
         
            image = cv2.imread(os.path.join('Final_Data/',img_path) , cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
            image = cv2.resize(image , (224,224))
            
            image = (image - 127.5)/127.5

            images.append(image)
            questions.append(np.array(question))
            answers.append(np.array(answer))
            ids.append(question_id)               
            
        questions = np.reshape(np.array(questions) , [batch,max_length,feat_dim])
        yield(np.array(images) , questions ,np.array(answers) , np.array(ids))
