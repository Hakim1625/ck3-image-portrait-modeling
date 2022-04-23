#!/usr/bin/env python
# coding: utf-8

# In[8]:


from deepface import DeepFace 


# In[9]:


parameters = {'model': 'VGG-Face',
              'backend': 'mtcnn',
              'actions': ['gender', 'age'],
              'size': (224, 224)
            }


# In[10]:


def face_detect(img_path):
    return DeepFace.detectFace(img_path = img_path, target_size = parameters['size'], detector_backend = parameters['backend'])

def face_analysis(img_path):
    return DeepFace.analyze(img_path = img_path, actions = parameters['actions'])

def face_features(img_path):
    return DeepFace.represent(img_path = img_path, model_name = parameters['model'])


# In[11]:


from torchvision import transforms


# In[12]:
