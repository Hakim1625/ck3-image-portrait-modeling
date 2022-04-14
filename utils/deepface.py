#!/usr/bin/env python
# coding: utf-8

# In[8]:


from deepface import DeepFace 


# In[9]:


parameters = {'model': 'Facenet',
              'backend': 'dlib',
              'actions': ['gender', 'age'],
              'size': (224, 224)
            }


# In[10]:


def face_detect(img_path, size, backend):
    return DeepFace.detectFace(img_path = img_path, target_size = size, detector_backend = backend)

def face_analysis(img_path, actions):
    return DeepFace.analyze(img_path = img_path, actions = actions)

def face_features(img_path, model):
    return DeepFace.represent(img_path = img_path, model_name = model)


# In[11]:


from torchvision import transforms


# In[12]:


def face_detect_img(input_img_path, output_img_path, size, backend):
    tensor = transforms.ToTensor()(face_detect(input_img_path, size, backend).copy())
    image = transforms.ToPILImage()(tensor)
    image.save(output_img_path, 'PNG')

