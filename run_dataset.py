#!/usr/bin/env python
# coding: utf-8

# In[1]:


import utils.macro as macro
import time


# In[2]:


generator = macro.dataset_generator(3, './portraits')


# In[ ]:


time.sleep(3)
generator._generate_dataset()

