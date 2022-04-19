#!/usr/bin/env python
# coding: utf-8

# In[1]:


import utils.macro as macro
import time


# In[2]:





# In[ ]:


time.sleep(3)


n_cycles = 1

for n in range(n_cycles):
    generator = macro.dataset_generator(3, './datasets')
    generator._generate_dataset(n)

