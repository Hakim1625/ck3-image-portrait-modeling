#!/usr/bin/env python
# coding: utf-8

# In[1]:


import utils.macro as macro
import time


# In[2]:





# In[ ]:





generator = macro.dataset_generator(3, './datasets')
n_cycles = 1
time.sleep(3)

for n in range(n_cycles):
    try: generator._generate_dataset(n)
    except: continue

