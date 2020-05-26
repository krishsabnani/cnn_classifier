#!/usr/bin/env python
# coding: utf-8

# In[1]:


f = open("notebook.py", "r")
contents = f.readlines()
f.close()

contents.insert(73, "newLayer()\n")

f = open("notebook.py", "w")
contents = "".join(contents)
f.write(contents)
f.close()


# In[ ]:




