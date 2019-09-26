#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import matplotlib.pyplot as plt 


# In[30]:





# In[36]:


#read data 
X = np.array([[1,3,3],
              [1,4,3],
              [1,1,1],
              [1,0,2]])
# labels
Y = np.array([[1],
              [1],
              [-1],
              [-1]])
# weight 3 行 1 列， 取值-1 到 1
W = (np.random.random([3,1])-0.5)*2
     
print(W)

#learning rate 
Ir = 0.11
#output 
O = 0

def update():
    global X,Y,W,Ir
    O = np.sign(np.dot(X,W)) #shape: (3,1)
    W_C = Ir*(X.T.dot(Y-O))/int(X.shape[0])
    W = W + W_C
for i in range(100):
    update()
    print(W)
    print(i)
    O = np.sign(np.dot(X,W))
    if(O == Y).all():
        print('Finished')
        print('epoch:',i)
        break
# 正样本
x1 = [3,4]
y1 = [3,3]
# 负样本
x2 = [1,0]
y2 = [1,2]

# intercpetion and k
k = -W[1]/W[2]
d = -W[0]/W[2]
print('k=',k)
print('d=',d)

xdata = (0,5)

plt.figure()
plt.plot(xdata,xdata*k+d,'r')
plt.scatter(x1,y1,c='b')
plt.scatter(x2,y2,c='y')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




