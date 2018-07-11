
# coding: utf-8

# In[ ]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

q_table = pd.read_csv('q_table(0.9)0.csv',index_col = 0)
data = pd.read_csv('latest\data_test.csv')

w = np.zeros(len(data))
bench = np.zeros(len(data))
w[0] = 10000 ; w[1] = 10000
bench[0] = 10000 ; bench[1] = 10000
m = np.arange(1,len(data)+1,1)
n = 1
while n < len(data)-1:
    s1 = data.iloc[n-1:n+1,2]
    s1_change = str(round((s1.iloc[1]-s1.iloc[0])/s1.iloc[0],2))
    s2 = data.iloc[n-1:n+1,4]
    s2_change = str(round((s2.iloc[1]-s2.iloc[0])/s2.iloc[0],2))
    s = ','.join([s1_change,s2_change])
    if s in q_table.index:
        action=q_table.loc[s,:]
        action=float(np.argmax(action))
        
    else:
        action = np.random.choice([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    s_1 = data.iloc[n:n+2,2]
    s_1_change = str(round((s_1.iloc[1]-s_1.iloc[0])/s_1.iloc[0],2))
    s_2 = data.iloc[n:n+2,4]
    s_2_change = str(round((s_2.iloc[1]-s_2.iloc[0])/s_2.iloc[0],2))
    w[n+1] = (float(s_1_change) * action + float(s_2_change) * (1-action))* w[n] + w[n]
    bench[n+1] = (float(s_1_change) + float(s_2_change)) * bench[n]/2 + bench[n]
    n += 1 
    
dic = pd.DataFrame({'q-learning': w, 'benchmark(0.5:0.5)': bench})
dic.to_csv('q_learning_test.csv')
plt.plot(m, w, label = 'q-learning')
plt.plot(m, bench, label = 'benchmark(0.5,0.5)')
plt.legend()
plt.show()
    
        
        

