import matplotlib.pyplot as plt
import requests
from lxml import etree
import numpy as np
infected = 1975    
suspected_case= 2684
allI=infected#+suspected_case
healer=56
death=49
time = 100



infection_rate = 0.057     # by Imai et al

mortality =  death/allI
cure_rate=healer/allI
t=1
f = open('C:\\sss\\za\\2.txt','a')
f.seek(0)
f.truncate()
f.close
while(t<time):
    if t<=10:
        k = 5         # the average contact by a person
    elif t>10:
        k=1
        
    allI=allI*k*infection_rate-allI*(mortality+cure_rate)+allI
    deathpeople=allI*mortality
    allI=int(allI)
    f = open('C:\\sss\\za\\2.txt','a')

    f.write(str(t)+' '+str(allI)+'\n')
    f.close
    t=t+1

import matplotlib.pyplot as plt
filename = 'C:\\sss\\za\\2.txt'
X,Y = [],[]
with open(filename, 'r') as f:#1
    lines = f.readlines()#2
    for line in lines:#3
        value = [float(s) for s in line.strip().split( )]#4
        X.append(value[0])#5
        Y.append(value[1])
  
print(X)
print(Y)
plt.plot(X, Y)
# for a, b in zip(X, Y):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=5)

plt.show()




