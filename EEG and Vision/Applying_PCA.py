
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




F = open("wake_data.txt","r")
s = F.read()

l=s.splitlines()

# r= l.
# print(r)

r = str.split(l[0], " ")
wake_data=np.zeros((l.__len__(),r.__len__()-1))
# print(float(wake_data[0,0]))

for i in range(l.__len__()):
    r = str.split(l[i], " ")
    # print(r)
    for j in range(r.__len__()-1):
        # print(r[j])
        wake_data[i][j]=float(r[j])


F = open("sleep_1_data.txt","r")
s = F.read()

l=s.splitlines()


sleep_1_data=np.zeros((l.__len__(),r.__len__()-1))

for i in range(l.__len__()):
    r = str.split(l[i], " ")
    for j in range(r.__len__()-1):
        sleep_1_data[i][j]=float(r[j])




# print(wake_data)
#
# print(sleep_1_data)
#
# print(wake_data.shape[0])
# print(wake_data.shape[1])
# print(sleep_1_data.shape[0])
# print(sleep_1_data.shape[1])






########################## Appling PCA ##################################




####### Wake Data ######

x = StandardScaler().fit_transform(wake_data)

pca = PCA(n_components=2)

wake_PCA_data = pca.fit_transform(x)

##################################


######### Sleep Data ##########


x = StandardScaler().fit_transform(sleep_1_data)

pca = PCA(n_components=2)

sleep_1_PCA_data = pca.fit_transform(x)

# print(principalComponents)

##############################################################################################################

################################################ Ploting #####################################################

fig, ax = plt.subplots()
x=[]
y=[]
x2=[]
y2=[]
for i in range(wake_PCA_data.__len__()):
    x.append(wake_PCA_data[i][0])
    y.append(wake_PCA_data[i][1])

for i in range(sleep_1_PCA_data.__len__()):
    x2.append(sleep_1_PCA_data[i][0])
    y2.append(sleep_1_PCA_data[i][1])

# Y = np.array([0.0, 0.001, 0.003, 0.2, 0.4, 0.5, 0.7, 0.88, 0.9, 1.0])
# Y2 = np.repeat(Y,4)
# print(Y2)
# plt.ylim(0.0000,0.00030 )
ax.scatter(x,y,color=['blue'])
ax.scatter(x2,y2,color=['red'])
plt.show()


#################################################################################