from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import random
import pickle



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

wake_sleep_data = np.concatenate((wake_data, sleep_1_data),axis=0)

####### Wake Data ######

x = StandardScaler().fit_transform(wake_sleep_data)
# print("X shape : " ,  x.shape)
pca = PCA(n_components=2)

wake_sleep_PCA_data = pca.fit_transform(x)




wake_PCA_data = wake_sleep_PCA_data[0:wake_data.shape[0], :]
sleep_1_PCA_data = wake_sleep_PCA_data[wake_data.shape[0]:(sleep_1_data.shape[0] + wake_data.shape[0]), :]






##################################


######### Sleep Data ##########


# x = StandardScaler().fit_transform(sleep_1_data)
#
# pca = PCA(n_components=2)
#
# sleep_1_PCA_data = pca.fit_transform(x)

# print(principalComponents)

##############################################################################################################


############################## Classification #####################################


##### I will change the number of wake data here


# overall_wake_data=np.zeros((wake_PCA_data.shape[0],wake_PCA_data.shape[1]+1))



# print (sleep_1_PCA_data.shape[0])



############################################# Adding CV values #####################################


wake_CV_data = np.zeros((sleep_1_PCA_data.shape[0],wake_PCA_data.shape[1]+3))
sleep_1_CV_data = np.zeros((sleep_1_PCA_data.shape[0],sleep_1_PCA_data.shape[1]+3))



np.random.shuffle(wake_PCA_data)

for i in range (sleep_1_PCA_data.shape[0]):
    for j in range (wake_PCA_data.shape[1]):
        wake_CV_data[i][j]=wake_PCA_data[i][j]

for i in range (sleep_1_PCA_data.shape[0]):
    for j in range (sleep_1_PCA_data.shape[1]):
        sleep_1_CV_data[i][j]=sleep_1_PCA_data[i][j]




F = open("EARValues.txt","r")
s = F.read()

l=s.splitlines()

# r= l.
# print(r)
# print(l.__len__())
# print(float(wake_data[0,0]))
counter_wake=0
counter_sleep=0
r = str.split(l[0], " ")
# print(r.__len__())
for i in range(l.__len__()):
    r = str.split(l[i], " ")
    for j in range(r.__len__()):
        # print(r[j])
        # print (r[2])
        if(int(r[2])==0):
            wake_CV_data[counter_wake][j+2]=float(r[j])
            if(j==2):
                counter_wake+=1
        else:
            sleep_1_CV_data[counter_sleep][j+2]=float(r[j])
            if (j == 2):
                counter_sleep += 1

# print(wake_CV_data)
# print(sleep_1_CV_data)
# print(counter_sleep)
# print(counter_wake)


overall_wake_data=wake_CV_data

overall_sleep_data=sleep_1_CV_data




##### I will change the number of wake data here

#
# for i in range(sleep_1_PCA_data.shape[0]):
# # for i in range(wake_PCA_data.shape[0]):
#
#     for j in range(wake_PCA_data.shape[1]):
#         overall_wake_data[i][j]=wake_PCA_data[i][j]
#     overall_wake_data[i][wake_PCA_data.shape[1]]=0
#
# for i in range(sleep_1_PCA_data.shape[0]):
#
#     for j in range(sleep_1_PCA_data.shape[1]):
#         overall_sleep_data[i][j]=sleep_1_PCA_data[i][j]
#     overall_sleep_data[i][sleep_1_PCA_data.shape[1]]=1


# print("wake : ", overall_wake_data.shape)
# print("sleep : ", overall_sleep_data.shape)

# fig, ax = plt.subplots()
# x=[]
# y=[]
# x2=[]
# y2=[]
# for i in range(overall_wake_data.__len__()):
#     x.append(overall_wake_data[i][0])
#     y.append(overall_wake_data[i][1])
#
# for i in range(overall_sleep_data.__len__()):
#     x2.append(overall_sleep_data[i][0])
#     y2.append(overall_sleep_data[i][1])
#
# # Y = np.array([0.0, 0.001, 0.003, 0.2, 0.4, 0.5, 0.7, 0.88, 0.9, 1.0])
# # Y2 = np.repeat(Y,4)
# # print(Y2)
# # plt.ylim(0.0000,0.00030 )
# ax.scatter(x,y,color=['blue'])
# ax.scatter(x2,y2,color=['red'])
# plt.show()

# print(overall_wake_data)
# print(overall_sleep_data)













final_data = np.concatenate((overall_wake_data,overall_sleep_data),axis=0)
# print(final_data)
# print(final_data)


# print(final_data)




########### DT ###################

np.random.shuffle(final_data)

# print(final_data)

# print(final_data)


# wake_counter = 0
# sleep_counter = 0
# for i in range(final_data.shape[0]):
#     if final_data[i][2] == 1:
#         sleep_counter += 1
#     elif final_data[i][2] == 0:
#         wake_counter += 1
#
# print("Wake Counter : ", wake_counter)
# print("Sleep Counter : ", sleep_counter)



# print(final_data.shape[0])






# temp=final_data[:,2]

# print(temp)

# test_size=0.9
# Train_set, Test_set = train_test_split(final_data, test_size=test_size)
# print type(Train_set)
# Input_train = Train_set[:, :4]     # input features
# Target_train = Train_set[:, 4]  # output labels

Input_test = final_data[:, :4]
Target_test = final_data[:, 4]

# clf = tree.DecisionTreeClassifier()
#
# clf = clf.fit(Input_train, Target_train)


###################### Loading Model ##########################
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
PredictedOutcome = loaded_model.predict(Input_test)





# PredictedOutcome = clf.predict(Input_test)


# Neighbours=5
#
# neigh = KNeighborsClassifier(n_neighbors=Neighbours,metric='euclidean')
#
# neigh.fit(Input_train, Target_train)
#
# PredictedOutcome = neigh.predict(Input_test)




# clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
#
# clf.fit(Input_train, Target_train)
#
# PredictedOutcome = clf.predict(Input_test)

################################# Saving the Model #######################################
# filename = 'finalized_model.sav'
# pickle.dump(clf, open(filename, 'wb'))


Number_of_Correct_Predictions = len([i for i, j in zip(PredictedOutcome, Target_test) if i == j])

print ('*******************************************')
print('Number of Correct Predictions using scikit-learn library DecisionTreeClassifier:', Number_of_Correct_Predictions, 'Out_of:', len(PredictedOutcome),
      'Number of Test Data')
print('Accuracy of Prediction in Percent:', (Number_of_Correct_Predictions/float(len(PredictedOutcome)))*100)

##############################################################################################################

wake_counter = 0
sleep_counter = 0
for i in range(Target_test.shape[0]):
    if Target_test[i] == 1:
        sleep_counter += 1
    elif Target_test[i] == 0:
        wake_counter += 1

print("Wake Counter : ", wake_counter)
print("Sleep Counter : ", sleep_counter)