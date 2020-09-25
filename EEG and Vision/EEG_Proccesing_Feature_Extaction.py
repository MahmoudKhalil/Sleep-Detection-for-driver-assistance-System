import pyedflib
import mne
import math
import scipy
import matplotlib.pyplot as plt
import numpy as np
import os

########################## FUNCTIONS #################################

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut /nyq
    high = highcut/nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    # print(b,a)
    y = scipy.signal.lfilter(b, a, data)
    return y

def apply_MMD(current_epoch):

    sum=0
    for j in range(0,10):
        # print(j)
        current_sub_epoch = current_epoch[(j * 100):(100 * (j + 1))]
        y_min=np.min(current_sub_epoch)
        y_max=np.max(current_sub_epoch)

        x_min=np.where(current_sub_epoch==y_min)[0]+1
        x_max=np.where(current_sub_epoch==y_max)[0]+1

        x_diff = x_max - x_min
        y_diff = y_max - y_min

        x_sq = math.pow(x_diff,2)
        y_sq = math.pow(y_diff,2)

        total_distance = x_sq + y_sq

        sum+= math.sqrt(total_distance)



    return sum

def apply_esis(current_epoch,v):

    sum=0
    for i in range (0,current_epoch.__len__()):

        sum+=math.pow(current_epoch[i],2)*v

    return sum


###########################################################################


rootDir = 'sleep-cassette\\'
for dirName, subdirList, fileList in os.walk(rootDir):
    print('Found directory: %s' % dirName)
    for fname in fileList:

        print('\t%s' % fname)

########################### READING FROM FILE #####################################

id=2

file = "sleep-cassette\\"+fileList[id]
data = mne.io.read_raw_edf(file)
raw_data = data.get_data()
# you can get the metadata included in the file and a list of all channels:
info = data.info
channels = data.ch_names
# print (info)
# print (channels)
# print (raw_data.shape[0])
# print (raw_data.shape[1])
# print (channels[0],raw_data[0][:])
# print (channels[0],raw_data[0].__len__())
# print (raw_data.shape[0])

time_domain_data=raw_data[0]
print (time_domain_data.__len__())


number_of_epoches=math.ceil((time_domain_data.__len__()/1000))

epoches_data=np.zeros((number_of_epoches,10))



# a=[1,2,3,4,5]
# b=scipy.fft(a)
# print(a)
# print(b)
# print(scipy.ifft(b))

# print(time_domain_data[0:5])

# plt.plot(time_domain_data)
# plt.show()

#########################################################################





################# DATA PROCESSING #######################





#epoches

# el mafroud asheel el 1 w a7ot number_of_epoches
for i in range(0,number_of_epoches):
    current_epoch=time_domain_data[(i*1000):(1000*(i+1))]
    frequency_domain_epoch=scipy.fft(current_epoch)
    # print(current_epoch[1])
    # print(frequency_domain_epoch[1])
    delta_frequency_domain = butter_bandpass_filter(frequency_domain_epoch, 0.00001, 4.0, 100)
    theta_frequency_domain = butter_bandpass_filter(frequency_domain_epoch, 4.0, 8.0, 100)
    alpha_frequency_domain = butter_bandpass_filter(frequency_domain_epoch, 8.0, 13.0, 100)
    beta_frequency_domain = butter_bandpass_filter(frequency_domain_epoch, 13.0, 22.0, 100)
    gamma_frequency_domain = butter_bandpass_filter(frequency_domain_epoch, 30.0, 45.0, 100)

    # plt.plot(current_epoch)
    # plt.show()

    # plt.plot(alpha_frequency_domain)
    # plt.show()
    # plt.plot(delta_frequency_domain)
    # plt.plot(theta_frequency_domain)
    # plt.plot(beta_frequency_domain)
    # plt.plot(gamma_frequency_domain)
    # plt.show()

    # plt.plot(current_epoch)
    # plt.show()

    # Time Domain
    delta_time_domain = np.real(scipy.ifft(delta_frequency_domain))
    theta_time_domain = np.real(scipy.ifft(theta_frequency_domain))
    alpha_time_domain = np.real(scipy.ifft(alpha_frequency_domain))
    beta_time_domain = np.real(scipy.ifft(beta_frequency_domain))
    gamma_time_domain = np.real(scipy.ifft(gamma_frequency_domain))



    ##########################################################################################



    ###################### Feature Extraction ##########################

    #MMD

    # print (delta_time_domain)
    delta_MMD = apply_MMD(delta_time_domain)
    theta_MMD = apply_MMD(theta_time_domain)
    alpha_MMD = apply_MMD(alpha_time_domain)
    beta_MMD = apply_MMD(beta_time_domain)
    gamma_MMD = apply_MMD(gamma_time_domain)

    # print (delta_MMD)


    #Esis

    delta_v = 100 * ((4+0)/2)
    theta_v = 100 * ((4+8)/2)
    alpha_v = 100 * ((8+13)/2)
    beta_v = 100 * ((13+22)/2)
    gamma_v = 100 * ((30+45)/2)


    delta_esis = apply_esis(delta_time_domain,delta_v)
    theta_esis = apply_esis(theta_time_domain,theta_v)
    alpha_esis = apply_esis(alpha_time_domain,alpha_v)
    beta_esis = apply_esis(beta_time_domain,beta_v)
    gamma_esis = apply_esis(gamma_time_domain,gamma_v)


    epoches_data[i,0]=delta_MMD
    epoches_data[i,1]=theta_MMD
    epoches_data[i,2]=alpha_MMD
    epoches_data[i,3]=beta_MMD
    epoches_data[i,4]=gamma_MMD

    epoches_data[i,5]=delta_esis
    epoches_data[i,6]=theta_esis
    epoches_data[i,7]=alpha_esis
    epoches_data[i,8]=beta_esis
    epoches_data[i,9]=gamma_esis



    # print(delta_esis)

    # print(epoches_data)
    # print(delta_MMD)





    # print(delta_time_domain)
    # plt.plot(delta_time_domain)
    # plt.show()


# a=np.zeros((2,2))
# a=[0]*4
# a[0]=1
# a[1]=2
# a[2]=5
# a[3]=-3
#
# print(np.where(a==np.min(a))[0])
# # print(np.where(a==1))


###############################################################################################


########################################### Gathering Data ##################################################


file_name = "sleep-cassette\\"+fileList[id+1]
f = pyedflib.EdfReader(file_name)
# print(f)
annotations = f.readAnnotations()
# print(annotations)
# print(annotations[0].__len__())
# print(annotations[2].__len__())
seconds= annotations[0]
identification=annotations[2]




wake_data=[]
sleep_1_data=[]
for i in range(identification.__len__()-1):
    if(identification[i]=='Sleep stage W' or identification[i]=='Sleep stage 1'):
        start = int(seconds[i])
        end = int(seconds[i + 1])
        interval = seconds[i + 1] - seconds[i]
        epoch_range = int(interval / 10)
        start_epoch = int(start / 10)
        end_epoch = int(end / 10)
        if(identification[i]=='Sleep stage W'):

            for j in range(start_epoch,end_epoch):
                wake_data.append(epoches_data[j])
            # print (wake_data[0])
            # print (wake_data.__len__())

        elif(identification[i]=='Sleep stage 1'):
            for j in range(start_epoch,end_epoch):
                sleep_1_data.append(epoches_data[j])

#
textfile=""
for i in range (wake_data.__len__()):
    for j in range (wake_data[i].__len__()):
        textfile+=str(wake_data[i][j])+" "
    textfile+="\n"
F = open("wake_data_2files.txt","a")
F.write(textfile)
F.close()



textfile=""
for i in range (sleep_1_data.__len__()):
    for j in range (sleep_1_data[i].__len__()):
        textfile+=str(sleep_1_data[i][j])+" "
    textfile+="\n"
F = open("sleep_1_data_2files.txt","a")
F.write(textfile)
F.close()



# print(wake_data)
# print(sleep_1_data)


##################################################################################################


########################## Appling PCA ##################################

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd


######## Wake Data ######

# x = StandardScaler().fit_transform(wake_data)
#
# pca = PCA(n_components=2)
#
# principalComponents = pca.fit_transform(x)
#
# principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
#
#
# print(principalDf)
#





#####################################################################






################################################ Ploting #####################################################

# fig, ax = plt.subplots()
# x=[]
# y=[]
# x2=[]
# y2=[]
# for i in range(wake_data.__len__()):
#     x.append(wake_data[i][2])
#     y.append(wake_data[i][7])
#
# for i in range(sleep_1_data.__len__()):
#     x2.append(sleep_1_data[i][2])
#     y2.append(sleep_1_data[i][7])
#
# # Y = np.array([0.0, 0.001, 0.003, 0.2, 0.4, 0.5, 0.7, 0.88, 0.9, 1.0])
# # Y2 = np.repeat(Y,4)
# # print(Y2)
# plt.ylim(0.0000,0.00030 )
# ax.scatter(x,y,color=['blue'])
# ax.scatter(x2,y2,color=['red'])
# plt.show()


#################################################################################





