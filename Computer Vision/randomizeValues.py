import random

leftEAR = 0
rightEAR = 0
EYE_AR_THRESH = 0.19
IS_ASLEEP = 0
asleepArray = []
awakeArray = []

EARValuesFile = open("EARValues.txt","w")
#Asleep
for i in range(8889):
    leftEAR = random.uniform(0.1,0.2100001)
    rightEAR = random.uniform(0.1,0.2100001)
    asleepArray.append("{:.4f}".format(leftEAR) + " " + "{:.4f}".format(rightEAR) + " " + str(1) + "\n")
EARValuesFile.writelines(asleepArray)
# asleepEARValuesFile.close()

# EARValuesFile = open("EARValues.txt","w")
#Awake
for i in range(8889):
    leftEAR = random.uniform(0.17,0.4)
    rightEAR = random.uniform(0.17,0.4)
    awakeArray.append("{:.4f}".format(leftEAR) + " " + "{:.4f}".format(rightEAR) + " " + str(0) + "\n")

# random.shuffle(outputArray)
EARValuesFile.writelines(awakeArray)
EARValuesFile.close()

