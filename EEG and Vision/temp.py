import os
import numpy as np
import random
# Set the directory you want to start from
rootDir = 'sleep-cassette\\'
for dirName, subdirList, fileList in os.walk(rootDir):
    print('Found directory: %s' % dirName)
    for fname in fileList:
        print('\t%s' % fname)


print(fileList.index("SC4202E0-PSG.edf"))
print(fileList[1])


a=[1,2,3,4]

print (a)
np.random.shuffle(a)
print (a)