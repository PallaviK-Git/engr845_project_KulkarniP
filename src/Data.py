import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import os
from os.path import dirname, join as pjoin
import glob
import scipy.io as sio
import re

logLevel = 0 # 0: No logging; 1: Func level; 2: Major steps; 3: Minor steps

class CSL_IO():
    def __init__(self, path, typeOP):
        self.tLP = path # tLP - topLevelPath
        # typeOP is type of output data format. 
        # If 0, the format is an array of size MxPx(N+1), 
        #       where M is number of labeled data and is obtained by concatenating
        #           (nSubjects*nSessions*nGestures)
        #       P is the number of channels 
        #       N is the length of data (typically 6144)
        #       Here, the trials are averaged
        # If 1, the format is an array of size MxPx(N+1), 
        #       where M is number of labeled data and is obtained by concatenating
        #           (nSubjects*nSessions*nTrials*nGestures)
        #       P is the number of channels 
        #       N is the length of data (typically 6144)
        # The last column in both above formats is the gesture label
        self.typeOP = typeOP
        self.nSubjects = 5#2#5
        self.nSessions = 5#2#5
        self.nTrials = 9#2#9 # 10; Changed coz Sub4-Sess4-Gest8 has only 9 trials data
        self.nGestures = 22 #14 works
        self.nChs = 168
        self.lenOfData = 6144

    def getData(self):
        if self.typeOP == 0:
            trialsData = np.zeros((self.nSubjects*self.nSessions*self.nGestures, self.nChs, self.lenOfData+1), dtype=np.float64)  #dtype=np.float32
        elif self.typeOP == 1:
            trialsData = np.zeros((self.nSubjects*self.nSessions*self.nTrials*self.nGestures, self.nChs, self.lenOfData+1), dtype=np.float64) #default dtype=np.float64
        with open('CSLData.txt', 'w') as fp:
            for sub in range(self.nSubjects):
                iSub = (sub*self.nSessions*self.nGestures)
                for sess in range(self.nSessions):
                    iSess = (sess*self.nGestures)
                    #for gest in range(self.nGestures):  #for 0-13 gestures
                    for gest in range(self.nGestures): #14,self.nGestures+14):   #for next 13-26 gestures
                        iGest = (gest*self.nTrials) #((gest-14)*self.nTrials)
                        fn = os.path.join(self.tLP, 'subject'+str(sub+1), 'session'+str(sess+1), 'gest'+str(gest)+'.mat')
                        # #if logLevel > 2:
                        #     if os.path.isfile(fn): fp.write(fn+' exists :) \n')
                        #     else: fp.write(fn+' doesnt exist !!! \n')
                        matDict = sio.loadmat(fn)
                        gestTrialData = matDict['gestures']
                        # if logLevel > 2:
                        #     fp.write(str(gestTrialData.shape)+'\n')
                        # print(fn) #For printing the gesture .mat file path
                        if self.typeOP == 0:
                            tData = np.zeros((self.nChs, self.lenOfData))
                            for trial in range(self.nTrials):
                                trialData = gestTrialData[trial,0]
                                trialData = np.delete(trialData,np.s_[7:192:8],0)
                                tData = tData + trialData
                                # if logLevel > 2:
                                #     fp.write(str(trialData.shape)+'\n')
                            tData = tData / self.nTrials
                            trialsData[iSub+iSess+gest, :, :-1] = tData
                            trialsData[iSub+iSess+gest, :, -1] = gest # TODO: Verify if gest needs to be converted to vector
                        elif self.typeOP == 1:
                            for trial in range(self.nTrials):
                                trialData = gestTrialData[trial,0]
                                trialData = np.delete(trialData,np.s_[7:192:8],0)
                                trialsData[iSub+iSess+iGest+trial, :, :-1] = trialData
                                trialsData[iSub+iSess+iGest+trial, :, -1] = gest # TODO: Verify if gest needs to be converted to vector
                                # if logLevel > 2:
                                #     fp.write(str(trialData.shape)+'\n')

        return trialsData