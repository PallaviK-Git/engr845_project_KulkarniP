import numpy as np
import sklearn

class FeatureExtract():
    def __init__(self, data, normalizeOpt=0):
        self.data = data
        self.zCrTh = -3.0e-10
        self.tTH = -3.0e-10
        self.normalizeOpt = normalizeOpt

    def normalize(self, vec):
        '''
        This function returs input vector normalized to the range [0, 1]
        '''
        if self.normalizeOpt == 0:
            P = len(vec)
            maxVal = max(vec)
            minVal = min(vec)
            m = 1/(maxVal-minVal+0.1) # slope of the linearly mapping line
            c = -1*m*minVal
            opVec = m*vec + c
        elif self.normalizeOpt == 1:
            opVec = sklearn.preprocessing.scale(vec)
        return opVec

    def ZCrossVec(self, vec, th):
        '''
        This function computes number of Zero-crossings in a vector
        and returns a scalar value
        '''
        adjMult = vec[:-1] * vec[1:]
        ZC = adjMult < th
        nZC = sum(ZC) 
        return nZC

    def zeroCrossing(self, array):
        P, N = array.shape
        nZC = np.zeros((P,1),dtype=int)
        for p in range(P):
            probeVec = array[p,:] # This corresponds to the N (6144) samples from a channel/probe
            nZC[p,0] = self.ZCrossVec(probeVec, self.zCrTh) # The right hand side should result in a scalar
        return nZC

    def firstOrdDiff(self, vec):
        adjSub = vec[1:] - vec[:-1]
        return adjSub

    def WAV(self, array):
        P, N = array.shape
        wav = np.zeros((P,1))
        for p in range(P):
            probeVec = array[p,:] # This corresponds to the N (6144) samples from a channel/probe
            fod = self.firstOrdDiff(probeVec)
            absFod = np.abs(fod)
            meanAbsFod = np.mean(absFod)
            wav[p,0] =  meanAbsFod #np.mean(np.abs(firstOrdDiff(probeVec))) # The right hand side should result in a scalar
        return wav

    def TC(self, array):
        P, N = array.shape
        nTC = np.zeros((P,1),dtype=int)
        for p in range(P):
            probeVec = array[p,:] # This corresponds to the N (6144) samples from a channel/probe
            fod = self.firstOrdDiff(probeVec)
            ZC = self.ZCrossVec(fod, self.tTH)
            nTC[p,0] = ZC #ZCrossVec(firstOrdDiff(probeVec), tTH) # The right hand side should result in a scalar
        return nTC

    def FeatureExtract1(self):
        # data is in the format MxPx(N+1)
        #       where M is number of labeled data and is obtained by concatenating
        #           (nSubjects*nSessions*nGestures) for typeOP=0 or (nSubjects*nSessions*nGestures*nTrials) typeOP=1 or
        #       P is the number of channels   
        #       N is the length of data (typically 6144)
        #       Here, the trials are average 
        M, P, NplusOne = self.data.shape
        numFeat = 5
        rms = np.zeros(P)
        X = np.zeros((M,P*numFeat), dtype=np.float64)
        Y = np.zeros((M,1), dtype=np.float64)
        
        for m in range(M):
            gestArray = self.data[m,:,:-1]
            mean = self.normalize(np.mean(gestArray, axis=1)) # TODO:axis=1 would compute mean along rows resulting in a vector of size P,1
            std = self.normalize(np.std(gestArray, axis=1)) #TODO:axis=1 would compute mean along rows resulting in a vector of size P,1
            # rms[m] += std #np.linalg.norm(trial[c,:]) / sqrt(len(trial[c,:])) / 10
            zc = self.normalize(self.zeroCrossing(gestArray)) #TODO:If training is not satisfactory, potentially change zCrTh
            tc = self.normalize(self.TC(gestArray))
            wav = self.normalize(self.WAV(gestArray))
            X[m,:] = np.hstack((mean.flatten(), std.flatten(), zc.flatten(), wav.flatten(), tc.flatten())) 
            Y[m,0] = self.data[m,0,-1] 
        #rms = rms/10
    #     for m in range(M):
    #         rms[m] += std #np.linalg.norm(trial[c,:]) / sqrt(len(trial[c,:])) / 10
    #     rms = rms/10
        print(X)
        print(Y)
        return [X, Y]

    # #reshaping to the correct shape
    #     rms = np.reshape(rms,(24,7))
    #     rms = np.flipud(np.transpose(rms))