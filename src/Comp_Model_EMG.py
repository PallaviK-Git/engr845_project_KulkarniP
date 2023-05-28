# example code to load data from the csl-hdemg dataset

import matplotlib.pyplot as plt
import scipy.io as sio
#import hdf5storage
from Features import *
from Data import *
from Models import *

# load the data
def main():
    csl = CSL_IO(path='E:\SFSU\Courses_Term_Wise\SPRING--22\ENGR-456_Computer-Systems--Qin\ResearchWorkon_sEMG\Datasets\CSL_DB-2015\CSL-HDEMG\data', typeOP=1)
    trialDataFromCSL = csl.getData()
    #sio.savemat('CSLFormattedData_gest14To26.mat', {"trailData":trialDataFromCSL})
    #trialDataFromCSL1 = sio.loadmat('CSLFormattedData_gest14To26.mat')
    #hdf5storage.savemat('CSLFormattedData_gest14To26.mat', {"trailData":trialDataFromCSL}, format=7.3, matlab_compatible=True, compress=False)
    #trialDataFromCSL1 = hdf5storage.loadmat('CSLFormattedData_gest14To26.mat')
    #FeatExt = FeatureExtract(data=trialDataFromCSL1['trailData'])
    FeatExt = FeatureExtract(data=trialDataFromCSL)#['trailData'])
    X, Y = FeatExt.FeatureExtract1()
    #SVMperformance = SVM(X, Y)
    #print("SVM performance",SVMperformance)
    #SVM(X, Y)
    #KNNPerformance = KNN(X, Y)
    #LDAPerformance = LDA(X, Y)
    #NaiveBayesPerformance = NB(X, Y)
    #MLPImplement = DNN(X, Y)
    #SVM(X, Y)
    #KNN(X, Y)
    #LDA(X, Y)
    #NB(X, Y)
    DNN(X, Y)

if __name__ == '__main__':
    main()