# engr845_project_KulkarniP : Comparing performance of classifier methods on EMG Hand Gesture Signals using Python.

	The prediction of gestures from EMG data is an active area of research. Such a capability assists amputees, stroke patients, physicians, and humans to control Robotic arms. The objective of this study is to use the publicly available EMG data corresponding to several gestures and predict them using Machine learning methods. 
This project focuses on comparing the performance of different Classifier methods, including both shallow learning and deep learning. The shallow learning methods include Linear Discriminant Analysis (LDA), Naïve Bayes, Support Vector Machine (SVM), and K-Nearest Neighbors (KNN). For the deep learning only classical CNN classifier will be used. 
The publicly available CSL-HDEMG, raw EMG data is used for this study. Features are computed from the EMG data to be fed to the shallow learning methods. In contrast, the EMG data is reformatted into images to be used as input to the CNN. Finally, the performance in predicting the gestures from EMG data using the above listed classifier methods is determined and compared.
**Block Diagram:** 


![image](https://github.com/PallaviK-Git/engr845_project_KulkarniP/assets/22448278/1bbdba59-13e8-43f9-a778-1915517076c1)


	The scope of this project is limited to the performance evaluation of classifier methods. As such, one of the publicly available raw CSL-HDEMG data obtained using the electrode array with 192 single electrodes arranged in a regular 8x24 grid with an inter-electrode distance of 10 mm is used in this study. 
In the feature extraction step, two types of inputs to the two classes of learning methods are generated. As inputs to the shallow learning methods four time-domain features MAV, ZC, SSC (also referred to as TC) and WAV are computed from the raw EMG data. As inputs to the deep learning method, the raw EMG data is reformatted to an image. 
	For classification, two categories of learning methods are chosen. The popular and still widely used shallow learning methods in the literature namely LDA, Naïve Bayes, SVM, and KNN are used in this study. In contrast to the inputs to the deep learning methods, the shallow learning methods are more suitable to certain EMG data obtained using fewer number of probes. The classical deep learning network, Resnet is used for this study.
The output of the classifier methods is the prediction of the gestures. The predicted gestures are compared with the known ground truth gestures during the influence step to determine confusion matrix as the performance metric. 

**Critical aspects:**
	Among the many publicly available data sets, the CSL-HDEMG is chosen because of the acquisition setup specifically the use of electrode array and its placement on the arm. Such a setup is a more realistic representation of the neuronal signals that is transmitted through the arm. The array data also makes it suitable to format it into images and use as inputs to the deep learning methods.
The four time-domain features namely mean absolute value (MAV), zero crossing (ZC), sign slope change (SSC) or turn count (TC) and Willison amplitude value (WAV) are chosen for their ease of implementation, low computational cost, and robustness.
The shallow learning methods are shown [**2**][**3**] to be among the best performing for the gesture recognition using EMG data. The use of deep learning to predict gestures is a very recent and active area of research.
Finally, the confusion matrix is used as the performance metric because of its robustness and simplicity to evaluate the performance of the classification methods.

**Tentative test plan:**
	As mentioned before, the publicly available CSL-HDEMG data set is downloaded from [**1**].
The data is split into training and testing set in the ratio of 80 to 20. The features computed on the training data set is used as input to train the shallow learning methods. The transformed dataset is used to train the deep learning method. The testing data will be similarly transformed and provided as inputs to the methods to compute and evaluate the performance. 
	All the implementation will be done using Python programming language and using Tensorflow library.

**Source:** CSL-HDEMG – The CSL dataset has the sEMG data from 5 subjects, each subject has 5 sessions. Each session has 27 gestures. Each gesture has 10 trials

**Expected Outcome:** With the help of confusion matrix, the performance evaluation of the classifier methods is studied. As for the expected outcome, comparing the accuracy of various classification methods will work. The implementation of the deep learning method using classical CNN classifier might work.

**Reference list entry:**
[1] Amma, Christoph; Krings, Thomas; Böer, Jonas; Schultz, Tanja (2015). [ACM Press the 33rd Annual ACM Conference - Seoul, Republic of Korea (2015.04.18-2015.04.23)] Proceedings of the 33rd Annual ACM Conference on Human Factors in Computing Systems - CHI '15 - Advancing Muscle-Computer Interfaces with High-Density Electromyography, (), 929–938. https://dl.acm.org/doi/abs/10.1145/2702123.2702501

CSL-HDEMG -- https://www.uni-bremen.de/en/csl/research/motion-recognition

[2] C R. N. Khushaba and S. Kodagoda, “Electromyogram (emg) feature reduction using mutual components analysis for multifunction prosthetic
fingers control,” in Control Automation Robotics & Vision (ICARCV),
2012 12th International Conference on. IEEE, 2012, pp. 1534–1539

[3] Chowdhury RH, Reaz MBI, Ali MABM, Bakar AAA, Chellappan K, Chang TG. Surface Electromyography Signal Processing and Classification Techniques. Sensors. 2013; 13(9):12431-12466. https://doi.org/10.3390/s130912431
