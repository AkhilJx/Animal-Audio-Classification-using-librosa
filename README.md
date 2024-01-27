# Audio-Classification-using-various-ML-Algorithms
A Comparative study to see the performance of various ML Algorithms in classifying an audio file.

In this program, we will be taking 50 audio files (can be increased based on the requirement) to train various ml models so that their efficiencies can be compared. The model will be evaluated to predict the sound of animal that is fed to the model.

# Dataset:
50 Audio files having the sound of 5 animals (each having 10 sounds) were taken for model building. All the audio fies are taken from internet (https://wavlist.com/). The audio file will be loaded into the librosa library to extract the relevant features needed for distinguishing an audio file.

The librosa library has inbuilt functions to extract the features of an audio file. The Important features used in this program for classifying the audio files are:

1. chroma_stft
2. rmse
3. spectral_centroid
4. spectral_bandwidth
5. spectral_rolloff
6. mfccs (Mel Frequency Cepstral Coefficients)

After extracting the above 6 features from the audio, it is then saved as a csv file which will be used as the dataset for building the models. 

# Libraries Used:
1.librosa                       : for extracting the audio features
2.pandas                        : for data handling
3.os                            : for accessing the folder
4.numpy                         : for array operation
5.re                            : for pattern matching
6.enum                          :  create sets of models used
7.IPython                       : for playing the audio file
8.matplotlib.pyplot and seaborn : for visualization
9.sklearn                       : for performing minmax scaler, test-train split, confusion matrix, model building etc..........
10.pickle                       : for saving the model.

# Algorithms Used:
1. LOGISTIC_REGRESSION
2. RIDGE Classifier
3. K_Nearest_NEIGHBORS
4. Support Vector Classifier
5. DECISION_TREE Classifier
6. RANDOM_FOREST Classifier
7. ADA_BOOST Classifier

# Steps Involved:
1. load the audio files into librosa library
2. extract the import features and save it as a csv file
3. load the csv file and scale it.
4. split the data into test and train and build model
5. evaluate the model
6. predict based on new audio
7. compare the various classification models used.


Note: The efficiency of the model can be increased by increasing the size of the trainig data.
 


