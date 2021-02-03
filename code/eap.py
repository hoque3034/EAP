# Run this program from the folder that contains this file
import sys
import numpy as np
import tensorflow as tf
import preparing_eap as Prep
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input
from keras import optimizers
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_absolute_error
from keras.models import model_from_json
from sklearn.utils import class_weight
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras.optimizers import SGD
from keras import regularizers
from keras.regularizers import l2, l1

ProteinName = sys.argv[1]
AngleName = sys.argv[2]  # phi_theta psi tau

LenSSHotvect, LenPSSM, LenPhysc, SinFlag, WinStep = 8, 20, 7, "", 1

if (AngleName == "phi" or AngleName == "theta" or AngleName == "tau"):
    SinFlag, NormVersion, WinSize, LenOut = "Direct", "range", 2, 4
else:
    SinFlag, NormVersion, WinSize, LenOut = "Sin", "zscore", 6, 8

LenFeatures = LenSSHotvect + LenPhysc + LenPSSM
LenInputNN = (2 * WinSize + 1) * LenFeatures
NNSetting = SinFlag

# Directories
DataFolder = "../inputs/"
BuffFolder = "../data/"
NNFile = BuffFolder + "NN_" + NNSetting + ".json"
WFile = BuffFolder + "W_" + NNSetting + ".hdf5"
MinMeanFile = BuffFolder + "MinMeanMaxSTD_" + NNSetting + ".txt"

# Reading Data
MinMean, MaxStd = Prep.ExtractMinMeanMaxStd(MinMeanFile)
Proteins = [ProteinName]

# Pre-processing
TestInput, TestOutput = Prep.SAPPreprocessing(DataFolder, Proteins, LenFeatures, LenInputNN, LenOut, WinStep, WinSize,
                                              NormVersion, MinMean, MaxStd, SinFlag, 1)

# Reading json file for testing
json_file = open(NNFile, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(WFile)
PredTestOutput = loaded_model.predict(TestInput)

# Printing Prediction

if (AngleName == "psi"):
    path = "../outputs/%s.psi" % ProteinName
    f = open(path, "w+")
    PsiMAE = Prep.CalAESinCos(TestOutput[:, 2:4], PredTestOutput[:, 2:4])
    print("MAE for Psi_Test = " + str(PsiMAE))
    f.write("MAE for Psi_Test = %s\n" % str(PsiMAE))
    # converting to the angle
    PredPsi = Prep.tan_to_angles(PredTestOutput[:, 2:4])
    for i in range(0, PredPsi.shape[0]):
        print(PredPsi[i])
        f.write("%f\n"% PredPsi[i])
    f.close()

elif (AngleName == "phi"):
    path = "../outputs/%s.phi" % ProteinName
    f = open(path, "w+")
    PhiMAE = Prep.CalAEAngles(TestOutput[:, 0:1], PredTestOutput[:, 0:1])
    print("MAE for Phi_Test = " + str(PhiMAE))
    f.write("MAE for Phi_Test = %s\n" % str(PhiMAE))
    for i in range(0, PredTestOutput.shape[0]):
        print(PredTestOutput[i, 0])
        f.write("%f\n"% PredTestOutput[i, 0])
    f.close()

elif (AngleName == "theta"):
    path = "../outputs/%s.theta" % ProteinName
    f = open(path, "w+")
    ThetaMAE = Prep.CalAEAngles(TestOutput[:, 2:3], PredTestOutput[:, 2:3])
    print("MAE for Theta_Test = " + str(ThetaMAE))
    f.write("MAE for Theta_Test = %s\n" % str(ThetaMAE))
    for i in range(0, PredTestOutput.shape[0]):
        print(PredTestOutput[i, 2])
        f.write("%f\n"% PredTestOutput[i, 2])
    f.close()

elif (AngleName == "tau"):
    path = "../outputs/%s.tau" % ProteinName
    f = open(path, "w+")
    TauMAE = Prep.CalAEAngles(TestOutput[:, 3:4], PredTestOutput[:, 3:4])
    print("MAE for Tau_Test = " + str(TauMAE))
    f.write("MAE for Theta_Test = %s\n" % str(TauMAE))
    for i in range(0, PredTestOutput.shape[0]):
        print(PredTestOutput[i, 3])
        f.write("%f\n" % PredTestOutput[i, 3])
    f.close()







































