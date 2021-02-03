# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import numpy as np
import   tensorflow as tf
import preparing_eap as Prep



from tensorflow.python import keras as TK

from keras import backend as K
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Activation, Input
from keras import optimizers
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_absolute_error
from tensorflow.python.keras.models import model_from_json
#tf.keras.models.model_from_json
from sklearn.utils import class_weight
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.regularizers import l2, l1

def testNumpy():
    arr = np.array([1, 2, 3, 4, 5])
    Xy = 464
    ui = 898.99
    print(Xy + ui)
    print(arr)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #print_hi('PyCharm')
    #testNumpy()
    #os.system('python eap.py 5fcnB phi')
    path = "../inputs/port_list.txt"
    with open(path, 'r') as reader:
        proteinlist = reader.readlines()
        reader.close()
        print(proteinlist)
        for line in proteinlist:
            line = line.strip()
            split_line = line.split()
            os.system('python eap.py %s phi' % split_line[0])
            os.system('python eap.py %s psi' % split_line[0])
            os.system('python eap.py %s theta' % split_line[0])
            os.system('python eap.py %s tau' % split_line[0])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
