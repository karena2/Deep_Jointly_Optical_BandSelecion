from functions import *
from models import *
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
import os.path as path

semilla = [42,78,22,70,35,58,22,20,0,23]

set_seed(0) #fixed for dataset
# @title RUN

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #(or "1" or "2")
gpus = tf.config.list_physical_devices('GPU')

dataset = "Sal"

if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit= 2*1024)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

if dataset == "IP":
    data_path = "Indian_pines_corrected"
    gt_path = "Indian_pines_gt"
    Spectral_data = loadmat('Data/' + data_path + '.mat').get('hyperimg')
    gt = loadmat('Data/' + gt_path + '.mat').get('hyperimg_gt')
if dataset == "Sal":
    data_path = "Salinas_corrected"
    gt_path = "Salinas_gt"
    Spectral_data = loadmat('Data/' + data_path + '.mat').get('hyperimg')
    gt = loadmat('Data/' + gt_path + '.mat').get('hyperimg_gt')

if dataset == "PU":
    data_path = "PaviaU"
    gt_path = "PaviaU_gt"
    Spectral_data = loadmat('Data/' + data_path + '.mat').get('paviaU')
    gt = loadmat('Data/' + gt_path + '.mat').get('paviaU_gt')

(M, N, L) = Spectral_data.shape
Spectral_data = np.reshape(Spectral_data, [M*N,L])
gt = np.reshape(gt, [M*N])

mean_all = np.mean(Spectral_data,0)
std_all = np.std(Spectral_data,0)
Spectral_data=(Spectral_data-mean_all)/std_all

Portion = [0.7,0.1,0.2] # Training, Validation, Test #@param {type:"raw"}
X_Train, Y_Train, X_Val, Y_Val, X_Test, Y_Test = split_data(Spectral_data,gt,Portion)
X_Train = np.expand_dims(X_Train, -1)
#X_Train = X_Train / np.max(X_Train)

X_Val = np.expand_dims(X_Val, -1)
#X_Val = X_Val / np.max(X_Val)

X_Test = np.expand_dims(X_Test, -1)
#X_Test = X_Test / np.max(X_Test)

batch_size = 32
N_epochs = 100
patience = 80
precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()

Seed = []
Acc = []
Loss = []
Bands = []

band_temps = [[7,11,20,24,32,38,41,46,55,68], #TRC-OC-FDPC
[8,15,25,32,45,58,93,119,125,135],#NC-OC-MVPCA
[8,15,25,34,45,58,93,120,125,135],#NC-OC-IE
[106,142,11,204,149,7,41,59,146,104],#SC-RDFBSS-SIDAM-MIN
[9,25,37,41,57,67,75,81,97,137],#Proposed
[np.arange(1,L+1)]] #Full

metodo = ['TRC-OC-FDPC','NC-OC-MVPCA','NC-OC-IE','SC-RDFBSS-SIDAM-MIN','Proposed','Full']

bandas = np.shape(band_temps[0])
for method in range(len(band_temps)):
    for band in range(len(band_temps[method])):
        band_temps[method][band] = band_temps[method][band]-1 

prueba = 0
for band_temp in band_temps:
    for ind_model in range(len(bandas)):
        name = 'Number_bands_' + str(bandas[ind_model])+"_Prueba_"+str(prueba)
        #model.summary()
        cwd = os.getcwd() + "/Results/Proposed_"+str(dataset)+"_final_bands_CNN/"
        try:
            os.stat(cwd)
        except:
            os.mkdir(cwd)
        print(name)
        save_path = cwd + name
        try:
            os.stat(save_path)
        except:
            os.mkdir(save_path)
        #name = 'Seed_' + str(seed)

        for seed in semilla:
            set_seed(seed)
            model = My_network200(input_size=(L, 1, 1), num_classes=np.max(np.unique(gt)))
            X_Train_temp = np.zeros(X_Train.shape)
            X_Train_temp[:,band_temp,:] =X_Train[:,band_temp,:]
            X_Test_temp = np.zeros(X_Test.shape)
            X_Test_temp[:, band_temp, :] = X_Test[:,band_temp, :]
            X_Val_temp = np.zeros(X_Val.shape)
            X_Val_temp[:, band_temp, :] = X_Val[:,band_temp, :]

            reload_callback = Reload2(save_path=save_path + '/', patience=patience, save= 0)
            optimizad = tf.keras.optimizers.Adam(learning_rate=1e-3)
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.9, patience=patience,
                                                             min_lr=1e-5, mode='max')

            model.compile(optimizer=optimizad, loss='categorical_crossentropy',
                          metrics=['accuracy', precision, recall])

            history = model.fit(np.expand_dims(X_Train_temp, -1), Y_Train, validation_data=(np.expand_dims(X_Val_temp, -1), Y_Val),
                                batch_size=batch_size, epochs=N_epochs,
                                callbacks=[reload_callback])

            ## Save Results
            model.save_weights(save_path + '/last_weights.h5')
            Results_last = model.evaluate(np.expand_dims(X_Test_temp, -1), Y_Test)
            model.load_weights(save_path + '/best_weights.h5')
            Results_Best = model.evaluate(np.expand_dims(X_Test_temp, -1), Y_Test)

            Seed.append(seed)
            Acc.append(Results_Best[1])
            Loss.append(Results_Best[0])
            Bands.append(bandas[ind_model])

            with open(save_path + '/Selected_complete_results.txt', 'a') as f:
                f.write(';' + (str(seed)))  # Semilla
                f.write('; ' + str(Results_Best[1]))  # Accuracy
                f.write('; ' + str(Results_Best[0]))  # Loss
                f.write('; ' + str(bandas[ind_model]))  # Bandas
                f.write('; ' + (str(np.squeeze(band_temp))) + "\n") #which bands

            Seed.append(seed)
            Acc.append(Results_Best[1])
            Loss.append(Results_Best[0])

            with open(cwd + '/Proposed_complete_results.txt', 'a') as f:
                f.write('; Semilla = ' + (str(seed)))  # Semilla
                f.write('; Accuracy = ' + str(Results_Best[1]))  # Accuracy
                f.write('; Loss = ' + str(Results_Best[0]))  # Loss
                f.write('; Metodo = ' + (str(metodo[prueba]))+ '\n'),  # Metodo

        with open(cwd + '/Final_Selected_results.txt', 'a') as f:
            f.write('\nNumber of bands = ' + str(bandas[ind_model]) + ' PACIENCE = ' + str(patience) + ' BATCH SIZE = ' + str(batch_size) + ' N. EPOCHS = ' + str(
                N_epochs))
            f.write('\nNumber of seeds = ' + (str(np.size(semilla)))),  # Semilla
            f.write('\nMean of the accuracy = ' + str(np.mean(Acc))),  # Accuracy
            f.write('\nStandard deviation of the accuracy = ' + str(np.std(Acc))),  # Accuracy
            f.write('\nMetodo = ' + (str(metodo[prueba]))),  # Metodo
            f.write('\nMean of the loss = ' + str(np.mean(Loss))),  # Loss
            f.write('\nStandard deviation of the loss = ' + str(np.std(Loss))),  # Accuracy
            f.write('\nMean of the number of bands = ' + str(np.mean(Bands))),  # Bandas
            f.write('\nStandard deviation of the number of bands = ' + str(np.std(Bands))),  # Accuracy

        Seed = []
        Acc = []
        Loss = []
        Bands = []
    prueba += 1






