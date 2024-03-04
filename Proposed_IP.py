from functions import *
from models import *
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os.path as path

semilla = [58]

set_seed(0) #fixed for dataset
# @title RUN

dataset = "IP"

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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #(or "1" or "2")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=2*1024)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

(M, N, L) = Spectral_data.shape #comentar para carbono
Spectral_data = np.reshape(Spectral_data, [M*N,L])
gt = np.reshape(gt, [M*N])
mean_all = np.mean(Spectral_data,0)
std_all = np.std(Spectral_data,0)
Spectral_data=(Spectral_data -mean_all)/std_all

Portion = [0.7,0.1,0.2] # Training, Validation Spectral_data=(Spectral_data -mean_all)/std_alln, Test #@param {type:"raw"}
X_Train, Y_Train, X_Val, Y_Val, X_Test, Y_Test = split_data(Spectral_data,gt,Portion)
X_Train = np.expand_dims(X_Train, -1)
X_Val = np.expand_dims(X_Val, -1)
X_Test = np.expand_dims(X_Test, -1)

batch_size = 32
N_epochs = 200
patience = 80
precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()

Seed = []
Acc = []
Loss = []
Bands = []
Accl = []
Lossl = []
Bandsl = []

temp = 3
alpha = 1
betas = [1] 
parm1s = [1e-6]
bandas = [10]
learning_rates = [1e-3]
p_aums =[2]
p_steps =[10]

for ind_model in range(len(bandas)):
    for beta in betas:
        for parm1 in parm1s:
            for p_aum in p_aums:
                for p_step in p_steps:
                    for learning_rate in learning_rates:
                        for seed in semilla:
                            name = "Seed_" + str(seed) + "_Number_bands_" + str(bandas[ind_model])+"_temp_"+str(temp)+"_beta_"+str(beta)+"_param1_"+str(parm1s)+"_lr_"+str(learning_rate)+"_psteps_"+str(p_step)
                            cwd = os.getcwd() + "/Results/Proposed_IP/"
                            beta_in = beta
                            parm1_in = parm1
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

                            set_seed(seed)
                            model = My_networkE2E(input_size=(L, 1, 1), num_classes=np.max(np.unique(gt)), alpha=alpha, bands=bandas[ind_model], parm1=parm1, beta=beta)
                            reload_callback = Reload(save_path=save_path + '/', patience=patience, save= 150)
                            inspector = Inspect_Training(X_Test, Y_Test, [], [], None, None, save_path, seed, 2)
                            optimizad = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                            aument = Aument_parameters(p_aum=p_aum, p_step=p_step,temp= temp, bandas=bandas[ind_model]) #Estaba en 8
                            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.9, patience=patience,
                                                                            min_lr=1e-5, mode='max')
                            model.compile(optimizer=optimizad, loss='categorical_crossentropy',
                                        metrics=['accuracy', precision, recall])

                            history = model.fit(np.expand_dims(X_Train, -1), Y_Train, validation_data=(np.expand_dims(X_Val, -1), Y_Val),
                                                batch_size=batch_size, epochs=N_epochs,
                                                callbacks=[reload_callback,aument,inspector, reduce_lr])

                            plt.plot(history.history['accuracy'])
                            plt.plot(history.history['val_accuracy'])
                            plt.title('Model accuracy')
                            plt.ylabel('Accuracy')
                            plt.xlabel('Epoch')
                            plt.legend(['Train', 'Validation'], loc='upper left')
                            #plt.show()
                            plt.savefig(save_path + '/Acc.png')


                            w = np.asarray(model.weights[0])
                            w[w>=0.5]=1
                            w[w<0.5]=0
                            bb1 = np.sum(w)
                            ## Save Results
                            model.save_weights(save_path + '/last_weights.h5')
                            w = np.asarray(model.weights[0])
                            w[w>=0.5]=1
                            w[w<0.5]=0
                            bb1 = np.sum(w)
                            model.layers[1].set_weights(tf.expand_dims(w,0))
                            Results_last = model.evaluate(np.expand_dims(X_Test, -1), Y_Test)
                            bandas_selected_last = np.where(w==1)

                            model.load_weights(save_path + '/best_weights.h5')
                            w = np.asarray(model.weights[0])
                            w[w>=0.5]=1
                            w[w<0.5]=0
                            bb2 = np.sum(w)
                            model.layers[1].set_weights(tf.expand_dims(w,0))
                            Results_Best = model.evaluate(np.expand_dims(X_Test, -1), Y_Test)
                            bandas_selected = np.where(w==1)

                            Seed.append(seed)
                            Acc.append(Results_Best[1])
                            Loss.append(Results_Best[0])
                            Bands.append(bb2)

                            Accl.append(Results_last[1])
                            Lossl.append(Results_last[0])
                            Bandsl.append(bb1)

                            with open(cwd + '/Proposed_complete_results.txt', 'a') as f:
                                f.write(';' + (str(seed)))  # Semilla
                                f.write('; ' + str(Results_Best[1]))  # Accuracy
                                f.write('; ' + str(Results_Best[0]))  # Loss
                                f.write('; ' + str(bb2))  # Bandas
                                f.write(' Beta = ' + (str(beta_in))),  # Semilla
                                f.write(' Param1 = ' + (str(parm1_in))),  # Semilla
                                f.write(' P_aum = ' + (str(p_aum))),  # Semilla
                                f.write(' P_step = ' + (str(p_step))),  # Semilla
                                f.write('; ' + (str(np.squeeze(bandas_selected[1]))) + "\n")  # which bands

                            with open(cwd + '/Proposed_complete_results_last.txt', 'a') as f:
                                f.write(';' + (str(seed)))  # Semilla
                                f.write('; ' + str(Results_last[1]))  # Accuracy
                                f.write('; ' + str(Results_last[0]))  # Loss
                                f.write('; ' + str(bb1))  # Bandas
                                f.write('; ' + (str(np.squeeze(bandas_selected_last[1]))) + "\n")  # which bands

                        with open(cwd + '/Final_Proposed_results.txt', 'a') as f:
                            f.write(
                                '\nNumber of bands = ' + str(bandas[ind_model]) + ' PACIENCE = ' + str(patience) + ' BATCH SIZE = ' + str(
                                    batch_size) + ' N. EPOCHS = ' + str(
                                    N_epochs))
                            f.write('\nNumber of seeds = ' + (str(np.size(semilla)))),  # Semilla
                            f.write('\nBeta = ' + (str(np.size(beta)))),  # Semilla
                            f.write('\nParam1 = ' + (str(np.size(parm1)))),  # Semilla
                            f.write('\nP_aum = ' + (str(np.size(p_aum)))),  # Semilla
                            f.write('\nP_step = ' + (str(np.size(p_step)))),  # Semilla
                            f.write('\nMean of the accuracy = ' + str(np.mean(Acc))),  # Accuracy
                            f.write('\nStandard deviation of the accuracy = ' + str(np.std(Acc))),  # Accuracy
                            f.write('\nMean of the loss = ' + str(np.mean(Loss))),  # Loss
                            f.write('\nStandard deviation of the loss = ' + str(np.std(Loss))),  # Accuracy
                            f.write('\nMean of the number of bands = ' + str(np.mean(Bands))),  # Bandas
                            f.write('\nStandard deviation of the number of bands = ' + str(np.std(Bands))),  # Accuracy

                        with open(cwd + '/Final_Proposed_results_last.txt', 'a') as f:
                            f.write(
                                '\nNumber of bands = ' + str(bandas[ind_model]) + ' PACIENCE = ' + str(patience) + ' BATCH SIZE = ' + str(
                                    batch_size) + ' N. EPOCHS = ' + str(
                                    N_epochs))
                            f.write('\nNumber of seeds = ' + (str(np.size(semilla)))),  # Semilla
                            f.write('\nMean of the accuracy = ' + str(np.mean(Accl))),  # Accuracy
                            f.write('\nStandard deviation of the accuracy = ' + str(np.std(Accl))),  # Accuracy
                            f.write('\nMean of the loss = ' + str(np.mean(Lossl))),  # Loss
                            f.write('\nStandard deviation of the loss = ' + str(np.std(Lossl))),  # Accuracy
                            f.write('\nMean of the number of bands = ' + str(np.mean(Bandsl))),  # Bandas
                            f.write('\nStandard deviation of the number of bands = ' + str(np.std(Bandsl))),  # Accuracy

                        Seed = []
                        Acc = []
                        Loss = []
                        Bands = []
                        Accl = []
                        Lossl = []
                        Bandsl = []