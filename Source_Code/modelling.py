import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#@title Library  { form-width: "30%" }
import tensorflow.keras as keras
from scipy.stats import entropy
import tensorflow as tf
import numpy as np
import time
from sklearn.metrics import roc_curve
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft
from scipy.stats import gennorm
import numpy as np
from scipy.special import gamma
import math

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing
from scipy.signal import savgol_filter
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D


# from utils.utils import save_logs
# from utils.utils import calculate_metrics

#@title DATA  { form-width: "30%" }
def find_energy(signal, snr):
  Y = fft(signal)
  abs_Y = abs(Y)**2
  r = np.mean(abs_Y)
  return 2*snr*np.sum(r)
def find_de(signal,beta):
  alpha = np.log(beta*(np.sum(abs(signal-np.mean(signal))**beta))/len(signal))/beta
  return 1/beta - np.log(beta/(2*math.gamma(1/beta))) + alpha
def find_gp(signal):
  return np.exp(np.mean(np.log(abs(np.array(signal)))))
def find_lp(signal,p):
  lp = np.mean(abs(signal)**p)
  return lp
def normalizeData(raw_data):
  data = (raw_data - np.mean(raw_data))/np.sqrt(np.var(raw_data))
  return data

batch = 50
df = pd.read_csv(r"dataset3.csv")
raw_data = df.iloc[:,0]
raw_data = raw_data.to_numpy()
signal = normalizeData(raw_data[0:50000])
lenSample = len(signal)
# beta = 1.5
TIME_PERIODS = 1

def createTrainTest(data,Features):
  x = data[:,0:Features]
  y = data[:,-1]
  return x,y

def createFeature(beta,snrDB,p=0.5):
  var = 1
  alpha = np.sqrt((var*gamma(1/beta))/(gamma(3/beta)))

  snr = 10**(snrDB/10);
  featuresMatrix = np.zeros(shape=(2000,5))

  for i in range(0,lenSample,batch):
    if i+batch <= lenSample:
      noise = gennorm.rvs(beta, size=batch,scale = alpha)

      h1 = noise + np.sqrt(snr)*signal[i:i+batch]
      h0 = gennorm.rvs(beta, size=batch,scale = alpha)
      # features for h1
      energy = find_energy(h1,snr)
      de = find_de(h1,beta)
      gp = find_gp(h1)
      lp = find_lp(h1,p)

      featuresMatrix[i//batch] = [gp,de,lp,energy,1]

      # features for h0
      energy = find_energy(h0,snr)
      de = find_de(h0,beta)
      gp = find_gp(h0)
      lp = find_lp(h0,p)

      featuresMatrix[(lenSample+i)//batch] = [gp,de,lp,energy,0]


  featuresMatrix[:,0] = featuresMatrix[:,0]/max(featuresMatrix[:,0])
  featuresMatrix[:,1] = featuresMatrix[:,1]/max(featuresMatrix[:,1])
  featuresMatrix[:,2] = featuresMatrix[:,2]/max(featuresMatrix[:,2])
  featuresMatrix[:,3] = featuresMatrix[:,3]/max(featuresMatrix[:,3])

  np.random.shuffle(featuresMatrix)

  return featuresMatrix

def findPd(pf,pd):
  temp = np.where((pf<=0.1) & (pf>0.0))[0]
  if temp.size>0 and pd[temp[-1]] != 0.0:
    return pd[temp[-1]]
  else:
    return 1.0

snrDB = np.flip(np.arange(-20,25,5))

beta = 1

pd_cnn = np.ones(len(snrDB))
pd_mlp = np.ones(len(snrDB))
pd_fcn = np.ones(len(snrDB))
pd_rnet =np.ones(len(snrDB))
pd_lstm =np.ones(len(snrDB))
pd_tcn = np.ones(len(snrDB))

num_time_periods = 1
num_classes = 1
num_sensors=4

# Supervised Machine Learning Algorithms


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
pd_knn=np.ones(len(snrDB))
pf_knn=np.ones(len(snrDB))
for i in range(len(snrDB)):

  featuresMatrix=createFeature(beta,snrDB[i])

  df_train,df_test = train_test_split(featuresMatrix, test_size=0.2, random_state=42)
  x_train, y_train = createTrainTest(df_train,num_sensors)
  x_test, y_test = createTrainTest(df_test,num_sensors)

  input_shape = (num_time_periods*num_sensors)

  x_train = x_train.reshape(x_train.shape[0], input_shape)

  x_train = x_train.astype("float32")
  y_train = y_train.astype("float32")

  x_test = x_test.reshape(x_test.shape[0], input_shape)

  x_test = x_test.astype("float32")
  y_test = y_test.astype("float32")
  print(snrDB[i])
  model_knn  = KNeighborsClassifier(n_neighbors=20);
  model_knn.fit(x_train, y_train)
  y_pred = model_knn.predict_proba(x_test)[: ,1]
  fpr_knn, tpr_knn, _ = roc_curve(y_test,y_pred,pos_label=1)
  pd_knn[i]=findPd(fpr_knn,tpr_knn)
  precision_knn=precision_score(y_test,y_pred.round())
  recall_knn=recall_score(y_test,y_pred.round())
  print("Accuracy is",model_knn.score(x_test,y_test))
  print("probablity of detection is ",pd_knn[i])
  print("precision is",precision_knn)
  print("recall is",recall_knn)

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
fig = plt.figure()
plt.plot(snrDB,pd_knn,label = 'KNN')
plt.xlim([-20,0])
plt.xlabel('SNR (dB)',fontsize=16)
plt.ylabel('Probability of detection (Pd)',fontsize=16)
plt.title("KNN algorithm")
plt.legend(fontsize=12)
plt.show()
fig.savefig('raw.png', dpi=300)


"""# SVM model"""

from sklearn import svm
from sklearn.metrics import confusion_matrix

pd_svm = np.ones(len(snrDB))
for i in range(len(snrDB)):

  featuresMatrix=createFeature(beta,snrDB[i])

  df_train,df_test = train_test_split(featuresMatrix, test_size=0.2, random_state=42)
  x_train, y_train = createTrainTest(df_train,num_sensors)
  x_test, y_test = createTrainTest(df_test,num_sensors)


  input_shape = (num_time_periods*num_sensors)

  x_train = x_train.reshape(x_train.shape[0], input_shape)

  x_train = x_train.astype("float32")
  y_train = y_train.astype("float32")

  x_test = x_test.reshape(x_test.shape[0], input_shape)

  x_test = x_test.astype("float32")
  y_test = y_test.astype("float32")
  model_svm = svm.SVC(kernel = 'linear')
  model_svm.fit(x_train, y_train)
  y_pred = model_svm.decision_function(x_test)
  fpr_svm, tpr_svm, thresholds_keras = roc_curve(y_test,y_pred)
  pd_svm[i]=findPd(fpr_svm,tpr_svm)
  precision_svm=precision_score(y_test,y_pred.round(),pos_label='positive',average='micro')
  recall_svm=recall_score(y_test,y_pred.round(),pos_label='positive',average='micro')
  print(snrDB[i])
  print("Accuracy is",model_svm.score(x_test,y_test))
  print("probablity of detection is ",pd_svm[i])
  print("precision is",precision_svm)
  print("recall is",recall_svm)

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
fig = plt.figure()
plt.plot(snrDB,pd_svm,label = 'SVM',linestyle="dashed",linewidth='2',c='r')
plt.xlim([-20,0])
plt.xlabel('SNR (dB)',fontsize=16)
plt.ylabel('Probability of detection (Pd)',fontsize=16)
plt.title("SVM algorithm")
plt.legend(fontsize=12)
plt.show()
fig.savefig('raw.png', dpi=300)


"""# LogisticRegression"""

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

pd_lr = np.ones(len(snrDB))
for i in range(len(snrDB)):

  featuresMatrix=createFeature(beta,snrDB[i])

  df_train,df_test = train_test_split(featuresMatrix, test_size=0.2, random_state=42)
  x_train, y_train = createTrainTest(df_train,num_sensors)
  x_test, y_test = createTrainTest(df_test,num_sensors)


  input_shape = (num_time_periods*num_sensors)

  x_train = x_train.reshape(x_train.shape[0], input_shape)

  x_train = x_train.astype("float32")
  y_train = y_train.astype("float32")

  x_test = x_test.reshape(x_test.shape[0], input_shape)

  x_test = x_test.astype("float32")
  y_test = y_test.astype("float32")
  model_lr=LogisticRegression()
  model_lr.fit(x_train,y_train)
  y_pred=model_lr.predict_proba(x_test)[:,1]
  fpr_lr, tpr_lr, thresholds = roc_curve(y_test, y_pred,pos_label=1)
  pd_lr[i]=findPd(fpr_lr,tpr_lr)
  precision_lr=precision_score(y_test,y_pred.round())
  recall_lr=recall_score(y_test,y_pred.round())
  print(snrDB[i])
  print("Accuracy is",model_lr.score(x_test,y_test))
  print("probablity of detection is ",pd_lr[i])
  print("precision is",precision_lr)
  print("recall is",recall_lr)

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
fig = plt.figure()
plt.plot(snrDB,pd_lr,label = 'Logistic Regression',linestyle='dashdot',linewidth='2',c='black')
plt.xlim([-20,0])
plt.xlabel('SNR (dB)',fontsize=16)
plt.ylabel('Probability of detection (Pd)',fontsize=16)
plt.title("Logistic Regression algorithm")
plt.legend(fontsize=12)
plt.show()
fig.savefig('raw.png', dpi=300)


"""# RandomForest"""

from sklearn.ensemble import RandomForestClassifier

pd_rf = np.ones(len(snrDB))
for i in range(len(snrDB)):

  featuresMatrix=createFeature(beta,snrDB[i])

  df_train,df_test = train_test_split(featuresMatrix, test_size=0.2, random_state=42)
  x_train, y_train = createTrainTest(df_train,num_sensors)
  x_test, y_test = createTrainTest(df_test,num_sensors)


  input_shape = (num_time_periods*num_sensors)

  x_train = x_train.reshape(x_train.shape[0], input_shape)

  x_train = x_train.astype("float32")
  y_train = y_train.astype("float32")

  x_test = x_test.reshape(x_test.shape[0], input_shape)

  x_test = x_test.astype("float32")
  y_test = y_test.astype("float32")
  model_rf=RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=5,n_estimators=100, oob_score=True)
  model_rf.fit(x_train,y_train)
  y_pred=model_rf.predict_proba(x_test)[:,0]
  fpr_lr, tpr_lr, thresholds = roc_curve(y_test, y_pred,pos_label=0)
  pd_rf[i]=findPd(fpr_lr,tpr_lr)
  precision_rf=precision_score(y_test,y_pred.round())
  recall_rf=recall_score(y_test,y_pred.round())
  print(snrDB[i])
  print("Accuracy is",model_rf.score(x_test,y_test))
  print("probablity of detection is ",pd_rf[i])
  print("precision is",precision_rf)
  print("recall is",recall_rf)

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
fig = plt.figure()
plt.plot(snrDB,pd_rf,label = 'Random Forest',linewidth='2',c='g')
plt.xlim([-20,0])
plt.xlabel('SNR (dB)',fontsize=16)
plt.ylabel('Probability of detection (Pd)',fontsize=16)
plt.title("Random Forest algorithm")
plt.legend(fontsize=12)
plt.show()
fig.savefig('raw.png', dpi=300)


# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
fig = plt.figure()
plt.plot(snrDB,pd_knn,label = 'KNN')
plt.plot(snrDB,pd_svm,label = 'SVM')
plt.plot(snrDB,pd_lr,label = 'LR')
plt.plot(snrDB,pd_rf,label = 'RF')
plt.xlim([-20,0])
plt.xlabel('SNR (dB)',fontsize=16)
plt.ylabel('Probability of detection (Pd)',fontsize=16)
plt.legend(fontsize=12)
plt.show()
fig.savefig('raw.png', dpi=300)


"""# DNN ARCHITECTURES"""

#@title Model train/test  { form-width: "30%" }
def trainingModel(model,tcn=False):
  if tcn:
    model.compile(
    loss='sparse_categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

    # Hyper-parameters
    BATCH_SIZE = 200
    EPOCHS = 5

    # Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
    history = model.fit(x_train,
                          y_train,
                          batch_size=BATCH_SIZE,
                          epochs=EPOCHS,

                          validation_split=0.2,
                          verbose=0
                          )
    y_pred = model.predict(x_test)

    y_pred2=[]
    for i in y_pred:
      # print(i[0])
      if(i[0]>i[1]):
        y_pred2.append(i[1])
      else:
        y_pred2.append(i[1])
    y_test2=[]
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test,y_pred2)
    return fpr_keras,tpr_keras

  model.compile(
      loss='binary_crossentropy',
                  optimizer='adam', metrics=['binary_accuracy'])

  # Hyper-parameters
  BATCH_SIZE = 10
  EPOCHS = 10

  # Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
  history = model.fit(x_train,
                        y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,

                        validation_split=0.2,
                        verbose=0
                        )
  accuracy_results = model.evaluate(x_test, y_test)
  print("Accuracy :",accuracy_results)

  y_pred = model.predict(x_test)
  fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test,y_pred)
  return fpr_keras,tpr_keras

"""# CNN MODEL"""

#@title CNN  { form-width: "30%" }
model_cnn = Sequential()
model_cnn.add(Reshape((num_sensors, 1), input_shape=(num_sensors,)))
model_cnn.add(Conv1D(50, 1, activation='relu', input_shape=(num_sensors,1)))
model_cnn.add(Conv1D(50, 1, activation='relu'))
model_cnn.add(MaxPooling1D(1))
model_cnn.add(Dense(50,activation='relu'))
model_cnn.add(Dense(50,activation='relu'))
model_cnn.add(GlobalAveragePooling1D())
model_cnn.add(Dense(num_classes, activation='sigmoid'))
model_cnn.save('CNN.h5')
print(model_cnn.summary())

for i in range(len(snrDB)):

  featuresMatrix=createFeature(beta,snrDB[i])

  df_train,df_test = train_test_split(featuresMatrix, test_size=0.2, random_state=42)
  x_train, y_train = createTrainTest(df_train,num_sensors)
  x_test, y_test = createTrainTest(df_test,num_sensors)

  print(snrDB[i])
  input_shape = (num_time_periods*num_sensors)

  x_train = x_train.reshape(x_train.shape[0], input_shape)

  x_train = x_train.astype("float32")
  y_train = y_train.astype("float32")

  x_test = x_test.reshape(x_test.shape[0], input_shape)

  x_test = x_test.astype("float32")
  y_test = y_test.astype("float32")

  fpr_cnn,tpr_cnn = trainingModel(model_cnn)
  pd_cnn[i]=findPd(fpr_cnn,tpr_cnn)
  print("probablity of detection is",pd_cnn[i])

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
fig = plt.figure()
plt.plot(snrDB,pd_cnn,label = 'CNN',linewidth='2',c='g')
plt.xlim([-20,0])
plt.xlabel('SNR (dB)',fontsize=16)
plt.ylabel('Probability of detection (Pd)',fontsize=16)
plt.title("Convolution Neural Network")
plt.legend(fontsize=12)
plt.show()
fig.savefig('raw.png', dpi=300)


"""# MLP MODEL"""

#@title MLP  { form-width: "30%" }
model_mlp = Sequential()
model_mlp.add(keras.Input(shape=(num_sensors,)))
model_mlp.add(Dense(100,activation='relu'))
model_mlp.add(Dense(100,activation='relu'))
model_mlp.add(Dense(100,activation='relu'))
model_mlp.add(Dense(1,activation='sigmoid'))
print(model_mlp.summary())

for i in range(len(snrDB)):

  featuresMatrix=createFeature(beta,snrDB[i])

  df_train,df_test = train_test_split(featuresMatrix, test_size=0.2, random_state=42)
  x_train, y_train = createTrainTest(df_train,num_sensors)
  x_test, y_test = createTrainTest(df_test,num_sensors)

  print(snrDB[i])
  input_shape = (num_time_periods*num_sensors)

  x_train = x_train.reshape(x_train.shape[0], input_shape)

  x_train = x_train.astype("float32")
  y_train = y_train.astype("float32")

  x_test = x_test.reshape(x_test.shape[0], input_shape)

  x_test = x_test.astype("float32")
  y_test = y_test.astype("float32")

  fpr_mlp,tpr_mlp = trainingModel(model_mlp)
  pd_mlp[i]=(findPd(fpr_mlp,tpr_mlp))
  print("probability od detection is ",pd_mlp[i])

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
fig = plt.figure()
plt.plot(snrDB,pd_mlp,label = 'MLP',linestyle='dashdot',linewidth='2',c='black')
plt.xlim([-20,0])
plt.xlabel('SNR (dB)',fontsize=16)
plt.ylabel('Probability of detection (Pd)',fontsize=16)
plt.title("Multi Layer Perceptron")
plt.legend(fontsize=12)
plt.show()
plt.savefig('roc_curve1.png')
fig.savefig('raw.png', dpi=300)



"""# LSTM MODEL"""

# Import Keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization

from keras.regularizers import l2
from time import time


N = num_sensors                 # number of features
EPOCH = 50                           # number of epochs
LR = 5e-2                            # learning rate of the gradient descent
LAMBD = 3e-2                         # lambda in L2 regularizaion
DP = 0.0                             # dropout rate
RDP = 0.0                            # recurrent dropout rate

model_lstm = Sequential()
model_lstm.add(Reshape((1,num_sensors), input_shape=(num_sensors,)))
model_lstm.add(LSTM(input_shape=(1,num_sensors), units=8,
               activation='tanh', recurrent_activation='hard_sigmoid',
               kernel_regularizer=l2(LAMBD), recurrent_regularizer=l2(LAMBD),
               dropout=DP, recurrent_dropout=RDP,
               return_sequences=True, return_state=False,
               stateful=False, unroll=False
              ))
model_lstm.add(BatchNormalization())
model_lstm.add(LSTM(units=8,
               activation='tanh', recurrent_activation='hard_sigmoid',
               kernel_regularizer=l2(LAMBD), recurrent_regularizer=l2(LAMBD),
               dropout=DP, recurrent_dropout=RDP,
               return_sequences=True, return_state=False,
               stateful=False, unroll=False
              ))
model_lstm.add(BatchNormalization())
model_lstm.add(LSTM(units=8,
               activation='tanh', recurrent_activation='hard_sigmoid',
               kernel_regularizer=l2(LAMBD), recurrent_regularizer=l2(LAMBD),
               dropout=DP, recurrent_dropout=RDP,
               return_sequences=False, return_state=False,
               stateful=False, unroll=False
              ))
model_lstm.add(BatchNormalization())
model_lstm.add(Dense(units=1, activation='sigmoid'))
model_lstm.save('LSTM.h5')
# model_m.summary()
from tensorflow.keras.utils import plot_model
# plot_model(model_m, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

for i in range(len(snrDB)):

  featuresMatrix=createFeature(beta,snrDB[i])

  df_train,df_test = train_test_split(featuresMatrix, test_size=0.2, random_state=42)
  x_train, y_train = createTrainTest(df_train,num_sensors)
  x_test, y_test = createTrainTest(df_test,num_sensors)

  print(snrDB[i])
  input_shape = (num_time_periods*num_sensors)

  x_train = x_train.reshape(x_train.shape[0], input_shape)

  x_train = x_train.astype("float32")
  y_train = y_train.astype("float32")

  x_test = x_test.reshape(x_test.shape[0], input_shape)

  x_test = x_test.astype("float32")
  y_test = y_test.astype("float32")

  fpr_lstm,tpr_lstm = trainingModel(model_lstm)
  pd_lstm[i]=(findPd(fpr_lstm,tpr_lstm))
  print("probability of detection is",pd_lstm[i])

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
fig = plt.figure()
plt.plot(snrDB,pd_lstm,label = 'LSTM')
plt.xlim([-20,0])
plt.xlabel('SNR (dB)',fontsize=16)
plt.ylabel('Probability of detection (Pd)',fontsize=16)
plt.title("Long Short Term Memory")
plt.legend(fontsize=12)
plt.show()
fig.savefig('raw.png', dpi=300)


# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
fig = plt.figure()
plt.plot(snrDB,pd_rnet,label = 'ResNet')
plt.plot(snrDB,pd_lstm,label = 'LSTM')
plt.plot(snrDB, pd_cnn, label = 'CNN')
plt.plot(snrDB,pd_mlp,label = 'MLP')
plt.xlim([-20,0])
plt.xlabel('SNR (dB)',fontsize=16)
plt.ylabel('Probability of detection (Pd)',fontsize=16)
plt.legend(fontsize=12)
plt.show()

plt.savefig('roc_curve.png')
import matplotlib.pyplot as plt

# Create a figure with subplots arranged in a 2x4 grid (7 algorithms, so one subplot will remain empty)
fig, axs = plt.subplots(2, 4, figsize=(16, 8))

# Plot KNN
axs[0, 0].plot(snrDB, pd_knn, marker='o', linestyle='-', color='blue')
axs[0, 0].set_title("KNN")
axs[0, 0].set_xlabel("SNR (dB)")
axs[0, 0].set_ylabel("Pd")
axs[0, 0].set_xlim([-20, 0])

# Plot SVM
axs[0, 1].plot(snrDB, pd_svm, marker='o', linestyle='--', color='red')
axs[0, 1].set_title("SVM")
axs[0, 1].set_xlabel("SNR (dB)")
axs[0, 1].set_xlim([-20, 0])

# Plot Logistic Regression
axs[0, 2].plot(snrDB, pd_lr, marker='o', linestyle='-.', color='black')
axs[0, 2].set_title("Logistic Regression")
axs[0, 2].set_xlabel("SNR (dB)")
axs[0, 2].set_xlim([-20, 0])

# Plot Random Forest
axs[0, 3].plot(snrDB, pd_rf, marker='o', linestyle='-', color='green')
axs[0, 3].set_title("Random Forest")
axs[0, 3].set_xlabel("SNR (dB)")
axs[0, 3].set_xlim([-20, 0])

# Plot CNN
axs[1, 0].plot(snrDB, pd_cnn, marker='o', linestyle='-', color='magenta')
axs[1, 0].set_title("CNN")
axs[1, 0].set_xlabel("SNR (dB)")
axs[1, 0].set_xlim([-20, 0])

# Plot MLP
axs[1, 1].plot(snrDB, pd_mlp, marker='o', linestyle='-', color='orange')
axs[1, 1].set_title("MLP")
axs[1, 1].set_xlabel("SNR (dB)")
axs[1, 1].set_xlim([-20, 0])

# Plot LSTM
axs[1, 2].plot(snrDB, pd_lstm, marker='o', linestyle='-', color='cyan')
axs[1, 2].set_title("LSTM")
axs[1, 2].set_xlabel("SNR (dB)")
axs[1, 2].set_xlim([-20, 0])

# Remove the unused subplot (bottom-right)
fig.delaxes(axs[1, 3])

fig.suptitle("Probability of Detection vs. SNR for Different Algorithms", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit the suptitle
plt.savefig("combined_algorithms.png")
plt.show()

fig = plt.figure()
plt.plot(snrDB, pd_knn, marker='o', label='KNN')
plt.xlim([-20, 0])
plt.xlabel('SNR (dB)', fontsize=16)
plt.ylabel('Probability of detection (Pd)', fontsize=16)
plt.title("KNN Algorithm")
plt.legend(fontsize=12)
plt.show()
fig.savefig('knn.eps')
