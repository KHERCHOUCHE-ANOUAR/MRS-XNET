from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras import activations, Model, Input
import tensorflow as tf
import numpy as np
import sklearn 
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import random
import numpy as np
from sklearn import preprocessing
from scipy import interp
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
tf.compat.v1.disable_eager_execution()
tf.random.set_seed(1234)
np.random.seed(3)
sklearn.utils.check_random_state(2)

print(tf.__version__)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'


mci=#mci data with a numpy file
mad=#mad data with a numpy file

#took only the real and the used part of the spectrum
mci=mci[:,:].real
mci=mci[:,:300]

mad=mad[:,:].real
mad=mad[:,:300]

# data augmentation using the average between two spectra from the same class (beside 5 for each class for the test)
c=True
for i in range(5,mci.shape[0]):
  for j in range(i+1,mci.shape[0]):
    if c:
      avg_mci=(mci[i]+mci[j])/2
      c=False
    else:
      avg_mci=np.vstack((avg_mci,(mci[i]+mci[j])/2))

c=True
for i in range(5,mad.shape[0]):
  for j in range(i+1,mad.shape[0]):
    if c:
      avg_mad=(mad[i]+mad[j])/2
      c=False
    else:
      avg_mad=np.vstack((avg_mad,(mad[i]+mad[j])/2))


# data normalization
for i in range(avg_mci.shape[0]):
  avg_mci[i]=avg_mci[i]/np.max(np.abs(avg_mci[i]))
  # avg_mci[i]=(avg_mci[i]-np.min(avg_mci[i]))/(np.max(avg_mci[i])-np.min(avg_mci[i]))

for i in range(avg_mad.shape[0]):
  avg_mad[i]=avg_mad[i]/np.max(np.abs(avg_mad[i]))
  # avg_mad[i]=(avg_mad[i]-np.min(avg_mad[i]))/(np.max(avg_mad[i])-np.min(avg_mad[i]))
data=np.vstack((avg_mci,avg_mad))

for i in range(mci.shape[0]):
  mci[i]=mci[i]/np.max(np.abs(mci[i]))
  # mci[i]=(mci[i]-np.min(mci[i]))/(np.max(mci[i])-np.min(mci[i]))
for i in range(mad.shape[0]):
  mad[i]=mad[i]/np.max(np.abs(mad[i]))
  mad[i]=(mad[i]-np.min(mad[i]))/(np.max(mad[i])-np.min(mad[i]))

# label the data
y_data=np.zeros(len(data))
y_data[len(avg_mci):]=np.ones(len(avg_mad))

X_test=np.vstack((mci[:5,:],mad[:5,:]))


y_test=np.zeros(np.size(X_test,0))
y_test[len(mci[:5,:]):]=np.ones(len(mad[:5,:]))

# prepare train validation and test data
X_train, X_val, y_train, y_val = train_test_split(data, y_data, test_size=0.2)
X_train,y_train = shuffle(X_train, y_train)

X_train=np.expand_dims(X_train, axis = 2)
X_val=np.expand_dims(X_val, axis = 2)
X_test=np.expand_dims(X_test, axis = 2)

num_classes=2
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')


y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_val = tf.keras.utils.to_categorical(y_val, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
#the used network
def network(x,n_classes):
    out = layers.Conv1D(32,8,name='conv0_0',padding='same')(x)
    out=layers.Activation(activations.relu,name='act0_0')(out)
    out=layers.BatchNormalization()(out)
    out = layers.Conv1D(64,5,name='conv0_1',padding='same')(out)
    out=layers.Activation(activations.relu,name='act0_1')(out)
    out=layers.BatchNormalization()(out)
    # out=layers.Dropout(0.4)(out)
    out = layers.Conv1D(64,8,name='conv0_2',padding='same')(out)
    out=layers.Activation(activations.relu,name='act0_2')(out)

    # out= layers.Conv1D(64,5,padding='same',name='conv1_2')(out)
    # out=layers.Activation(activations.relu,name='act1_2')(out)
    out=layers.BatchNormalization()(out)
    out=layers.Dropout(0.5)(out)
    pred_out = layers.Lambda(lambda z:0.5*layers.GlobalAveragePooling1D()(z)+ 0.5*layers.GlobalMaxPooling1D()(z))(out)
    # pred_out=layers.Dropout(0.5)(pred_out)
    pred_out=layers.Dense(n_classes,name='dense1_0')(pred_out)
    pred_out = layers.Softmax(name='attention_branch')(pred_out)

    return pred_out


def get_model(input_shape, n_classes):
    img_input = Input(shape=input_shape)
    pred_out = network(img_input,n_classes)
    
    model = Model(inputs=img_input, outputs=pred_out)
    return model

model=get_model((300,1),2)

model.summary()
# the used hyper-param
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=200000,
    decay_rate=0.9)

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
          optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
          metrics=['accuracy'])
history=model.fit(X_train, y_train,
      batch_size=16,
      epochs=164,
      validation_data=(X_test, y_test))

score = model.evaluate(X_test,y_test, verbose=0)
print('total val and accuracy:', score)
#saving the trained model
model.save('icip2022/model_mci_vs_mad.h5')

