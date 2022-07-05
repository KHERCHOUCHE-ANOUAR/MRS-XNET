from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras import activations, Model, Input
import tensorflow as tf
import numpy as np
from scipy.interpolate import interp1d
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import random
from scipy.signal import chirp, find_peaks, peak_widths
import numpy as np
from sklearn import preprocessing
from scipy import interp
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import sklearn
tf.compat.v1.disable_eager_execution()
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
# plt.style.use('fivethirtyeight')
print(tf.__version__)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


num_classes=2
mci=#load the mci data with a numpy file
mad=#load the mad data with a numpy file

max_length=200000

#took only the real part of the spectra
mci=mci[:,:].real
mci=mci[:,:300]

mad=mad[:,:].real)
mad=mad[:,:300]

#data normalization
for i in range(mci.shape[0]):
  mci[i]=mci[i]/np.max(np.abs(mci[i]))
  # mci[i]=(mci[i]-np.min(mci[i]))/(np.max(mci[i])-np.min(mci[i]))
for i in range(mad.shape[0]):
  mad[i]=mad[i]/np.max(np.abs(mad[i]))
  mad[i]=(mad[i]-np.min(mad[i]))/(np.max(mad[i])-np.min(mad[i]))

#our mrs_cam
def visualize_class_activation_map(X_test):
    model1 = tf.keras.models.load_model('/home/anouar/Bureau/icip2022/model_mci_vs_mad.h5')
    model=Model(model1.layers[0].output,model1.layers[-2].output)
    X_test=np.reshape(X_test,(1,300,1))

    # print(model.layers[-1].get_weights()[0])
    class_weights = model.layers[-1].get_weights()[0]
    final_conv_layer = model.get_layer("conv0_2")
    get_output = K.function([model.layers[0].input], \
                [final_conv_layer.output, 
    model.layers[-1].output])
    [conv_outputs, predictions] = get_output(X_test)
    conv_outputs = conv_outputs[0, :, :]

    #Create the class activation map.
    cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:1])
    target_class =0
    # print(conv_outputs.shape,cam.shape)
    for i, w in enumerate(class_weights[:,target_class]):
            cam += w * conv_outputs[:,i]
    return cam
t=visualize_class_activation_map(mci[5])

minimum = np.min(t)
t = t - minimum
t = t / max(t)
t = t * 100
x = np.linspace(0, 300 - 1, max_length, endpoint=True)
# linear interpolation to smooth
f = interp1d(range(300), mci[5])
y = f(x)

f = interp1d(range(300), t)
t = f(x).astype(float)
zz=t
# plt.plot(X_test[7:8,:,0])
z={}
# print(y.shape[0])
for i in range(1,y.shape[0]-1,1):
    if y[i-1]<y[i] and y[i]>y[i+1] and y[i]>0.15:
        z[i]=y[i]
# w=z
# print(z,len(z))
k=0
for key in list(z):
    # print(key)
    if k!=0:
        if old_key+10000>=key:
            if z[old_key]<z[key]:
                del z[old_key]
                old_key=key
            else:
                del z[key]
        else:
            old_key=key
    else:
        old_key=key
        k=1
    # print(key,old_key)
z=dict(sorted(z.items(), key=lambda item: item[1],reverse=True))
# print(z,len(z))

# print(len(z))
peaks=list(z.keys())[0:6]
print(peaks)
# step=int(max_length/300)
step_left=[]
step_right=[]
y_inv=-y+1
for i in peaks:
    left=True
    right=True
    # print(i)
    j=0
    while left or right:
    
        # print(j)
        if left and y_inv[i-j-2]<y_inv[i-j] and y_inv[i-j]>y_inv[i-j+2] and y_inv[i-j]>0.6:
            left=False
            step_left.append(j)
            # j_prime=np.where(y_inv[i:]>=y_inv[i-j])
            # print(i-j,i,j_prime)
            # step_right.append(j_prime[0][0])
        if right and y_inv[i+j-2]<y_inv[i+j] and y_inv[i+j]>y_inv[i+j+2] and y_inv[i-j]>0.6:
            right=False
            step_right.append(j)
            # print((y_inv[i+j] in y_inv[:i-j]))
            # j_prime=np.where(y_inv[:i]>=y_inv[i+j])
            # # print(j_prime[0][-1],i,i+j)
            # step_left.append(i-j_prime[0][-1])
        j+=1

results_half = peak_widths(y, peaks, rel_height=1)
# print(*results_half[1:])
# print(results_half[1:])
# print(results_half)
# plt.hlines(*results_half[1:], color="C2")
# plt.plot(mad[2],color="black")
# min_iso=np.array(peaks)-np.array(step_left)
# max_iso=np.array(peaks)+np.array(step_right)
# plt.scatter(x=x[max_iso], y=y[max_iso],color="red")

# plt.scatter(x=x[min_iso], y=y[min_iso],color="green")
# plt.plot(x,t)
plt.scatter(x=x, y=y, c=100-t, cmap='jet',marker='.', s=2, vmin=0, vmax=100, linewidths=1)
# plt.scatter(x=x, y=y, c=t, cmap='jet',marker='.', s=2, vmin=0, vmax=100)
#plt.scatter(x=x, y=y, c=zz, cmap='jet',marker='.', s=2, vmin=0, vmax=100, linewidths=1)
for i in range(len(peaks)):
    plt.text(x[peaks[i]],y[peaks[i]], ' '+str("{:.0f}".format(t[peaks[i]]))+'%', horizontalalignment='center', verticalalignment='bottom',rotation=90, fontdict={ 'size':10})
cbar = plt.colorbar()
plt.ylim(-0.1,1.19999)

# cbar.ax.set_yticklabels([100,75,50,25,0])
plt.savefig('icip2022/class0.png',
            bbox_inches='tight', dpi=1080)
#mad 6
#mci 5