# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Generates the simulated TPSF voxel data (FLIM) using functions included along with
# IRF (deconvolved via software) and MNIST data.
#
# Jason T. Smith, Rensselaer Polytechnic Institute, August 23, 2019
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
import scipy.io as sio
import os
PATH = '/home/antoine/Documents/Code/DL4FLI/TPSFsimulation_FLIM/python/dataset/'
if not os.path.exists(PATH):
    os.makedirs(PATH)
import h5py
from  generate_intensity import generate_intensity
from generate_lifetime import generate_lifetime
from generate_tpsfs import generate_tpsfs

irf_whole = np.array(sio.loadmat('/home/antoine/Documents/Code/DL4FLI/TPSFsimulation_FLIM/python/FLIM_IRF.mat')['irf_whole'])
train_images = np.array(sio.loadmat('/home/antoine/Documents/Code/DL4FLI/TPSFsimulation_FLIM/python/train_binary.mat')['train_images'])

t1 = []
t2 = []
rT = []
sigD = []
sigD_nn = []
I = []

# Number of TPSF voxels to create
N_total = 300
k = 1
# nTG = 256;
while k <= N_total:

    # Take 28x28 subset of random 32x32 MNIST image
    im_binary = train_images[2:32 - 2,2:32 - 2,np.random.randint(low = 0, high = train_images.shape[2]-1)]
    # Make sure it is not too sparse (we want voxels with more TPSFs than
# less)
    if sum(sum(im_binary)) < 250:
        continue
    # Generate intensity image map
    inten = generate_intensity(im_binary)
    # Generate t1, t2 and AR image maps
    tau1,tau2,ratio = generate_lifetime(im_binary)
    data, data_nn = generate_tpsfs(inten,tau1,tau2,ratio,irf_whole)
    m = im_binary.shape[0]
    n = im_binary.shape[1]
    t1.append(tau1)
    t2.append(tau2)
    rT.append(ratio)
    sigD_nn.append(data_nn)
    sigD.append(data)
    #I.append(np.multiply(inten,im_binary))
    I.append(inten)
    k = k + 1 


hf = h5py.File(PATH+'data'+'.h5', 'w')
hf.create_dataset('sigD', data=sigD)
hf.create_dataset('sigD_nn', data=sigD_nn)
hf.create_dataset('t1', data=t1)
hf.create_dataset('t2', data=t2)
hf.create_dataset('rT', data=rT)
hf.create_dataset('intensity', data=I)
hf.close()

