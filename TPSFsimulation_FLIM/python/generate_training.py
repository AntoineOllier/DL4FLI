# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Generates the simulated TPSF voxel data (FLIM) using functions included along with
# IRF (deconvolved via software) and MNIST data.
#
# Jason T. Smith, Rensselaer Polytechnic Institute, August 23, 2019
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
import scipy.io as sio
irf_whole = np.array(sio.loadmat('/home/antoine/Documents/Code/DL4FLI/TPSFsimulation_FLIM/python/FLIM_IRF.mat')['irf_whole'])
train_images = np.array(sio.loadmat('/home/antoine/Documents/Code/DL4FLI/TPSFsimulation_FLIM/python/train_binary.mat')['train_images'])
from  generate_intensity import generate_intensity
from generate_lifetime import generate_lifetime
from generate_tpsfs import generate_tpsfs

# Number of TPSF voxels to create
N_total = 100
k = 1
# nTG = 256;
while k <= N_total:

    # Take 28x28 subset of random 32x32 MNIST image
    im_binary = train_images[3:31 - 2,3:31 - 2,np.random.randint(low = 0, high = train_images.shape[2]-1)]
    #round(rand()*(size(train_images,3)-1))+1)
    # Make sure it is not too sparse (we want voxels with more TPSFs than
# less)
    if sum(sum(im_binary)) < 250:
        continue
    # Generate intensity image map
    inten = generate_intensity(im_binary)
    # Generate t1, t2 and AR image maps
    tau1,tau2,ratio = generate_lifetime(im_binary)
    data = generate_tpsfs(inten,tau1,tau2,ratio,irf_whole)
    m = im_binary.shape[1-1]
    n = im_binary.shape[2-1]
    t1 = tau1
    t2 = tau2
    rT = ratio
    sigD = data
    I = np.multiply(inten,im_binary)
    # Making sure sample numbers are assigned like 00001, 00002,.... 01001,
# 01002, etc.
    if k >= 0 and k < 10:
        n = np.array(['0000',str(k)])
    else:
        if k >= 10 and k < 100:
            n = np.array(['000',str(k)])
        else:
            if k >= 100 and k < 1000:
                n = np.array(['00',str(k)])
            else:
                if k >= 1000 and k < 10000:
                    n = np.array(['0',str(k)])
                else:
                    n = str(k)
    # Assign path along with file name.
    #pathN = ''
    #filenm = np.array([pathN,'\','a_',n,'_',num2str(1)])
    # Save .mat file. It is important to note the end '-v7.3' - this is one
# of the more convenient ways to facillitate easy python upload of
# matlab-created data.
    #save(filenm,'sigD','I','t1','t2','rT','-v7.3')
    k = k + 1
    print(sigD.shape)
