import numpy as np
    
def generate_tpsfs(intensity = None,tau1 = None,tau2 = None,ratio = None,irf_whole = None): 
    # dimension: image * time gate (28x28xnTG)
# irf: unit Instrumental Response Function (sum=1)
    
    M = intensity.shape[0]
    N = intensity.shape[1]
    # Number of time-points/gates
    nTG = 160
    width = 0.075
    
    time = np.array([np.arange(1,nTG+1,1)]) * width
    # Pre-allocate memory for each TPSF voxel
    full_data0 = np.zeros((M,N,nTG))
    full_data = np.zeros((M,N,nTG))
    # Pre-allocate memory for each IRF voxel
    irf_full = np.zeros((M,N,nTG))
    # Loop over all pixels spatially
    for i in np.arange(1,M).reshape(-1):
        for j in np.arange(1,N).reshape(-1):
            #         Only loop at locations from which TPSFs can be created.
            if tau1[i,j] != 0:
                #             Create initial bi-exponential given the tau1, tau2 and ratio
#             values at the image position (i,j)
                decay = ratio[i,j] * np.exp(- time / tau1[i,j]) + (1 - ratio[i,j]) * np.exp(- time / tau2[i,j])
                #             Grab IRF from library
                #irf = irf_whole[:,np.round(np.random.rand() * (irf_whole.shape[2-1] - 1)) + 1]
                irf = irf_whole[:,np.random.randint(low = 0, high = irf_whole.shape[1])-1]
                #             Convolve IRF with our exp. decay
                decay = np.convolve(decay[0,:], irf / np.sum(irf))
            
                #             Sample back to the original number of time-points by including random
#             effects due to laser-jitter (point of TPSF ascent).
                r = np.random.rand()
                if r > -1: # Je bypass le laser jitter (Pour l'instant)
                    decay = decay[0:nTG]
                else:
                    if r < 0.25:
                        rC = np.round(np.multiply(np.random.rand(),3))
                        decay = np.array([[np.zeros((rC,1))],[decay(np.arange(1,nTG - rC+1))]])
                    else:
                        rC = np.round(np.multiply(np.random.rand(),3))
                        decay = decay[np.arange(1 + rC,nTG + rC+1).astype('int')]
                             #Multiple the decay by its corresponding intensity value
#             (maximum photon count)
                decay = decay * intensity[i,j]
                #             Add poisson noise
                #cur = np.round(poissrnd(decay))
                cur = np.random.poisson(decay)
                full_data[i,j,:] = cur
                #             Assign the decay to its corresponding pixel location
                full_data0[i,j,:] = cur/np.max(cur)
    
    return full_data0, full_data
    