import numpy as np
    
def generate_intensity(image = None): 
    # generate random intensity for input binary image
    m = image.shape[1-1]
    n = image.shape[2-1]
    #     random matrix of intensity values possessing values within maximum
#     photon count threshold.
    int1 = np.random.rand(m,n) * 250 + 50
    
    intensity = np.multiply(int1,image)
    return intensity
    