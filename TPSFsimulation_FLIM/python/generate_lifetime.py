import numpy as np
    
def generate_lifetime(image = None): 
    # generate random lifetime values for the 28x28 binary image
    
    m = image.shape[1-1]
    n = image.shape[2-1]
    #     Create randomly generated value matrices for the tau1 and tau2
#     thresholds of interest.
    tau1 = np.random.rand(m,n) * 0.3 + 0.2
    
    tau2 = np.random.rand(m,n) * 1 + 2
    
    tau1 = np.multiply(tau1,image)
    tau2 = np.multiply(tau2,image)
    ratio = np.multiply(np.random.rand(m,n),image)
    return tau1,tau2,ratio