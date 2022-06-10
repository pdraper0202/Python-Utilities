import numpy as np
import random

def biweight_estimate(x, c=5.0, max_iter=15, convergence_percent=0.005, return_history=False):
    '''
    Compute the Tukey Biweight estimate of location and scale
    
    Parameters
    ----------
    x: 1D numpy array
    
    c: float
        Biweight tuning constant
    
    max_iter: int
        Maximum number of algorithm iterations
    
    convergence_percent: float in [0, 1)
        early stopping condition
    
    return_history: bool
        if False, return only last iteration results
        
    Returns
    -------
    loc: numpy array
        biweight location estimate(s)
        
    scl: numpy array
        biweight scale estimate(s)
        
    wgt: numpy array
        weights
    '''
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    
    # store the weight estimates. Initialize with 1's to align with loc and scl:
    wgt = {0 : np.repeat(1.0, x.shape[0])}
    
    # store the location estimates:
    u = np.median(x)
    loc = np.array([u])
    
    # store the scale estimates:
    s = np.median(np.abs(x - u))
    scl = np.array([s])
    
    # main loop
    idx = 0
    while (idx < max_iter):
        # calculate and update weights
        if s == 0:
            break
        else:
            d = ((x - u) / np.array([c * s]))**2
            w = np.where(d < 1, (1 - d)**2, 0)
            wgt.update({(idx + 1) : w})
        
        # update location estimate
        u = np.sum(x * w) / np.sum(w)
        loc = np.append(loc, u)
        
        # update scale estimate (weighted std dev)
        s = np.sqrt(np.cov(x, aweights = w))
        scl = np.append(scl, s)
        
        if (np.abs(u - loc[idx]) <= loc[idx] * convergence_percent):
            break
        
        idx += 1
    
    # convert weights to numpy array
    wgt=np.stack(tuple(wgt.values()), axis=1)
    
    # results
    if not return_history:
        loc = loc[-1]
        scl = scl[-1]
        wgt = wgt[:, -1]
    
    return loc, scl, wgt
