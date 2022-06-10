import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

def dbscan_tune_eps(df, ylim):
    '''
    Tune epsilon parameter for DBScan using the 'elbow method' - Choose the value of epsilon
    where there is maximal rate of change in the plot
    
    Parameters
    ----------
    df: dataframe
        Feature matrix for DBScan
        
    ylim: float
        plot y-axis maximum
    '''
    nbrs = NearestNeighbors(n_neighbors=2).fit(df)
    distances, indices = nbrs.kneighbors(df)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    
    plt.plot(distances)
    plt.ylim(bottom=0, top=ylim)
    plt.ylabel("eps")
    plt.grid(color='k', linestyle='--', linewidth=0.5)
    plt.show()
	
def plot_pca_variance_explained(df):
    '''
    Plot the percentage of variance explained as a function of
    the number of principle components
    
    Parameters
    ----------
    df: dataframe
    '''
    pca = PCA().fit(df)
    variance = pca.explained_variance_ratio_ 
    var = np.cumsum(np.round(variance, 3) * 100)
    plt.figure(figsize=(12, 6))
    plt.ylabel('% Variance Explained')
    plt.xlabel('# of Features')
    plt.title('PCA - % Variance Explained vs. Number of Features')
    plt.ylim(0, 100.5)
    plt.plot(var)
    plt.show()
	
def confusion_matrix_annot():
    '''
    Returns a printout of confusion matrix statistics
    '''
    print('##          Predicted')
    print('##          N   P')
    print('## Truth N [TN, FP]')
    print('##       P [FN, TP]')
    print('## ')
    print('## FP - Type I Error')
    print('## FN - Type II Error')
    print('## Sensitivity=TP / P')
    print('## Specificity=TN / N')
    print('## PPV=TP / (TP + FP)')
    print('## NPV=TN / (TN + FN)')
    return
	
