import itertools
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns

def expand_grid(*itrs, return_df=False, header_prefix='Var'):
    '''
    Returns all possible combinations of inputs
    
    Parameters
    ----------
    itrs: list, numpy array
    
    return_df: boolean
        If True, return results as a pandas dataframe, otherwise return a dict
        
    header_prefix: str
        prefix of each column name
    
    Returns
    -------
    grid: dictionary or dataframe
    '''
    product = list(itertools.product(*itrs))
    grid = {header_prefix + str(i + 1) : [x[i] for x in product] for i in range(len(itrs))}
    if return_df:
        grid = pd.DataFrame(grid)
    return grid

def dataframe_summary(df):
    '''
    Returns a summary of a dataframe
    
    Parameters
    ----------
    df: dataframe
    
    Returns
    -------
    summary: dataframe
    '''
    summary = pd.DataFrame({
        
        'DType' : df.dtypes,
        'N_Missing' : df.isnull().sum(),
        'N_Nonmissing' : df.notnull().sum(),
        'N_Unique' : df.nunique(),
        'Min' :  df.min(),
        'Median' : df.median(),
        'Mean' : df.mean(),
        'Max' : df.max(),
        'StdDev' : df.std()     
   
    })
    return summary

def plot_hist(df, var, nbins, lower_line=None, upper_line=None, plot_trim_percentile=1.0, plot_trim_method='neither'):
    '''
    Returns a histogram of dataframe feature. 
    
    Parameters
    ----------
    df: dataframe
    
    var: string
        Name of column to plot
        
    nbins: int

    lower_line: float
        x-intercept of lower plot line

    upper_line: float
        x-intercept of upper plot line
    
    plot_trim_percentile: float
        Value between 0 and 1. Proportion of data to trim from distribution
        
    plot_trim_method: string
        'neither', 'top', 'bottom', or 'both'. Region from which to trim
        extreme values.
    '''
    if (plot_trim_percentile < 0 or plot_trim_percentile > 1):
        raise Exception('Percentile should be a number between 0 and 1.')
    
    data = df[var]
    
    if plot_trim_method == 'neither':
        pass
    elif plot_trim_method == 'top':
        data = data[data < data.quantile(plot_trim_percentile)]
    elif plot_trim_method == 'bottom':
        data = data[data > data.quantile(1 - plot_trim_percentile)]
    elif plot_trim_method == 'both':
        data = data[(data < data.quantile(plot_trim_percentile)) & (data > data.quantile(1 - plot_trim_percentile))]
    else:
        raise Exception('plot_trim_method should be one of "neither", "top", "bottom", or "both".')
    
    n, bins, patches = plt.hist(data, nbins, facecolor='black', edgecolor='black', alpha=0.5)
    
    if (lower_line is not None):
        plt.axvline(x=lower_line, color='r', linestyle='dashed', linewidth=2)
    if (upper_line is not None):
        plt.axvline(x=upper_line, color='r', linestyle='dashed', linewidth=2)
    
    plt.xlabel(var)
    plt.ylabel('Count')
    plt.title('Histogram of ' + var)
    plt.show()

def plot_hist_biv(df, var, class_labels, nbins, plot_trim_percentile=1.0, plot_trim_method='neither'):
    '''
    Returns a histogram of dataframe feature divided into two distributions based on class_labels
    
    Parameters
    ----------
    df: dataframe
    
    var: string
        Name of column to plot
        
    class_labels: list
        Dataframe instance class indicators (2 unique values)
    
    nbins: int
    
    percentile: float
        Value between 0 and 1. Proportion of data to trim from distribution
        
    plot_trim_method: string
        'neither', 'top', 'bottom', or 'both'. Region from which to trim
        extreme values.
    '''
    plt.style.use('seaborn-deep')
    
    if (plot_trim_percentile < 0 or plot_trim_percentile > 1):
        raise Exception('Percentile should be a number between 0 and 1.')
    
    df_full = pd.concat(
        [df.reset_index(), pd.DataFrame(class_labels, columns=['class_labels'])],
        axis=1,
        sort=False
    )
        
    if plot_trim_method == 'neither':
        pass
    elif plot_trim_method == 'top':
        df_full = df_full[df_full[var] < df_full[var].quantile(plot_trim_percentile)]
    elif plot_trim_method == 'bottom':
        df_full = df_full[df_full[var] > df_full[var].quantile(1 - plot_trim_percentile)]
    elif plot_trim_method == 'both':
        df_full = df_full[(df_full[var] < df_full[var].quantile(plot_trim_percentile)) & (df_full[var] > df_full[var].quantile(1 - plot_trim_percentile))]
    else:
        raise Exception('plot_trim_method should be one of "neither", "top", "bottom", or "both".')
    
    unique_class_labels = list(set(list(df_full['class_labels'])))
    if (len(unique_class_labels) != 2):
        raise Exception('This function requires two distinct classes')
    
    dist_1 = df_full[df_full['class_labels'] == unique_class_labels[0]][var]
    dist_2 = df_full[df_full['class_labels'] == unique_class_labels[1]][var]
    
    plt.hist([dist_1, dist_2], bins=bins, label=unique_class_labels, density=True)
    plt.title('Density of ' + var)
    plt.xlabel(var)
    plt.legend(loc='upper right')
    plt.show()

def scatterplot_3d(df, columns, class_labels=None, opacity=0.5, size=5, height=1000, width=1000, colorscale='Viridis'):
    '''
    Produce a 3d scatterplot
    
    Parameters
    ----------
    df: dataframe
    
    columns: list of str (length three)
        List of column names of form ['x_col', 'y_col', 'z_col']
    
    class_labels: list of int
        class labels for dataframe for point colouring. Of size df.shape[0]
    '''
    scene = dict(
        xaxis=dict(title=columns[0]), 
        yaxis=dict(title=columns[1]),
        zaxis=dict(title=columns[2])
    )

    trace = go.Scatter3d(
        x=df[columns[0]], 
        y=df[columns[1]], 
        z=df[columns[2]], 
        mode='markers',
        marker=dict(
            color=class_labels, 
            colorscale=colorscale, 
            opacity=opacity, 
            size=size
        )
    )

    layout = go.Layout(margin=dict(l=0, r=0), scene=scene, height=height, width=width)
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    fig.show()
