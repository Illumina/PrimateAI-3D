


import numpy
import scipy

def rank_to_normal(rank, c, n):
    ''' convert ranks to normal distribution
    '''
    # Standard quantile function
    x = (rank - c) / (n - 2 * c + 1)
    return scipy.stats.norm.ppf(x).astype(numpy.float32)

def inverse_rank_normalize(arr, c=3.0 / 8, stochastic=True):
    """ Perform rank-based inverse normal transformation on numpy array.
        If stochastic is True ties are given rank randomly, otherwise ties will
        share the same value. NaN values are ignored.
    
    Args:
        arr:  numpy array to transform
        c: Constant parameter (Bloms constant)
        stochastic:  Whether to randomise rank of ties
    
    Returns:
        numpy.array
    """
    # Set seed
    numpy.random.seed(123)
    arr = numpy.array(arr)
    
    notnan = ~numpy.isnan(arr)
    arr = arr[notnan]
    
    # shuffle by index depnding if we want a stochastic sample or not
    method = 'ordinal' if stochastic else 'average'
    idx = numpy.arange(len(arr))
    shuffle_idx = numpy.random.permutation(idx) if stochastic else idx
    unshuffle_idx = numpy.argsort(shuffle_idx)
    
    # Get rank, if stochastic ties are randomly assigned
    rank = scipy.stats.rankdata(arr[shuffle_idx], method=method)
    
    # Convert rank to normal distribution
    transformed = rank_to_normal(rank, c=c, n=len(rank))
    
    # refill to original positions (including nans)
    complete = numpy.empty(len(notnan), dtype=numpy.float32)
    complete[:] = numpy.nan
    complete[notnan] = transformed[unshuffle_idx]
    return complete
