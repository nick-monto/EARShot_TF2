from numpy import zeros,arange

def are_neighbors(x,y):
    '''
    Computes the edit distance between two input iterables x and y, but
    does NOT compute or return the actual alignment.  This works on either
    strings or lists, so if you have multi-character phoneme codes pass in
    a list of the phonemes instead of the strings themselves.
    '''
    m = len(x)
    n = len(y)
    E = zeros((n+1,m+1),dtype=int)
    # first row/col
    E[0,:] = arange(0,m+1)
    E[:,0] = arange(0,n+1)
    # fill in the subproblem matrix
    for i in range(1,n+1):
        for j in range(1,m+1):
            cx,cy = x[j-1],y[i-1]
            subprob = [1 + E[i-1,j], 1 + E[i,j-1], (1 - int(cx == cy)) + E[i-1,j-1]]
            E[i,j] = min(subprob)
    if E[n,m] == 1:
        return True
    return False


def are_cohorts(x,y):
    '''
    Takes two phonological transcriptions (strings for single-letter dictionaries,
    lists of phonemes for multi-letter codes like CSAMPA) and returns True if
    they are cohorts (identical up to the first two phonemes) AND x != y
    '''
    if x == y:
        return False
    if x[0:2] == y[0:2]:
        return True
    return False


def are_rhymes(x,y):
    '''
    Takes two phonological transcriptions (strings for single-letter dictionaries,
    lists of phonemes for multi-letter codes like CSAMPA) and returns True if
    they are rhymes (identical after the inital phoneme but not homophones).
    '''
    if x == y:
        return False
    if x[1:] == y[1:]:
        return True
    return False
