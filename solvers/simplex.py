############################################################################
################################## Imports #################################
############################################################################
## CVX opt
import cvxopt
from cvxopt import matrix, spmatrix, spdiag, sqrt, mul, div

## Numpy
import numpy as np

## Time
import time
############################################################################
############################################################################
############################################################################


############################################################################
############################### Type checking ##############################
############################################################################
def type_checking( g, A, b, x_0 ):
    try:
        g = matrix(g)
        n = g.size[0]
    except:
        print( 'InputError: System is not properly defined.' )

    try:
        A  = matrix(A)
        b  = matrix(b)
        ma = b.size[0]
    except:
        A  = matrix( 0.0, (n,0) )
        b  = matrix( 0.0, (0,1) )
        ma = 0

    try:
        x_0 = matrix( x_0 )
    except:
        x_0 = matrix( 0.0, (n,1) )

    return g, A, b, x_0, n, ma
############################################################################
############################################################################
############################################################################



def simplex( \
    ## System matrices
    g, \
    ## Equality constraint matrices
    A=None, b=None, \
    ## Initial guess(es)
    x_0=None,
    ## ALgirithm parameters
    it_max=100, ):

    ## Type checking
    g, A, b, x_0, n, ma = type_checking( g, A, b, x_0 )

    ## Form basic set
    A_inv = matrix(np.linalg.pinv(A)).T
    x_    = A_inv*b
    B_idx = matrix(list(filter( lambda i: x_[i] >= 0, range(n) ) ) )
    N_idx = matrix(list(filter( lambda i: x_[i] <  0, range(n) ) ) )
    print( x_    )
    print( x_idx )
