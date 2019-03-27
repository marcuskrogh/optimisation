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
def type_checking( g, A, b, B_0 ):
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
        B_0 = matrix( B_0 )
    except:
        B_0 = matrix( 0, (1,1) )

    return g, A, b, B_0, n, ma
############################################################################
############################################################################
############################################################################



def simplex( \
    ## System matrices
    g, \
    ## Equality constraint matrices
    A=None, b=None, \
    ## Initial guess(es)
    B_0 = None,
    ## ALgirithm parameters
    it_max=100, ):

    ## Type checking
    g, A, b, B_0, n, m = type_checking( g, A, b, B_0 )

    ## Form basic and non-basic sets
    B = B_0
    N = matrix( list( filter( lambda i: i not in B, range(m) ) ) )

    ## ~
    B_  = matrix( [ [A[:,i]] for i in B ] )
    N_  = matrix( [ [A[:,i]] for i in N ] )

    g_B = matrix( [ [g[i]  ] for i in B ] )
    g_N = matrix( [ [g[i]  ] for i in N ] )

    ## Form


    ## Main loop
    converged = False
    it        = 0
    while not converged and it < it_max:
        B_  = matrix( [ [A[:,i]] for i in B ] )
        N_  = matrix( [ [A[:,i]] for i in N ] )

        g_B = matrix( [ [g[i]  ] for i in B ] )
        g_N = matrix( [ [g[i]  ] for i in N ] )

        ## Solve for lagrange multipliers
        B_inv = matrix( np.linalg.pinv( B ) )
        mu = B_inv.T*g_B

        lambda_N = g_N - N_.T*mu
        print( lambda_N )
        if min(lambda_N) >= 0:
            print( 'Optimal solution found.' )
            converged = True

        ## Iterate
        it += 1
