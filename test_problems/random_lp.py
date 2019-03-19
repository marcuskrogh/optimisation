import numpy as np
from cvxopt import matrix


def generate_lp( n=200, m=50, seed=5330 ):

    ## Seed the random LP generator
    #np.random.seed(seed)

    A = matrix( np.random.randn(n,m) )

    x = matrix( 0.0, (n,1) )
    x[:m] = matrix( abs(np.random.rand(m,1)) )

    z = matrix( 0.0, (n,1) )
    z[m:n] = matrix( abs(np.random.rand(n-m,1)) )

    y = matrix( np.random.rand(m,1) )

    g = A*y + z
    b = A.T*x

    ## Return statement
    return g, A, b, x, y, z
