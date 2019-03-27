## Imports
from cvxopt         import matrix
import numpy as np

## Import test function
from visualisation  import lc_contour

## Main function
def driver( g, A=None, b=None, C=None, d=None ):
    try:
        g = matrix(g)
        n = g.size[0]
    except:
        print( 'Error: System matrices not properly defined.' )
        return None, None

    try:
        A = matrix(A)
        b = matrix(b)
        _, ma = A.size
    except:
        A = matrix( 0.0, (n,0) )
        b = matrix( 0.0, (0,1) )
        ma = 0

    try:
        C = matrix(C)
        d = matrix(d)
        _, mc = C.size
    except:
        C = matrix( 0.0, (n,0) )
        d = matrix( 0.0, (0,1) )
        mc = 0

    ## Grid definition
    N = 100

    ## Define x-region
    x_min, x_max = -0.5, 2.0
    x_int  = x_max-x_min
    x_step = x_int/N
    x = matrix(np.arange( x_min, x_max+x_step, x_step ))

    ## Define y-region
    y_min, y_max = -0.5, 2.0
    y_int = y_max-y_min
    y_step = y_int/N
    y = matrix(np.arange( y_min, y_max+y_step, y_step ))

    ## Define mesh
    X, Y = np.meshgrid( x, y )
    X = matrix(X)
    Y = matrix(Y)

    ## Define objective values
    Z = g[0]*X + g[1]*Y

    ## Plot linear constraints in contour plot
    fig, ax = lc_contour( X, Y, Z, A, b, C, d, 25, )

    return fig, ax
