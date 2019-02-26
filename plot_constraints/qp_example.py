## Imports
from cvxopt         import matrix
from cvxopt.lapack  import gesv
import numpy as np

## Import test function
from qp_function import objective, constraints
from linear_constraints_contour import linear_constraints_contour

## Main function
def driver():
    ## Define quadratic problem
    H, g = objective()
    A, b = constraints()

    ## Grid definition
    N = 100

    ## Define x-region
    x_min, x_max = -5, 10
    x_int  = x_max-x_min
    x_step = x_int/N
    x = matrix(np.arange( x_min, x_max+x_step, x_step ))
    nx, _ = x.size

    ## Define y-region
    y_min, y_max = -5, 10
    y_int = y_max-y_min
    y_step = y_int/N
    y = matrix(np.arange( y_min, y_max+y_step, y_step ))
    ny, _ = y.size

    ## Define mesh
    X, Y = np.meshgrid( x, y )
    X = matrix(X)
    Y = matrix(Y)

    ## Define objective function
    obj_fun = lambda x: x.T*H*x + g.T*x

    ## Compute contours
    Z = matrix( 0.0, (nx,ny) )
    for i in range(ny):
        for j in range(nx):
            x_ = matrix( [ x[j], y[i] ] )
            Z[j,i] = obj_fun( x_ )

    ## Plot linear constraints in contour plot
    fig, ax = linear_constraints_contour( X, Y, Z, A, b, 25, True )
