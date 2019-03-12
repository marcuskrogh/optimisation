## Imports
from cvxopt         import matrix
from cvxopt.lapack  import gesv

import numpy as np

## Import test function
from visualisation.qp_function                import objective, constraints
from visualisation.linear_constraints_contour import linear_constraints_contour

## Main function
def driver():
    ## Define quadratic problem
    H, g = objective()
    A, b = constraints()

    ## Grid definition
    Nx = 500
    Ny = 500

    ## Define x-region
    x_min, x_max =  0, 5
    x_int  = x_max-x_min
    x_step = x_int/Nx
    x = matrix( np.arange( x_min, x_max+x_step, x_step ) )
    nx, _ = x.size

    ## Define y-region
    y_min, y_max =  0, 6
    y_int = y_max-y_min
    y_step = y_int/Ny
    y = matrix( np.arange( y_min, y_max+y_step, y_step ) )
    ny, _ = y.size

    ## Define mesh
    X, Y = np.meshgrid( x, y )
    X = matrix(X)
    Y = matrix(Y)

    ## Define objective function
    #clear
    #obj_fun = lambda x: (x[0]**2 + x[1] - 7)**2 + (x[0] + x[1]**2 - 11)**2
    obj_fun = lambda x: 1/2*x.T*H*x + g.T*x

    ## Compute contours
    Z = matrix( 0.0, (ny,nx) )
    for i in range(nx):
        for j in range(ny):
            x_ = matrix( [ X[j,i], Y[j,i] ] )
            Z[j,i] = obj_fun( x_ )

    ## Plot linear constraints in contour plot
    fig, ax = linear_constraints_contour( X, Y, Z, A, b, 40 )

    return fig, ax
