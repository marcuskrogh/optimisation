## Imports
from cvxopt  import matrix, spmatrix
import numpy as np

## Import test function
from visualisation import nlp_contour

## Main function
def driver( obj, eq_cons, in_cons, ):
    ## Grid definition
    Nx = 500
    Ny = 500

    ## Define x-region
    
    x_min, x_max =  -5.50, 5.30
    x_int  = x_max-x_min
    x_step = x_int/Nx
    x = matrix( np.arange( x_min, x_max+x_step, x_step ) )
    nx, _ = x.size

    ## Define y-region
    y_min, y_max =  -5.15, 5.00
    y_int = y_max-y_min
    y_step = y_int/Ny
    y = matrix( np.arange( y_min, y_max+y_step, y_step ) )
    ny, _ = y.size

    ## Define mesh
    X, Y = np.meshgrid( x, y )
    X = matrix(X)
    Y = matrix(Y)

    ## Compute contours
    Z = matrix( 0.0, (ny,nx) )
    for i in range(nx):
        for j in range(ny):
            x_ = matrix( [ X[j,i], Y[j,i] ] )
            z, _, _ = obj( x_, ['f'] )
            Z[j,i]  = z

    ## Plot linear constraints in contour plot
    v_1 = matrix( np.arange(   0.0,  10.0,   2.50 ) )
    v_2 = matrix( np.arange(  10.0, 200.0,  10.00 ) )
    #v_3 = matrix( np.arange( 200.0, 300.0,  25.00 ) )
    v = matrix( [ v_1, v_2] )
    C = matrix( spmatrix( 1.0, range(2), range(2) ) )
    d = matrix( [ -4.0, -4.0 ] )
    fig, ax = nlp_contour( X, Y, Z, eq_cons, C, d, v )

    return fig, ax
