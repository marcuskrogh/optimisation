## Imports
from cvxopt         import matrix

import numpy                          as np
import matplotlib.pyplot              as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib


## Import test function
from lp_function import objective, constraints
from linear_constraints_contour import linear_constraints_contour

## Main function
def driver():
    ## Define LP
    g    = objective()
    A, b = constraints()

    ## Grid definition
    N = 100

    ## Define x-region
    x_min, x_max = -5, 10
    x_int  = x_max-x_min
    x_step = x_int/N
    x = matrix(np.arange( x_min, x_max+x_step, x_step ))

    ## Define y-region
    y_min, y_max = -5, 10
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
    fig, ax = linear_constraints_contour( X, Y, Z, A, b, 25, )

    return fig, ax
