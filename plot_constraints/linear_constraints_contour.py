## Linear algebra imports
import cvxopt
from cvxopt import matrix, spmatrix
import numpy as np

## Plot imports
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches     import Polygon
from matplotlib.collections import PatchCollection


## Function for plotting linear constraints in a contour plot
def linear_constraints_contour(
        ## Objective function variables
        X, Y, Z, A, b,
        ## Plot parameters
        n_contours=25, show_plot=False, ):
    """
    LINEAR_CONSTRAINTS_CONTOUR
    Function for plotting linear inequality constraints in a contour plot.

    The Linear constraints must be supplied as A and b, from
    constraints of the form
        A.T * x - b >= 0

    Inputs:
        X           ->  Meshgrid of x of size (nx x nx)
        Y           ->  Meshgrid of y of size (ny x ny)
        Z           ->  Objective function values (contour values)

        A           ->  Matrix defining linear inequality constriants
        b           ->  Vector defining linear inequality constraints

        n_contours  ->  Number of contour lines, equaly spaced
        show_plot   ->  Show the plot (boolean)

    Outputs:
        fig         ->  Figure holding the contour plot
        ax          ->  Axes of the contour plot
    """

    ## Define sizes
    nx, ny = X.size[0], Y.size[0]

    ## Define variable intervals as diagonals of meshgrid variables
    x = X[::nx+1]
    y = Y[::ny+1]

    ## Define maximum and minimum variables
    x_min, x_max = cvxopt.min(x), cvxopt.max(x)
    y_min, y_max = cvxopt.min(y), cvxopt.max(y)

    ## Define large value for patches
    M = 1e+10

    ## Constraint patches
    x_ = matrix( [ x_min-0.1, x, x_max+0.1 ] ).T
    y_ = matrix( [ y_min-0.1, y, y_max+0.1 ] ).T
    x_val = matrix( [ x_, y_ ] )
    start_l = matrix([x_[0] ,-M])
    end_l   = matrix([x_[-1],-M])
    start_u = matrix([x_[0] , M])
    end_u   = matrix([x_[-1], M])
    patches = []
    for i in range(b.size[0]):
        ## Reset invalid boolean
        invalid = False

        ## Validity check
        if sum(abs(A[i,:])) <= 1e-15:
            invalid = True
        ## a_2 = 0
        elif abs(A[i,1]) <= 1e-15:
            c_x = matrix( 0.0*x_ + b[i] )
            c_val = matrix( [ c_x, y_ ] )
            if A[i,0] < 0:
                c_val = np.array( \
                    matrix( [ [end_l], [c_val], [end_u] ] ) ).T
            else:
                c_val = np.array( \
                    matrix( [ [start_l], [c_val], [start_u] ] ) ).T
        ## a_1 = 0
        elif abs(A[i,0]) <= 1e-15:
            c_y = matrix( 0.0*x_ + b[i]/A[i,1] )
            c_val = matrix( [ x_, c_y ] )
            if A[i,1] < 0:
                c_val = np.array( \
                    matrix( [ [start_u], [c_val], [end_u] ] ) ).T
            else:
                c_val = np.array( \
                    matrix( [ [start_l], [c_val], [end_l]] )).T
        else:
            if A[i,1] < 0:
                c_y = matrix( ( A[i,0]*x_ - b[i] ) / -A[i,1] )
                c_val = matrix( [ x_, c_y ] )
                c_val = np.array( \
                    matrix( [ [start_u], [c_val], [end_u] ] ) ).T
            else:
                c_y = matrix( ( A[i,0]*x_ - b[i] ) / -A[i,1] )
                c_val = matrix( [ x_, c_y ] )
                c_val = np.array( \
                    matrix( [ [start_l], [c_val], [end_l] ] ) ).T

        ## Do not plot invalid constraints, e.g., 0-constraints
        if not invalid:
            polygon = Polygon( c_val, True )
            patches.append(polygon)

    ## Define patches for plot
    p = PatchCollection(patches, alpha=0.5 )
    p.set_facecolor( 'black' )
    p.set_edgecolor( 'black' )

    ## Plot objective and constraint patches
    fig, ax = plt.subplots()
    ax.set_xlim( x_min, x_max )
    ax.set_ylim( y_min, y_max )

    normalise = matplotlib.colors.Normalize( \
        vmin=cvxopt.min(Z), vmax=cvxopt.max(Z) )
    color_map = plt.cm.get_cmap("jet")
    ax.contour( X, Y, Z, n_contours, cmap=color_map, norm=normalise )
    ax.add_collection(p)

    ## Show plot
    if show_plot:
        plt.show()

    ## Return statement
    return fig, ax
