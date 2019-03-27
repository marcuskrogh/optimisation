## CVXOPT import
import numpy as np
import cvxopt
from cvxopt                       import matrix, spmatrix
from cvxopt.solvers               import lp, options
options['show_progress'] = False

## Random LP problems and visualisation
from test_problems.random_lp import generate_lp
from visualisation.lp_plot   import driver


## Import custom LP solver
from solvers import interior_point_lp as interior_point
from solvers import                      simplex

## Pyplot for visualisation
import matplotlib.pyplot as plt


############################################################################
################################ QP Example ################################
############################################################################
def main():

    ## Define problem size
    n = 2
    m = 2

    ## Generate problem
    g, A, b, x_opt, _, _ = generate_lp( n, m )
    C = matrix( spmatrix( 1.0, range(n), range(n) ) ) # x >= 1
    d = matrix( 0.0, (n,1) )                          # x >= 1

    ## Simplex algorithm
    simplex( g, A, b, [0] )

    ## Plot problem
    fig, ax = driver( g, A, b, C, d )

    ## Compute solution
    res = interior_point( g, A, b, matrix( 1.0, x_opt.size ) )

    ## Plot iteration sequence
    ax.plot( x_opt[0]     , x_opt[1]     , 'k2' , markersize=20 )
    ax.plot( res['X'][:,0], res['X'][:,1], 'r-1', markersize=20 )
    plt.show()


## Execution
if __name__ == '__main__':
    main()
