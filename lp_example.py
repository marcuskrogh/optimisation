## CVXOPT import
import cvxopt
from cvxopt                       import matrix
from cvxopt.solvers               import lp, options
options['show_progress'] = False

## QP related functions
from visualisation.lp_function    import objective, constraints
from visualisation.lp_plot        import driver

## Possible custom LP solver
#import custom_lp_solver

## Pyplot for visualisation
import matplotlib.pyplot as plt


############################################################################
################################ QP Example ################################
############################################################################
def main():
    ## Initialisation of QP (Inequality constrained)
    g    = objective()
    C, d = constraints()

    ## Visualisation of QP with constraints
    fig, ax = driver()

    ## Solution via CVXOPT
    x_opt_cvxopt = lp( g, -C.T, -d )['x']
    print( 'Optimal solution (from CVXOpt): \n', x_opt_cvxopt )

    """
    ## Solution via custom algorithm - Primal active set
    x_opt, lambda_opt, W_opt, X = custom_lp_solver( g, C=C, d=d, \
        x_0=[3.5,2.5], W_0=[4] )
    print( 'Optimal solution (from Primal Active Set): ' )
    print( 'x:                  \n', x_opt      )
    #print( 'lambda:             \n', lambda_opt )
    #print( 'Working set:        \n', W_opt      )
    #print( 'Iteration sequence: \n', X          )
    """

    ## Visualisation of iteration sequence and optimal point
    ax.plot( x_opt_cvxopt[0], x_opt_cvxopt[1], 'k2' , markersize=10 )
    #ax.plot( X[:,0]         , X[:,1]         , 'r1-', markersize=10 )
    plt.show()


## Execution
if __name__ == '__main__':
    main()