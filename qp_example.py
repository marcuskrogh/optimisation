## CVXOPT import
import cvxopt
from cvxopt                       import matrix
from cvxopt.solvers               import qp, options
options['show_progress'] = False

## QP related functions
from visualisation.qp_function    import objective, constraints
from visualisation.qp_plot        import driver

## QP Solver - primal active set
from solvers.active_set_primal_qp import active_set
from solvers.interior_point_qp    import interior_point

## Pyplot for visualisation
import matplotlib.pyplot as plt


############################################################################
################################ QP Example ################################
############################################################################
def main():
    ## Initialisation of QP (Inequality constrained)
    H, g = objective()
    C, d = constraints()

    ## Visualisation of QP with constraints
    fig, ax = driver()


    ## Redefine prolem


    ## Solution via cvxopt
    print( '------------------------' )
    x_opt_cvxopt = qp( H, g, -C.T, -d )['x']
    print( 'C V X O P T solution:              \n', x_opt_cvxopt )
    print( '------------------------' )
    print()


    ## Solution via interior point algorithm
    print( '------------------------' )
    res_ip = interior_point( H, g, C=C, d=d, x_0=[2,2] )
    print( 'Custom interior point solution:    \n', res_ip['x']  )
    print( 'Iterations: %d     ' % res_ip['N']        )
    print( 'CPU-time:   %.2f ms' % (res_ip['T']*1000) )
    print( '------------------------' )
    print()

    ## Solution via custom algorithm - Primal active set
    print( '------------------------' )
    res_as = active_set( H, g, C=C, d=d, x_0=[3.5,2.5], w_0=[4] )
    print( 'Custom primal active set solution: \n', res_as['x']  )
    print( 'Iterations: %d     ' % res_as['N']        )
    print( 'CPU-time:   %.2f ms' % (res_as['T']*1000) )
    print( '------------------------' )
    print()


    ## Visualisation of iteration sequence and optimal point
    ax.plot( x_opt_cvxopt[0] , x_opt_cvxopt[1] , 'k1' , markersize=20 )
    ax.plot( res_ip['X'][:,0], res_ip['X'][:,1], 'r2-', markersize=20 )
    ax.plot( res_as['X'][:,0], res_as['X'][:,1], 'b3-', markersize=20 )
    plt.show()


## Execution
if __name__ == '__main__':
    main()
