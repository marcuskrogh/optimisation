## CVXOPT import
import cvxopt
from cvxopt                       import matrix
from cvxopt.solvers               import qp, options
options['show_progress'] = False

## QP related functions
from test_problems.qp_function    import objective, constraints
from visualisation.qp_plot        import driver

## QP Solver - primal active set
from solvers import active_set
from solvers import interior_point

## Pyplot for visualisation
import matplotlib.pyplot as plt


############################################################################
################################ QP Example ################################
############################################################################
def main():
    ## Initialisation of QP (Inequality constrained)
    H, g = objective()
    C, d = constraints()


    ########################################################################
    ######################### Unconstrained Example ########################
    ########################################################################
    ## Visualisation of QP with constraints
    fig, ax = driver( H, g )

    ## Solution via cvxopt
    print( '------------------------' )
    x_opt_cvxopt = qp( H, g )['x']
    print( 'C V X O P T solution:              \n', x_opt_cvxopt )
    print( '------------------------' )
    print()

    ## Solution via interior point algorithm
    print( '------------------------' )
    res_ip = interior_point( H, g )
    print( 'Custom interior point solution:    \n', res_ip['x']  )
    print( 'Iterations: %d     ' % res_ip['N']        )
    print( 'CPU-time:   %.2f ms' % (res_ip['T']*1000) )
    print( '------------------------' )
    print()

    ## Solution via custom algorithm - Primal active set
    print( '------------------------' )
    res_as = active_set( H, g )
    print( 'Custom primal active set solution: \n', res_as['x']  )
    print( 'Iterations: %d     ' % res_as['N']        )
    print( 'CPU-time:   %.2f ms' % (res_as['T']*1000) )
    print( '------------------------' )
    print()


    ## Visualisation of iteration sequence and optimal point
    ax.plot( x_opt_cvxopt[0] , x_opt_cvxopt[1] , 'k1' , markersize=20 )
    ax.plot( res_ip['X'][:,0], res_ip['X'][:,1], 'r2-', markersize=20 )
    ax.plot( res_as['X'][:,0], res_as['X'][:,1], 'b3-', markersize=20 )

    plt.savefig( 'unconstrained_qp.pdf' )
    plt.show()


    ########################################################################
    ##################### Equality Constrained Example #####################
    ########################################################################
    ## Visualisation of QP with constraints
    fig, ax = driver( H, g, C[:,3], d[3] )

    ## Solution via cvxopt
    print( '------------------------' )
    x_opt_cvxopt = qp( H, g, A=C[:,3].T, b=matrix(d[3]) )['x']
    print( 'C V X O P T solution:              \n', x_opt_cvxopt )
    print( '------------------------' )
    print()

    ## Solution via interior point algorithm
    print( '------------------------' )
    res_ip = interior_point( H, g, A=C[:,3], b=d[3] )
    print( 'Custom interior point solution:    \n', res_ip['x']  )
    print( 'Iterations: %d     ' % res_ip['N']        )
    print( 'CPU-time:   %.2f ms' % (res_ip['T']*1000) )
    print( '------------------------' )
    print()

    ## Solution via custom algorithm - Primal active set
    print( '------------------------' )
    res_as = active_set( H, g, A=C[:,3], b=d[3] )
    print( 'Custom primal active set solution: \n', res_as['x']  )
    print( 'Iterations: %d     ' % res_as['N']        )
    print( 'CPU-time:   %.2f ms' % (res_as['T']*1000) )
    print( '------------------------' )
    print()


    ## Visualisation of iteration sequence and optimal point
    ax.plot( x_opt_cvxopt[0] , x_opt_cvxopt[1] , 'k1' , markersize=20 )
    ax.plot( res_ip['X'][:,0], res_ip['X'][:,1], 'r2-', markersize=20 )
    ax.plot( res_as['X'][:,0], res_as['X'][:,1], 'b3-', markersize=20 )

    plt.savefig( 'equality_qp.pdf' )
    plt.show()


    ########################################################################
    #################### Inequality Constrained Example ####################
    ########################################################################
    ## Visualisation of QP with constraints
    fig, ax = driver( H, g, C=C, d=d )

    ## Solution via cvxopt
    print( '------------------------' )
    x_opt_cvxopt = qp( H, g, -C.T, -d )['x']
    print( 'C V X O P T solution:              \n', x_opt_cvxopt )
    print( '------------------------' )
    print()

    ## Solution via interior point algorithm
    print( '------------------------' )
    res_ip = interior_point( H, g, C=C, d=d, x_0=[1,1] )
    print( 'Custom interior point solution:    \n', res_ip['x']  )
    print( 'Iterations: %d     ' % res_ip['N']        )
    print( 'CPU-time:   %.2f ms' % (res_ip['T']*1000) )
    print( '------------------------' )
    print()

    ## Solution via custom algorithm - Primal active set
    print( '------------------------' )
    res_as = active_set( H, g, C=C, d=d, x_0=[1.0,1.0], w_0=[] )
    print( 'Custom primal active set solution: \n', res_as['x']  )
    print( 'Iterations: %d     ' % res_as['N']        )
    print( 'CPU-time:   %.2f ms' % (res_as['T']*1000) )
    print( '------------------------' )
    print()

    ## Visualisation of iteration sequence and optimal point
    ax.plot( x_opt_cvxopt[0] , x_opt_cvxopt[1] , 'k1' , markersize=20 )
    ax.plot( res_ip['X'][:,0], res_ip['X'][:,1], 'r2-', markersize=20 )
    ax.plot( res_as['X'][:,0], res_as['X'][:,1], 'b3-', markersize=20 )

    plt.savefig( 'inequality_qp.pdf' )
    plt.show()


    ########################################################################
    ################ Inequality-Equality Constrained Example ###############
    ########################################################################
    ## Visualisation of QP with constraints
    fig, ax = driver(H, g, C[:,0], d[0], C[:,1:], d[1:] )

    ## Solution via cvxopt
    print( '------------------------' )
    x_opt_cvxopt = qp( H, g, -C[:,1:].T, -d[1:], \
                       C[:,0].T, matrix(d[0]) )['x']
    print( 'C V X O P T solution:              \n', x_opt_cvxopt )
    print( '------------------------' )
    print()

    ## Solution via interior point algorithm
    print( '------------------------' )
    res_ip = interior_point( H, g, A=C[:,0], b=d[0], C=C[:,1:], d=d[1:], \
                             x_0=[1.0,1.0] )
    print( 'Custom interior point solution:    \n', res_ip['x']  )
    print( 'Iterations: %d     ' % res_ip['N']        )
    print( 'CPU-time:   %.2f ms' % (res_ip['T']*1000) )
    print( '------------------------' )
    print()

    ## Solution via custom algorithm - Primal active set
    print( '------------------------' )
    res_as = active_set( H, g, A=C[:,0], b=d[0], C=C[:,1:], d=d[1:], \
                         x_0=[1.0,1.5], w_0=[] )
    print( 'Custom primal active set solution: \n', res_as['x']  )
    print( 'Iterations: %d     ' % res_as['N']        )
    print( 'CPU-time:   %.2f ms' % (res_as['T']*1000) )
    print( '------------------------' )
    print()

    ## Visualisation of iteration sequence and optimal point
    ax.plot( x_opt_cvxopt[0] , x_opt_cvxopt[1] , 'k1' , markersize=20 )
    ax.plot( res_ip['X'][:,0], res_ip['X'][:,1], 'r2-', markersize=20 )
    ax.plot( res_as['X'][:,0], res_as['X'][:,1], 'b3-', markersize=20 )

    plt.savefig( 'inequality_equality_qp.pdf' )
    plt.show()


## Execution
if __name__ == '__main__':
    main()
