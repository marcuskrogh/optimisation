## CVXOPT import
import cvxopt
from cvxopt                       import matrix
from cvxopt.solvers               import qp, options
options['show_progress'] = False

## QP related functions
from test_problems.qp_function    import objective, constraints
from visualisation.qp_plot        import driver

## QP Solver - primal active set
from solvers import active_set_qp     as active_set
from solvers import interior_point_qp as interior_point

## Pyplot for visualisation
import matplotlib.pyplot as plt


############################################################################
####################### Generate Quadratic Programme #######################
############################################################################
def generate_qp():
    ## Initialisation of QP (Inequality constrained)
    H, g = objective()
    C, d = constraints()

    return H, g, C, d


############################################################################
########################### Unconstrained Example ##########################
############################################################################
def unconstrained_qp():
    ## Generate QP
    H, g, _, _ = generate_qp()

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

    plt.savefig( 'figures/unconstrained_qp.png', dpi=600 )
    plt.show()


############################################################################
####################### Equality Constrained Example #######################
############################################################################
def equality_constrained_qp():
    ## Generate QP
    H, g, C, d = generate_qp()

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

    plt.savefig( 'figures/equality_qp.png', dpi=600 )
    plt.show()
    plt.close()


############################################################################
###################### Inequality Constrained Example ######################
############################################################################
def inequality_constrained_qp():
    ## Generate QP
    H, g, C, d = generate_qp()

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
    res_ip = interior_point( H, g, C=C, d=d, x_0=[2.0,0.0] )
    print( 'Custom interior point solution:    \n', res_ip['x']  )
    print( 'Iterations: %d     ' % res_ip['N']        )
    print( 'CPU-time:   %.2f ms' % (res_ip['T']*1000) )
    print( '------------------------' )
    print()

    ## Solution via custom algorithm - Primal active set
    print( '------------------------' )
    res_as = active_set( H, g, C=C, d=d, x_0=[2.0,0.0], w_0=[2,4] )
    print( 'Custom primal active set solution: \n', res_as['x']  )
    print( 'Iterations: %d     ' % res_as['N']        )
    print( 'CPU-time:   %.2f ms' % (res_as['T']*1000) )
    print( '------------------------' )
    print()

    ## Visualisation of iteration sequence and optimal point
    ax.plot( x_opt_cvxopt[0] , x_opt_cvxopt[1] , 'k1' , markersize=20 )
    ax.plot( res_ip['X'][:,0], res_ip['X'][:,1], 'r2-', markersize=20 )
    ax.plot( res_as['X'][:,0], res_as['X'][:,1], 'b3-', markersize=20 )

    plt.savefig( 'figures/inequality_qp.png', dpi=600 )
    plt.show()
    plt.close()


############################################################################
################## Inequality-Equality Constrained Example #################
############################################################################
def inequality_equality_constrained_qp():
    ## Generate QP
    H, g, C, d = generate_qp()
    A = C[:,1]
    b = matrix(d[1])
    C_ = matrix( [ [C[:,0]], [C[:,2:]] ] )
    d_ = matrix( [  d[0]   ,  d[2:]    ] )

    ## Visualisation of QP with constraints
    fig, ax = driver(H, g, A, b, C_, d_ )

    ## Solution via cvxopt
    print( '------------------------' )
    x_opt_cvxopt = qp( H, g, -C_.T, -d_, A.T, b )['x']
    print( 'C V X O P T solution:              \n', x_opt_cvxopt )
    print( '------------------------' )
    print()

    ## Solution via interior point algorithm
    print( '------------------------' )
    res_ip = interior_point( H, g, A=A, b=b, C=C_, d=d_, x_0=[1.0,1.0] )
    print( 'Custom interior point solution:    \n', res_ip['x']  )
    print( 'Iterations: %d     ' % res_ip['N']        )
    print( 'CPU-time:   %.2f ms' % (res_ip['T']*1000) )
    print( '------------------------' )
    print()

    ## Solution via custom algorithm - Primal active set
    print( '------------------------' )
    res_as = active_set( H, g, A=A, b=b, C=C_, d=d_, x_0=[4.0,1.0], w_0=[3] )
    print( 'Custom primal active set solution: \n', res_as['x']  )
    print( 'Iterations: %d     ' % res_as['N']        )
    print( 'CPU-time:   %.2f ms' % (res_as['T']*1000) )
    print( '------------------------' )
    print()

    ## Visualisation of iteration sequence and optimal point
    ax.plot( x_opt_cvxopt[0] , x_opt_cvxopt[1] , 'k1' , markersize=20 )
    ax.plot( res_ip['X'][:,0], res_ip['X'][:,1], 'r2-', markersize=20 )
    ax.plot( res_as['X'][:,0], res_as['X'][:,1], 'b3-', markersize=20 )

    plt.savefig( 'figures/inequality_equality_qp.png', dpi=600 )
    plt.show()
    plt.close()


############################################################################
################################# Execution ################################
############################################################################
if __name__ == '__main__':
    unconstrained_qp()
    equality_constrained_qp()
    inequality_constrained_qp()
    inequality_equality_constrained_qp()
