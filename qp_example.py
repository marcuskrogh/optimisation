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
    x_opt_cvxopt = qp( H, g, -C.T, -d )['x']
    print( 'Optimal solution (C V X O P T):              \n', x_opt_cvxopt )


    ## Solution via interior point algorithm
    res_ip = interior_point( H, g, C=C, d=d, x_0=[2,2] )
    print( 'Optimal solution (custom interior point):    \n', res_ip['x']  )

    ## Solution via custom algorithm - Primal active set
    res_as = active_set( H, g, C=C, d=d, x_0=[3.5,2.5], w_0=[4] )
    print( 'Optimal solution (custom primal active set): \n', res_as['x']  )


    ## Visualisation of iteration sequence and optimal point
    ax.plot( x_opt_cvxopt[0] , x_opt_cvxopt[1] , 'k1' , markersize=20 )
    ax.plot( res_ip['X'][:,0], res_ip['X'][:,1], 'r2-', markersize=20 )
    ax.plot( res_as['X'][:,0], res_as['X'][:,1], 'b3-', markersize=20 )
    plt.show()


## Execution
if __name__ == '__main__':
    main()
