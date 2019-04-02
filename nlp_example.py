## CVXOPT import
import cvxopt
from cvxopt                            import matrix, spmatrix, log, exp
from cvxopt.solvers                    import qp, options
options['show_progress'] = False

import numpy as np

## QP related functions
from test_problems.himmelblau_function import objective
from test_problems.himmelblau_function import equality_constraints
from test_problems.himmelblau_function import inequality_constraints
from visualisation.nlp_plot            import driver

## QP Solver - Sequencial Quadratic Programming (SQP) Solver
from solvers import sqp

## Pyplot for visualisation
import matplotlib.pyplot as plt


############################################################################
####################### Equality Constrained Example #######################
############################################################################
def nlp_example():
    ## Visualisation of QP with constraints
    fig, ax = driver( \
        objective, equality_constraints, inequality_constraints )

    ## Solve problem
    #x_0 = matrix( 1.0*np.random.randn(2,1) )
    x_0 = matrix( [-1.0, 0.0] )
    y_0 = matrix( [ 1.0 ] )
    z_0 = matrix( [ 1.0, 1.0 ] )
    B_0 = matrix( spmatrix( 1.0, range(2), range(2) ) )
    tmp = matrix([[1.0,2.0],[0.0,2.0]])
    nargout = ['f','c','df','dc']

    res = sqp( objective, equality_constraints, inequality_constraints, \
        x_0, y_0, z_0, B_0, nargout=nargout )
    print( res['X'] )
    print( res['Alpha'] )
    print( res['converged'] )
    print( res['N'] )
    print( res['dL'] )

    p1, = ax.plot( x_0[0]       , x_0[1]       , 'bo', markersize=10.0, \
        label='initial point', zorder=4 )
    p2, = ax.plot( res['X'][:,0], res['X'][:,1], 'rx-'                , \
        label='iteration sequence', zorder=3 )
    p3, = ax.plot( res['x'][0]  , res['x'][1]  , 'go', markersize=10.0, \
        label='optimal point', zorder=5 )
    ax.legend( handles=[p1,p2,p3] )
    ax.set_title( 'SQP Solution to Himmelblau Problem', fontsize=15 )
    fig.tight_layout()


    ## Convergence plot parameters
    P_conv = 0.25
    N_conv = int(P_conv*res['DL'].size[0])

    ## Define data points
    e_k   = res['DL'][-N_conv:-1]
    e_kp1 = res['DL'][(-N_conv+1):]

    ## Fit straight line in log-log space
    A_conv = matrix([ [ matrix( 1.0, (N_conv-1,1) ) ], [ log(e_k) ],])
    b_conv = log(e_kp1)
    beta   = matrix(np.linalg.pinv(A_conv))*b_conv
    print( beta )

    ## Plot convergence
    fig_, ax_ = plt.subplots( ncols=1, nrows=1, figsize=(10,6) )
    ax_.loglog( e_k, e_kp1, 'k.' )

    plt.show()



    strippage = "'{: abcdefghijklmnopqrstuvwxyz"
    for i in range(len(strippage)):
        .strip()



nlp_example()
