import cvxopt
from cvxopt                import matrix
from cvxopt.solvers        import qp, lp, options
options['show_progress'] = False

from qp_function           import objective   as qp_objective
from qp_function           import constraints as qp_constraints
from qp_example            import driver      as qp_driver

from active_set_primal_qp  import active_set

import matplotlib.pyplot as plt

def kktsolver( H, g, A, b, ):
    n, m = A.size

    O     = matrix( 0.0, (m, m) )
    KKT   = matrix( [ [ H, -A.T ], [ -A, O ] ] ) # KKT matrix
    res   = matrix( [ -g, -b ] )                 # Right-hand side
    I     = matrix( range(n+m) )

    cvxopt.lapack.sytrf( KKT, I )
    cvxopt.lapack.sytrs( KKT, I, res )
    x       = res[:n]
    lambda_ = res[n:]
    print( res )

    return x, lambda_

def unqpsolver( H, g, ):
    ## H is positive semi-definite, should be Cholesky, not LDL
    x = -g
    I = matrix( range(H.size[0]) )
    cvxopt.lapack.sytrf( H, I )
    cvxopt.lapack.sytrs( H, I, x )
    return x


## Initialisation and plotting of feasible region
H, g    = qp_objective()
C, d    = qp_constraints()
fig, ax = qp_driver()

## Solution to inequality constrained QP
x_opt_cvxopt = qp( H, g, -C.T, -d )['x']
print( 'Optimal solution (from CVXOpt): \n', x_opt_cvxopt )

x_opt, lambda_opt, W_opt, X = active_set( H, g, C=C, d=d, \
    x_0=[3.5,2.5], W_0=[4] )
print( 'Optimal solution (from Primal Active Set): ' )
print( 'x:                  \n', x_opt      )
print( 'lambda:             \n', lambda_opt )
print( 'Working set:        \n', W_opt      )
print( 'Iteration sequence: \n', X          )

## Visualisation
ax.plot( x_opt_cvxopt[0], x_opt_cvxopt[1], 'k2' , markersize=10 )
ax.plot( X[:,0]         , X[:,1]         , 'r1-', markersize=10 )
plt.show()
