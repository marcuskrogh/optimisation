############################################################################
################################## Imports #################################
############################################################################
## CVX opt
import cvxopt
from cvxopt import matrix, spmatrix, spdiag, sqrt, mul, div

## Numpy
import numpy as np

## Time
import time
############################################################################
############################################################################
############################################################################


############################################################################
############################### Type checking ##############################
############################################################################
def type_checking( g, A, b, x_0 ):
    try:
        g = matrix(g)
        n = g.size[0]
    except:
        print( 'InputError: System is not properly defined.' )

    try:
        A  = matrix(A)
        b  = matrix(b)
        ma = b.size[0]
    except:
        A  = matrix( 0.0, (n,0) )
        b  = matrix( 0.0, (0,1) )
        ma = 0

    try:
        x_0 = matrix( x_0 )
    except:
        x_0 = matrix( 0.0, (n,1) )

    return g, A, b, x_0, n, ma
############################################################################
############################################################################
############################################################################


############################################################################
########################### Interior Point Method ##########################
############################################################################
def interior_point_lp( \
    ## System matrices
    g, \
    ## Equality constraint matrices
    A=None, b=None, \
    ## Initial guess(es)
    x_0=None,
    ## ALgirithm parameters
    eta=0.995, tol=1e-9, it_max=100, ):
    """
############################################################################
####### Primal-Dual Predictor-Corrector Interior Point Method for LPs ######
############################################################################
    Description:
        Primal-dual predictor-corrector interior-point method for solving
        standard form linear programmes:

            min     g' x
             x
            s.t.    A' x == b
                       x >= 0
############################################################################


############################################################################
    Inputs:
        g           ->      Linear objective vector         |   n  x 1
        A           ->      Equality constraint matrix      |   n  x ma
        b           ->      Equality constraint vector      |   ma x 1
        x_0         ->      Initial guess of x              |   n  x 1
        eta         ->      Step length                     |   float
        tol         ->      Tolerance of optimality         |   float
        it_max      ->      Maximum allowed iterations      |   integer

    Outputs:
        res         ->      Result dictionary
            Optimal Variables:
                x   ->      State variables                 |   n  x 1
                y   ->      Lagrange multiplier (Eq.)       |   ma x 1
                z   ->      Lagrange multiplier (In.)       |   n  x 1
                mu  ->      Duality gap                     |   float
            Iteration data:
                X   ->      State variables                 |   N  x n
                Y   ->      Lagrange multiplier (Eq)        |   N  x ma
                Z   ->      Lagrange multiplier (Ineq)      |   N  x n
                Mu  ->      Duality gap                     |   N  x 1
            Congergence information:
                converged   ->  Did the algorithm converge  |   boolean
                N           ->  Number of iterations        |   integer
                T           ->  Time used                   |   ms
############################################################################
    """
    ## Start timing
    cpu_time_start = time.process_time()

    ########################################################################
    ################### Type checking and initialisation ###################
    ########################################################################
    ## Check matrix and vector types
    g, A, b, x_0, n, ma = \
        type_checking( g, A, b, x_0 )

    ## Check problem type
    if ma > 0:
        pass
    else:
        print( 'Problem is unconstrained.' )
        return False

    ## Initialisation
    x = x_0
    y = matrix( 0.0, (ma,1) )   # Lagrange multipliers (Eq.)
    z = matrix( 1.0, (n ,1) )   # Lagrange multipliers (In.)


    ## Iteration data
    X  = matrix( [ x.T  ] )
    Y  = matrix( [ y.T  ] )
    Z  = matrix( [ z.T  ] )
    ########################################################################
    ########################################################################
    ########################################################################


    ########################################################################
    ############################ Initialisation ############################
    ########################################################################
    ## Definition of tolerances
    tol_L  = tol
    tol_A  = tol
    tol_mu = tol

    ## Compute residuals
    r_L  = g - A*y - z          # Lagrange gradient
    r_A  = A.T*x - b            # Equality constraint
    r_C  = mul(x,z)             # Complementarity
    mu   = matrix(sum(r_C)/n)   # Duality gap

    ## Storage of duality gap
    Mu   = matrix( [ mu.T ] )

    ## Define convergence measure
    def converged( r_L, r_A, mu ):
        res = ( ( max(matrix([0.0,r_L    ]) ) <= tol_L  ) and    \
                ( max(matrix([0.0,r_A    ]) ) <= tol_A  ) and    \
                ( max(matrix([0.0,abs(mu)]) ) <= tol_mu )        )
        return res

    ########################################################################
    ########################################################################
    ########################################################################


    ########################################################################
    ############################ Main algorithm ############################
    ########################################################################
    it = 0
    while (not converged(r_L, r_A, mu)) and (it < it_max):
        ## Iterate
        it += 1

        ## Form hessian and compute Cholesky factorisation
        x_div_z = div(x,z)
        xz_diag = matrix(spdiag(x_div_z))
        H = A.T*xz_diag*A
        L = +H
        cvxopt.lapack.potrf(L)  # Cholesky decompostion

        ## Affine step
        tmp = div(mul(x,r_L) + r_C,z)
        rhs = -r_A + A.T*tmp
        dy = +rhs
        cvxopt.lapack.potrs( L, dy )   # Solve

        dx = mul(x_div_z,A*dy) - tmp
        dz = -div(r_C + mul(z,dx),x)

        x_idx   = matrix( list( filter( \
            lambda i: dx[i] < 0.0, range(n) ) ) )  # Limit in x
        z_idx   = matrix( list( filter( \
            lambda i: dz[i] < 0.0, range(n) ) ) )  # Limit in z
        alpha = min( matrix([ 1.0, -div(x[x_idx],dx[x_idx])]) )
        beta  = min( matrix([ 1.0, -div(z[z_idx],dz[z_idx])]) )

        ## Centering step
        x_aff  = x + alpha*dx
        z_aff  = z + beta *dz
        mu_aff = sum( mul(x_aff,z_aff) )/n

        sigma = (mu_aff/mu)**3
        tau   = sigma*mu

        ## Center-corrector step
        r_C = r_C + mul(dx,dz) - tau

        tmp = div( mul(x,r_L) + r_C, z )
        rhs = -r_A + A.T*tmp
        dy = +rhs
        cvxopt.lapack.potrs( L, dy )   # Solve
        dz  = -div( r_C + mul(z,dx), x )

        x_idx   = matrix( list( filter( \
            lambda i: dx[i] < 0.0, range(n) ) ) )  # Limit in x
        z_idx   = matrix( list( filter( \
            lambda i: dz[i] < 0.0, range(n) ) ) )  # Limit in z
        alpha = min( matrix([ 1.0, -div(x[x_idx],dx[x_idx])]) )
        beta  = min( matrix([ 1.0, -div(z[z_idx],dz[z_idx])]) )

        ## Take step
        x = x + eta*alpha*dx;
        y = y + eta*beta *dy;
        z = z + eta*beta *dz;

        ## Re-compute residuals
        r_L  = g - A*y - z          # Lagrange gradient
        r_A  = A.T*x - b            # Equality constraint
        r_C  = mul(x,z)             # Complementarity
        mu   = matrix(sum(r_C)/n)   # Duality gap

        ## Store iteration data
        X  = matrix( [ X , x.T  ] )
        Y  = matrix( [ Y , y.T  ] )
        Z  = matrix( [ Z , z.T  ] )
        Mu = matrix( [ Mu, mu.T ] )
    ########################################################################
    ########################################################################
    ########################################################################

    ## Finish timing
    cpu_time = time.process_time() - cpu_time_start

    ########################################################################
    ###################### Construct result dictionary #####################
    ########################################################################
    res =   { \
        ## Optimal variables
        'x'         : x,                            \
        'y'         : y,                            \
        'z'         : z,                            \
        'mu'        : mu,                           \
        ## Iteration data
        'X'         : X,                            \
        'Y'         : Y,                            \
        'Z'         : Z,                            \
        'Mu'        : Mu,                           \
        ## Convergence information
        'converged' : converged(r_L, r_A, mu),      \
        'N'         : it,                           \
        'T'         : cpu_time,                     \
            }
    ########################################################################
    ########################################################################
    ########################################################################


    return res
