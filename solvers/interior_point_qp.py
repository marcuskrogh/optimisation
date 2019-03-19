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
########################## Unconstrained QP solver #########################
############################################################################
def ucqpsolver( H, g, ):
    L = +H
    x = +(-g)

    ## Cholesky-decomposition
    cvxopt.lapack.potrf( L    )

    ## Solve linear system of equation via Cholesky-decomposition
    cvxopt.lapack.potrs( L, x )

    ## Return statement
    return x
############################################################################
############################################################################
############################################################################


############################################################################
################################# KKT solver ###############################
############################################################################
def kktsolver( H, g, A, b, ):
    ## Problem size
    n, m = A.size

    ## Setup of KKT-system
    O   = matrix( 0.0, (m, m) )                # Zeros for KKT matrix
    KKT = matrix( [ [ H, A.T ], [ A, O ] ] ) # KKT matrix
    res = matrix( [ g, -b ] )                 # Right-hand side
    I   = matrix( range(n+m) )                 # Identity for LDL

    ## LDL-decomposition
    cvxopt.lapack.sytrf( KKT, I )

    ## Solution via LDL-decompotision
    cvxopt.lapack.sytrs( KKT, I, res )

    ## Solution
    x       = -res[:n]  # Optimisation variable
    lambda_ = res[n:]   # Lagrange multipliers

    return x, lambda_
############################################################################
############################################################################
############################################################################


############################################################################
############################### Type checking ##############################
############################################################################
def type_checking( H, g, A, b, C, d, x_0, y_0, z_0, s_0 ):
    try:
        H = matrix(H)
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
        C  = matrix(C)
        d  = matrix(d)
        mc = d.size[0]
    except:
        C  = matrix( 0.0, (n,0) )
        d  = matrix( 0.0, (0,1) )
        mc = 0

    try:
        x_0 = matrix( x_0 )
    except:
        x_0 = matrix( 0.0, (n,1) )

    try:
        y_0 = matrix( y_0 )
    except:
        y_0 = matrix( 0.0, (ma,1) )

    try:
        z_0 = matrix( z_0 )
    except:
        z_0 = matrix( 1.0, (mc,1) )

    try:
        s_0 = matrix( s_0 )
    except:
        s_0 = matrix( 1.0, (mc,1) )


    return H, g, A, b, C, d, x_0, y_0, z_0, s_0, n, ma, mc
############################################################################
############################################################################
############################################################################


############################################################################
########################### Interior Point Method ##########################
############################################################################
def interior_point_qp( \
    ## System matrices
    H, g, \
    ## Equality constraint matrices
    A=None, b=None, \
    ## Inequality constraint matrices
    C=None, d=None, \
    ## Initial guess
    x_0=None, y_0=None, z_0=None, s_0=None,
    ## ALgirithm parameters
    eta=0.995, tol=1e-10, it_max=100, ):
    """
############################################################################
### Primal-Dual Predictor-Corrector Interior Point Method for Convex QPs ###
############################################################################
    Description:
        Primal-dual predictor-corrector interior-point method for solving
        convex constrained quadratic programmes in the form:

            min     1/2 x' H x + g' x
             x
            s.t.    A' x == b
                    C' x >= d
############################################################################


############################################################################
        If the input is an unconstrained quadratic programme:

            min     1/2 x' H x + g' x
             x

        the solution is directly computed by solving the equation

            H * x = -g
############################################################################


############################################################################
        If the input is an equality constrained quadratic programme:

            min     1/2 x' H x + g' x
             x
            s.t.    A' x == b

        the solution is directly computed via the Karush-Kuhn-Tucker (KKT)
        conditions. The solution is obtained by solving the KKT system

            [  H  -A ] [ x ]    [ -g ]
            [ -A'  0 ] [ y ] == [ -b ]
############################################################################


############################################################################
    Inputs:
        H           ->      Quadratic objective matrix      |   n  x n
        g           ->      Linear objective vector         |   n  x 1
        A           ->      Equality constraint matrix      |   n  x ma
        b           ->      Equality constraint vector      |   ma x 1
        C           ->      Inequality constraint matrix    |   n  x mc
        d           ->      Inequality constraint vector    |   mc x 1
        x_0         ->      Initial guess of x              |   n  x 1
        y_0         ->      Initial guess of y              |   n  x 1
        z_0         ->      Initial guess of z              |   n  x 1
        s_0         ->      Initial guess of s              |   n  x 1
        eta         ->      Step length                     |   float
        tol         ->      Tolerance of optimality         |   float
        it_max      ->      Maximum allowed iterations      |   integer

    Outputs:
        res         ->      Result dictionary
            Optimal Variables:
                x   ->      State variables                 |   n  x 1
                y   ->      Lagrange multiplier (Eq.)       |   ma x 1
                z   ->      Lagrange multiplier (In.)       |   mc x 1
                s   ->      Slack variables     (In.)       |   mc x 1
            Iteration data:
                X   ->      State variables                 |   N  x n
                Y   ->      Lagrange multiplier (Eq.)       |   N  x ma
                Z   ->      Lagrange multiplier (In.)       |   N  x mc
                S   ->      Slack variables     (In.)       |   N  x mc
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
    H, g, A, b, C, d, x_0, y_0, z_0, s_0, n, ma, mc = \
        type_checking( H, g, A, b, C, d, x_0, y_0, z_0, s_0 )


    ## Check problem type
    if ma > 0:
        if mc > 0:
            #Full problem
            print( 'Equality-inequality constrained QP.' )
            pass
        else:
            #Equality constrained problem
            print( 'Equality constrained QP. ' )
            print( 'Solving via KKT solver...' )

            ## Solve via KKT system
            x, y = kktsolver( H, g, A, b )
            X = matrix( [ x.T ] )
            Y = matrix( [ y.T ] )

            ## Finish timing
            cpu_time = time.process_time() - cpu_time_start

            res =   { \
                ## Optimal variables
                'x'         : x,            \
                'y'         : y,            \
                'z'         : None,         \
                's'         : None,         \
                ## Iteration data
                'X'         : X,            \
                'Y'         : Y,            \
                'Z'         : None,         \
                'S'         : None,         \
                ## Convergence information
                'converged' : True,         \
                'N'         : 0,            \
                'T'         : cpu_time,     \
                    }

            return res
    else:
        print( mc )
        if mc > 0:
            #Inequality constrained problem
            print( 'Inequality constrained QP.' )
            pass
        else:
            #Unconstrained problem
            print( 'Unconstrained QP.' )

            ## Solve via normal equations
            x = ucqpsolver( H, g )
            X = matrix( [ x.T ] )

            ## Finish timing
            cpu_time = time.process_time() - cpu_time_start

            res =   { \
                ## Optimal variables
                'x'         : x,            \
                'y'         : None,         \
                'z'         : None,         \
                's'         : None,         \
                ## Iteration data
                'X'         : X,            \
                'Y'         : None,         \
                'Z'         : None,         \
                'S'         : None,         \
                ## Convergence information
                'converged' : True,         \
                'N'         : 0,            \
                'T'         : cpu_time,     \
                    }

            return res

    ## Initialisation
    x    = x_0
    y    = y_0
    z    = z_0
    s    = s_0

    ## Iteration data
    X    = matrix( [ x.T ] )
    Y    = matrix( [ y.T ] )
    Z    = matrix( [ z.T ] )
    S    = matrix( [ s.T ] )
    ########################################################################
    ########################################################################
    ########################################################################


    ########################################################################
    ####################### Initial guess correction #######################
    ########################################################################
    ## Compute residuals
    r_L  = H*x + g - A*y - C*z
    r_A  = b - A.T*x
    r_C  = s + d - C.T*x
    r_SZ = mul( s, z )

    ## Setup KKT system and LDL-decomposition
    H_bar = H + C*(spdiag(div(z,s)))*C.T
    KKT   = matrix( [ [ H_bar, -A.T                   ] , \
                      [ -A   , matrix( 0.0, (ma,ma) ) ] ] )
    p     = matrix( range(n+ma) )   # Permutation vector
    cvxopt.lapack.sytrf( KKT, p )   # LDL-decomposition

    ## Affine seach direction
    r_bar_L   = r_L - C * mul( div(z,s), r_C - div(r_SZ,z) )
    r_A_bar_L = matrix( [ -r_bar_L, -r_A ] )
    d_aff     = +r_A_bar_L
    cvxopt.lapack.sytrs( KKT, p, d_aff )    # Solve system via LDL
    dx_aff    = d_aff[:n]
    dz_aff    = - mul( div(z,s), C.T*dx_aff )   \
                + mul( div(z,s), r_C - div(r_SZ,z) )
    ds_aff    = - div(r_SZ,z) - mul( div(s,z), dz_aff )

    ## Update to initial guess
    #x = x  # z and s determines feasibility
    #y = y  # z and s determines feasibility
    z = matrix( [ max( 1, (z[i] + dz_aff[i]) ) for i in range(mc) ] )
    s = matrix( [ max( 1, (s[i] + ds_aff[i]) ) for i in range(mc) ] )
    ########################################################################
    ########################################################################
    ########################################################################


    ########################################################################
    ##################### Definition of stop criterion #####################
    ########################################################################
    ## Compute residuals
    r_L  = H*x + g - A*y - C*z
    r_A  = b - A.T*x
    r_C  = s + d - C.T*x
    r_SZ = mul( s, z )
    mu   = sum(r_SZ)/mc

    ## Define tolerances
    tol_L  = tol * max( matrix([1.0,H[::],g[::],A[::],C[::]]) )
    tol_A  = tol * max( matrix([1.0,A[::],b[::]]) )
    tol_C  = tol * max( matrix([1.0,d[::],C[::]]) )
    tol_mu = tol * mu

    ## Define convergence measure
    def converged( r_L, r_A, r_C, mu ):
        res = ( ( max(matrix([0.0,r_L    ]) ) <= tol_L  ) and    \
                ( max(matrix([0.0,r_A    ]) ) <= tol_A  ) and    \
                ( max(matrix([0.0,r_C    ]) ) <= tol_C  ) and    \
                ( max(matrix([0.0,abs(mu)]) ) <= tol_mu )        )
        return res
    ########################################################################
    ########################################################################
    ########################################################################


    ########################################################################
    ############################ Main algorithm ############################
    ########################################################################
    it = 0
    while (not converged(r_L, r_A, r_C, mu)) and (it < it_max):
        ## Iterate
        it += 1

        ## Form KKT system and compute LDL-factorisation
        H_bar = H + C*(spdiag(div(z,s)))*C.T
        KKT   = matrix([[ H_bar, -A.T                   ], \
                        [ -A   , matrix( 0.0, (ma,ma) ) ]] )
        p   = matrix( range(n+ma) )     # Permutation vector
        cvxopt.lapack.sytrf( KKT, p )   # Overwrites KKT and p

        ## Compute affine seach direction
        r_bar_L   = r_L - C * mul( div(z,s), r_C - div(r_SZ,z) )
        r_A_bar_L = matrix( [ -r_bar_L, -r_A ] )
        d_aff     = +r_A_bar_L
        cvxopt.lapack.sytrs( KKT, p, d_aff )    # Solves via LDL
        dx_aff    = d_aff[:n]
        dz_aff    = - mul( div(z,s), C.T*dx_aff )   \
                    + mul( div(z,s), r_C - div(r_SZ,z) )
        ds_aff    = - div(r_SZ,z) - mul( div(s,z), dz_aff )

        ## Compute affine step length
        z_idx = matrix( list( filter( \
            lambda i: dz_aff[i] < 0.0, range(mc) ) ) )  # Limits in z
        s_idx = matrix( list( filter( \
            lambda i: ds_aff[i] < 0.0, range(mc) ) ) )  # Limits in s

        alpha_z_aff = min( matrix([ 1.0, -div(z[z_idx],dz_aff[z_idx])]) )
        alpha_s_aff = min( matrix([ 1.0, -div(s[s_idx],ds_aff[s_idx])]) )
        alpha_aff   = min( matrix([alpha_z_aff, alpha_s_aff]) )

        ## Compute duality gap and centering parameter
        mu_aff = (z + alpha_aff*dz_aff).T * (s + alpha_aff*ds_aff) / mc
        sigma  = (mu_aff/mu)**3

        ## Computate affine-centering-correction search direction
        r_bar_SZ  = r_SZ + mul( ds_aff, dz_aff ) - sigma*mu
        r_bar_L   = r_L - C*mul( div(z,s), r_C - div(r_bar_SZ,z) )
        r_A_bar_L = matrix( [ -r_bar_L, -r_A ] )
        d_        = +r_A_bar_L
        cvxopt.lapack.sytrs( KKT, p, d_ )   # Solves via LDL
        dx        = d_[:n]
        dy        = d_[n:]
        dz        = - mul( div(z,s), C.T*dx ) \
                    + mul( div(z,s), r_C - div(r_bar_SZ,z) )
        ds        = - div(r_bar_SZ,z) - mul( div(s,z), dz )

        ## Compute affine-centering-correction step length
        z_idx = matrix( list( filter( \
            lambda i: dz[i] < 0.0, range(mc) ) ) )
        s_idx = matrix( list( filter( \
            lambda i: ds[i] < 0.0, range(mc) ) ) )

        alpha_z   = min( matrix( [ 1 , -div(z[z_idx],dz[z_idx]) ] ) )
        alpha_s   = min( matrix( [ 1 , -div(s[s_idx],ds[s_idx]) ] ) )
        alpha     = min( matrix( [ alpha_z, alpha_s ] ) )
        alpha_bar = eta*alpha

        ## Update step
        x = x + alpha_bar*dx
        y = y + alpha_bar*dy
        z = z + alpha_bar*dz
        s = s + alpha_bar*ds

        ## Re-compute residuals
        r_L  = H*x + g - A*y - C*z;
        r_A  = b - A.T*x;
        r_C  = s + d - C.T*x
        r_SZ = mul( s, z )
        mu   = sum(r_SZ)/mc

        ## Store iteration data
        X = matrix( [ X, x.T ] )
        Y = matrix( [ Y, y.T ] )
        Z = matrix( [ Z, z.T ] )
        S = matrix( [ S, s.T ] )
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
        's'         : s,                            \
        ## Iteration data
        'X'         : X,                            \
        'Y'         : Y,                            \
        'Z'         : Z,                            \
        'S'         : S,                            \
        ## Convergence information
        'converged' : converged(r_L, r_A, r_C, mu), \
        'N'         : it,                           \
        'T'         : cpu_time,                     \
            }
    ########################################################################
    ########################################################################
    ########################################################################


    return res
