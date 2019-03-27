############################################################################
################################## Imports #################################
############################################################################
## CVX opt
import cvxopt
from cvxopt import matrix, spmatrix, spdiag, sqrt, mul, div, max
from cvxopt.solvers import qp

## Numpy
import numpy as np

## Import QP solver
from solvers import interior_point_qp

## Time
import time
############################################################################
############################################################################
############################################################################


############################################################################
################################# Line Search ##############################
############################################################################
## Just backtracking right now.
##
## Cannot get polynimoal approximation to work properly.
def line_search( \
        ## Objective and constraints
        obj, eq_cons, in_cons, \
        ## Initial evaluations
        f, df, c_eq, c_in, \
        ## Currect step and lagrange multipliers
        y, z, x, dx, mu_, lambda_,\
        ## Algorithm parameters
        it_max=100 ):

    ## Update merits
    tmp     = 0.5*( mu_ + abs(y) )
    mu_     = matrix( np.max( np.array( \
        [ np.array(abs(y)), np.array(tmp) ] ), axis=0 ) )
    tmp     = 0.5*( lambda_ + abs(z) )
    lambda_ = matrix( np.max( np.array( \
        [ np.array(abs(z)), np.array(tmp) ] ), axis=0 ) )

    ## Compute c=phi(0) and b=phi'(0)
    h_xk = matrix(abs(c_eq))
    g_xk = matrix([abs(min(0.0,c_in[i])) for i in range(c_in.size[0])])
    phi_0  = f       + mu_.T*h_xk + lambda_.T*g_xk
    dphi_0 = df.T*dx - mu_.T*h_xk - lambda_.T*g_xk

    ## Main loop
    alpha     = 1.0
    converged = False
    it = 0
    nargout = ['f','c']
    while (not converged) and it < it_max:
        ## Take step
        x_ = x + alpha*dx;

        ## Re-evaluate objective and constraints
        f    = obj(     x_, nargout )[0]
        c_eq = eq_cons( x_, nargout )[0]
        c_in = in_cons( x_, nargout )[0]

        ##
        h_xk = matrix(abs(c_eq))
        g_xk = matrix([abs(min(0.0,c_in[i])) for i in range(c_in.size[0])])

        ##
        phi_alpha = f + mu_.T*h_xk + lambda_.T*g_xk
        phi_tmp   = phi_0 + 0.1*dphi_0*alpha
        if phi_alpha[0] <= phi_tmp[0]:
            converged = True
        else:
            a = ( phi_alpha - ( phi_0 + alpha*dphi_0 ) ) / ( alpha*alpha )
            alpha_min = - dphi_0[0] / ( 2.0*a[0] )
            #alpha = min( [ 0.9*alpha, max( [ alpha_min, 0.1*alpha ] ) ] )
            alpha *= 0.95

        ## Iterate
        it += 1

    return alpha, mu_, lambda_
############################################################################
############################################################################
############################################################################


############################################################################
########################### BFGS Update to Hessian #########################
############################################################################
def bfgs(B, p, q):
    if (p.T*q)[0] < (0.2*p.T*(B*p))[0]:
        theta = div( 0.8*p.T*(B*p), ( p.T*(B*p) - p.T*q ) )
    else:
        theta = 1.0
    r    = theta*q + ( 1.0 - theta )*(B*p);
    B += div( r*r.T, p.T*r ) - div( (B*p)*(B*p).T, p.T*(B*p) )
    return B
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
def type_checking( x_0, y_0, z_0, B_0, obj, eq_cons, in_cons ):
    ## Initial guesses
    try:
        x_0 = matrix( x_0 )
        y_0 = matrix( y_0 )
        z_0 = matrix( z_0 )
        B_0 = matrix( B_0 )
        n , _ = x_0.size
        me, _ = y_0.size
        mi, _ = z_0.size
    except:
        print( 'InitialGuessError: Initial guesses could not be converted to cvxopt.matrix.' )

    ## Objective check
    try:
        f, df, ddf = obj(x_0)
        obj_check = True
    except:
        obj_check = False
        print( 'ObjectiveError: Objective function not evaluating properly.' )

    ## Equality constraints
    try:
        c_eq, dc_eq, ddc_eq = eq_cons(x_0)
        eq_check = True
    except:
        eq_check = False
        print( 'ConstraintError: Equality constraints not evaluating properly.' )

    ## Inequality constraints
    try:
        c_in, dc_in, ddc_in = in_cons(x_0)
        in_check = True
    except:
        in_check = False
        print( 'ConstraintError: Inequality constraints not evaluating properly.' )

    return x_0, y_0, z_0, B_0, n, me, mi, obj_check, eq_check, in_check
############################################################################
############################################################################
############################################################################


############################################################################
############ Non-Linear Sequantial Quadratic Programming Solver ############
############################################################################
def sqp( \
    ## Objective function
    obj=None,
    ## Equality constraints
    eq_cons=None,
    ## Inequality constraints
    in_cons=None,
    ## Initial guess
    x_0=None, y_0=None, z_0=None, B_0=None,
    ## ALgirithm parameters
    tol=1e-8, it_max=100, **kwargs):
    """
############################################################################
############ Non-Linear Sequantial Quadratic Programming Solver ############
############################################################################
    Description:

    Inputs:

    Output:

    """
    ## Start timing
    cpu_time_start = time.process_time()

    ## Type checking
    x_0, y_0, z_0, B_0, n, me, mi, obj_check, eq_check, in_check = \
        type_checking( x_0, y_0, z_0, B_0, obj, eq_cons, in_cons )

    ## Define variables
    x = x_0
    y = y_0
    z = z_0
    B = B_0

    ## Initial evaluation of objective and constraints
    f   , df   , ddf    = obj(     x, **kwargs )
    c_eq, dc_eq, ddc_eq = eq_cons( x, **kwargs )
    c_in, dc_in, ddc_in = in_cons( x, **kwargs )
    dL = df - dc_eq*y - dc_in*z

    ## Evaluate convergence measures
    converged = \
        ( max( abs( dL   ) ) < tol ) and \
        ( max( abs( c_eq ) ) < tol )

    ## Storage of relevant data
    X  = matrix( x.T )
    Y  = matrix( y.T )
    Z  = matrix( z.T )
    DL = matrix( max(abs(dL)) )

    ## Main loop
    mu_     = abs(y)
    lambda_ = abs(z)
    it = 0
    while (not converged) and it < it_max:
        ## Compute SQP by solvering equality constrained QP
        #dx, y = kktsolver( H, df, dc_eq, -c_eq )
        res = interior_point_qp( \
            B, df, dc_eq, -c_eq, dc_in, -c_in, x, y, abs(z), abs(z) )
        #res = qp( B, df, -dc_in.T, c_in, dc_eq.T, -c_eq )
        dx = res['x']

        ## Update step
        y    = res['y']
        z    = res['z']
        dL_  = df - dc_eq*y - dc_in*z    # For BFGS update

        ## Line search
        alpha, mu_, lambda_ = line_search(  \
            obj, eq_cons, in_cons,          \
            f, df, c_eq, c_in,              \
            y, z, x, dx, mu_, lambda_ )

        x += alpha*dx

        ## Re-evaluation objective and constraints
        f   , df   , _ = obj(     x, **kwargs )
        c_eq, dc_eq, _ = eq_cons( x, **kwargs )
        c_in, dc_in, _ = in_cons( x, **kwargs )

        ## BFGS-update to Hessian
        dL = df - dc_eq*y - dc_in*z
        q  = dL - dL_
        B  = bfgs( B, dx, q )

        ## Evaluate convergence measures
        converged = \
            ( max( abs( dL   ) ) < tol ) and \
            ( max( abs( c_eq ) ) < tol )

        ## Store relevant data
        X  = matrix( [ X, x.T ] )
        Y  = matrix( [ Y, y.T ] )
        Z  = matrix( [ Z, z.T ] )
        DL = matrix( [ DL, max(abs(dL)) ] )

        ## Iterate
        it += 1
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
        'dL'        : max(abs(dL)),                 \
        ## Iteration data
        'X'         : X,                            \
        'Y'         : Y,                            \
        'Z'         : Z,                            \
        'DL'        : DL,                           \
        ## Convergence information
        'converged' : converged,                    \
        'N'         : it,                           \
        'T'         : cpu_time,                     \
            }
    ########################################################################
    ########################################################################
    ########################################################################


    ## Return statement
    return res
############################################################################
############################################################################
############################################################################
