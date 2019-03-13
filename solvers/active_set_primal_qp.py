## CVXopt import
import cvxopt
from cvxopt         import matrix, spmatrix, sqrt
from cvxopt.solvers import qp

## Additional imports
import numpy as np


############################################################################
################################# KKT solver ###############################
############################################################################
def kktsolver( H, g, A, b, ):
    ## Problem size
    n, m = A.size

    ## Setup of KKT-system
    O   = matrix( 0.0, (m, m) )                # Zeros for KKT matrix
    KKT = matrix( [ [ H, A.T ], [ A, O ] ] ) # KKT matrix
    res = matrix( [ g, b ] )                 # Right-hand side
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
############################ Euclidean distance ############################
############################################################################
def norm( x, ):
    """
    Computes the 2-norm of a vector
        || x ||_2 = sqrt( sum( x.T * x ) )
    """
    return sqrt(x.T*x)
############################################################################
############################################################################
############################################################################


############################################################################
############################### Type checking ##############################
############################################################################
def type_checking( H, g, A, b, C, d, x_0, w_0 ):
    try:
        H   = matrix(H)
        n, _ = H.size

        g   = matrix(g)
    except:
        print( 'System matrices are not properly defined.' )

    try:
        A = matrix(A)
        _, ma = A.size
        b = matrix(b)
        eq = True
    except:
        ma = 0
        A = matrix( 0.0, (n,ma) )
        b = matrix( 0.0, (ma,1) )
        eq = False

    try:
        C = matrix(C)
        _, mc = C.size
        d = matrix(d)
        ineq = True
    except:
        mc = 0
        C = matrix( 0.0, (n,mc) )
        d = matrix( 0.0, (mc,1) )
        ineq = False

    try:
        x_0 = matrix( x_0 )
    except:
        x_0 = matrix( 0.0, (n,1) ) # Default: 0.0

    try:
        w_0 = matrix( w_0 )
    except:
        w_0 = matrix( [] )      # Default: Empty set

    return H, g, A, b, C, d, x_0, w_0, n, ma, mc, eq, ineq
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
############### Active Set Algorithm for Quadratic Programmes ##############
############################################################################
def active_set(      \
    ## Objective matrices
    H, g,               \
    ## Equality constraint matrices
    A=None, b=None,     \
    ## Inequality constraint matrices
    C=None, d=None,     \
    ## Initial values
    x_0=None, w_0=None, \
    ## Algorithm parameters
    it_max=100, kktsolver_=kktsolver):
    """
############################################################################
################## Primal Active Set Method for Convex QPs #################
############################################################################

    Description:
        The function solves the inequality constraint quadratic programme
        of the form:

            min  1/2 x' H x + g' x
            s.t. A' x == b
                 C' x >= d

        By treating a subset of inequality constraints as equality
        constriants, then iterating through subsets until the optimal
        solution is found.

    Inputs:
        H           ->      Quadratic objective matrix      |   n  x n
        g           ->      Linear objective vector         |   n  x 1
        A           ->      Equality constraint matrix      |   n  x ma
        b           ->      Equality constraint vector      |   ma x 1
        C           ->      Inequality constraint matrix    |   n  x mc
        d           ->      Inequality constraint vector    |   mc x 1
        x_0         ->      Initial guess on x              |   n  x 1
        w_0         ->      Initial guess on working set    |   Indecies
        it_max      ->      Maximum allowed iterations      |   integer
            Default: 100
        kktsolver_  ->      Solver for KKT system
            Default: KKT solver using LDL-decomposition.
            The KKT solver is chosen as an input, such that the solver
            can be constructed to optimally use specific problem
            structures, e.g., sparse systems, block-diagonal systems.

    Outputs:
        x_opt   ->      Optimal point                   |   n  x 1
        w_opt   ->      Optimal working set             |   Indecies
    """
    ## Type checking
    H, g, A, b, C, d, x_0, w_0, n, ma, mc, eq, ineq = \
        type_checking( H, g, A, b, C, d, x_0, w_0 )

    ## Problem type
    if eq:
        if ineq:
            #Full problem
            print( 'Equality-inequality constrained QP.' )
            pass
        else:
            #Equality constrained problem
            print( 'Equality constrained QP. ' )
            print( 'Solving via KKT solver...' )

            ## Solve via KKT system
            x_opt, lambda_opt = kktsolver_( H, g, A, b )
            w_opt = matrix( range( A.size[1] ) )
            X = matrix( [ x_0.T, x_k.T ] )

            return x_opt, lambda_opt, w_opt, X
    else:
        if ineq:
            #Inequality constrained problem
            print( 'Inequality constrained QP.' )
            pass
        else:
            #Unconstrained problem
            print( 'Unconstrained QP.' )

            ## Solve via normal equations
            x_opt = ucqpsolver( H, g )
            X = matrix( [ x_0.T, x_opt.T ] )

            ## Return statement
            return x_opt, matrix([]), matrix([]), X

    ## Initialisation
    x_k = x_0
    w_k = w_0
    eps = 1e-13

    _, me = A.size
    _, mc = C.size
    #...

    ## Storage initialisation
    X = matrix( x_k.T     )
    Y = matrix( matrix( 0.0, (1,ma+mc) ) )
    W = []

    ## While-loop
    it = 0
    converged = False
    while it < it_max:
        """
        Solve the QP from 16.39

            Solve the equality constrained QP:

                min     1/2 p_k' H p_k + g_k' p_k
                p_k

                s.t.    A_bar' p_k = 0,

            where p_k is the step direction, such that the next iterate
            x_{k+1} = x_k + alpha * p_k, where 0 <= alpha <= 1.
        """
        ## Setup sub-problem
        g_k   = H*x_k + g
        if len(w_k) == 0:
            C_w = matrix( 0.0, (n,0) )
        else:
            C_w   = matrix( [ [C[:,i]] for i in w_k ] )

        ## Define equality constrants including Working set (W_k)
        A_bar = matrix( [ [A], [C_w] ] )
        O_bar = matrix( 0.0, (A_bar.size[1],1) )
        p_k, lambda_k = kktsolver_( H, g_k, A_bar, O_bar )

        ## Check if current point is optimal
        if norm(p_k)[0] < eps:
            ## Lambda_k satisfies 16.42 from KKT solution
            if min(lambda_k[ma:]) >= 0.0:
                ## Optimal solution found
                print( 'Optimal solution found.' )
                converged  = True
                x_opt      = x_k
                if len(lambda_k[ma:]) == 0:
                    lambda_opt = matrix( 0.0, (1,ma+mc) )
                else:
                    lambda_opt = matrix( 0.0, (1,ma+mc) )
                    lambda_opt[:ma]    = lambda_k[:ma]
                    lambda_opt[ma+w_k] = lambda_k[ma:]
                w_opt      = w_k

                ## Construct result dictionary
                res =   { \
                    ## Optimal variables
                    'x'         : x_opt,        \
                    'y'         : lambda_opt,   \
                    'w'         : w_opt,        \
                    ## Iteration data
                    'X'         : X,            \
                    'Y'         : Y,            \
                    'W'         : W,            \
                    ## Convergence information
                    'converged' : converged,    \
                    'N'         : it,           \
                        }
                return res
            else:
                ## Find constraint with negative Lagrange multiplier
                j   = list(filter( \
                    lambda j: lambda_k[ma+j]==min(lambda_k[ma:]), \
                    range( len(lambda_k[ma:]) ) ) )[0]

                ## Remove chosen constraint
                w_k = matrix( list( filter( \
                    lambda i: not i == w_k[j], w_k ) ) )
                lambda_k = matrix(lambda_k[ma+j])
        else:
            ## Construct inactive constraints
            nw_k = matrix( list( filter( \
                lambda i: i not in w_k, range(mc) ) ) )
            nC_w    = matrix( [ [C[:,i]] for i in nw_k ] )
            nd_w    = matrix( [ d[i] for i in nw_k ] )

            ## Line search parameter. Compute from 16.41
            res = []
            for i in range(len(nw_k)):
                tmp_ = nC_w[:,i].T*p_k
                if tmp_[0] < 0.0:
                    tmp_ = (nd_w[i] - nC_w[:,i].T*x_k)[0] / tmp_[0]
                    res.append( tmp_ )
                else:
                    res.append( np.inf )
            res   = matrix( res )
            alpha = min(res)
            j     = list( filter( \
                lambda j: res[j] == alpha, range(len(res)) ) )

            ## Update step and working set
            if alpha < 1.0:
                x_k += alpha * p_k
                w_k = matrix( [w_k, nw_k[j[0]]] )
            else:
                x_k += p_k
                #w_k = w_k

        it += 1

        ## Store iteration data
        X = matrix( [ X, x_k.T     ] )
        if len(lambda_k[ma:]) == 0:
            lambda_opt = matrix( 0.0, (1,ma+mc) )
        else:
            lambda_opt = matrix( 0.0, (1,ma+mc) )
            lambda_opt[:ma]    = lambda_k[:ma]
            lambda_opt[ma+w_k] = lambda_k[ma:]
        Y = matrix( [ Y, lambda_opt ] )
        W.append([w_k])


    ########################################################################
    ###################### Construct result dictionary #####################
    ########################################################################
    res =   { \
        ## Optimal variables
        'x'         : x_opt,        \
        'y'         : lambda_opt,   \
        'w'         : w_opt,        \
        ## Iteration data
        'X'         : X,            \
        'Y'         : Y,            \
        'W'         : W,            \
        ## Convergence information
        'converged' : converged,    \
        'N'         : it,           \
            }
    ########################################################################
    ########################################################################
    ########################################################################


    return res
