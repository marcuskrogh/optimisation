## CVXopt import
import cvxopt
from cvxopt         import matrix, spmatrix, sqrt
from cvxopt.solvers import qp

## Additional imports
import numpy as np

## Time
import time

############################################################################
################################# KKT solver ###############################
############################################################################
def kktsolver( H, g, A, b, ):
    ## Problem size
    n, m = A.size

    ## Setup of KKT-system
    O   = matrix( 0.0, (m, m) )                 # Zeros for KKT matrix
    KKT = matrix( [ [ H, A.T ], [ A, O ] ] )    # KKT matrix
    res = matrix( [ g, -b ] )                   # Right-hand side
    I   = matrix( range(n+m) )                  # Permutation vector

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
        H = matrix(H)
        g = matrix(g)
        n = g.size[0]
    except:
        print( 'System matrices are not properly defined.' )

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
        x_0 = matrix( 0.0, (n,1) ) # Default: 0.0

    try:
        w_0 = matrix( w_0 )
    except:
        w_0 = matrix( [] )      # Default: Empty set

    return H, g, A, b, C, d, x_0, w_0, n, ma, mc
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
        res         ->      Result dictionary
            Optimal Variables:
                x   ->      State variables                 | n  x 1
                y   ->      Lagrange multiplier (Eq)        | ma x 1
                w   ->      Initial guess on working set    | Indecies
            Iteration data:
                X   ->      State variables                 | N  x n
                Y   ->      Lagrange multiplier (Eq)        | N  x ma
                W   ->      Working set                     | N  x n_w(k)
            Congergence information:
                converged   ->  Did the algorithm converge  | boolean
                N           ->  Number of iterations        | integer
                T           ->  Time used                   | Seconds
    """
    ## Start timing
    cpu_time_start = time.process_time()

    ## Type checking
    H, g, A, b, C, d, x_0, w_0, n, ma, mc = \
        type_checking( H, g, A, b, C, d, x_0, w_0 )

    ## Problem type
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
            x_k, lambda_opt = kktsolver_( H, g, A, b )
            w_k = matrix( range( A.size[1] ) )
            X = matrix( [ x_k.T ] )

            ## Finish timing
            cpu_time = time.process_time() - cpu_time_start

            res =   { \
                ## Optimal variables
                'x'         : x_k,          \
                'y'         : lambda_opt,   \
                'w'         : w_k,          \
                ## Iteration data
                'X'         : X,            \
                'Y'         : None,         \
                'W'         : None,         \
                ## Convergence information
                'converged' : True,         \
                'N'         : 0,            \
                'T'         : cpu_time,     \
                    }

            return res
    else:
        if mc > 0:
            #Inequality constrained problem
            print( 'Inequality constrained QP.' )
            pass
        else:
            #Unconstrained problem
            print( 'Unconstrained QP.' )

            ## Solve via normal equations
            x_k = ucqpsolver( H, g )
            X = matrix( [ x_k.T ] )

            ## Finish timing
            cpu_time = time.process_time() - cpu_time_start

            res =   { \
                ## Optimal variables
                'x'         : x_k,          \
                'y'         : None,         \
                'w'         : None,         \
                ## Iteration data
                'X'         : X,            \
                'Y'         : None,         \
                'W'         : None,         \
                ## Convergence information
                'converged' : True,         \
                'N'         : 0,            \
                'T'         : cpu_time,     \
                    }

            ## Return statement
            return res

    ## Initialisation
    x_k = x_0
    w_k = w_0
    eps = 1e-13

    ## Storage initialisation
    X = matrix( x_k.T     )
    Y = matrix( matrix( 0.0, (1,ma+mc) ) )
    W = []

    ## While-loop
    it = 0
    converged = False
    while (not converged) and (it < it_max):
        """
        Solve the QP from 16.39

            Solve the equality constrained QP:

                min     1/2 p_k' H p_k + g_k' p_k
                p_k

                s.t.    A_bar' p_k = 0,

            where p_k is the step direction, such that the next iterate
            x_{k+1} = x_k + alpha * p_k, where 0 <= alpha <= 1.
        """
        ## Iterate
        it += 1

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
            if min(matrix([0.0,lambda_k[ma:]])) >= 0.0:
                ## Optimal solution found
                converged  = True
                if len(lambda_k[ma:]) == 0:
                    lambda_opt = matrix( 0.0, (1,ma+mc) )
                else:
                    lambda_opt = matrix( 0.0, (1,ma+mc) )
                    lambda_opt[:ma]    = lambda_k[:ma]
                    lambda_opt[ma+w_k] = lambda_k[ma:]
                w_opt      = w_k
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


    ## Finish timing
    cpu_time = time.process_time() - cpu_time_start

    ########################################################################
    ###################### Construct result dictionary #####################
    ########################################################################
    res =   { \
        ## Optimal variables
        'x'         : x_k,          \
        'y'         : lambda_opt,     \
        'w'         : w_k,          \
        ## Iteration data
        'X'         : X,            \
        'Y'         : Y,            \
        'W'         : W,            \
        ## Convergence information
        'converged' : converged,    \
        'N'         : it,           \
        'T'         : cpu_time,     \
            }
    ########################################################################
    ########################################################################
    ########################################################################


    return res
