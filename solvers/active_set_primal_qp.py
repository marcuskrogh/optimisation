## CVXopt import
import cvxopt
from cvxopt         import matrix, spmatrix, sqrt
from cvxopt.solvers import qp

## Additional imports
import numpy as np


############################################################################
############################# Utility Functions ############################
############################################################################
## KKT solver
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


## Euclidean norm
def norm( x, ):
    """
    Computes the 2-norm of a vector
        || x ||_2 = sqrt( sum( x.T * x ) )
    """
    return sqrt(x.T*x)


## Unconstrained QP solver
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
    x_0=None, W_0=None, \
    ## Algorithm parameters
    it_max=10, kktsolver_=kktsolver):
    """
    Active set algorithm for quadratic programmes.

    Description:
        The function solves the inequality constraint quadratic programme
        of the form:

            min  1/2 x' H x + g' x
            s.t. A' x - b =  0
                 C' c - d >= 0

        By treating a subset of inequality constraints as equality
        constriants, then iterating through subsets until the optimal
        solution is found.

    Inputs:
        H           ->      Quadratic objective matrix      |   n  x n
        g           ->      Linear objective vector         |   n  x 1
        A           ->      Equality constraint matrix      |   n  x me
        b           ->      Equality constraint vector      |   me x 1
        C           ->      Inequality constraint matrix    |   n  x mi
        d           ->      Inequality constraint vector    |   mi x 1
        x_0         ->      Initial guess on optimal point  |   n  x 1
        W_0         ->      Initial guess on working set    |   Indecies
        it_max      ->      Maximum number of iterations    |   integer
        kktsolver_  ->      Solver for KKT system
            Default: KKT solver using LDL-decomposition.
            The KKT solver is chosen as an input, such that the solver
            can be constructed to optimally use specific problem
            structures, e.g., sparse systems, block-diagonal systems.

    Outputs:
        x_opt   ->      Optimal point                   |   n  x 1
        W_opt   ->      Optimal working set             |   Indecies
    """

    ## Type checking
    try:
        H   = matrix(H)
        n, _ = H.size

        g   = matrix(g)
    except:
        print( 'System matrices are not properly defined.' )

    try:
        A = matrix(A)
        b = matrix(b)
        eq_cons = True
    except:
        A = matrix( 0.0, (n,0) )
        b = matrix( 0.0, (0,1) )
        eq_cons = False

    try:
        C = matrix(C)
        d = matrix(d)
        ineq_cons = True
    except:
        C = matrix( 0.0, (n,0) )
        d = matrix( 0.0, (0,1) )
        ineq_cons = False

    try:
        x_0 = matrix( x_0 )
    except:
        x_0 = matrix( 0.0, (n,1) ) # Default initial guess on optimal point

    try:
        W_0 = matrix( W_0 )
    except:
        W_0 = matrix( [ 0 ] )      # Default initial guess on working set

    ## Problem type
    if eq_cons:
        if ineq_cons:
            #Full problem
            print( 'Equality-inequality constrained QP.' )
            pass
        else:
            #Equality constrained problem
            print( 'Equality constrained QP. ' )
            print( 'Solving via KKT solver...' )

            ## Solve via KKT system
            x_opt, lambda_opt = kktsolver_( H, g, A, b )
            W_opt = matrix( range( A.size[1] ) )
            X = matrix( [ x_0.T, x_k.T ] )

            return x_opt, lambda_opt, W_opt, X
    else:
        if ineq_cons:
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
    W_k = W_0
    eps = 1e-13

    _, me = A.size
    _, mi = C.size
    #...

    ## Storage initialisation
    X = matrix(x_k.T)

    ## While-loop
    it = 0
    while it < it_max:
        """
        Solve the QP from 16.39

            Solve the equality constrained QP:

                min  1/2 p_k' H p_k + g_k' p_k
                s.t. A_bar' p_k = 0

            to compute the step p_k = x - x_k
        """
        ## Setup sub-problem
        g_k   = H*x_k + g
        if len(W_k) == 0:
            C_W = matrix( 0.0, (n,0) )
        else:
            C_W   = matrix( [[C[j,i] for i in W_k] for j in range(n)] ).T

        ## Define equality constrants including Working set (W_k)
        A_bar = matrix( [ [A], [C_W] ] )
        O_bar = matrix( 0.0, (A_bar.size[1],1) )
        p_k, lambda_k = kktsolver_( H, g_k, A_bar, O_bar )

        ## Check if current point is optimal
        if norm(p_k)[0] < eps:
            ## Lambda_k satisfies 16.42 from KKT solution
            if min(lambda_k) >= 0.0:
                ## Optimal solution found
                print( 'Optimal solution found.' )
                x_opt      = x_k
                lambda_opt = matrix( 0.0, (mi,1) )
                lambda_opt[W_k] = lambda_k
                W_opt      = W_k

                ## Return statement
                return x_opt, lambda_opt, W_opt, X
            else:
                ## Find constraint with negative Lagrange multiplier
                j   = list(filter( \
                    lambda j: lambda_k[j]==min(lambda_k), \
                    range( len(lambda_k) ) ) )[0]

                ## Remove chosen constraint
                W_k = matrix( list( filter( \
                    lambda i: not i == W_k[j], W_k ) ) )
        else:
            ## Construct inactive constraints
            nW_k = matrix( list( filter( \
                lambda i: i not in W_k, range(mi) ) ) )
            nC_W    = matrix( [ [ C[j,i] for i in nW_k ] \
                for j in range(n) ] )
            nd_W    = matrix( [ d[i] for i in nW_k ] )

            ## Line search parameter. Compute from 16.41
            res = []
            for i in range(len(nW_k)):
                tmp_ = nC_W[i,:]*p_k
                if tmp_[0] < 0.0:
                    tmp_ = (nd_W[i] - nC_W[i,:]*x_k)[0] / tmp_[0]
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
                W_k = matrix( [W_k, nW_k[j[0]]] )
            else:
                x_k += p_k
                #W_k = W_k

        it += 1

        ## Storage
        X = matrix( [X, x_k.T] )

    if it == it_max:
        print( 'Reached maximum iterations.' )
        ## Return statement
        return x_k, lambda_k, W_k, X
