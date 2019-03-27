## Imports
import cvxopt
from cvxopt import matrix


## Objective function
def objective( x, nargout=['f','df','ddf'] ):
    f = None
    df = None
    ddf = None

    ## Function
    if 'f' in nargout:
        f_1 = x[0]*x[0] + x[1]      - 11.0
        f_2 = x[0]      + x[1]*x[1] -  7.0
        f   = matrix( f_1*f_1 + f_2*f_2 )

    ## Gradient
    if 'df' in nargout:
        if 'f' in nargout:
            df  = matrix( [ \
                2.0*x[1] + 4.0*x[0]*f_1 + 2.0*x[1]*x[1] - 14.0,     \
                2.0*x[1] + 4.0*x[1]*f_2 + 2.0*x[0]*x[0] - 22.0,     \
                    ] )
        else:
            f_1 = x[0]*x[0] + x[1]      - 11.0
            f_2 = x[0]      + x[1]*x[1] -  7.0
            df  = matrix( [ \
                2.0*x[1] + 4.0*x[0]*f_1 + 2.0*x[1]*x[1] - 14.0,     \
                2.0*x[1] + 4.0*x[1]*f_2 + 2.0*x[0]*x[0] - 22.0,     \
                    ] )

    ## Hessian
    if 'ddf' in nargout:
        ddf = matrix( [ \
            [ 12.0*x[0]*x[0] + 4.0*x[1] - 42.0, 4.0*( x[0] + x[1] ) ],  \
            [ 4.0*( x[0] + x[1] ), 12.0*x[1]*x[1] + 4.0*x[0] - 26.0 ],  \
                ] )

    ## Return statement
    return f, df, ddf


## Equality constraints
def equality_constraints( x, nargout=['c','dc','ddc'] ):
    c = None
    dc = None
    ddc = None

    ## Function
    if 'c' in nargout:
        c_ = x[0] + 2.0
        c  = matrix( c_*c_ - x[1] )

    ## Gradient
    if 'dc' in nargout:
        dc = matrix( [  2.0*x[0] + 4.0, -1.0 ] )

    ## Hessian
    if 'ddc' in nargout:
        ddc_1 = matrix( [   \
            [ 2.0, 0.0 ],   \
            [ 0.0, 0.0 ],   \
                ] )
        ddc = [ddc_1]

    ## Return statement
    return c, dc, ddc


## Inequality constraints
def inequality_constraints( x, nargout=['c','dc','ddc'] ):
    c = None
    dc = None
    ddc = None

    ## Function
    if 'c' in nargout:
        c  = matrix( [ x[0]+5.0, x[1]+5.0 ] )

    ## Gradient
    if 'dc' in nargout:
        dc = matrix( [
            [ 1.0, 0.0 ], \
            [ 0.0, 1.0 ], \
                ], )

    ## Hessian
    if 'ddc' in nargout:
        ddc_1 = matrix( [   \
            [ 0.0, 0.0 ],   \
            [ 0.0, 0.0 ],   \
                ] )
        ddc_2 = +ddc_1
        ddc = [ddc_1,ddc_2]

    ## Return statement
    return c, dc, ddc
