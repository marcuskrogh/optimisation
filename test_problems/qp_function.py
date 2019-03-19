## Imports
from cvxopt         import matrix

## Objective function
def objective( ):
    ## System matrix and vector
    H = matrix( [
        [  2.0,  0.0, ],
        [  0.0,  2.0, ],
        ] )

    g = matrix( [ -2.0, -5.0 ] )

    return H, g

## Constraint function
def constraints( ):
    ## System matrix and vector
    C = matrix( [
        [  1.0, -1.0, -1.0,  1.0,  0.0 ],
        [ -2.0, -2.0,  2.0,  0.0,  1.0 ],
        ] ).T
    d = matrix( [ -2.0, -6.0, -2.0,  0.0,  0.0 ] )

    return C, d
