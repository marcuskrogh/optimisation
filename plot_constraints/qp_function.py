## Imports
from cvxopt         import matrix

## Objective function
def objective( ):
    ## System matrix and vector
    H = matrix( [
        [ 6.0, 2.0, ],
        [ 2.0, 5.0, ],
        ] )
    g = matrix( [ -8.0, -3.0 ] )

    return H, g

## Constraint function
def constraints( ):
    ## System matrix and vector
    A = matrix( [
        [ 1.0, 0.0 ],
        [ 0.0, 1.0 ],
        ] )
    b = matrix( [ 3.0, 0.0 ] )

    return A, b
