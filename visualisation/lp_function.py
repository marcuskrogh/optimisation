## Imports
from cvxopt         import matrix

## Objective function
def objective( ):
    ## System vector
    g = matrix( [ 1.0, -2.0 ] )

    return g

## Constraint function
def constraints( ):
    ## System matrix and vector
    C = matrix( [
        [  1.0,  0.0,  1.0,  1.0, -5.0 ],
        [  0.0,  1.0, -1.0, -5.0,  1.0 ],
        ] ).T
    d = matrix( [  1.0,  1.0, -2.0, -20.0, -15.0 ] )

    return C, d
