## Import visualisation library
import matplotlib.pyplot as plt

## Import example driver
from visualisation.qp_plot import driver

## Main function
def main():
    fig, ax = driver()
    plt.show()

## Execution
if __name__ == '__main__':
    main()
