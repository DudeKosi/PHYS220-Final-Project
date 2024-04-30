import matplotlib.pyplot as plt
import numpy as np
from HeatProblem import HeatProblem


def main():

    # 3 second solution
    tmax = 3
    
    # 3 meters
    xmax = 3
    ymax = 3

    # Defaults
    dt = 1e-2
    dx = 1e-2
    dy = 1e-2

    # Boundary conditions
    x0_boundary = 100 * np.ones(int(xmax/dx))
    y0_boundary = 100 * np.ones(int(ymax/dy))

    xmax_boundary = np.zeros(int(xmax/dx))
    ymax_boundary = np.zeros(int(ymax/dy))

    # Construct HeatProblem object using parameters
    sample = HeatProblem(tmax, xmax, ymax, x0_boundary, y0_boundary, xmax_boundary, ymax_boundary, dt, dx, dy)

    # Solve using Crank-Nicolson Finite differentiation
    sample.CrankNicolson()   

    
    plt.figure()

    plt.pcolormesh(sample.x, sample.t, sample.solution, cmap="viridis", shading='auto', vmin=0, vmax=100)

    plt.title("Sample Solution")
    plt.xlabel("X, in Meters")
    plt.ylabel("T, in Seconds")
    plt.grid()
    cbar = plt.colorbar()
    cbar.set_label("Temperature, in Kelvin")

    plt.savefig("Sample.png", dpi=300)


    return

if __name__ == '__main__':
    main()