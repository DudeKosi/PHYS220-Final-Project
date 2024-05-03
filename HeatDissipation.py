import matplotlib.pyplot as plt
import numpy as np
from HeatProblem import HeatProblem
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def X0_Boundary(x,y,t):
    return 50  

def Y0_Boundary(x,y,t):
    return 50

def XMax_Boundary(x,y,t):
    return 0

def YMax_Boundary(x,y,t):
    return 0

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


    # Construct HeatProblem object using parameters
    sample = HeatProblem(tmax, xmax, ymax, X0_Boundary, Y0_Boundary, XMax_Boundary, YMax_Boundary, dt, dx, dy)

    # Solve using Crank-Nicolson Finite differentiation
    sample.CrankNicolson()   
    
    fig = plt.figure()
    ax = fig.add_subplot()

    U = sample.solution
    x, y, t = sample.x, sample.y, sample.t

    frame_interval = 100
    slice = 0
    frame_plot = ax.pcolormesh(sample.x, sample.y, U[slice, :, :], cmap='viridis', shading='auto', vmin=0, vmax=100)

    ax.set_title("2 Dimensional Heat Equation")

    ax.set_xlabel("X, in Meters")
    ax.set_ylabel("Y, in Meters")
    fig.colorbar(frame_plot, ax=ax)

    time = fig.text(0.05,0.05, "Time: 0", ha="left", va="top")
    
    def FrameUpdate(frame):
        nonlocal slice
        slice = frame
        frame_plot.set_array(U[slice, :, :])
        time.set_text(f"Time: {frame/frame_interval}")
        
    
    anim = FuncAnimation(fig, FrameUpdate, frames=len(t), interval=frame_interval)

    anim.save('Figure01.mp4', writer='ffmpeg')
    
    return


if __name__ == '__main__':
    main()