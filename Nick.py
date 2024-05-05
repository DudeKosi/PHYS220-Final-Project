import matplotlib.pyplot as plt
import numpy as np
from HeatProblem import HeatProblemLinear, HeatProblemRadial
from matplotlib.animation import FuncAnimation

def Center_Boundary(t):
    return 30+20*t
def Edge_Boundary(r, t):
    return 30


def main():

    # 3 second solution
    tmax = 3
    
    # 1 meter max distance from center, radius
    rmax = 1

    dt = 1e-2
    dr = 1e-2

    N, M, O = int(tmax / dt), 2 * int(rmax / dr), 2 * int(rmax / dr)

    initial_conditions = 30 * np.ones((M, O))

    # Construct HeatProblem object using parameters
    sample = HeatProblemRadial(tmax, rmax, Center_Boundary, Edge_Boundary, initial_conditions=initial_conditions, dt=dt, dr=dr)

    # Solve using Crank-Nicolson Finite differentiation
    sample.CrankNicolson()   
    
    fig = plt.figure()
    ax = fig.add_subplot()

    U = sample.solution
    x, y, t = sample.x, sample.y, sample.t

    frame_interval = 100
    slice = 0
    frame_plot = ax.pcolormesh(sample.x, sample.y, U[slice, :, :], cmap='viridis', shading='auto', vmin=0, vmax=300)

    ax.set_title("2 Dimensional Heat Equation")

    ax.set_xlabel("X, in Meters")
    ax.set_ylabel("Y, in Meters")
    ax.set_xlim((-sample.rmax, sample.rmax))
    fig.colorbar(frame_plot, ax=ax)

    time = fig.text(0.05,0.05, "Time: 0", ha="left", va="top")
    
    def FrameUpdate(frame):
        nonlocal slice
        slice = frame
        frame_plot.set_array(U[slice, :, :])
        time.set_text(f"Time: {frame/frame_interval}")
        
    
    anim = FuncAnimation(fig, FrameUpdate, frames=len(t), interval=frame_interval)

    anim.save('Figure02.mp4', writer='ffmpeg')
    
    return

if __name__ == '__main__':
    main()