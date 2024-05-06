import matplotlib.pyplot as plt
import numpy as np
from HeatProblem import HeatProblemLinear, HeatProblemRadial
from matplotlib.animation import FuncAnimation

# Environment
ambient_temperature = 21 # 21 Celcius


#Intel Alder Lake Specifications
CPU_dimensions = 0.045 #Meters

CPU_min_boost = 2.5 # 2.5 GHz clock rate
CPU_max_boost = 5 # 5 GHz boost clock rate

CPU_boost_amplitude = CPU_max_boost - CPU_min_boost



CPU_frequency = CPU_min_boost
CPU_temp = ambient_temperature
CPU_boost_status = "NO"
CPU_boost_time = 0

CPU_enable_temp = 40
CPU_disable_temp = 70


def BoostClockProblem():

    # Experiment Runtime
    tmax = 30
    rmax = CPU_dimensions / 2

    dt = 1e-3
    dr = 1e-2
    
    CPU = HeatProblemRadial(tmax, rmax, CPU_Temperature, Ambient_Temperature, dt=dt, dr=dr)
    CPU.CrankNicolson()

    fig = plt.figure()
    ax = fig.add_subplot()

    U = CPU.solution
    x, y, t = CPU.x, CPU.y, CPU.t

    animation_runtime = 30
    frame_rate = len(t) / animation_runtime
    slice = 0
    frame_plot = ax.pcolormesh(CPU.x*1e3, CPU.y*1e3, U[slice, :, :], cmap='viridis', shading='auto', vmin=-100, vmax = 120)

    ax.set_title("2 Dimensional Heat Equation")

    ax.set_xlabel("X, in Millimeters")
    ax.set_ylabel("Y, in Millimeters")
    fig.colorbar(frame_plot, ax=ax)

    time = fig.text(0.05,0.05, "Time: 0", ha="left", va="top")
    
    def FrameUpdate(frame):
        nonlocal slice
        slice = frame
        frame_plot.set_array(U[slice, :, :])
        time.set_text(f"Time: {t[slice]}")
        
    
    anim = FuncAnimation(fig, FrameUpdate, frames=len(t), interval=1000/frame_rate)

    anim.save('CPU Boost Cycling.mp4', writer='ffmpeg')
    
    return

ignore = False


def CPU_Temperature(t, i, U=None):
    

    global CPU_boost_status, CPU_boost_time, CPU_frequency, CPU_temp, ignore

    # Ignore bad iterations, return last reported time
    if (i % 2 == 0):
        return CPU_temp
    
    if (ignore == False):
        ignore = True
        return CPU_temp
    
    ignore = False
    
    if not (isinstance(U, np.ndarray)):
        return ambient_temperature
    
    # Good iteration of time
    else:
        # Compute the current IHS average temperature
        avg_temp = np.average(U.flatten())

        
        if (200 <= i < 300):
            print(f"Iter: {i} Avg Temp: {avg_temp} CPU Temp: {CPU_temp}")
            print(CPU_boost_status)
            print(f"CPU Frequency: {CPU_frequency} GHz")
            print()
        

        if (CPU_boost_status == "NO"):

            # Enable boost
            if (CPU_temp < CPU_enable_temp):
                CPU_boost_status = "INCREASING"
                CPU_boost_time = t
        

        elif (CPU_boost_status == "INCREASING"):

            CPU_frequency = CPU_min_boost + 10 * CPU_boost_amplitude * (t - CPU_boost_time)
            
            # Fully boosted, change status
            if (CPU_frequency >= CPU_max_boost):
                CPU_frequency = CPU_max_boost
                CPU_boost_status = "YES"
           
        

        elif (CPU_boost_status == "YES"):

            if (CPU_temp >= CPU_disable_temp):
                CPU_boost_status = "DECREASING"
                CPU_boost_time = t
        

        elif (CPU_boost_status == "DECREASING"):

            CPU_frequency = CPU_max_boost - 10 * CPU_boost_amplitude * (t - CPU_boost_time)

            # Fully off boost, change status
            if (CPU_frequency <= CPU_min_boost):
                CPU_frequency = CPU_min_boost
                CPU_boost_status = "NO"
        

        #print(f"CPU Frequency: {CPU_frequency} GHz")
        CPU_temp = ambient_temperature + 50 *(CPU_frequency - CPU_min_boost)
        return CPU_temp


def Ambient_Temperature(r, t, i, U=None):
    return ambient_temperature


if __name__ == '__main__':
    #main()
    #BoostClock()
    BoostClockProblem()
