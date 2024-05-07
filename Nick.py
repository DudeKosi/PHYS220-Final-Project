import matplotlib.pyplot as plt
import numpy as np
from HeatProblem import HeatProblem2DLinear, HeatProblem2DRadial, HeatProblem1DLinear
from matplotlib.animation import FuncAnimation

# Environment
ambient_temperature = 21 # 21 Celcius


#Intel Alder Lake Specifications
CPU_dimensions = 45 #Meters

CPU_min_boost = 2.5 # 2.5 GHz clock rate
CPU_max_boost = 5 # 5 GHz boost clock rate

CPU_boost_amplitude = CPU_max_boost - CPU_min_boost



CPU_frequency = CPU_min_boost
CPU_temp = ambient_temperature
CPU_boost_status = "NO"
CPU_boost_time = 0
CPU_min_temp = ambient_temperature
CPU_max_temp = ambient_temperature

CPU_enable_temp = 40
CPU_disable_temp = 110

CPU_frequency_history = []
CPU_temperature_history = []
IHS_temperature_history = []

def BoostClockProblem():

    # Experiment Runtime
    tmax = 100
    rmax = CPU_dimensions / 2

    dt = 1e-1
    dr = 1e-1
    
    CPU = HeatProblem2DRadial(tmax, rmax, CPU_Temperature, Ambient_Temperature, dt=dt, dr=dr)
    x, y, t = CPU.x, CPU.y, CPU.t
    M, N, O = CPU.shape

    initial_conditions = Ambient_Temperature(rmax, 0, 0) * np.ones((N, O))
    initial_conditions[np.where(CPU.center)] = CPU_Temperature(0, 0)

    CPU.initial_conditions = initial_conditions

    CPU.CrankNicolson()

    print("Finished Solving")

    fig = plt.figure()
    ax = fig.add_subplot()

    U = CPU.solution

    animation_runtime = 30
    frame_rate = len(t)/10 / animation_runtime
    slice = 0
    frame_plot = ax.pcolormesh(CPU.x, CPU.y, U[slice, :, :], cmap='viridis', shading='auto', vmin=0, vmax = 120)

    ax.set_title("CPU IHS Heat Dispersion")

    ax.set_xlabel("X, in Millimeters")
    ax.set_ylabel("Y, in Millimeters")
    cbar = fig.colorbar(frame_plot, ax=ax)
    cbar.set_label("Temperature, in C")

    time = fig.text(0.05,0.05, "Time: 0", ha="left", va="top")
    
    def FrameUpdate(frame):
        nonlocal slice
        slice = 10 * frame
        frame_plot.set_array(U[slice, :, :])
        time.set_text("Time: {:.2f}".format(t[slice]))
        
    
    anim = FuncAnimation(fig, FrameUpdate, frames=int(len(t)/10), interval=1000/frame_rate)

    anim.save('CPU Boost Cycling.gif', writer='ffmpeg')

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(t[1::], CPU_frequency_history, linewidth=2)
    ax1.set_xlabel("Time, in Seconds")
    ax1.set_ylabel("CPU Frequency, in GHz")
    ax1.grid()
    ax1.title.set_text("CPU Boost Frequency over Time")

    ax2.plot(t[1::], CPU_temperature_history, linewidth=2)
    ax2.set_xlabel("Time, in Seconds")
    ax2.set_ylabel("CPU Temperature, in C")
    ax2.grid()
    ax2.title.set_text("CPU Die Temperature over Time")

    fig.tight_layout()

    fig.savefig("CPU Boost Cycling.png", dpi=300)


    plt.figure()
    plt.plot(t[1::], IHS_temperature_history, linewidth=2)
    plt.xlabel("Time, in Seconds")
    plt.ylabel("IHS Average Temperature, in C")
    plt.grid()
    plt.title("IHS Average Temperature over Time")

    plt.savefig("IHS Average Temperature.png", dpi=300)

    return

ignore = False

def CPU_Temperature(t, i, U=None):
    
    global CPU_boost_status, CPU_boost_time, CPU_frequency, CPU_temp, CPU_min_temp, CPU_max_temp, ignore

    # Ignore bad iterations, return last reported time
    #if (i % 2 == 0):
    #    return CPU_temp
    
    if (ignore == False):
        ignore = True
        return CPU_temp
    
    ignore = False
    
    if not (isinstance(U, np.ndarray)):
        return ambient_temperature
    
    # Good iteration of time
    else:

        if (CPU_boost_status == "NO"):
            
            # Enable boost
            if (CPU_temp < CPU_enable_temp):
                CPU_boost_status = "INCREASING"
                CPU_boost_time = t
            else:
                CPU_temp = CPU_min_temp - (t - CPU_boost_time)


        elif (CPU_boost_status == "INCREASING"):

            CPU_frequency = min(CPU_min_boost + (1/10) * CPU_boost_amplitude * (t - CPU_boost_time), CPU_max_boost)
            
            # Fully boosted, change status
            if (CPU_frequency >= CPU_max_boost):
                CPU_frequency = CPU_max_boost
                CPU_boost_status = "YES"
                CPU_boost_time = t

            CPU_temp = CPU_min_temp + 30 * (CPU_frequency - CPU_min_boost)
           
        
        elif (CPU_boost_status == "YES"):
                        
            if (CPU_temp >= CPU_disable_temp):
                CPU_boost_status = "DECREASING"
                CPU_max_temp = CPU_min_temp + 30 * (CPU_frequency - CPU_min_boost) + (t - CPU_boost_time)
                CPU_boost_time = t
                CPU_temp = CPU_max_temp

            else:
                CPU_temp = CPU_min_temp + 30 * (CPU_frequency - CPU_min_boost) + (t - CPU_boost_time)


        elif (CPU_boost_status == "DECREASING"):

            prev_f = CPU_frequency

            CPU_frequency = max(CPU_max_boost - (1/10) * CPU_boost_amplitude * (t - CPU_boost_time), CPU_min_boost)

            # Fully off boost, change status
            if (CPU_frequency <= CPU_min_boost):
                CPU_frequency = CPU_min_boost
                CPU_boost_status = "NO"
                CPU_min_temp = CPU_temp
                CPU_boost_time = t

            else:
                CPU_temp = CPU_max_temp - 30 * (CPU_max_boost - prev_f)
        
        
        """
        print(f"CPU Boost Status: {CPU_boost_status}")
        print(f"CPU Frequency: {CPU_frequency} GHz")
        print(f"CPU Temp: {CPU_temp} C")
        print()
        """

        CPU_frequency_history.append(CPU_frequency)
        CPU_temperature_history.append(CPU_temp)
        IHS_temperature_history.append(np.average(U.flatten()))


        return CPU_temp


def Ambient_Temperature(r, t, i, U=None):
    return ambient_temperature


def PCBTraceHeat():
    
    # Experiment Runtime
    tmax = 5

    # Tracelength in mmeters
    xmax = 3

    dx = 1e-2
    dt = 1e-2
    

    Trace = HeatProblem1DLinear(tmax, xmax, TraceLeftEdge, TraceRightEdge, dt=dt, dx=dx)
    x, t = Trace.x, Trace.t

    for i, x in enumerate(Trace.x[1: -1], start=1):
        Trace.initial_conditions[i] = TraceRightEdge(x, 0)


    Trace.CrankNicolson()

    print("Finished Solving")
    

    plt.figure()
    plt.pcolormesh(Trace.x, Trace.t, Trace.solution, cmap="viridis", shading='auto', vmin = 0, vmax = 100)
    plt.title("PCB Trace Heat Diffusion")
    plt.xlabel("X, in Millimeters")
    plt.ylabel("Time, in Seconds")
    cbar = plt.colorbar()
    cbar.set_label("Temperature, in C")
    plt.grid()
    plt.savefig("PCB Trace Heating.png", dpi = 300)

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot()

    U = Trace.solution
    
    trace_width = 0.5
    y = np.arange(0, trace_width, dx)


    animation_runtime = 15
    frame_rate = len(t) / animation_runtime
    slice = 0
    frame_plot = ax.pcolormesh(Trace.x, y, np.tile(U[0, :], (len(y), 1)) , cmap='viridis', shading='auto', vmin=0, vmax = 100)
    
    ax.set_title("PCB Trace Heat Diffusion")

    ax.set_xlabel("X, in Millimeters")
    ax.set_ylabel("Y, in Millimeters")
    cbar = fig.colorbar(frame_plot, ax=ax)
    cbar.set_label("Temperature, in C")

    time = fig.text(0.01,0.075, "Time: 0", ha="left", va="top")
    
    def FrameUpdate(frame):
        nonlocal slice
        slice = frame
        frame_plot.set_array(np.tile(U[slice, :], (len(y), 1)))
        time.set_text("Time: {:.2f}".format(t[slice]))
        
    
    anim = FuncAnimation(fig, FrameUpdate, frames=int(len(t)), interval=1000/frame_rate)

    anim.save('PCB Trace Heating.gif', writer='ffmpeg')
    
    return

def TraceLeftEdge(x, t):
    return 80

def TraceRightEdge(x, t):
    return 21

if __name__ == '__main__':
    #main()
    BoostClockProblem()
    #PCBTraceHeat()
