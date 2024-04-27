import numpy as np

class HeatProblem:

    """
    HeatProblem Constructor

    :param tmax (float): Max time index
    :param xmax (float): Max x index
    :param ymax (float): Max y index

    :param xboundary (np.array): Boundary value at each x val
    :param yboundary (np.array): Boundary value at each y val

    :param dt (float): Timestep, computational accuracy
    :param dx (float): X step, computational accuracy
    :param dy (float): Y step, computational accuracy


    :return (HeatProblem): HeatProblem object
    """
    def __init__(self, tmax, xmax, ymax,
                        xboundary, yboundary,
                        dt=1e-2, dx=1e-2, dy=1e-2):
        
        self.rx = (dt) / (dx ** 2)
        self.ry = (dt) / (dy ** 2)

        #Unconditional Stability for Crank-Nicolson
        assert(self.rx > 0)
        assert(self.ry > 0)

        # Type check for boundary conditions
        if (type(xboundary) != np.array or type(yboundary) != np.array):

            # Attempt type coercion
            try:
                xboundary = np.array(xboundary)
                yboundary = np.array(yboundary)
            except:
                raise ValueError("Boundary  conditions must be of type np.array")
        

        # 3 Dimensional Solution Framework

        # Time
        self.t_lim = (0, tmax)
        self.t = np.arange(0, tmax, dt)
        N = len(self.t)

        # X Coordinate
        self.xlim = (0, xmax)
        self.x = np.arange(0, xmax, dx)
        M = len(self.x)

        # Y Coordinate
        self.ylim=(0, ymax)
        self.y = np.arange(0, ymax, dy)
        O = len(self.y)

        if (len(xboundary) != M or len(yboundary) != O):
            raise ValueError(f"Boundary Conditions (X, Y)  must be of dimensions ({M} x {O})")
        
        # Geometric complexity 
        self.size = (N, M, O)

        # Temperature Boundary Conditions
        self.xboundry = xboundary
        self.yboundry = yboundary

        return self
    


    def CrankNicolson(self):
        
        return










