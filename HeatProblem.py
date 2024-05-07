import numpy as np
import matplotlib.pyplot as plt


class HeatProblem1DLinear:

    """
    HeatProblem Constructor

    :param tmax (float): Max time index
    :param xmax (float): Max x index
    :param ymax (float): Max y index

    :param x0_boundary (function ptr): Boundary value at each (x0, y) val
    :param xmax_boundary (function ptr): Boundary value at each (xmax, y) val

    :param initial_conditions (np.array): Initial temperature at t=0

    :param dt (float): Timestep, computational accuracy
    :param dx (float): X step, computational accuracy
    :param dy (float): Y step, computational accuracy
    
    :return (HeatProblem): HeatProblem object
    """
    def __init__(self, tmax, xmax,
                        x0_boundary, xmax_boundary,
                        initial_conditions=None,
                        dt=1e-2, dx=1e-2):
        

        # Type check for boundary conditions, using DeMorgans
        if not(callable(x0_boundary) or callable(xmax_boundary)):

            raise ValueError("Boundaries must be a function of type f(x, t)")

        # 3 Dimensional Solution Framework

        # Time
        self.t_lim = (0, tmax)
        self.t = np.arange(0, tmax, dt)
        N = len(self.t)


        # X Coordinate
        self.xlim = (0, xmax)
        self.x = np.arange(0, xmax, dx)
        M = len(self.x)

        
        # Geometric complexity 
        self.shape = (N, M)
        
        # Temperature Boundary Conditions
        self.x0_boundary = x0_boundary
        self.xmax_boundary = xmax_boundary

        self.dt = dt
        self.dx = dx

        # Check if initial conditions
        if (initial_conditions is not None):

            # Typecheck for initial conditions
            if (not isinstance(initial_conditions, np.ndarray) or initial_conditions.shape != (M)):
                raise ValueError(f"Initial condition must be an ({M}) np.array")
        
        else:
            initial_conditions = np.zeros((M))

        # Apply Boundary Conditions to Initial Conditions for greater continuity
                
        initial_conditions[0] = x0_boundary(0, 0)
        initial_conditions[-1] = xmax_boundary(self.x[-1], 0)


        self.initial_conditions = initial_conditions


        # Gets set in solving method
        self.solution = None

        return
    

    """
    Iterates through time solving the Crank-Nicolson linear equation for the solution 
    vector/matrix at each timeslice.

    :params NONE
    :returns NONE
    """
    def CrankNicolson(self):
        
        N, M = self.shape

        r_gamma = ((self.dt) / (self.dx **2))
        s_gamma = 2 * (1 + r_gamma)
        s_gamma_prime = 2 * (1 - r_gamma)

        # M X O Toeplitz matrix, left side matrix
        left_matrix = ConstructToeplitz(M, M, s_gamma, -r_gamma, -r_gamma)

        # M X O Toeplitz matrix, right side matrix
        right_matrix = ConstructToeplitz(M, M, s_gamma_prime, r_gamma, r_gamma) 

        
        # Initialize N X M X O matrix, solution matrix
        U = np.zeros((N, M))

        U[0, :] = self.initial_conditions

        """
        plt.figure()
        plt.pcolormesh(self.x, self.y, self.initial_conditions, cmap='viridis', shading='auto')
        plt.colorbar()
        plt.savefig("InitialConditions.png", dpi=300)
        """
        

        # Loop through all time steps, starting at t=dT
        for i, t in enumerate(self.t[1::], start=1):

            boundary_conditions = np.zeros(M)
            boundary_conditions[0] = (self.x0_boundary(self.xlim[0], t) + self.x0_boundary(self.xlim[0], t + self.dt))
            boundary_conditions[-1] = (self.xmax_boundary(self.xlim[1], t) + self.xmax_boundary(self.xlim[1], t + self.dt))
                            

            # Derived equation takes the form Ax = B
            # A = Toeplitz matrix w/ s and -r
            # x = Solution for timeslice k+1, U[k+1]
            # B = Toeplitz matrix w/ s' and r * U[k] + corrective vector

            # Left side of derived equation, x dim
            
            A = left_matrix
            
            # Right side of derived equation, x dim
            B = np.transpose(np.dot(right_matrix, U[i - 1]) + (r_gamma) * boundary_conditions)


            # solve for X in Ax = B
            U[i, :] = np.linalg.solve(A, B)


        self.solution = U
        return 


class HeatProblem2DLinear:

    """
    HeatProblem Constructor

    :param tmax (float): Max time index
    :param xmax (float): Max x index
    :param ymax (float): Max y index

    :param x0_boundary (function ptr): Boundary value at each (x0, y) val
    :param y0_boundary (function ptr): Boundary value at each (x, y0) val
    :param xmax_boundary (function ptr): Boundary value at each (xmax, y) val
    :param ymax_boundary (function ptr): Boundary value at each (x, ymax) val

    :param initial_conditions (np.array): Initial temperature at t=0

    :param dt (float): Timestep, computational accuracy
    :param dx (float): X step, computational accuracy
    :param dy (float): Y step, computational accuracy
    
    :return (HeatProblem): HeatProblem object
    """
    def __init__(self, tmax, xmax, ymax,
                        x0_boundary, y0_boundary,
                        xmax_boundary, ymax_boundary,
                        initial_conditions=None,
                        dt=1e-2, dx=1e-2, dy=1e-2):
        

        # Type check for boundary conditions, using DeMorgans
        if not(callable(x0_boundary) or callable(y0_boundary) or
            callable(xmax_boundary) or callable(ymax_boundary)):

            raise ValueError("Boundaries must be a function of type f(x,y,t)")

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

        
        # Geometric complexity 
        self.shape = (N, M, O)
        
        # Temperature Boundary Conditions
        self.x0_boundary = x0_boundary
        self.xmax_boundary = xmax_boundary

        self.y0_boundary = y0_boundary
        self.ymax_boundary = ymax_boundary

        self.dt = dt
        self.dx = dx
        self.dy = dy

        # Check if initial conditions
        if (initial_conditions is not None):

            # Typecheck for initial conditions
            if (not isinstance(initial_conditions, np.ndarray) or initial_conditions.shape != (M, O)):
                raise ValueError(f"Initial condition must be an ({M}, {O}) np.array")
        
        else:
            initial_conditions = np.zeros((M, O))

        # Apply Boundary Conditions to Initial Conditions for greater continuity
        for i, y in enumerate(self.y):
                
            initial_conditions[0, i] = x0_boundary(0, y, 0)
            initial_conditions[-1, i] = xmax_boundary(self.x[-1], y, 0)

        for j, x in enumerate(self.x):
            initial_conditions[j, 0] = y0_boundary(x, 0, 0)
            initial_conditions[j, -1] = ymax_boundary(x, self.y[-1], 0)

        self.initial_conditions = initial_conditions


        # Gets set in solving method
        self.solution = None

        return
    

    """
    Iterates through time solving the Crank-Nicolson linear equation for the solution 
    vector/matrix at each timeslice.

    :params NONE
    :returns NONE
    """
    def CrankNicolson(self):
        
        N, M, O = self.shape

        r_gamma = ((self.dt) / (self.dx **2)) + ((self.dt) / (self.dy ** 2))
        s_gamma = 2 * (1 + r_gamma)
        s_gamma_prime = 2 * (1 - r_gamma)

        # M X O Toeplitz matrix, left side matrix
        left_matrix = ConstructToeplitz(M, O, s_gamma, -r_gamma, -r_gamma)

        # M X O Toeplitz matrix, right side matrix
        right_matrix = ConstructToeplitz(M, O, s_gamma_prime, r_gamma, r_gamma) 

        
        # Initialize N X M X O matrix, solution matrix
        U = np.zeros((N, M, O))

        U[0, :, :] = self.initial_conditions
      
        plt.figure()
        plt.pcolormesh(self.x, self.y, self.initial_conditions, cmap='viridis', shading='auto')
        plt.colorbar()
        plt.savefig("InitialConditions.png", dpi=300)

        

        # Loop through all time steps, starting at t=dT
        for i, t in enumerate(self.t[1:], start=1):

            boundary_conditions = np.zeros((M, O))
            for j, y in enumerate(self.y):
                boundary_conditions[0, j] = (self.x0_boundary(self.xlim[0], y, t) + self.x0_boundary(self.xlim[0], y, t + self.dt))
                boundary_conditions[-1, j] = (self.xmax_boundary(self.xlim[1], y, t) + self.xmax_boundary(self.xlim[1], y, t + self.dt))
            
            for k, x in enumerate(self.x):
                boundary_conditions[k, 0] = (self.y0_boundary(x, self.ylim[0], t) + self.y0_boundary(x, self.ylim[0], t + self.dt) )
                boundary_conditions[k, -1] = (self.ymax_boundary(x, self.ylim[1], t) + self.ymax_boundary(x, self.ylim[1], t + self.dt) )
            
                      

            # Derived equation takes the form Ax = B
            # A = Toeplitz matrix w/ s and -r
            # x = Solution for timeslice k+1, U[k+1]
            # B = Toeplitz matrix w/ s' and r * U[k] + corrective vector

            # Left side of derived equation, x dim
            
            A = left_matrix
            
            # Right side of derived equation, x dim
            B = np.transpose(np.dot(right_matrix, U[i - 1]) + (r_gamma) * boundary_conditions)

            # solve for X in Ax = B
            U[i, :, :] = np.linalg.solve(A, B)


        self.solution = U
        return 


    



class HeatProblem2DRadial:

    """
    HeatProblem Constructor

    :param tmax (float): Max time index
    :param rmax (float): Max r index

    :param center_boundary (function ptr): Boundary value at r=0 val
    :param edge_boundary (function ptr): Boundary value at r=rmax val

    :param initial_conditions (np.array): Initial temperature at each point at t=0

    :param dt (float): Timestep, computational accuracy
    :param dx (float): X step, computational accuracy
    :param dy (float): Y step, computational accuracy
    
    :return (HeatProblem): HeatProblem object
    """
    def __init__(self, tmax, rmax,
                    center_boundary, edge_boundary,
                    initial_conditions=None,
                    dt=1e-2, dr=1e-2):
        
        dt = dt / 2
        # Type check for boundary conditions, using DeMorgans
        if not(callable(center_boundary) or callable(edge_boundary)):
            raise ValueError("Boundaries must be a function of type f(t)")

        # 3 Dimensional Solution Framework

        # Time
        self.t_lim = (0, tmax)
        self.t = np.arange(0, tmax, dt)
        N = len(self.t)


        # R
        self.rmax = rmax

        # X Coordinate
        self.xlim = (-rmax, rmax)
        self.x = np.arange(-rmax, rmax, dr)
        M = len(self.x)


        # Y Coordinate
        self.ylim=(-rmax, rmax)
        self.y = np.arange(-rmax, rmax, dr)
        O = len(self.y)

        
        # Geometric complexity 
        self.shape = (N, M, O)
        
        # Temperature Boundary Conditions
        self.center_boundary = center_boundary
        self.edge_boundary = edge_boundary

        self.dt = dt
        self.dr = dr


        # Check if initial conditions
        if (initial_conditions is not None):

            # Typecheck for initial conditions
            if (not isinstance(initial_conditions, np.ndarray) or initial_conditions.shape != (M, O)):
                raise ValueError(f"Initial condition must be an ({M}, {O}) np.array")
        
        # Lets create some initial conditions for the user using X, Y boundaries and 0
        else:
            initial_conditions = np.zeros((M, O))

        
        # M x N (square) matrix where 1 if element in that index is in the center
        self.center = np.zeros((M, N))

        # Even, set center as 2x2
        if (M % 2 == 0):
            self.center[M // 2 - 1 : M // 2 + 1, M // 2 - 1 : M // 2 + 1] = 1
        # Odd, set center as singular element
        else:
            self.center[M // 2,  M // 2] = 1


        # Apply center condition
        initial_conditions[np.where(self.center)] = center_boundary(0, 0)

        # Apply edge condition to outside edge
        initial_conditions[0, :] = edge_boundary(rmax, 0, 0)
        initial_conditions[-1, :] = edge_boundary(rmax, 0, 0)
        initial_conditions[:, 0] = edge_boundary(rmax, 0, 0)
        initial_conditions[:, -1] = edge_boundary(rmax, 0, 0)

        self.initial_conditions = initial_conditions

        

        # Gets set in solving method
        self.solution = None

        return
    

    """
    Iterates through time solving the Crank-Nicolson linear equation for the solution 
    vector/matrix at each timeslice.

    :params NONE
    :returns NONE
    """
    def CrankNicolson(self):
        
        N, M, O = self.shape

        r_gamma = ((self.dt) / (self.dr **2)) + ((self.dt) / (self.dr ** 2))
        s_gamma = 2 * (1 + r_gamma)
        s_gamma_prime = 2 * (1 - r_gamma)

        # M X O Toeplitz matrix, left side matrix
        left_matrix = ConstructToeplitz(M, O, s_gamma, -r_gamma, -r_gamma)

        # M X O Toeplitz matrix, right side matrix
        right_matrix = ConstructToeplitz(M, O, s_gamma_prime, r_gamma, r_gamma) 

        
        # Initialize N X M X O matrix, solution matrix
        U = np.zeros((N, M, O))

        U[0, :, :] = self.initial_conditions
      
        plt.figure()
        plt.pcolormesh(self.x, self.y, self.initial_conditions, cmap='viridis', shading='auto')
        plt.colorbar()
        plt.savefig("InitialConditions.png", dpi=300)

        
        # Loop through all time steps, starting at t=dT
        for i, t in enumerate(self.t[1:], start=1):

            boundary_conditions = np.zeros((M, O))

            # Apply edge condition to outside edge
            boundary_conditions[0, :] = self.edge_boundary(self.rmax, t, i) + self.edge_boundary(self.rmax, t + self.dt, i)
            boundary_conditions[-1, :] = self.edge_boundary(self.rmax, t, i) + self.edge_boundary(self.rmax, t + self.dt, i)
            boundary_conditions[:, 0] = self.edge_boundary(self.rmax, t, i) + self.edge_boundary(self.rmax, t + self.dt, i)
            boundary_conditions[:, -1] = self.edge_boundary(self.rmax, t, i) + self.edge_boundary(self.rmax, t + self.dt, i)

            # Apply center boundary condition

            boundary_conditions[np.where(self.center)] = self.center_boundary(t, i, U[i - 1, :, :]) + self.center_boundary(t + self.dt, i, U[i - 1, :, :])
                     

            # Derived equation takes the form Ax = B
            # A = Toeplitz matrix w/ s and -r
            # x = Solution for timeslice k+1, U[k+1]
            # B = Toeplitz matrix w/ s' and r * U[k] + corrective vector

            # Left side of derived equation, x dim
            
            A = left_matrix
            
            # Right side of derived equation, x dim
            B = np.transpose(np.dot(right_matrix, U[i - 1]) + (r_gamma) * boundary_conditions)

            # solve for X in Ax = B
            U[i, :, :] = np.linalg.solve(A, B)

        #self.t = self.t[0::2]
        #self.solution = U[0::2, :, :]

        self.solution = U
        return





"""
Constructs a Toeplitz-style matrix where only the 
main diagonal, k+1 diagonal and k-1 diagonal are non-zero

:params N (int): Number of Rows
:params M (int): Number of Columns

:params a (int): Value along main diagonal
:params b (int): Value along the upper diagonal, k+1
:params c (int): Value along the lower diagonal, k-1

:returns matrix (np.array): Toeplitz-style matrix with specified params
"""
def ConstructToeplitz(N, M, a, b, c):

    matrix = np.zeros((N, M))

    i,j = np.indices(matrix.shape)

    # Main diagonal
    matrix[i==j] = a

    # Upper diagonal
    matrix[i==j-1] = b

    # Lower diagonal
    matrix[i==j+1] = c

    return matrix

