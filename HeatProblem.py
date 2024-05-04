import numpy as np
import matplotlib.pyplot as plt

class HeatProblem:

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
        
            self.initial_conditions = initial_conditions

        
        # Lets create some initial conditions for the user using X, Y boundaries and 0
        else:
            initial_conditions = np.zeros((M, N))

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
        s_gamma = 1 + 2 * r_gamma
        s_gamma_prime = 2 * (1 - r_gamma)

        # M X O Toeplitz matrix, left side matrix
        left_matrix = self.ConstructToeplitz(M, O, s_gamma, -r_gamma, -r_gamma)

        # M X O Toeplitz matrix, right side matrix
        right_matrix = self.ConstructToeplitz(M, O, s_gamma_prime, r_gamma, r_gamma) 

        
        # Initialize N X M X O matrix, solution matrix
        U = np.zeros((N, M, O))

        U[0, :, :] = self.initial_conditions
      
        plt.figure()
        plt.pcolormesh(self.x, self.y, self.initial_conditions, cmap='viridis', shading='auto', vmin=0, vmax=100)
        plt.colorbar()
        plt.savefig("InitialConditions.png", dpi=300)

        # Loop through all time steps, starting at t=dT
        for i, t in enumerate(self.t[1:], start=1):

            boundary_conditions = np.zeros((M, O))

            for j, y in enumerate(self.y):
                boundary_conditions[0, j] = (r_gamma) * (self.x0_boundary(self.xlim[0], y, t) + self.x0_boundary(self.xlim[0], y, t + self.dt))
                boundary_conditions[-1, j] = (r_gamma) * (self.xmax_boundary(self.xlim[1], y, t) + self.xmax_boundary(self.xlim[1], y, t + self.dt))
            
            for k, x in enumerate(self.x):
                boundary_conditions[k, 0] = (r_gamma) * ( self.y0_boundary(x, self.ylim[0], t) + self.y0_boundary(x, self.ylim[0], t + self.dt) )
                boundary_conditions[k, -1] = (r_gamma) * ( self.ymax_boundary(x, self.ylim[1], t) + self.ymax_boundary(x, self.ylim[1], t + self.dt) )
            
            
            # Display the boundary condition matrix
            if (i == 1):    
                plt.figure()
                plt.pcolormesh(self.x, self.y, boundary_conditions)
                plt.savefig("BoundaryConditions.png", dpi=300)
            

            # Derived equation takes the form Ax = B
            # A = Toeplitz matrix w/ s and -r
            # x = Solution for timeslice k+1, U[k+1]
            # B = Toeplitz matrix w/ s' and r * U[k] + corrective vector

            # Left side of derived equation, x dim
            
            A = left_matrix
            
            # Right side of derived equation, x dim
            B = np.transpose(np.dot(right_matrix, U[i - 1]) + boundary_conditions)

            # solve for X in Ax = B
            U[i] = np.linalg.solve(A, B)


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
    def ConstructToeplitz(self, N, M, a, b, c):

        matrix = np.zeros((N, M))
 
        i,j = np.indices(matrix.shape)

        # Main diagonal
        matrix[i==j] = a

        # Upper diagonal
        matrix[i==j-1] = b

        # Lower diagonal
        matrix[i==j+1] = c

        return matrix







