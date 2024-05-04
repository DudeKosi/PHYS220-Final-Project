import numpy as np

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
        
        self.rx = (dt) / (dx ** 2)
        self.ry = (dt) / (dy ** 2)

        #Unconditional Stability for Crank-Nicolson
        assert(self.rx > 0)
        assert(self.ry > 0)

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

            for i, x in enumerate(self.x):
                initial_conditions[i, 0] = y0_boundary(x, 0, 0)
                initial_conditions[i, -1] = ymax_boundary(x, self.y[-1], 0)

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

        rx = self.rx
        sx = 2 * (1 + rx)
        sx_p = 2 * (1 - rx)

        ry = self.ry
        sy = 2 * (1 + ry)
        sy_p = 2 * (1 - ry)

        # N X M Toeplitz matrix, left side matrix
        left_matrix_x = self.ConstructToeplitz(N, M, sx, -rx, -rx)
        left_matrix_y = self.ConstructToeplitz(N, O, sy, -ry, -ry)


        # N X M Toeplitz matrix, right side matrix
        right_matrix_x = self.ConstructToeplitz(N, M, sx_p, rx, rx)
        right_matrix_y = self.ConstructToeplitz(N, O, sy_p, ry, ry)

        
        # Initialize N X M X O matrix, solution matrix 
        U = np.zeros((N, M, O))
        Ux = np.zeros((N, M))
        Uy = np.zeros((N, O))
  
        # Initialize 1 X M vector, boundary corrective vector for x
        corrective_vector_x = np.zeros(N)

        # Initialize 1 X O vector, boundary corrective vector for y
        corrective_vector_y = np.zeros(O)

        #U[0, :, :] = self.initial_conditions
        
        #Ux[0] = U[0, :, 0]
        #Uy[0] = U[0, 0, :]

        # Loop through all time steps, starting at t=dT
        for i, t in enumerate(self.t[1::], 1):

            # XXX Not positive how we define this corrective vector
            corrective_vector_x[0] = rx * self.x0_boundary(self.xlim[0], 0, t) + rx * self.x0_boundary(self.xlim[0], 0, self.t[i]+self.dt)
            corrective_vector_x[-1] = rx * self.xmax_boundary(self.xlim[1], 0, t) + rx * self.xmax_boundary(self.xlim[1], 0, self.t[i]+self.dt)

            corrective_vector_y[0] = ry * self.y0_boundary(0, self.ylim[0], t) + ry * self.y0_boundary(0, self.ylim[0], self.t[i]+self.dt)
            corrective_vector_y[-1] = ry * self.ymax_boundary(0, self.ylim[1], t) + ry * self.ymax_boundary(0, self.ylim[1], self.t[i]+self.dt)
            
            # Derived equation takes the form Ax = B
            # A = Toeplitz matrix w/ s and -r
            # x = Solution for timeslice k+1, U[k+1]
            # B = Toeplitz matrix w/ s' and r * U[k] + corrective vector

            # Left side of derived equation, x dim
            
            Ax = left_matrix_x
            
            # Right side of derived equation, x dim
            Bx = np.transpose(np.dot(right_matrix_x, Ux[i - 1]) + corrective_vector_x)

            # solve for X in Ax = B
            Ux[i] = np.linalg.solve(Ax, Bx)


            Ay = left_matrix_y
            By = np.transpose(np.dot(right_matrix_y, Uy[i - 1]) + corrective_vector_y)
            Uy[i] = np.linalg.solve(Ay, By)

            # XXX Perhaps a more Pythonic approach exists 
            # Turn a 1xM and 1xO vector into MxO matrix using additon
            for j in range(M):
               for k in range(O):
                    U[i, j, k] = Ux[i, j] + Uy[i, k]


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







