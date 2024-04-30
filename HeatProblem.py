import numpy as np

class HeatProblem:

    """
    HeatProblem Constructor

    :param tmax (float): Max time index
    :param xmax (float): Max x index
    :param ymax (float): Max y index

    :param x0_boundary (np.array): Boundary value at each (x0, y) val
    :param y0_boundary (np.array): Boundary value at each (x, y0) val
    :param xmax_boundary (np.array): Boundary value at each (xmax, y) val
    :param ymax_boundary (np.array): Boundary value at each (x, ymax) val

    :param dt (float): Timestep, computational accuracy
    :param dx (float): X step, computational accuracy
    :param dy (float): Y step, computational accuracy


    :return (HeatProblem): HeatProblem object
    """
    def __init__(self, tmax, xmax, ymax,
                        x0_boundary, y0_boundary,
                        xmax_boundary, ymax_boundary,
                        dt=1e-2, dx=1e-2, dy=1e-2):
        
        self.rx = (dt) / (dx ** 2)
        self.ry = (dt) / (dy ** 2)

        #Unconditional Stability for Crank-Nicolson
        assert(self.rx > 0)
        assert(self.ry > 0)

        # Type check for boundary conditions
        if (type(x0_boundary) != np.array or type(y0_boundary) != np.array
            or type(xmax_boundary) != np.array or type(ymax_boundary) != np.array):

            # Attempt type coercion
            try:
                x0_boundary = np.array(x0_boundary)
                y0_boundary = np.array(y0_boundary)
                xmax_boundary = np.array(xmax_boundary)
                ymax_boundary = np.array(ymax_boundary)
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


        if (len(x0_boundary) != M or len(xmax_boundary) != M):
            raise ValueError(f"Boundary Conditions for X must be of dimensions (1 X {M})")
        
        if (len(y0_boundary) != O or len(ymax_boundary) != O):
             raise ValueError(f"Boundary Conditions for Y must be of dimensions (1 X {N})")
        
        # Geometric complexity 
        self.shape = (N, M, O)

        # Temperature Boundary Conditions
        self.x0_boundary = x0_boundary
        self.xmax_boundary = xmax_boundary

        self.y0_boundary = y0_boundary
        self.ymax_boundary = ymax_boundary

        # Gets get in solving method
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

        # N X M Toeplitz matrix, left side matrix
        left_matrix = self.ConstructToeplitz(N, M, sx, -rx, -rx)


        # N X M Toeplitz matrix, right side matrix
        right_matrix = self.ConstructToeplitz(N, M, sx_p, rx, rx)

        
        # Initialize N X M matrix, solution matrix
        U = np.zeros((N, M))
  
        # Initialize 1 X M vector, boundary corrective vector
        corrective_vector = np.zeros(N)

        # XXX Not positive how we define this corrective vector
        corrective_vector[0] = rx * self.x0_boundary[0]
        corrective_vector[-1] = rx * self.xmax_boundary[0]


        # Loop through all time steps, starting at t=dT
        for i, t in enumerate(self.t[1::]):
            
            # Derived equation takes the form Ax = B
            # A = Toeplitz matrix w/ s and -r
            # x = Solution for timeslice k+1, U[k+1]
            # B = Toeplitz matrix w/ s' and r * U[k] + corrective vector

            # Left side of derived equation
            A = left_matrix
            
            # Right side of derived equation
            B = np.transpose(np.dot(right_matrix, U[i - 1]) + corrective_vector)

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







