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
            raise ValueError(f"Boundary Conditions X and Y  must be of dimensions (1 X {M}) and (1 x {O}) respectively")
        
        # Geometric complexity 
        self.shape = (N, M, O)

        # Temperature Boundary Conditions
        self.xboundry = xboundary
        self.yboundry = yboundary

        # Gets get in solving method
        self.solution = None

        return
    


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
        #corrective_vector[0] = rx * 
        #corrective_vector[-1] = rx


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







