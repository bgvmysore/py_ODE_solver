import numpy as np
from tqdm import tqdm

class linDiffEqnSys:
    '''
    Differential Equation Object
    
    Xdot = A . X + B . U
    
    here, X has n state variables
          U is a function strictly dependent on time which returns vector of lenght 'm'

    1. U is callable python Object
    2. A is n x n Matrix
    3. B is n x m Matrix
    4. X is State Variables
    '''

    def __init__(self, A, B, U):
        self.A = np.asarray(A)
        self.B = np.asarray(B)
        self.U = U # callabe

    def F(self, t, Xn0):
        ''' 
        Differential Function
        
        dX/dt = A . X(t) + B . U(t)
        
        returns dX/dt at time 't'
        '''
        A = np.asarray(self.A)
        B = np.asarray(self.B)
        Xn0 = np.asarray(Xn0)
        dxdt = A.dot(Xn0) + B.dot(self.U(t)) 
        return dxdt

class ODESol:
    '''
    This solves linear system of diff eqns

    Xdot = A . X + B . U 
    
    1. for 'n' states i.e. length of vector X is 'n', A is n x n matrix.
    2. for 1 input of 'U' matrix B is n X 1.

    Solution:
    if X0 or initial conditions are given,
    
    X1 = X0 + ODEsolver_method(t1,X0)
    X2 = X1 + ODEsolver_method(t2,X1)
    ...
    Xn+1 = Xn + ODEsolver_method(tn+1,Xn)
    '''
    def __init__(self, Diff_Eqn_Sys_F):
        self.diffsys = Diff_Eqn_Sys_F
        self.F = Diff_Eqn_Sys_F.F # callable object
        
    def initConditions(self, t0, X0):
        '''
        X0 is intial conditions of state variables at t0 
        
        It is same size as X and dX/dt
        '''
        self.t0 = t0
        self.X0 = np.asarray(X0)

        self.Xn0 = self.X0 # used to iterate in for loop
        self.Xn0 = self.Xn0.reshape((self.Xn0.size, 1))

    def genTimePoints_dt(self, dt, t_end_point):
        '''
        Given t0, t_end, dt we can find n -> time points
        '''
        self.t_end = t_end_point
        self.dt = dt
        self.n = abs(self.t0 - self.t_end) / dt

    def genTimePoints_n(self, n, t_end_point):
        '''
        Given t0, t_end, n we can find dt -> time points
        '''
        self.t_end = t_end_point
        self.n = n
        self.dt = abs(self.t0 - self.t_end) / n

    def setup_sol(self):
        self.tout = np.zeros( (self.n + 1,) )
        self.x_states = np.zeros( (self.Xn0.size, self.n + 1) )
        
    def solverSolve(self):
        '''
        Contains for loop w.r.t time to solve the differential eqn.
        It takes time points as input and returns 'X(t)' solution matrix
        and time points tout.
            
        1. time(time points), F(function) -> Inputs
        2. Xsol -> solution w.r.t time(time points)

        Returns tout, Xsol
        tout -> time points of length n+1
        Xsol -> Matrix State Variable Soln of dim ( 'no. od states' x (n + 1) )
        '''
        self.setup_sol()
        for k in tqdm( range(self.n + 1) ):
            tin = self.t0 + self.dt*k
            self.tout[k] = tin 
            tmp = self.solverMethod(tin, self.Xn0)
            self.Xn0 = tmp
            for ii in range(self.Xn0.size):
                self.x_states[ii,k] = tmp[ii,0]
        return self.tout,self.x_states

    def solverMethod(self, tin, X0):
        '''
        This is usually of form 
            X0 + dt * someFunction( F(t,X0) )

        This is the actual function that solves the diff eqn; runs for every
        time point
        '''
        raise NotImplementedError

class EulerForwardSolver(ODESol):
    def solverMethod(self, tin, X0):
        return self.Xn0 + self.dt * self.F(tin, X0)

class EulerModifiedSolver(ODESol):
    def solverMethod(self, tin, X0):
        tmp = X0 + self.dt * self.F(tin, X0)
        tmp = X0 + self.dt/2 *( self.F(tin + self.dt, tmp) + self.F(tin, X0) )
        return tmp

class RungeKutta4Solver(ODESol):
    def solverMethod(self, tin, X0):
        dt2 = self.dt/2
        k1 = self.dt * self.F( tin, X0)
        k2 = self.dt * self.F( tin + dt2, X0 + (k1*0.5) )
        k3 = self.dt * self.F( tin + dt2, X0 + (k2*0.2) )
        k4 = self.dt * self.F( tin + self.dt, X0 + k3)
        tmp = X0 + (1/6)*(k1+2*(k2+k3)+k4)
        return tmp