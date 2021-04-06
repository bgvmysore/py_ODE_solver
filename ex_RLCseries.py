"""
 ---------wwww------(\(\(\(\(---------||-----------
 |           R           L     ---->    C         |
 |                              i(t)              |
(V(t))                                            |
 |                                                |
 |                                                |
 --------------------------------------------------
_|_
 - GND
 `
"""
import matplotlib.pyplot as plt
from bgvODEsolvr.bgvODEsolvr import np, linDiffEqnSys, RungeKutta4Solver

# setting circuit params
R = 10
L = 1e-3
C = 2e-6

# definig A, B Matrix
A = np.array([
             [ 0       ,  1   ],
             [ -1/(C*L), -R/L ]
            ])

B = np.array([
             [0  ],
             [1/L]
             ])

# initial conditions
t0 = 0
X0 = np.array([
              [0],
              [0]
              ])

# time info
n = 255
t_end = 2e-3

# definig input
def V(t):
    return np.heaviside(t - 0.5e-3, 0.99)

def U(t):
    dt = ( t0 - t_end ) * ( 1/n )
    return ( V(t) - V(t - dt) )/dt

# Solving
RLC_series = linDiffEqnSys(A, B, U)
Sol_RLC_Series = RungeKutta4Solver(RLC_series)
Sol_RLC_Series.initConditions(t0,X0)
Sol_RLC_Series.genTimePoints_n(n,t_end)
x,y = Sol_RLC_Series.solverSolve()

# Plotting
fig, (ax1,ax2) = plt.subplots(2,1,sharex='col')
ax1.plot(x,V(x),'orangered',label = '$V(t)$') # subplot 1
ax1.set_ylabel('Voltage (V)')
ax1.grid(1)
ax1.legend()
# subplot 2
ax2.set_ylabel('Current (A)')
ax2.set_xticks([0,0.5e-3,1e-3,1.5e-3,2e-3])
ax2.set_xticklabels(['0','0.5m','1m','1.5m','2m'])
ax2.set_xlim([0e-3,2e-3])
ax2.set_xlabel('time (s)')
ax2.plot(x,y[0],label=str("RungeKuttaSolver"))
ax2.grid(1)
ax2.legend()
plt.show() # show plot
