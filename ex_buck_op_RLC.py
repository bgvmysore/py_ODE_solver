import matplotlib.pyplot as plt
from bgvODEsolvr.bgvODEsolvr import np, linDiffEqnSys, RungeKutta4Solver

"""
  --------(\(\(\(------------------wwwwww-------
  | --->     L          |   --->      R        |
  |  i1(t)              |   i2(t)              |
  |                     |                      |
(V(t))                 ---                     |
  |                    --- C                   |
  |                     |                      |
  ----------------------------------------------
  |
 ---
 GND
"""

# setting circuit params
R = 10
L = 100e-6
C = 4e-6

# definig A, B Matrix
A = np.array([
             [ 0       , 1,  0       ],
             [ -1/(L*C), 0,  1/(L*C) ],
             [ 1/(R*C) , 0, -1/(R*C) ]
            ])

B = np.array([
             [0  ],
             [1/L],
             [0  ]
             ])

# initial conditions
t0 = 0
X0 = np.array([
              [0],
              [0],
              [0]
              ])

# time info
n = 255
t_end = 2e-3

# definig input
def V(t):
    return np.heaviside(t - 0.5e-3, 0.99)
    # return abs(np.sin(2*np.pi*5e3*t))

def U(t):
    dt = ( t0 - t_end ) * ( 1/n )
    t = t + dt
    return ( V(t) - V(t - dt) )/dt

# Solving
RL_series = linDiffEqnSys(A, B, U)
Sol_RL_Series = RungeKutta4Solver(RL_series)
Sol_RL_Series.initConditions(t0,X0)
Sol_RL_Series.genTimePoints_n(n,t_end)
x,y = Sol_RL_Series.solverSolve()

# Plotting
fig, (ax1,ax2) = plt.subplots(2,1,sharex='col')
ax1.plot(x,V(x),'orangered',label = '$V_{in}(t)$') # subplot 1
ax1.set_ylabel('Voltage (V)')
ax1.grid(1)
ax1.legend()
# subplot 2
ax2.plot(x,y[2],'royalblue',label='$i_R(t)$')
ax2.set_ylabel('Current (A)')
ax2.set_xticks(np.linspace(0,2e-3,9))
ax2.set_xticklabels(['0','0.25m','0.5m','0.75m','1m','1.25m','1.5m','1.75m','2m'])
ax2.set_xlim([0e-3,2e-3])
ax2.set_xlabel('time (s)')
ax2.grid(1)
ax2.legend()
plt.show() # show plot
