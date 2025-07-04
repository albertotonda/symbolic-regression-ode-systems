import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

sns.set_style("darkgrid")

# ---------------------------
# Aizawa attractor definition
# ---------------------------
def aizawa(t, state, a=0.95, b=0.7, c=0.6, d=3.5, e=0.25, f=0.1):
    x, y, z = state
    dxdt = (z - b) * x - d * y
    dydt = d * x + (z - b) * y
    dzdt = c + a * z - (z**3) / 3 - (x**2 + y**2) * (1 + e * z) + f * z * x**3
    return [dxdt, dydt, dzdt]

# ---------------------------
# Initial conditions
# ---------------------------
state0 = [0.1, 0.0, 0.0]
dT = 0.1
num_frames = 100
t_eval_dense = np.linspace(0, dT, 500)

# ---------------------------
# Storage for time & states
# ---------------------------
trajectory = []
times = []

# ---------------------------
# Set up 2D figure
# ---------------------------
fig, ax = plt.subplots(figsize=(10, 6))
line_x, = ax.plot([], [], label='x(t)') #, color='blue')
line_y, = ax.plot([], [], label='y(t)') #, color='green')
line_z, = ax.plot([], [], label='z(t)') #, color='red')

ax.set_xlim(0, 10)
ax.set_ylim(-3, 3)
ax.set_xlabel('Time')
ax.set_ylabel('State Variables')
ax.set_title("Aizawa Attractor - State Variables Over Time")
ax.legend()

# ---------------------------
# Animation update function
# ---------------------------
def update(frame):
    global trajectory, times
    T_start = frame * dT
    T_end = T_start + dT

    # If first frame, use state0
    sol = solve_ivp(aizawa, [0, dT], state0 if frame == 0 else trajectory[-1], t_eval=t_eval_dense)

    # Append times adjusted to global time
    if frame == 0:
        times = sol.t.tolist()
        trajectory = sol.y.T.tolist()
    else:
        t_offset = T_start
        times.extend((sol.t + t_offset).tolist())
        trajectory.extend(sol.y.T.tolist())

    traj_array = np.array(trajectory)

    line_x.set_data(times, traj_array[:, 0])
    line_y.set_data(times, traj_array[:, 1])
    line_z.set_data(times, traj_array[:, 2])

    ax.set_xlim(0, 10)

    return line_x, line_y, line_z

# ---------------------------
# Create animation
# ---------------------------
anim = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=True)

# ---------------------------
# Save or show
# ---------------------------
# To save as GIF:
anim.save("aizawa_states.gif", writer='pillow', fps=15)
#plt.show()
