# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
# ---

# %%
# 2D Pond wave equation

# %%
# Import libraries
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation, colors, ticker, image
import seaborn as sns
import pandas as pd
import os
import time
from datetime import datetime

# !%matplotlib widget

os.chdir(os.path.dirname(__file__))
print("Import complete.")

# %%

# User parameters
T = 10                             # total simulation time
save_csv = False
save_figures = True
apply_smoothing = False
output_folder = "test"


start_time = time.time()
timestamp = datetime.now().strftime(r"%Y.%m.%d-%H.%M.%S")

# problem parameters


#############################################################################
########### Variable library ################################################
# u             water level
# x             x location
# y             y location
# t             time
# L_x           Pond length on x axis
# L_y           Pond length on y axis
# c             wave constant
# v             friction factor
# n             number of spatial intervals
# N_x           x grid points
# N_y           y grid points
# h             spatial mesh spacing
# T             total simulation time (s)
# dt            time step duration (s)
# cfl_max       maximum Courant number
#############################################################################

# Problem parameters
L_x = 10.0      # total length on x axis
L_y = 10.0      # total length on y axis
c = 1.0         # wave speed
v = 0.002       # friction factor


# Spatial parameters
n = 100         # number of spatial intervals
N_x = n + 1     # x grid points
N_y = n + 1     # y grid points
h = L_x / n     # mesh spacing

# Time parameters
sf = 0.7                            # safety factor
cfl_max = 1/np.sqrt(2)              # maximum Courant number
dt = h/c * (sf * cfl_max)           # time step satisfying CFL condition

fps_gif = 10.0                        # gif frames per second

n_frames = int(fps_gif * T) + 1
time_per_frame = 1 / fps_gif
frame_interval = max(1, int(round(time_per_frame / dt)))
interval = frame_interval * dt * 1000  # ms

# Grid
x = np.linspace(0, L_x, N_x)
y = np.linspace(0, L_y, N_y)
XX, YY = np.meshgrid(x, y)

# Initialise the 3 states of the water level u at the:
# previous, current and next timestep
u_prev = np.zeros((N_x, N_y))           # u at t-dt
u = np.zeros((N_x, N_y))                # u at t
u_next = np.zeros((N_x, N_y))           # u at t+dt

# set time and energy history lists to check energy conservation
time_history = []
energy_history = []


u_all = np.zeros((n_frames, N_x, N_y))  # 3D array with all u values

def get_u(f: int, x_coord: float, y_coord: float, u: np.ndarray = u_all) -> float:
    """Get the value of u at a given frame index f and coordinates (x, y).

    Parameters:
        f (int): frame index
        x_coord (float): x coordinate
        y_coord (float): y coordinate
        u (np.ndarray): 3D array with all u values (default: u_all)

    Returns:
        float: value of u at the given frame index and coordinates
    """

    x_i = np.argwhere(x==x_coord)[0,0]
    y_i = np.argwhere(y==y_coord)[0,0]
    return(u[f,x_i,y_i])

print("Problem setup complete.")


# Set up disturbance
x_loc = 5.0     # x coordinate
y_loc = 5.0     # y coordinate

def find_nearest(axis: np.ndarray, location: float) -> float:
    """Find the nearest node on the meshgrid to a given location.

    Parameters:
        axis (np.ndarray): 1D array of x or y coordinates
        location (float): location to find the nearest node to

    Returns:
        float: nearest node on the meshgrid to the given location
    """

    axis = np.asarray(axis)
    idx = (np.abs(axis - location)).argmin()
    return(axis[idx])


if np.argwhere(x==x_loc).size==0:
    x_loc = find_nearest(x, x_loc)
    print(f"Adjusted disturbance location for the available grid: {x_loc = :.1f}")

if np.argwhere(y==y_loc).size==0:
    y_loc = find_nearest(y, y_loc)
    print(f"Adjusted disturbance location for the available grid: {y_loc = :.1f}")

# grid array indices of disturbance location
x_idx = np.argwhere(x==x_loc)[0,0]  # index on x grid array
y_idx = np.argwhere(y==y_loc)[0,0]  # index on y grid array



def disturb(t: float) -> float:
    """Generate a sinusoidal disturbance active in the first 2.5 seconds.

    Parameters:
        t (float): time in seconds

    Returns:
        float: disturbance
    """

    if t<=2.5:
        dist = np.sin(t/10)
    else:
        dist = 0
    return(dist)


# unnormalised gaussian mask for smoothing
sigma = 0.05     # radius of disturbance region

def gaussian_smoothing(x_d: float, y_d: float, sigma: float) -> np.ndarray[float]:
    """Generate a 2D Gaussian mask for smoothing the disturbance.

    Parameters:
        x_d (float): x coordinate of the disturbance location
        y_d (float): y coordinate of the disturbance location
        sigma (float): standard deviation of the Gaussian mask

    Returns:
        np.ndarray: 2D Gaussian mask for smoothing the disturbance
    """

    G = np.exp(-((XX-x_d)**2 + (YY-y_d)**2) / (2 * sigma**2))
    return(G)


G = gaussian_smoothing(x_loc, y_loc, sigma)

if apply_smoothing:
    print(f"Applying Gaussian smoothing with: {sigma = :.2f}")
else:
    print("Smoothing not applied on disturbance.")

print("Disturbance preparation complete.")


# %%

def rem_b(array: np.ndarray) -> np.ndarray:
    """Remove boundary elements from n-D array.

    Parameters:
        array (np.ndarray): array to remove boundary elements from

    Returns:
        np.ndarray: array with boundary elements removed
    """

    if len(array.shape)==1:
        return(array[1:-1])
    elif len(array.shape)==2:
        return(array[1:-1, 1:-1])
    elif len(array.shape)==3:
        return(array[1:-1, 1:-1, 1:-1])
    else:
        raise Exception(f"Array given has {len(array.shape)} dimensions.")




def update_water_level(frame_idx: int) -> image.AxesImage:
    """Update the water level using finite differences method.

    Parameters:
        frame_idx (int): current frame index

    Returns:
        AxesImage: updated water level visualisation
    """


    global u, u_prev, u_next

    # timestep calculations
    for _ in range(frame_interval):
        t = frame_idx * frame_interval * dt

        # Use finite differences method

        # In space:
        # Laplacian
        # five-point stencil approximation

        # skipping the boundary points: so indexing is [1:-1]

        # Laplacian using i=1 and j=1
        # lap(u_11) = ( u_21 + u_01 + u_12 + u_10 - 4*u_11 ) / h**2

        u_11 = u[1:-1, 1:-1]        # centre

        u_21 = u[2:,   1:-1]        # below (i+1, j)
        u_01 = u[:-2,  1:-1]        # above (i-1, j)

        u_12 = u[1:-1, 2:]          # right (i, j+1)
        u_10 = u[1:-1, :-2]         # left  (i, j-1)

        lap = (u_21 + u_01 + u_12 + u_10 - 4*u_11) / h**2


        # use finite differences method in time to calculate new water level u
        leapfrog_term = 2*rem_b(u) - rem_b(u_prev)
        spatial_term = c**2 * dt**2 * lap
        friction = v * dt * (rem_b(u) - rem_b(u_prev))

        # calculate the disturbance and apply Gaussian smoothing if needed
        if apply_smoothing:
            disturbance = disturb(t) * G
            dist_term = dt**2 * rem_b(disturbance)
        else:
            # without smoothing
            disturbance = dt**2 * disturb(t)
            disturbance_matrix = np.zeros_like(u)
            disturbance_matrix[x_idx, y_idx] = disturbance
            dist_term = rem_b(disturbance_matrix)

        u_next[1:-1, 1:-1] = leapfrog_term + spatial_term - friction + dist_term


        # Setting boundary conditions before next step
        u_next[0, :] = 0
        u_next[-1,:] = 0
        u_next[:, 0] = 0
        u_next[:,-1] = 0

        # update the data to calculate the next step
        u_prev = u.copy()
        u = u_next.copy()


    # frame actions
    t_display = frame_idx * time_per_frame
    time_history.append(t_display)

    print(f"Generating frame {frame_idx} / {n_frames-1}")

    # write data every 5 frames
    if save_csv and frame_idx % (fps_gif//2) == 0:
        # print(f"Writing CSV: frame {frame_idx} / {n_frames-1}")

        # flatten arrays
        rows = np.column_stack([
            np.full(u.size, t_display),
            XX.ravel(),
            YY.ravel(),
            u.ravel()
        ])

        with open(csv_path, "a") as f:
            np.savetxt(f, rows, delimiter=",")


    # save u in u_df dataframe every frame
    u_all[frame_idx] = u.copy()

    # update animation by one frame interval
    ax.set_title(f"2D Pond Wave simulation\nFrame = {frame_idx} --- t = {t_display:.2f} s")
    # update the image
    cax.set_data(u)

    # NO LONGER SCALING LIMITS
    # update color limits after the first frame
    # if frame_idx > 0:
    #     cax.set_clim(vmin=np.min(u), vmax=np.max(u))
    # cbar.update_ticks()

    return([cax])

print("Simulation functions prepared.")

# %%

# Create csv file
cwd = os.getcwd()
csv_filename = f"simulation_data_{timestamp}.csv"

os.makedirs(output_folder, exist_ok=True)
os.chdir(output_folder)

print(f"Directory: {os.getcwd()}")

csv_path = os.path.join(os.getcwd(), csv_filename)

if save_csv:
    print(f"Saving results in {csv_filename}.\n")
    with open(csv_path, "w") as f:
        f.write("t,x,y,u\n")



# Initialise visualisation
fig, ax = plt.subplots()

# add padding around axes
fig.subplots_adjust(left=0.1, right=0.85, top=0.85, bottom=0.1)



# colormap limits
amp = 0.0003

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(ls=":")

# set up colorbar
norm = colors.Normalize(vmin=-amp, vmax=amp)
cax = ax.imshow(
    u,
    # cmap= "berlin",
    cmap= "seismic",
    origin= "lower",
    extent= [0, L_x, 0, L_y],
    animated = True,
    norm= norm,
)

cbar = fig.colorbar(cax)

# set colorbar label manually to avoid jittering movement as the scale changes
cbar_label = "u(x,y,t)"
fig.text(
    x = cbar.ax.get_position().xmax*1.005,
    y = 0.5,
    s = cbar_label,
    rotation = 90,
    ha="center",        # horizontal alignment
    va="center"         # vertical alignment
)

cbar.formatter = ticker.ScalarFormatter(useMathText=True)
cbar.formatter.set_powerlimits((0, 0))  # always show scientific notation





# Run simulation
def initialise_animation() -> image.AxesImage:
    """
    Initialise the animation by setting up the initial image data.

    Returns:
        AxesImage: initial image data
    """

    cax.set_data(u)
    return (cax)


interval_ms = time_per_frame * 1000  # ms per frame
anim = animation.FuncAnimation(
    fig=fig,
    func= update_water_level,
    frames= n_frames,
    interval=interval_ms,
    blit=True,
    init_func= initialise_animation,
)

gif_name = f"2D_pond_wave_simulation_{timestamp}.gif"

# save animation
if save_figures:
    print(f"Saving animation as {gif_name}.\n")
    anim.save(gif_name, writer="pillow", fps=fps_gif, dpi=120)



fig.show()
plt.close(fig)


# for data inspection
# df = pd.read_csv(csv_path)

# %%
# Calculate system energy and plot

def calculate_energy(
    u_all: np.ndarray = u_all,
    dtime: float = time_per_frame,
    gspace: float = h,
    c: float = c
) -> pd.DataFrame:
    """Calculate the kinetic and potential energies of the system.

    Args:
        u_all (np.ndarray, optional): 3D array with all u values. Defaults to u_all.
        dtime (float, optional): Time step duration (s). Defaults to time_per_frame.
        gspace (float, optional): Grid spacing. Defaults to h.
        c (float, optional): wave speed. Defaults to c.

    Returns:
        pd.DataFrame: Dataframe with the different energy types in the system
    """

    num = u_all.shape[0]
    kinetic = np.zeros(num)
    potential = np.zeros(num)
    frames = range(0,num)

    for k in frames[1:]:
        u_curr = u_all[k]
        u_pr = u_all[k-1]

        velocity = (u_curr - u_pr) / dtime
        kinetic[k] = 0.5 * np.sum(velocity**2) * gspace**2

        # kinetic[k] = 0.5 * np.sum(((u_curr - u_pr)/dtime)**2)

        gradx = (u_curr[2:,1:-1] - u_curr[:-2,1:-1])/(2*gspace)
        grady = (u_curr[1:-1,2:] - u_curr[1:-1,:-2])/(2*gspace)
        potential[k] = 0.5 * c**2 * np.sum(gradx**2 + grady**2) * gspace**2


    df = pd.DataFrame({"Kinetic": kinetic, "Potential": potential })
    df["Total"] = df["Kinetic"] + df["Potential"]
    df["Frame"] = list(frames)
    return(df)


# energy_T, energy_K, energy_P = calculate_energy()
energy_df = calculate_energy()
energy_df["Time"] = time_history

energy_df = energy_df.set_index("Time")

# energy plot
en_fig = plt.figure(figsize=(6,5))
sns.lineplot(data = energy_df[["Total", "Potential", "Kinetic"]])

plt.grid(ls=":")
plt.xlabel("Time (s)")
plt.ylabel("Energy")
plt.xlim(left=0, right=T)
plt.ylim(bottom=0)
plt.title("History of the energy in the system")
plt.ticklabel_format(axis="y",useMathText=True, style="sci", scilimits=(0,0))
plt.legend()
plt.tight_layout()


en_fig_name = f"Energy_History_{timestamp}.png"
if save_figures:
    print(f"Saving energy history plot as {en_fig_name}.\n")
    en_fig.savefig(en_fig_name, format="png", dpi=300)

plt.show()

# %%
# Final print statement
os.chdir(cwd)
print("Script finished.")
print(f"--- {time.time() - start_time:.2f} seconds ---")

