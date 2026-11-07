import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# import matplotlib.ticker as ticker
from matplotlib.animation import FuncAnimation
import librosa as lib
import scipy.io.wavfile as wav

import Lowson_solver as ls
import Poda_data as pod
import Gibson_sound_analysys as gib


# global settings
fname = "/Users/max/UCLA/Research/Codes/Data/Poda2024/Database S1/all_data.mat"
path = "/Users/max/UCLA/Research/Codes"
pre_path = f"{path}/Data/Poda_sound"

mono = False
pole_str = "monopole" if mono else "dipole"

fs = 44100
start_time = 950 # 1010.20 # 900
target_duration = 10 # 0.1 # 10  # s
# for n_harm in range(1,4):
n_harm = 4

mic_pos = np.array([np.sqrt(0.89**2 - 0.27**2), -0.27, 0.0]) # m
# mic_pos = np.array([0.0, 0.0, 1.0], [0.0, 0.0, -1.0]) # m

# initialize swarm data
m = pod.mat(fname)
# fix trial number
tno = 4
obj_ids = m.obj_ids(tno)


swarm_data = m.format_all_smooth_insects(tno, n_harmonics=n_harm, t_start=start_time, duration=target_duration, monopole=mono)

# if mono:
#     t_audio,p = ls.swarm_sound_monopole(mic_pos, swarm_data)
# else:
#     t_audio,p = ls.swarm_sound_dipole(mic_pos, swarm_data)

# np.savez_compressed(f"{pre_path}/trial_{tno+1}_{pole_str}.npz", t=t_audio, p=p, fs=fs, mic_pos=mic_pos)
# p_clean = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
# p_max = np.max(np.abs(p_clean))
# p_norm = p / p_max
# wav.write(f"{pre_path}/trial_{tno+1}_{pole_str}_audio.wav", fs, np.int16(p_norm * 32767))

###########################################################################
###########################################################################

# on=0
# t = m.time(tno,on)
# x = m.x(tno, on)

# fig = plt.figure(figsize=(10,8))

# plt.scatter(t[:100], x[:100], s=3)
# plt.xlabel("Time [s]")
# plt.ylabel("X [m]")

# plt.show()


###########################################################################
###########################################################################

# fig,ax = plt.subplots(figsize=(10,6))

# if mono:
#     plt.plot(t_audio, 1e3 * p, lw=1)
#     plt.ylabel(r"Pressure [mPa]")
# else:
#     plt.plot(t_audio, 1e6 * p, lw=1)
#     plt.ylabel(r"Pressure [$\mu$Pa]")

# # ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
# plt.xlabel("Time [s]")
# plt.grid(True)
# plt.tight_layout()

# plt.savefig(f"{path}/Figures/Poda_dual_mic_{pole_str}.png", dpi=800)

# plt.show()

###########################################################################
###########################################################################

# oid = max(obj_ids) - 5
# # for oid in obj_ids:
# swarm_data = [ m.format_insect_data(tno, obj) for obj in obj_ids[oid:oid+1] ]

# # Extract active time bounds for this specific insect
# t_s = swarm_data[0]['t']
# t_start, t_end = t_s[0], t_s[-1]
# print(f"Object {oid} active time range: {t_start:.2f} s to {t_end:.2f} s")

# # select target time inside its active window (e.g. midpoint)
# t_target = (t_start + t_end) / 2.0  

# # get insect position at t_target to center the view frame
# x_center = np.interp(t_target, t_s, swarm_data[0]['y'][:, 0])
# y_center = np.interp(t_target, t_s, swarm_data[0]['y'][:, 1])

# # set spatial extent centered around the insect
# half_width = 1.5  # 1.5 m window around mosquito
# x_range = (x_center - half_width, x_center + half_width)
# y_range = (y_center - half_width, y_center + half_width)

###########################################################################
###########################################################################


block = -99


if block == 0:
    gib.spectrum_analyze(
        f"{pre_path}/trial_{tno+1}_{pole_str}_audio.wav",
        show=True,
        print_info=False,
        add_trim=True,
        fft=True,
        extract_peaks=True,
        lHz=5e2,
        save_path=f"{path}/Figures/Poda_{tno+1}_{pole_str}_spec_trim_fft"
    )

elif block == 1:
    cross_sec = 'xy' # 'xy' or 'xz'
    d = 2
    x_range = (-d,d)
    y_range = (-d,d)
    resol = 0.0125
    n_grid = int(2 * d / resol) + 1
    fpi = 2
    zlice = 0.6 # 0.1 # m

    duration = 0.015 # s
    dt_frame = 0.0000625  # s per frame
    # dt_frame = 0.0025  # s per frame
    t_frames = np.arange(start_time, start_time + duration + dt_frame, dt_frame)
    num_frames = len(t_frames)

    # if longitudinal slice, center z-range on swarm CM
    if cross_sec == 'xz':
        z0 = []
        for insect in swarm_data:
                    if insect['t'][0] <= t_frames[0] <= insect['t'][-1]:
                            z0.append(np.interp(t_frames[0], insect['t'], insect['y'][:, 2]))
        z_CM = np.mean(z0)
        y_range = tuple(z + z_CM for z in y_range)

    print(f"Pre-computing {num_frames} frames...")

    if mono:
        X, Y, P0 = ls.get_2d_pressure_snapshot_monopole(
            swarm_data, t_0=t_frames[0], x_range=x_range, y_range=y_range, res=resol, fp_iterations=fpi, slice=0.1
        )
    else:
        X, Y, P0 = ls.get_2d_pressure_snapshot(
            swarm_data, t_0=t_frames[0], x_range=x_range, y_range=y_range, res=resol, fp_iterations=fpi, slice=0.1
        )
    
    P_stack = np.zeros((num_frames, X.shape[0], X.shape[1]))
    P_stack[0] = P0

    for i in range(1, num_frames):
        print(i)
        if mono:
            _, _, P_stack[i] = ls.get_2d_pressure_snapshot_monopole(swarm_data, t_0=t_frames[i], x_range=x_range, y_range=y_range, res=resol, fp_iterations=fpi, slice=zlice, plane=cross_sec)
        else:
            _, _, P_stack[i] = ls.get_2d_pressure_snapshot(swarm_data, t_0=t_frames[i], x_range=x_range, y_range=y_range, res=resol, fp_iterations=fpi, slice=zlice, plane=cross_sec)
        # _, _, P_stack[i] = ls.get_2d_pressure_snapshot([swarm_data[-1]], t_0=t_frames[i], x_range=x_range, y_range=y_range, res=resol, fp_iterations=fpi, slice=0.1, plane=cross_sec)
    print("Pre-computation complete!")

    fig, ax = plt.subplots(figsize=(8, 7))
    
    # if mono:
    #     P_stack_uPa = [ 1e3 * P for P in P_stack ]
    # else:
    #     P_stack_uPa = [ 1e6 * P for P in P_stack ]
    P_stack_uPa = [ 1e6 * P for P in P_stack ]

    vmax = np.percentile(np.abs(P_stack_uPa), 98)
    if vmax == 0:
        vmax = 1.0

    # initialize plot on first frame
    # P_normed = [ P / np.max(P) for P in P_stack ]
    # mesh = ax.pcolormesh(X, Y, P_normed[0], cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
    # cbar = fig.colorbar(mesh, ax=ax, label=r'Pressure [$P_0$]')
    # linthresh = 0.5  # uPa
    # cscale = np.max([ np.percentile(np.abs(P), 98) for P in P_stack_uPa ])
    # norm = mcolors.SymLogNorm(
    #     linthresh=linthresh, 
    #     linscale=1.0, 
    #     vmin=-cscale, 
    #     vmax=cscale, 
    #     base=10
    # )
    # mesh = ax.pcolormesh(X, Y, P_stack_uPa[0], cmap='RdBu_r', shading='auto')

    mesh = ax.pcolormesh(X, Y, P_stack_uPa[0], cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
    # if mono:
    #     cbar = fig.colorbar(mesh, ax=ax, label=r'Pressure [mPa]')
    # else:
    #     cbar = fig.colorbar(mesh, ax=ax, label=r'Pressure [$\mu$Pa]')
    cbar = fig.colorbar(mesh, ax=ax, label=r'Pressure [$\mu$Pa]')

    # insect locations
    if mono:
        dots, = ax.plot([], [], 'ko', markersize=3.25, label='Monopoles')
    else:
        dots, = ax.plot([], [], 'ko', markersize=3.25, label='Dipoles')

    axis_labels = list(cross_sec)
    ax.set_xlabel(f"{axis_labels[0]} [m]")
    ax.set_ylabel(f"{axis_labels[1]} [m]")
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.legend(loc='upper right')
    
    def update(frame_idx):
        t_curr = t_frames[frame_idx]
        
        # Update pressure mesh array
        mesh.set_array(P_stack_uPa[frame_idx].ravel())
        # mesh.set_array(P_normed[frame_idx].ravel())
        
        x_dots, y_dots = [], []
        for insect in swarm_data:
            if insect['t'][0] <= t_curr <= insect['t'][-1]:
                x_dots.append(np.interp(t_curr, insect['t'], insect['y'][:, 0]))
                
                if axis_labels[1] == 'y':
                    # if transverse plane:
                    y_dots.append(np.interp(t_curr, insect['t'], insect['y'][:, 1]))
                else:
                    # if longitudinal plane:
                    y_dots.append(np.interp(t_curr, insect['t'], insect['y'][:, 2]))
        dots.set_data(x_dots, y_dots)

        ax.set_title(f'Acoustic Pressure Field (t = {t_curr:.4f} s)\n{fpi} fixed-point iteration{['s' if fpi != 1 else ''][0]}, {'y' if axis_labels[1] == 'z' else 'z'}-slice = {zlice:.2f} m')
        return mesh, dots

    anim = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=False)

    anim.save(f"{path}/Animations/Poda_{tno+1}_{pole_str}_{cross_sec}_{zlice:.2f}_m_fpi_{fpi}.mp4", fps=30, extra_args=['-vcodec', 'libx264'])
    
    plt.show()


elif block == 10:
    t_target = 1010.54 # s

    d = 2
    x_range = (-d,d)
    y_range = (-d,d)

    fpi = 2
    X, Y, P = ls.get_2d_pressure_snapshot(swarm_data, t_0=t_target, x_range=x_range, y_range=y_range, res=0.025, fp_iterations=fpi)
    # X, Y, P = ls.get_2d_pressure_snapshot([swarm_data[-1]], t_0=t_target, x_range=x_range, y_range=y_range, res=0.025, fp_iterations=fpi)
    fig, ax = plt.subplots(figsize=(8, 7))

    P_uPa = 1e6 * P
    vmax = np.percentile(np.abs(P_uPa), 98)
    if vmax == 0: vmax = 1.0

    # initialize plot on first frame
    mesh = ax.pcolormesh(X, Y, 1e6 * P, cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
    # mesh = ax.pcolormesh(X, Y, 1e6 * (P - np.mean(P)), cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')

    # linthresh = 0.5  # uPa
    # cscale = np.max(np.abs(P_uPa))
    # norm = mcolors.SymLogNorm(
    #     linthresh=linthresh, 
    #     linscale=1.0, 
    #     vmin=-cscale, 
    #     vmax=cscale, 
    #     base=10
    # )
    # mesh = ax.pcolormesh(X, Y, 1e6 * P, cmap='RdBu_r', shading='auto')


    cbar = fig.colorbar(mesh, ax=ax, label=r'Pressure [$\mu$Pa]')
    # cbar.formatter.set_scientific(True)
    # cbar.formatter.set_powerlimits((0, 0))
    # cbar.formatter.set_useMathText(True)
    # cbar.update_ticks()

    # # Plot insect markers
    # xi = np.interp(t_target, t_audio, swarm_data[0]['y'][:, 0])
    # yi = np.interp(t_target, t_audio, swarm_data[0]['y'][:, 1])
    # insect_dot, = ax.plot(xi,yi, 'ko', markersize=8, label='Source')

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    # ax.set_xlim(extent)
    # ax.set_ylim(extent)
    ax.set_title(f'Acoustic Pressure Field (t = {t_target:.2f} s)\n{fpi} fixed-point iteration{['s' if fpi != 1 else ''][0]}')

    plt.show()

    # print(f"Saving iteration {fpi}...")
    # plt.savefig(f"{pre_path}/Figures/soundfield_fast_moving_dipole_fpi_{fpi}.png", dpi=800)
    # print("Done!")
