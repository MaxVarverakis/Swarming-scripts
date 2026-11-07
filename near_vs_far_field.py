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

plt.rcParams['font.size'] = 16

# global settings
fname = "/Users/max/UCLA/Research/Codes/Data/Poda2024/Database S1/all_data.mat"
path = "/Users/max/UCLA/Research/Codes"
pre_path = f"{path}/Data/Poda_sound"

mono_bool = False
pole_str = "monopole" if mono_bool else "dipole"

fs = 44100
start_time = 945 # s
target_duration = 10 # s
n_harm = 4

# initialize swarm data
m = pod.mat(fname)
# fix trial number
tno = 4

block = 3

if block == 0:
    # extract clean, interpolated swarm data
    swarm_data = m.format_all_smooth_insects(tno, n_harmonics=n_harm, t_start=start_time, duration=target_duration, monopole=mono_bool)
    # swarm_data = [ m.generate_test_insect(duration=target_duration, n_harmonics=n_harm, monopole=mono_bool) ] # start time = 0 s here
    
    # snapshot parameters
    t_curr = start_time + target_duration / 2.0 # 950.015
    # t_curr = 0.0023 # target_duration / 2.0 # 950.015
    limits = ["near", "far", "both"]
    titles = ["Near", "Far", "Full"]
    cross_sec = 'xy' # 'xy' or 'xz'
    axes = list(cross_sec)

    zlice = 0.0
    d = 0.5
    resol = 0.005
    x_range = (-d,d)
    y_range = (-d,d)
    fpi = 3

    P_stack = []

    for _,lim in enumerate(limits):
        if mono_bool:
            X, Y, Psnap = ls.get_2d_pressure_snapshot_monopole(
                swarm_data,
                t_0=t_curr,
                x_range=x_range,
                y_range=y_range,
                res=resol,
                fp_iterations=fpi,
                slice=zlice,
                plane=cross_sec,
                limit=lim
            )
        else:
            X, Y, Psnap = ls.get_2d_pressure_snapshot(
                swarm_data,
                t_0=t_curr,
                x_range=x_range,
                y_range=y_range,
                res=resol,
                fp_iterations=fpi,
                slice=zlice,
                plane=cross_sec,
                limit=lim
            )
        
        P_stack.append(Psnap)

    P_stack_uPa = [ 1e6 * P for P in P_stack ]

    # check computation by plotting entire field
    # vmax = np.percentile(np.abs(P_stack_uPa), 98)
    vmax = [ np.percentile(np.abs(P_uPa), 98) for P_uPa in P_stack_uPa ]

    for i in range(3):
        fig, ax = plt.subplots(figsize=(8, 7))

        mesh = ax.pcolormesh(X, Y, P_stack_uPa[i], cmap='RdBu_r', vmin=-vmax[i], vmax=vmax[i], shading='auto')
        cbar = fig.colorbar(mesh, ax=ax, label=r'Pressure [$\mu$Pa]')

        x_src = [ np.interp(t_curr, insect['t'], insect['y'][:, 0]) for insect in swarm_data if insect['t'][0] <= t_curr <= insect['t'][-1] ]
        if axes[1] == 'y':
            y_src = [ np.interp(t_curr, insect['t'], insect['y'][:, 1]) for insect in swarm_data if insect['t'][0] <= t_curr <= insect['t'][-1] ]
        else:
            y_src = [ np.interp(t_curr, insect['t'], insect['y'][:, 2]) for insect in swarm_data if insect['t'][0] <= t_curr <= insect['t'][-1] ]
        ax.plot(x_src, y_src, 'ko', markersize=3.25, label="Dipoles" if not mono_bool else "Monopoles")


        ax.set_xlabel(f"{axes[0]} [m]")
        ax.set_ylabel(f"{axes[1]} [m]")
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.legend(loc='upper right')

        plt.suptitle(f"{titles[i]} field solution")

        plt.tight_layout()
        
        # save_path = f"{path}/Figures/{pole_str}_solution_comparison/{titles[i].lower()}_field_solution_soundfield_snapshot_range_{d:.2f}_trial_{tno+1}"
        # plt.savefig(f"{save_path}.png", dpi=800)

        plt.show()

elif block == 1:
    swarm_data = [ m.generate_test_insect(duration=target_duration, n_harmonics=n_harm, monopole=mono_bool) ] # start time = 0 s here
    
    limits = ["near", "far", "both"]
    lidx = 2
    
    cross_sec = 'xz' # 'xy' or 'xz'
    zlice = 0.0
    d = 0.5
    resol = 0.0025
    x_range = (-d,d)
    y_range = (-d,d)
    fpi = 3

    duration = 0.005 # s
    dt_frame = 0.0000625  # s per frame
    t_frames = np.arange(0, duration + dt_frame, dt_frame)
    num_frames = len(t_frames)

    print(f"Pre-computing {num_frames} frames...")

    print(1)
    X, Y, P0 = ls.get_2d_pressure_snapshot(
            swarm_data,
            t_0=t_frames[0],
            x_range=x_range,
            y_range=y_range,
            res=resol,
            fp_iterations=fpi,
            slice=zlice,
            plane=cross_sec,
            limit=limits[lidx]
        )

    P_stack = np.zeros((num_frames, X.shape[0], X.shape[1]))
    P_stack[0] = P0

    for i in range(1, num_frames):
        print(i+1)
        _, _, P_stack[i] = ls.get_2d_pressure_snapshot(
            swarm_data,
            t_0=t_frames[i],
            x_range=x_range,
            y_range=y_range,
            res=resol,
            fp_iterations=fpi,
            slice=zlice,
            plane=cross_sec,
            limit=limits[lidx]
        )
    print("Pre-computation complete!")

    fig, ax = plt.subplots(figsize=(8, 7))

    P_stack_uPa = [ 1e6 * P for P in P_stack ]
    vmax = np.percentile(np.abs(P_stack_uPa), 98)
    if vmax == 0:
        vmax = 1.0

    mesh = ax.pcolormesh(X, Y, P_stack_uPa[0], cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
    cbar = fig.colorbar(mesh, ax=ax, label=r'Pressure [$\mu$Pa]')

    # insect locations
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

    # anim.save(f"{path}/Animations/________Poda_{tno+1}_{pole_str}_{cross_sec}_fpi_{fpi}.mp4", fps=30, extra_args=['-vcodec', 'libx264'])

    plt.show()

elif block == 2:
    swarm_data = m.format_all_smooth_insects(tno, n_harmonics=n_harm, t_start=start_time, duration=target_duration, monopole=mono_bool)
    # swarm_data = [ m.generate_test_insect(duration=target_duration, n_harmonics=n_harm, monopole=mono_bool) ] # start time = 0 s here
    # snapshot parameters
    # t_curr = 0.0045 # 0.0023 # target_duration / 2.0 # 950.015
    # t_currs = np.arange(0.0040, 0.005005, 0.00005)
    # t_currs = [ 0.0040 ]
    t_currs = [ start_time + target_duration / 2.0 ] # 950.015
    # print(t_currs)

    limits = ["near", "far", "both"]
    titles = ["Near", "Far", "Full"]
    cross_sec = 'xz' # 'xy' or 'xz'
    axes = list(cross_sec)

    zlice = 0.0
    d = 1.0
    resol = 0.005
    x_range = (-d,d)
    y_range = (-d,d)
    fpi = 3

    # if plotting actual swarm data, start 1D snapshot at swarm CM
    z0 = []
    for insect in swarm_data:
                if insect['t'][0] <= t_currs[0] <= insect['t'][-1]:
                        z0.append(np.interp(t_currs[0], insect['t'], insect['y'][:, 2]))
    z_CM = np.mean(z0)
    y_range = tuple(z + z_CM for z in y_range)

    for t_curr in t_currs:
        P_stack = []

        for lim in limits:
            if mono_bool:
                X, Y, Psnap = ls.get_2d_pressure_snapshot_monopole(
                            swarm_data,
                            t_0=t_curr,
                            x_range=x_range,
                            y_range=y_range,
                            res=resol,
                            fp_iterations=fpi,
                            slice=zlice,
                            plane=cross_sec,
                            limit=lim
                        )
            else:
                X, Y, Psnap = ls.get_2d_pressure_snapshot(
                            swarm_data,
                            t_0=t_curr,
                            x_range=x_range,
                            y_range=y_range,
                            res=resol,
                            fp_iterations=fpi,
                            slice=zlice,
                            plane=cross_sec,
                            limit=lim
                        )
            
            P_stack.append(Psnap)

        P_stack_uPa = [ 1e6 * P for P in P_stack ]

        # extract 1D slice along x = 0
        # zero_slice_idx = X.shape[1]//2
        x_zero_idx = np.argmin(np.abs(X[0, :]))            # Column index where x approx 0
        z_start_idx = np.argmin(np.abs(Y[:, 0] - z_CM))    # Row index where z approx z_CM
        z_coords = Y[z_start_idx:, x_zero_idx] - z_CM      # relative z to z_CM

        # print(zero_slice_idx, np.argmin(abs(X), axis=1)[0])
        # print(X[:,zero_slice_idx])
        # print(Y[:,zero_slice_idx])

        # plim = [12, 100, 100] # uPa
        plim = 200 * np.ones(3) # uPa
        colors = [ "tab:red", "b", "k" ]
        lss = [ '--', '-.', '-' ]

        fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6), sharex=True)

        
        for i in range(3):
            # ax.plot(Y[zero_slice_idx:, zero_slice_idx], P_stack_uPa[i][zero_slice_idx:, zero_slice_idx], label=f"{titles[i]} field solution", lw=1.75, c=colors[i], ls=lss[i])
            ax.plot(z_coords, P_stack_uPa[i][z_start_idx:, x_zero_idx], label=f"{titles[i]} field solution", lw=1.75, c=colors[i], ls=lss[i])
        
        ax.set_ylim(-plim[-1], plim[-1])
        ax.set_xlim(left=0)
        ax.grid(True, alpha=0.3, linestyle='-', color='gray')
        
        ax.legend()
        
        # ax.set_xlabel(f"{axes[1]} [m]", fontsize=20)
        ax.set_xlabel(r"$z - c_{\text{CM}}$ [m]", fontsize=20)
        ax.set_ylabel(r"Pressure [$\mathregular{\mu}$Pa]", fontsize=20)
        
        # ax.set_title(f"t = {t_curr*1e3:.2f} ms")
        ax.set_title(f"t = {t_curr:.1f} s")

        ax_inset = ax.inset_axes([0.5, 0.05, 0.49, 0.3])
        for i in range(2):
            ax_inset.plot(
                z_coords,
                P_stack_uPa[i][z_start_idx:, x_zero_idx] - P_stack_uPa[-1][z_start_idx:, x_zero_idx],
                lw=1.75,
                c=colors[i],
                ls=lss[i]
            )
        ax_inset.set_ylim(-200, 200)
        ax_inset.tick_params(axis='both', labelsize=10)
        ax_inset.text(
            x=0.5,
            y=0.95,
            transform=ax_inset.transAxes,
            ha="center", va="top",
            s=r"Difference from full solution [$\mathregular{\mu}$Pa]",
            fontsize=10,
            fontweight="normal",
            bbox=dict(boxstyle='round,pad=0.5', facecolor='w', alpha=0.7, edgecolor='k', linewidth=0),
        )
        ax_inset.grid(True, alpha=0.3, linestyle='-', color='gray')
        
        # ax_inset.set_xlim(0.0, 0.5)
        # ax.indicate_inset_zoom(ax_inset, edgecolor="black")

        # if want on different (vetically stacked) subplots
        # for i in range(3):
        #     ax[i].plot(Y[zero_slice_idx:, zero_slice_idx], P_stack_uPa[i][zero_slice_idx:, zero_slice_idx], label=f"{titles[i]} field solution")
        #     ax[i].set_ylim(-plim[i], plim[i])
        #     ax[i].set_xlim(left=0)
        #     ax[i].legend(handlelength=0, handletextpad=0)

        plt.show()

elif block == 3:
    # swarm_data = m.format_all_smooth_insects(tno, n_harmonics=n_harm, t_start=start_time, duration=target_duration, monopole=mono_bool)
    swarm_data = [ m.generate_test_insect(duration=target_duration, n_harmonics=n_harm, monopole=mono_bool) ]

    limits = ["near", "far", "both"]
    titles = ["Near", "Far", "Full"]
    colors = ["tab:red", "b", "k"]
    lss = ['--', '-.', '-']
    # lss = ['-', '-', '-']

    cross_sec = 'xz' # 'xy' or 'xz'
    axes = list(cross_sec)

    zlice = 0.0
    d = 1.0
    resol = 0.005
    x_range = (-d, d)
    y_range = (-d, d)
    fpi = 3

    t_currs = np.arange(0.0040, 0.006005, 0.0000125)
    # t_currs = [ 0.004 ]
    
    # st = start_time + target_duration / 2.0
    # t_currs = np.arange(st, st + 0.002005, 0.0000125)
    # t_currs = [ st ]
    
    num_frames = len(t_currs)

    z0 = []
    for insect in swarm_data:
                if insect['t'][0] <= t_currs[0] <= insect['t'][-1]:
                        z0.append(np.interp(t_currs[0], insect['t'], insect['y'][:, 2]))
    z_CM = np.mean(z0)
    y_range = tuple(z + z_CM for z in y_range)
    # z_CM = 0

    # ---------------------------------------------------------
    # 1. Pre-compute Grid Geometry & All Time Steps
    # ---------------------------------------------------------
    print(f"Pre-computing acoustic fields for {num_frames} frames across 3 limits...")

    # Sample frame 0 to get spatial dimensions (X, Y)
    if mono_bool:
        X, Y, _ = ls.get_2d_pressure_snapshot_monopole(
            swarm_data,
            t_0=t_currs[0],
            x_range=x_range,
            y_range=y_range,
            res=resol,
            fp_iterations=fpi,
            slice=zlice,
            plane=cross_sec,
            limit=limits[0]
        )
    else:
        X, Y, _ = ls.get_2d_pressure_snapshot(
            swarm_data,
            t_0=t_currs[0],
            x_range=x_range,
            y_range=y_range,
            res=resol,
            fp_iterations=fpi,
            slice=zlice,
            plane=cross_sec,
            limit=limits[0]
        )

    x_zero_idx = np.argmin(np.abs(X[0, :]))            # Column index where x approx 0
    z_start_idx = np.argmin(np.abs(Y[:, 0] - z_CM))    # Row index where z approx z_CM
    z_coords = Y[z_start_idx:, x_zero_idx] - z_CM      # relative z to z_CM

    # Pre-allocate array for all frames: shape (num_frames, 3_limits, Nz, Nx)
    P_all_frames = np.zeros((num_frames, 3, X.shape[0], X.shape[1]))

    for f_idx, t_curr in enumerate(t_currs):
        print(f_idx+1)
        for l_idx, lim in enumerate(limits):
            if mono_bool:
                _, _, Psnap = ls.get_2d_pressure_snapshot_monopole(
                    swarm_data,
                    t_0=t_curr,
                    x_range=x_range,
                    y_range=y_range,
                    res=resol,
                    fp_iterations=fpi,
                    slice=zlice,
                    plane=cross_sec,
                    limit=lim
                )
            else:
                _, _, Psnap = ls.get_2d_pressure_snapshot(
                    swarm_data,
                    t_0=t_curr,
                    x_range=x_range,
                    y_range=y_range,
                    res=resol,
                    fp_iterations=fpi,
                    slice=zlice,
                    plane=cross_sec,
                    limit=lim
                )
            P_all_frames[f_idx, l_idx] = Psnap * 1e6  # uPa

    print("Pre-computation complete!")

    # ---------------------------------------------------------
    # 2. Figure Setup & Line Initialization
    # ---------------------------------------------------------
    plim = 100 # 300  # uPa axis limit

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.set_ylim(-plim, plim)
    ax.set_xlim(left=0, right=d)
    ax.grid(True, alpha=0.3, linestyle='-', color='gray')

    ax.set_xlabel(f"{axes[1]} [m]", fontsize=20)
    # ax.set_xlabel(r"$z - z_{\text{CM}}$ [m]", fontsize=20)
    ax.set_ylabel(r"Pressure [$\mathregular{\mu}$Pa]", fontsize=20)

    title_text = ax.text(
        0.5, 0.98, f"", 
        transform=ax.transAxes, 
        ha='center', va='top',
        # fontsize=16,
        fontweight='bold',
    )


    # inset stuff
    if not mono_bool:
        ax_inset = ax.inset_axes([0.49, 0.05, 0.49, 0.3])
        ax_inset.set_ylim(-100, 100)
        ax_inset.set_xlim(0, d)
        ax_inset.tick_params(axis='both', labelsize=10)
        inset_text = ax_inset.text(
            x=0.5,
            y=0.95,
            transform=ax_inset.transAxes,
            ha="center", va="top",
            s=r"Difference from full solution [$\mathregular{\mu}$Pa]",
            fontsize=10,
            fontweight="normal",
            bbox=dict(boxstyle='round,pad=0.5', facecolor='w', alpha=0.7, edgecolor='k', linewidth=0),
        )
        ax_inset.grid(True, alpha=0.3, linestyle='-', color='gray')

    # Initialize 3 empty line objects for Near, Far, Full
    nscl = 1e2 # optional scale for small near fields
    lines = []
    inset_lines = []
    full_slice = P_all_frames[0, 2, z_start_idx:, x_zero_idx]
    for i in range(3):
        initial_slice = P_all_frames[0, i, z_start_idx:, x_zero_idx]
        line, = ax.plot(
            z_coords,
            initial_slice,
            label=f"{titles[i]} field solution{r" $(\times 10^2)$" if (mono_bool and i==0) else ""}",
            lw=1.75,
            # lw=1.75 if i!=1 else 3,
            c=colors[i],
            ls=lss[i],
        )
        lines.append(line)

        if i != 2 and not mono_bool:
            i_line, = ax_inset.plot(
                        z_coords,
                        initial_slice - full_slice,
                        lw=1.75,
                        c=colors[i],
                        ls=lss[i]
                    )
            inset_lines.append(i_line)

    ax.legend(loc='upper right')

    # ---------------------------------------------------------
    # 3. Animation Update Function
    # ---------------------------------------------------------
    def update(frame_idx):
        t_curr = t_currs[frame_idx]

        # Update y-data for all 3 field lines
        full_p_slice = P_all_frames[frame_idx, 2, z_start_idx:, x_zero_idx]
        for i in range(3):
            p_slice = P_all_frames[frame_idx, i, z_start_idx:, x_zero_idx]
            lines[i].set_ydata(p_slice)
            # lines[i].set_ydata(p_slice * (1.0 if i!=0 else nscl))

            if i != 2 and not mono_bool:
                inset_lines[i].set_ydata(p_slice - full_p_slice)

        title_text.set_text(f"t = {(t_curr - t_currs[0])*1e3:.3f} ms")
        
        if mono_bool:
            return (*lines, title_text)
        else:
            return (*lines, *inset_lines, title_text, inset_text)

    # ---------------------------------------------------------
    # 4. Render Animation
    # ---------------------------------------------------------
    anim = FuncAnimation(fig, update, frames=num_frames, interval=80, blit=True)

    plt.tight_layout()

    # anim.save(
    #     # f"{path}/Animations/swarm_near_vs_far_{pole_str}_comparison_xz_slice_{zlice:.2f}_{n_harm}_harmonics.mp4",
    #     f"{path}/Animations/near_vs_far_{pole_str}_comparison_xz_slice_{zlice:.2f}_{n_harm}_harmonics.mp4",
    #     fps=30,
    #     extra_args=[
    #         '-vcodec', 'libx264',
    #     ],
    # )

    plt.show()
