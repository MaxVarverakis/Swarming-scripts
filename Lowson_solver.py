import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import matplotlib.colors as mcolors


plt.rcParams['font.size'] = 16

def swarm_sound_monopole(x_det, swarm_data, fs=44100, c0=343.0, t_start=None, duration=np.inf, print_progress=False):
    """
    Computes acoustic pressure at a detector location using the Push (Source-Time) method.
        
        Parameters:
        -----------
        x_det : ndarray (3,)
            Detector position [x, y, z]
        swarm_data : list of dicts
            [{ 't': t_k, 'y': y_k, 'v': v_k, 'a': a_k, 'F': F_k, 'dFdt': dFdt_k }, ...]
        fs : int, default 44100
            Audio sample rate (Hz)
        c0 : float, default 343.0
            Speed of sound
        
        Returns:
        --------
        t_global : ndarray of shape (N_samples)
            global time data
        p_total : ndarray of shape (N_samples)
            Pressure time series for the detector.
    """

    if t_start is None:
        t_start = min(data['t'][0] for data in swarm_data)
    
    t_max_data = max(data['t'][-1] for data in swarm_data)
    t_end = min(t_max_data, t_start + duration) if np.isfinite(duration) else t_max_data

    max_propagation_delay = 0.0 # s

    t_global = np.arange(t_start, t_end + max_propagation_delay, 1.0 / fs)
    p_total = np.zeros_like(t_global)

    for iid,insect in enumerate(swarm_data):
        if print_progress:
            print(f"{iid+1} / {len(swarm_data)}")
        
        # obtain data range
        t_raw = insect['t'] # (N_samples,)
        t_idx = np.argmin(abs(t_raw - t_end))
        
        # extract source information
        t_s = t_raw[:t_idx+1]
        y_s = insect['y'][:t_idx+1] # (N, 3)
        v_s = insect['v'][:t_idx+1] # (N, 3)
        a_s = insect['a'][:t_idx+1] # (N, 3)
        
        # scalar mass rate Q [kg/s] instead of Force
        Q_s = insect['Q'][:t_idx+1] # (N, 3)
        dQdt_s = insect['dQdt'][:t_idx+1] # (N, 3)

        r_vec = x_det - y_s # (N,3)
        r = np.linalg.norm(r_vec, axis=1) # distance/magnitude, shape (N,)
        r_hat = r_vec / r[..., None]
    
        M_vec = v_s / c0  # mach number (N, 3)
        M_sq = np.sum(M_vec**2, axis=1) # (N,)
        M_r = np.sum(M_vec * r_hat, axis=1) # (N,)
    
        # acceleration along radiation direction
        a_vec = a_s / c0  # (N, 3)
        dMr_dt = np.sum(a_vec * r_hat, axis=1)  # (N,)
    
        # arrival time
        t_arrive = t_s + r / c0
    
        doppler_factor = (1.0 - M_r)**2
        prefactor = 1.0 / (4.0 * np.pi * doppler_factor)

        p_far = prefactor / r * (dQdt_s + Q_s * (dMr_dt / (1.0 - M_r)))

        p_near = prefactor * c0 / r**2 * Q_s * ((M_r - M_sq) / (1.0 - M_r))
        
        p_emit = p_far + p_near
        p_emit = np.nan_to_num(p_emit, nan=0.0, posinf=0.0, neginf=0.0)

        p_k = np.interp(t_global, t_arrive, p_emit, left=0.0, right=0.0) # ensure padding with zeros so there's no DC pressure off-time
        p_total += p_k # superposition

    return t_global, p_total

def get_2d_pressure_snapshot_monopole(swarm_data, t_0, x_range=(-2,2), y_range=(-2,2), res=0.02, slice=0.0, plane='xy', c0=343.0, fp_iterations=2, limit="both"):
    """
    t_0 : float
        time stamp of snapshot (global time, must be within time range of recording!)
    """
    x = np.arange(x_range[0], x_range[1]+res, res)
    y = np.arange(y_range[0], y_range[1]+res, res)
    X,Y = np.meshgrid(x, y)

    if plane == 'xz':
        # xz-plane
        grid_pts = np.column_stack([X.ravel(), np.full(X.size, slice), Y.ravel()])
    else:
        # xy-plane
        grid_pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, slice)])

    p_grid = np.zeros(X.size)

    for insect in swarm_data:
        t_s = insect['t']
        pos = insect['y']

        # approximate delayed time based off current positions
        # y_guess = np.column_stack([np.interp(t_0, t_s, insect['y'][:, i]) for i in range(3)])
        y_guess = np.column_stack([np.interp(t_0, t_s, pos[:, i]) for i in range(3)])
        r_vec = grid_pts - y_guess
        r = np.linalg.norm(r_vec, axis=1)
        tau = t_0 - (r / c0)

        # fixed-point iteration
        for _ in range(fp_iterations):
            # y_tau = np.column_stack([np.interp(tau, t_s, insect['y'][:, i]) for i in range(3)])
            y_tau = np.column_stack([np.interp(tau, t_s, pos[:, i]) for i in range(3)])
            r_vec = grid_pts - y_tau
            r = np.linalg.norm(r_vec, axis=1)
            # r = np.maximum(np.linalg.norm(r_vec, axis=1), 1e-6) # clamp to prevent divergences
            tau = t_0 - (r / c0)  # Refined delayed time

        r_hat = r_vec / r[:, np.newaxis]

        # Approximate insect position and forces at delayed time tau
        v_tau    = np.column_stack([np.interp(tau, t_s, insect['v'   ][:, i], left=0.0, right=0.0) for i in range(3)])
        a_tau    = np.column_stack([np.interp(tau, t_s, insect['a'   ][:, i], left=0.0, right=0.0) for i in range(3)])
        
        Q_tau    = np.interp(tau, t_s, insect['Q'   ], left=0.0, right=0.0)
        dQdt_tau = np.interp(tau, t_s, insect['dQdt'], left=0.0, right=0.0)

        M_vec = v_tau / c0
        M_r = np.sum(M_vec * r_hat, axis=1)
        M_sq = np.sum(M_vec**2, axis=1) # (N,)
        dMr_dt = np.sum((a_tau / c0) * r_hat, axis=1)
        
        doppler_factor = (1.0 - M_r)**2
        prefactor = 1.0 / (4.0 * np.pi * doppler_factor)

        if limit in ["both", "far"]:
            # far-field contribution
            p_far = prefactor / r * (dQdt_tau + Q_tau * (dMr_dt / (1.0 - M_r)))
            p_grid += p_far

        if limit in ["both", "near"]:
            # near-field contribution
            p_near = prefactor * c0 / r**2 * Q_tau * ((M_r - M_sq) / (1.0 - M_r))
            p_grid += p_near

    return X, Y, p_grid.reshape(X.shape)

def swarm_sound_dipole(x_det, swarm_data, fs=44100, c0=343.0, t_start=None, duration=np.inf, print_progress=False):
    """
    Computes acoustic pressure at a detector location using the Push (Source-Time) method.
        
        Parameters:
        -----------
        x_det : ndarray (3,)
            Detector position [x, y, z]
        swarm_data : list of dicts
            [{ 't': t_k, 'y': y_k, 'v': v_k, 'a': a_k, 'F': F_k, 'dFdt': dFdt_k }, ...]
        fs : int, default 44100
            Audio sample rate (Hz)
        c0 : float, default 343.0
            Speed of sound (m/s)
        
        Returns:
        --------
        t_global : ndarray of shape (N_samples)
            global time data
        p_total : ndarray of shape (N_samples)
            Pressure time series for the detector.
    """

    if t_start is None:
        t_start = min(data['t'][0] for data in swarm_data)
    
    t_max_data = max(data['t'][-1] for data in swarm_data)
    t_end = min(t_max_data, t_start + duration) if np.isfinite(duration) else t_max_data


    # max_dist = max(np.max(np.linalg.norm(x_det - insect['y'], axis=1)) for insect in swarm_data)
    # max_propagation_delay = (max_dist / c0) + 0.05  # Distance delay + 50 ms safety buffer
    max_propagation_delay = 0.0 # s

    t_global = np.arange(t_start, t_end + max_propagation_delay, 1.0 / fs)
    p_total = np.zeros_like(t_global)

    for iid,insect in enumerate(swarm_data):
        if print_progress:
            print(f"{iid+1} / {len(swarm_data)}")
        
        # obtain data range
        t_raw = insect['t'] # (N_samples,)
        t_idx = np.argmin(abs(t_raw - t_end))
        
        # extract source information
        t_s = t_raw[:t_idx+1]
        y_s = insect['y'][:t_idx+1] # (N, 3)
        v_s = insect['v'][:t_idx+1] # (N, 3)
        a_s = insect['a'][:t_idx+1] # (N, 3)
        F_s = insect['F'][:t_idx+1] # (N, 3)
        dFdt_s = insect['dFdt'][:t_idx+1] # (N, 3)

        r_vec = x_det - y_s # (N,3)
        r = np.linalg.norm(r_vec, axis=1) # distance/magnitude, shape (N,)
        # r = np.maximum(np.linalg.norm(r_vec, axis=1), 1e-6)
        r_hat = r_vec / r[:, np.newaxis] # to preserve shape (N,3), the new axis was created to apply the 1/r to each of the 3 components
    
        # M_r = rhat . M
        M_vec = v_s / c0  # mach number (N, 3)
        M_sq = np.sum(M_vec**2, axis=1) # (N,)
        M_r = np.sum(M_vec * r_hat, axis=1) # (N,)
    
        # acceleration along radiation direction
        a_vec = a_s / c0  # (N, 3)
        dMr_dt = np.sum(a_vec * r_hat, axis=1)  # (N,)
    
        # arrival time
        t_arrive = t_s + r / c0

        doppler_factor = (1.0 - M_r)**2
        prefactor = 1.0 / (4.0 * np.pi * doppler_factor)

        # far-field contribution
        F_prime_r = np.sum(r_hat * dFdt_s, axis=1) # (N,)
        F_r = np.sum(r_hat * F_s, axis=1) # (N,)
        p_far = prefactor / (c0 * r) * (F_prime_r + F_r * (dMr_dt / (1.0 - M_r)))

        # near-field contribution
        F_dot_M = np.sum(F_s * M_vec, axis=1)
        p_near = prefactor / r**2 * ( F_r * ((1.0 - M_sq) / (1.0 - M_r)) - F_dot_M )

        p_emit = p_near + p_far
        p_emit = np.nan_to_num(p_emit, nan=0.0, posinf=0.0, neginf=0.0)
        p_k = np.interp(t_global, t_arrive, p_emit, left=0.0, right=0.0) # ensure padding with zeros so there's no DC pressure off-time
        p_total += p_k # superposition

    return t_global, p_total

def get_2d_pressure_snapshot(swarm_data, t_0, x_range=(-2,2), y_range=(-2,2), res=0.02, slice=0.0, plane='xy', c0=343.0, fp_iterations=2, limit="both"):
    """
    t_0 : float
        time stamp of snapshot (global time, must be within time range of recording!)
    """
    x = np.arange(x_range[0], x_range[1]+res, res)
    y = np.arange(y_range[0], y_range[1]+res, res)
    X,Y = np.meshgrid(x, y)

    if plane == 'xz':
        # xz-plane
        grid_pts = np.column_stack([X.ravel(), np.full(X.size, slice), Y.ravel()])
    else:
        # xy-plane
        grid_pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, slice)])

    p_grid = np.zeros(X.size)

    for insect in swarm_data:
        t_s = insect['t']
        pos = insect['y']

        # approximate delayed time based off current positions
        # y_guess = np.column_stack([np.interp(t_0, t_s, insect['y'][:, i]) for i in range(3)])
        y_guess = np.column_stack([np.interp(t_0, t_s, pos[:, i]) for i in range(3)])
        r_vec = grid_pts - y_guess
        r = np.linalg.norm(r_vec, axis=1)
        tau = t_0 - (r / c0)

        # fixed-point iteration
        for _ in range(fp_iterations):
            # y_tau = np.column_stack([np.interp(tau, t_s, insect['y'][:, i]) for i in range(3)])
            y_tau = np.column_stack([np.interp(tau, t_s, pos[:, i]) for i in range(3)])
            r_vec = grid_pts - y_tau
            r = np.linalg.norm(r_vec, axis=1)
            # r = np.maximum(np.linalg.norm(r_vec, axis=1), 1e-6) # clamp to prevent divergences
            tau = t_0 - (r / c0)  # Refined delayed time

        r_hat = r_vec / r[:, np.newaxis]

        # Approximate insect position and forces at delayed time tau
        v_tau    = np.column_stack([np.interp(tau, t_s, insect['v'   ][:, i], left=0.0, right=0.0) for i in range(3)])
        a_tau    = np.column_stack([np.interp(tau, t_s, insect['a'   ][:, i], left=0.0, right=0.0) for i in range(3)])
        F_tau    = np.column_stack([np.interp(tau, t_s, insect['F'   ][:, i], left=0.0, right=0.0) for i in range(3)])
        dFdt_tau = np.column_stack([np.interp(tau, t_s, insect['dFdt'][:, i], left=0.0, right=0.0) for i in range(3)])

        M_vec = v_tau / c0
        M_sq = np.sum(M_vec**2, axis=1) # (N,)
        M_r = np.sum(M_vec * r_hat, axis=1)
        dMr_dt = np.sum((a_tau / c0) * r_hat, axis=1)

        doppler_factor = (1.0 - M_r)**2
        prefactor = 1.0 / (4.0 * np.pi * doppler_factor)

        F_r = np.sum(r_hat * F_tau, axis=1) # (N,)
        
        if limit in ["both", "far"]:
            # far-field contribution
            F_prime_r = np.sum(r_hat * dFdt_tau, axis=1) # (N,)
            p_far = prefactor / (c0 * r) * (F_prime_r + F_r * (dMr_dt / (1.0 - M_r)))
            p_grid += p_far

        if limit in ["both", "near"]:
            # near-field contribution
            F_dot_M = np.sum(F_tau * M_vec, axis=1)
            p_near = prefactor / r**2 * ( F_r * ((1.0 - M_sq) / (1.0 - M_r)) - F_dot_M )
            p_grid += p_near

    return X, Y, p_grid.reshape(X.shape)

if __name__ == "__main__":
    pre_path = "/Users/max/UCLA/Research/Codes"
    path = f"{pre_path}/Animations/"
    
    test = -99

    if test == -1:
        # FFT test!!!
        fs_source = 44100
        duration = 0.02  # 20 ms trajectory
        t_src = np.linspace(0, duration, int(fs_source * duration))
        N = len(t_src)

        zero_cols = np.zeros(N)
        one_cols = np.ones(N)
        
        mic_pos = np.array([0.0, 0.0, 0.0])

        f1 = 864.0 # Hz
        omega = 2.0 * np.pi * f1
        phi1 = 0
        # v_x1 = 50.0  # m/s
        # x1_traj = -4.0 + v_x1 * t_src

        F1 = np.sin(omega * t_src + phi1)
        dF1dt = omega * np.cos(omega * t_src + phi1)
        
        phi2 = 2*phi1 + np.pi / 2.0
        A2 = np.power(10, -12/20) / 2
        F2 = A2 * np.sin(2*omega * t_src + phi2)
        dF2dt = A2 * 2.0 * omega * np.cos(omega * t_src + phi2)

        swarm_data = [{
            't': t_src,
            'y': np.column_stack([zero_cols, zero_cols, 0.1 * one_cols]),
            'v': np.column_stack([zero_cols, zero_cols, zero_cols]),
            'a': np.zeros((N, 3)),
            'F': np.column_stack([zero_cols, zero_cols, F1 + F2]),
            'dFdt': np.column_stack([zero_cols, zero_cols, dF1dt + dF2dt]),
        }]

        t_audio,p = swarm_sound_dipole(mic_pos, swarm_data)
        
        fig = plt.figure(figsize=(10,6))

        plt.plot(1e3 * t_audio, p)

        plt.xlabel("Time [ms]")
        plt.ylabel("Pressure [Pa]")
        plt.grid(True)
        plt.tight_layout()

        plt.show()

    elif test == 0:
        # t_target = 0.080 # t = 20 ms field
        
        mic_pos = np.array([0.0, -5.0, 0.0])

        fs_source = 44100
        duration = 0.2 # s
        t = np.linspace(0, duration, int(fs_source * duration))

        wingbeat_freq = 400 # Hz
        omega = 2.0 * np.pi * wingbeat_freq
        phase = 0 # rad

        zero_cols = np.zeros_like(t)
        one_cols = np.ones_like(t)
        
        F_x = np.sin(omega * t + phase)
        F_y = zero_cols
        F_z = zero_cols

        dFdt_x = omega * np.cos(omega * t + phase)
        dFdt_y = zero_cols
        dFdt_z = zero_cols

        N_samples = len(t)
        F_matrix = F_matrix = np.column_stack([F_x, F_y, F_z])
        dFdt_matrix = np.column_stack([dFdt_x, dFdt_y, dFdt_z])

        v_x = 200.0  # m/s
        x_pos = -10.0 + v_x * t  # Shape (N_samples,)
        y1 = np.column_stack([x_pos, zero_cols, zero_cols])
        v1 = np.column_stack([v_x * one_cols, zero_cols, zero_cols])
        # y2 = np.column_stack([5.0 * one_cols, 5.0 * one_cols, zero_cols])

        source  = [
            {
                't': t,
                'y': y1,
                'v': v1,
                'a': np.zeros((N_samples, 3)),
                'F': F_matrix,
                'dFdt': dFdt_matrix,
            },
            # {
            #     't': t,
            #     'y': y2,
            #     'v': np.zeros((N_samples, 3)),
            #     'a': np.zeros((N_samples, 3)),
            #     'F': F_matrix,
            #     'dFdt': dFdt_matrix,
            # },
        ]

        # X, Y, P_2d = get_2d_pressure_snapshot(source, t_0=t_target, x_range=(-2, 2), y_range=(-2, 2), res=200)

        # --- 3. Pre-compute 3D Data Stack (num_frames, res, res)
        x_rng, y_rng = (-2.0, 2.0), (-2.0, 2.0)

        # plot_type = '2d snapshot'
        # plot_type = 'push pull compare'
        plot_type = 'snapshot convergence'


        if plot_type == 'anim':
            res = 0.02
            dt_frame = 0.0002  # 0.2 ms per frame (captures 10 frames per 500 Hz wave cycle)
            t_frames = np.arange(0.0, 0.090, dt_frame)  # Evaluate from 0 ms to 90 ms
            num_frames = len(t_frames)
            for fpi in [2]:
                print(f"Pre-computing {num_frames} frames...")
                P_stack = np.zeros((num_frames, res, res))
                
                for i, t_val in enumerate(t_frames):
                    X, Y, P_stack[i] = get_2d_pressure_snapshot(
                        source, t_0=t_val, x_range=x_rng, y_range=y_rng, res=res, fp_iterations=fpi
                    )
                print("Pre-computation complete!")

                fig, ax = plt.subplots(figsize=(8, 7))
                
                vmax = np.max(np.abs(P_stack)) / 40.0
                if vmax == 0: vmax = 1.0

                # initialize plot on first frame
                mesh = ax.pcolormesh(X, Y, P_stack[0], cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
                cbar = fig.colorbar(mesh, ax=ax, label='Pressure [Pa]')

                # Plot insect marker
                insect_dot, = ax.plot([], [], 'ko', markersize=8, label='Source')
                
                ax.set_xlabel('x [m]')
                ax.set_ylabel('y [m]')
                ax.set_xlim(x_rng)
                ax.set_ylim(y_rng)
                ax.legend(loc='upper right')

                def update(frame_idx):
                    t_curr = t_frames[frame_idx]
                    
                    # Update pressure mesh array
                    mesh.set_array(P_stack[frame_idx].ravel())
                    
                    # Update insect position dot
                    x_inst = np.interp(t_curr, t, source[0]['y'][:, 0])
                    y_inst = np.interp(t_curr, t, source[0]['y'][:, 1])
                    insect_dot.set_data([x_inst], [y_inst])
                    
                    ax.set_title(f'Acoustic Pressure Field (t = {t_curr * 1e3:.1f} ms)')
                    return mesh, insect_dot

                anim = FuncAnimation(fig, update, frames=num_frames, interval=30, blit=False)
                
                plt.show()
                
                # print(f"Saving...")
                # anim.save(f"{path}soundfield_fast_moving_dipole_fpi_{fpi}.mp4", fps=30, extra_args=['-vcodec', 'libx264'])
                # print(f"Done!")
        elif plot_type == '2d snapshot':
            res=0.005
            t_target = 49.8 * 1e-3 # target time
            for fpi in [1, 2, 5, 10, 15]:
            # for fpi in np.arange(2,3):
                X, Y, P = get_2d_pressure_snapshot(source, t_0=t_target, x_range=x_rng, y_range=y_rng, res=res, fp_iterations=fpi)

                fig, ax = plt.subplots(figsize=(8, 7))

                vmax = 2 # np.max(np.abs(P)) / 20.0 # saturate peaks for visability
                if vmax == 0: vmax = 1.0

                # initialize plot on first frame
                mesh = ax.pcolormesh(X, Y, P, cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
                cbar = fig.colorbar(mesh, ax=ax, label='Pressure [Pa]')

                # Plot insect marker
                xi = np.interp(t_target, t, source[0]['y'][:, 0])
                yi = np.interp(t_target, t, source[0]['y'][:, 1])
                insect_dot, = ax.plot(xi,yi, 'ko', markersize=3.25, label='Dipole')
                
                ax.set_xlabel('x [m]')
                ax.set_ylabel('y [m]')
                ax.set_xlim(x_rng)
                ax.set_ylim(y_rng)
                ax.legend(loc='upper right')
                ax.set_title(f'Acoustic Pressure Field (t = {t_target * 1e3:.1f} ms)\n{fpi} fixed-point iteration{['s' if fpi != 1 else ''][0]}')
                
                # plt.show()

                print(f"Saving iteration {fpi}...")
                plt.savefig(f"{pre_path}/Figures/fpi_convergence/soundfield_fast_moving_dipole_fpi_{fpi}.png", dpi=800)
                print("Done!")
                plt.close()

        elif plot_type == 'push pull compare':
            d = 0.5
            mic_pos = np.array([d, d, 0.0])
            t_audio, p_1D = swarm_sound_dipole(mic_pos, source)

            # Extract single pixel from 2D snapshots across time
            p_2D_extracted = []
            sample_times = t_audio[::10]  # Subsample every n-th frame to save time

            for t_val in sample_times:
                X, Y, P_grid = get_2d_pressure_snapshot(
                    source, t_0=t_val, x_range=(0, 1), y_range=(0, 1), res=0.02,
                    fp_iterations=2
                )
                # Pick pixel closest to mic_pos
                i_x = np.argmin(np.abs(X[0, :] - mic_pos[0]))
                i_y = np.argmin(np.abs(Y[:, 0] - mic_pos[1]))
                p_2D_extracted.append(P_grid[i_y, i_x])

            fig = plt.figure(figsize=(10,8))
            plt.plot(t_audio * 1e3, p_1D, label='Push (1D Solver)')
            plt.plot(sample_times * 1e3, p_2D_extracted, '--', lw=1, label='Pull (2D Snapshot)')
            plt.xlabel('Time [ms]')
            plt.ylabel('Pressure [Pa]')
            plt.legend()
            plt.show()
        elif plot_type == 'snapshot convergence':
            res = 0.005

            t_target = 49.8 * 1e-3 # target time

            ref_fpi = 30
            d = np.arange(0.5, 10.5, 0.5)

            cmap_l = plt.colormaps['jet']
            colors = cmap_l(np.linspace(0,1, len(d)))
            cmap_name = "jet"  # Replace with the colormap you used to make colors
            cmap = plt.get_cmap(cmap_name)
            norm = mcolors.Normalize(vmin=min(d), vmax=max(d))

            fpi_range = np.arange(1,ref_fpi)
            rel_error = np.zeros((len(d),len(fpi_range)))
            
            for i,dlim in enumerate(d):
                print(f"Distance {i+1}/{len(d)}")
                x_rng, y_rng = (-dlim, dlim), (-dlim, dlim)
                X_ref,Y_ref,P_ref = get_2d_pressure_snapshot(source, t_0=t_target, x_range=x_rng, y_range=y_rng, res=res, fp_iterations=ref_fpi)

                for j,fpi in enumerate(fpi_range):
                    print(f"Current fpi = {fpi}")
                    X, Y, P = get_2d_pressure_snapshot(source, t_0=t_target, x_range=x_rng, y_range=y_rng, res=res, fp_iterations=fpi)

                    # L2 error of the pressure field relative to reference
                    rel_error[i,j] = np.linalg.norm(P - P_ref) / np.linalg.norm(P_ref)

            
            fig,ax = plt.subplots(figsize=(10,8))
            
            for i in range(len(d)):
                ax.scatter(fpi_range, rel_error[i,:], color=colors[i], s=10)
            ax.set_xlabel("Fixed-point iterations")
            ax.set_ylabel(f"Relative $L^2$ error from reference ({ref_fpi})")
            ax.set_yscale('log')

            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, pad=0.02)
            cbar.set_label("Domain extent [m]")

            # plt.legend(title="Domain extent (m)", ncols=2)
            # plt.title(fr"Domain size: {2*dlim**2} m$^2$")
            # plt.xticks(fpi_range)
            plt.title(f"Mach {v_x / 343.0:.2f}")
            
            plt.savefig(f"{pre_path}/Figures/fpi_convergence/soundfield_fast_moving_dipole_fpi_convergence_with_distance.pdf", dpi=500)
            plt.show()

    elif test == 1:
        mic_pos = np.array([0.0, -5.0, 0.0])
        
        fs_source = 44100
        duration = 0.1 # 20 ms window
        t = np.linspace(0, duration, int(fs_source * duration))

        wingbeat_freq = 500 # Hz
        omega = 2.0 * np.pi * wingbeat_freq
        phase = 0 # rad

        zero_cols = np.zeros_like(t)
        one_cols = np.ones_like(t)
        
        F_x = np.sin(omega * t + phase)
        F_y = np.sin(omega * t + phase)
        F_z = zero_cols

        dFdt_x = omega * np.cos(omega * t + phase)
        dFdt_y = omega * np.cos(omega * t + phase)
        dFdt_z = zero_cols

        N_samples = len(t)
        F_matrix = F_matrix = np.column_stack([F_x, F_y, F_z])
        dFdt_matrix = np.column_stack([dFdt_x, dFdt_y, dFdt_z])

        v_x = 100.0  # m/s
        x_pos = -10.0 + v_x * t  # Shape (N_samples,)
        y1 = np.column_stack([x_pos, zero_cols, zero_cols])
        v1 = np.column_stack([v_x * one_cols, zero_cols, zero_cols])
        # y2 = np.column_stack([5.0 * one_cols, 5.0 * one_cols, zero_cols])

        source  = [
            {
                't': t,
                'y': y1,
                'v': v1,
                'a': np.zeros((N_samples, 3)),
                'F': F_matrix,
                'dFdt': dFdt_matrix,
            },
            # {
            #     't': t,
            #     'y': y2,
            #     'v': np.zeros((N_samples, 3)),
            #     'a': np.zeros((N_samples, 3)),
            #     'F': F_matrix,
            #     'dFdt': dFdt_matrix,
            # },
        ]

        t_audio,p = swarm_sound_dipole(mic_pos, source)

        fig = plt.figure(figsize=(10,6))

        plt.plot(1e3 * t_audio, p)

        plt.axvline(np.sqrt(2) * 5.0 / v_x * 1e3, 0, 1, c='k', ls='--')

        plt.xlabel("Time [ms]")
        plt.ylabel("Pressure [Pa]")
        plt.grid(True)
        plt.tight_layout()

        plt.show()

    elif test == 2:
        fs_source = 44100
        duration = 0.20  # 200 ms trajectory
        t_src = np.linspace(0, duration, int(fs_source * duration))
        N = len(t_src)

        zero_cols = np.zeros(N)
        one_cols = np.ones(N)

        # --- Agent 1: Flying Left-to-Right (+x) at y = +0.5 m (500 Hz) ---
        f1 = 500.0
        w1 = 2.0 * np.pi * f1
        v_x1 = 50.0  # m/s
        x1_traj = -4.0 + v_x1 * t_src
        
        phase1 = -np.pi / 2.0  # Smooth zero-pressure start
        F1_x = np.sin(w1 * t_src + phase1)
        F1_y = np.sin(w1 * t_src + phase1)
        dF1dt_x = w1 * np.cos(w1 * t_src + phase1)
        dF1dt_y = w1 * np.cos(w1 * t_src + phase1)

        agent1 = {
            't': t_src,
            'y': np.column_stack([x1_traj, 0.5 * one_cols, zero_cols]),
            'v': np.column_stack([v_x1 * one_cols, zero_cols, zero_cols]),
            'a': np.zeros((N, 3)),
            'F': np.column_stack([F1_x, F1_y, zero_cols]),
            'dFdt': np.column_stack([dF1dt_x, dF1dt_y, zero_cols]),
        }

        # --- Agent 2: Flying Right-to-Left (-x) at y = -0.5 m (505 Hz) ---
        f2 = 505.0  # 5 Hz beat offset
        w2 = 2.0 * np.pi * f2
        v_x2 = -50.0  # m/s
        x2_traj = 4.0 + v_x2 * t_src

        phase2 = -np.pi / 2.0
        F2_x = np.sin(w2 * t_src + phase2)
        F2_y = -np.sin(w2 * t_src + phase2)
        dF2dt_x = w2 * np.cos(w2 * t_src + phase2)
        dF2dt_y = w2 * -np.cos(w2 * t_src + phase2)

        agent2 = {
            't': t_src,
            'y': np.column_stack([x2_traj, -0.5 * one_cols, zero_cols]),
            'v': np.column_stack([v_x2 * one_cols, zero_cols, zero_cols]),
            'a': np.zeros((N, 3)),
            'F': np.column_stack([F2_x, F2_y, zero_cols]),
            'dFdt': np.column_stack([dF2dt_x, dF2dt_y, zero_cols]),
        }

        swarm_data = [agent1, agent2]

        # --- 2. Pre-compute Frame Stack (Slow-Motion) ---
        dt_frame = 0.00025  # 0.25 ms per frame (8 frames per 500 Hz wave cycle)
        t_frames = np.arange(0.00, 0.140, dt_frame)
        num_frames = len(t_frames)

        res = 160
        x_rng, y_rng = (-2.5, 2.5), (-2.5, 2.5)

        print(f"Pre-computing {num_frames} spatial pressure snapshots...")
        P_stack = np.zeros((num_frames, res, res))

        for i, t_val in enumerate(t_frames):
            X, Y, P_stack[i] = get_2d_pressure_snapshot(
                swarm_data, t_0=t_val, x_range=x_rng, y_range=y_rng, res=res
            )
        print("Pre-computation finished! Rendering animation...")

        # --- 3. Build Animation Figure ---
        fig, ax = plt.subplots(figsize=(8,6), layout='constrained')

        # Calculate colorbar symmetry scale
        vmax = np.max(np.abs(P_stack)) * 0.1  # saturate peaks for contrast
        if vmax == 0: vmax = 1.0

        mesh = ax.pcolormesh(X, Y, P_stack[0], cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
        cbar = fig.colorbar(mesh, ax=ax, label='Acoustic Pressure [Pa]', extend='both')

        # Mosquito markers
        m1_dot, = ax.plot([], [], 'ko', markersize=8, label=r'Source 1 (500 Hz)')
        m2_dot, = ax.plot([], [], 'mo', markersize=8, label=r'Source 2 (505 Hz)')

        ax.set_xlabel('x [m]')
        ax.set_ylabel('z [m]')
        ax.set_xlim(x_rng)
        ax.set_ylim(y_rng)
        ax.legend(loc='upper right')

        def update(frame_idx):
            t_curr = t_frames[frame_idx]

            # Update 2D colormap mesh
            mesh.set_array(P_stack[frame_idx].ravel())

            # Update mosquito positions
            x1_now = np.interp(t_curr, t_src, agent1['y'][:, 0])
            y1_now = np.interp(t_curr, t_src, agent1['y'][:, 1])
            x2_now = np.interp(t_curr, t_src, agent2['y'][:, 0])
            y2_now = np.interp(t_curr, t_src, agent2['y'][:, 1])

            m1_dot.set_data([x1_now], [y1_now])
            m2_dot.set_data([x2_now], [y2_now])

            ax.set_title(f'Counter-Propagating Pass-By (t = {t_curr * 1e3:.1f} ms)')
            return mesh, m1_dot, m2_dot

        anim = FuncAnimation(fig, update, frames=num_frames, interval=25, blit=False)
        
        plt.show()

        # anim.save(f"{path}soundfield_counter_prop_dipole_interference.mp4", fps=30, extra_args=['-vcodec', 'libx264'])
