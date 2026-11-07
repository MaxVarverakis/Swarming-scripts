import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline, CubicSpline
import h5py

fname = "/Users/max/UCLA/Research/Codes/Data/Poda2024/Database S1/all_data.mat"

pre_path = "/Users/max/UCLA/Research/Codes"
path = f"{pre_path}/Animations/"

class mat:
    def __init__(self, filename: str, seed=None):
        self.fps = 50
        self.dt = 1/self.fps # s
        self.f = h5py.File(filename)
        self.raw_data = self.f["all_data"]
        self.trials = list(self.raw_data['exp'].keys())
        self.objs = { i : list(self.raw_data['exp'][t]['tracks']['object'].keys()) for i,t in enumerate(self.trials) }
        self.metadata = self.raw_data['exp'][self.trials[0]]['metadata']
        self.metadata_keys = list(self.metadata.keys())
        self.data_keys = list(self.raw_data['exp'][self.trials[0]]['tracks']['object'][self.objs[0][0]].keys())
        self.scalar_keys = list(self.raw_data['exp'][self.trials[0]]['tracks']['object'][self.objs[0][0]]['scalars'].keys())
        self.max_trials = len(self.trials)
        self.max_obj = [ len(self.objs[i]) for i in range(self.max_trials) ]
        self.rng = np.random.default_rng(seed=1243 if seed is None else seed)

    def obj_ids(self, trial: int):
        return list(range(self.max_obj[trial]))

    def t_start(self, trial: int):
        return min(self.time(trial, i)[0] for i in range(self.max_obj[trial])) # s

    def generate_wbf_drift(self, t_audio, drift_std_Hz=15.0):
        """
        Generates smooth, amplitude-preserved wingbeat frequency drift directly on t_audio.
        """

        duration = t_audio[-1] - t_audio[0]

        # ensure control points strictly cover [0, duration + buffer]
        # semi-randomly distribute control points for more peak width variation
        t_control = [ 0.0 ]
        while t_control[-1] < duration:
            control_step = self.rng.uniform(0.1, 0.35)
            t_control.append(t_control[-1] + control_step)
        # t_control[-1] = duration # uncomment if desire no buffer
        
        drift_control = self.rng.normal(0.0, drift_std_Hz, size=len(t_control))
        
        # control_dt = 0.08
        # t_control = np.arange(0.0, duration + 2.0 * control_dt, control_dt)
        # drift_control = self.rng.uniform(-max_drift_hz, max_drift_hz, size=len(t_control))

        # fit spline over control points
        cs = CubicSpline(t_control, drift_control, bc_type='natural')

        # evaluate directly on normalized t_audio grid
        t_rel = t_audio - t_audio[0]
        return cs(t_rel)

    def format_raw_insect_data(self, trial: int, objno: int, wbf: float=864.0, mass=1e-6, n_harmonics=1):
        """
        Parameters:
        -----------
        trial : int between 0 and 5
            trial number from experimental data
        objno : int between 0 and max_obj[trial]-1
            object/insect of interest
        wbf : float, default 864.0 Hz
            wingbeat frequency in Hz
        mass : float, default 1 mg = 10^-6 kg
            approximate mass in kg
        n_harmonics : int, default 1 (minimum)
            number of harmonics included in forcing function (currently max is 3)

        Returns:
        --------
        insect_data : dict
            contains 't', 'y', 'v', 'a', 'F', 'dFdt'
        """
        t = self.time(trial, objno)
        x = np.column_stack((self.x(trial, objno),self.y(trial, objno),self.z(trial, objno))) # shape (max_step, 3)
        # x = np.column_stack((0.0 * self.x(trial, objno),0.0 * self.y(trial, objno),0.0 * self.z(trial, objno))) # shape (max_step, 3)
        v = np.column_stack((self.x_vel(trial, objno),self.y_vel(trial, objno),self.z_vel(trial, objno))) # shape (max_step, 3)
        # v = np.column_stack((0.0 * self.x_vel(trial, objno),0.0 * self.y_vel(trial, objno),0.0 * self.z_vel(trial, objno))) # shape (max_step, 3)
        a = np.gradient(v, self.dt, axis=0)  # shape (max_step, 3)
        g_vec = np.array([0.0, 0.0, -9.81])
        thrust_vec = a - g_vec
        thrust_mag = np.linalg.norm(thrust_vec, axis=1, keepdims=True) # shape (max_step)
        nhat = thrust_vec / thrust_mag
        # v_mag = np.linalg.norm(v, axis=1, keepdims=True) # shape (max_step)
        # nhat = v / v_mag

        omega = 2.0 * np.pi * wbf
        # for debugging
        # rng = np.random.default_rng(seed=42)
        # rng = np.random.default_rng(seed=objno)
        phi1 = (2.0 * self.rng.random() - 1.0) * np.pi # [-pi, pi) # with fixed seed, each call is still distinct
        # phi1 = (2.0 * np.random.rand() - 1.0) * np.pi # [-pi, pi)
        phi1 = np.pi / 2.0
        F_mag = mass * thrust_mag.squeeze()
        F_scalar = F_mag * np.sin(omega * t + phi1)
        dFdt_scalar = F_mag * omega * np.cos(omega * t + phi1)

        # at some point should consider adding second harmonic to F (but should refer to literature on separate orientation and phase locking!)
        Delta_dB = [ -13, -25 ] # n-th harmonic change in dB from fundamental (Arthur 2014 Fig. 4)
        phases = [ np.pi / 2.0, 0.0 ] # sus, needs to be thoroughly studied
        # 2nd harmonic allegedly has max drag normal which is distinct from max thrust

        for n in range(2, n_harmonics+1):
            phi_n = n * phi1 + phases[n-2] # phase-locked
            An = np.power(10, Delta_dB[n-2]/20) / n
            Fn = An * F_mag * np.sin(n*omega * t + phi_n)
            dFndt = An * F_mag * n * omega * np.cos(n*omega * t + phi_n)
            F_scalar += Fn
            dFdt_scalar += dFndt

        F = F_scalar[:, np.newaxis] * nhat
        dFdt = dFdt_scalar[:, np.newaxis] * nhat

        return {
            't': t,
            'y': x,
            'v': v,
            'a': a,
            'F': F,
            'dFdt': dFdt,
        }

    def format_smooth_insect_data(self, trial: int, objno: int, window_length: int = 15, polyorder: int = 3, wbf: float=864.0, mass=1e-6, n_harmonics=1, target_fs=44100, c0=343.0, t_start=None, duration=None, monopole=False):
        """
        Processes raw swarm data, smoothens the position data, 
        and then interpolates to the desired sampling 
        frequency using quintic splines. Then the velocity 
        and acceleration are calculated from the interpolated
        position data. Finally, the forcing functions for an
        acoustic dipole are applied with the desired number
        of harmonics of the wingbeat frequency.


        Parameters:
        -----------
        trial : int between 0 and 5
            trial number from experimental data
        objno : int between 0 and max_obj[trial]-1
            object/insect of interest
        window_length : int (default 15)
            length of window for Savitzky-Golay filter
        polyorder : int (default 3)
            polynomial order for Savitzky-Golay filter
        wbf : float, default 864.0 Hz
            wingbeat frequency in Hz
        mass : float, default 1 mg = 10^-6 kg = 1 mg
            approximate mass in kg
        n_harmonics : int, default 1 (minimum)
            number of harmonics included in forcing function (currently max is 3)
        target_fs : float or int
            target sampling frequency for interpolation to produce realistic audio
        t_start : float, optional
            start time in seconds to return data for
        duration : float, optional
            duration in seconds for the dataset

        Returns:
        --------
        insect_data : dict
            contains 't', 'y', 'v', 'a', 'F', 'dFdt'
        """
        t_raw = self.time(trial, objno)
        x_raw = np.column_stack((self.x(trial, objno), self.y(trial, objno), self.z(trial, objno))) # shape (max_step, 3)

        half_win = window_length//2

        if t_start is not None and duration is not None:
            t_end = t_start + duration
            # include a buffer margin to prevent SG boundary trimming from shortening the requested window
            margin = (half_win + 2) * self.dt
            slice_mask = (t_raw >= t_start - margin) & (t_raw <= t_end + margin)

            t_raw = t_raw[slice_mask]
            x_raw = x_raw[slice_mask]

            if len(t_raw) == 0:
                return []
        
        else: t_end = None

        # remove NaNs
        valid_frames = ~np.isnan(x_raw).any(axis=1)
        v_mask = valid_frames.copy()

        # trim dataset where SG filter lacks window support
        for i in range(half_win, len(t_raw) - half_win):
            if np.sum(valid_frames[i - half_win : i + half_win + 1]) < window_length:
                v_mask[i] = False
        
        # trim edges
        v_mask[:half_win] = False
        v_mask[-half_win:] = False
        
        # extract longest contiguous segment of valid tracking
        valid_indices = np.where(v_mask)[0]
        if len(valid_indices) == 0:
            # raise ValueError(f"No valid trajectory points for trial {trial}, object {objno} after mask trimming.")
            return []

        splits = np.where(np.diff(valid_indices) > 1)[0] + 1
        segments = np.split(valid_indices, splits)
        # longest_seg = max(segments, key=len)

        # t_clean = t_raw[longest_seg]
        # x_clean = x_raw[longest_seg] # contiguous 3D positions

        # ensure window length is odd and does not exceed segment length
        # seg_len = len(longest_seg)
        # effective_win = min(window_length, seg_len if seg_len % 2 != 0 else seg_len - 1)
        # if effective_win < polyorder + 2:
        #     # raise ValueError(f"Longest segment for object {objno} ({seg_len} frames) is too short for spline fitting.")
        #     return

        formatted_segments = []
        for seg in segments:
            if (seg_len := len(seg)) < 6: continue
            
            # ensure window length is odd and does not exceed segment length
            effective_win = min(window_length, seg_len if seg_len % 2 != 0 else seg_len - 1)
            
            # Skip micro-segments shorter than window requirement
            if effective_win < polyorder + 2:
                continue

            t_clean = t_raw[seg]
            x_clean_raw = x_raw[seg]

            # Smooth and spline *this* segment only
            x_clean = np.column_stack([
                savgol_filter(x_clean_raw[:, dim], window_length=effective_win, polyorder=polyorder, delta=self.dt, mode='interp')
                for dim in range(3)
            ])

            t_audio_start = max(t_clean[0], t_start) if t_start is not None else t_clean[0]
            t_audio_end = min(t_clean[-1], t_end) if t_end is not None else t_clean[-1]
            if t_audio_start >= t_audio_end: continue

            t_audio = np.arange(t_audio_start, t_audio_end, 1.0 / target_fs)
            if len(t_audio) == 0: continue

            # # smooth position data with Savitzky-Golay filter
            # x_clean = np.column_stack([
            #     savgol_filter(x_clean[:,dim], window_length=effective_win, polyorder=polyorder, delta=self.dt, mode='interp') for dim in range(3)
            # ])
            # x_clean = x_clean * v_mask[..., None]
            # t_clean = t_raw[v_mask]

            # t_audio = np.arange(t_clean[0], t_clean[-1], 1.0 / target_fs)
            x_audio = np.zeros((len(t_audio), 3))
            v_audio = np.zeros((len(t_audio), 3))
            a_audio = np.zeros((len(t_audio), 3))

            # interpolate with quintic splines (ensures data is C^4 continuous)
            for dim in range(3):
                spl = UnivariateSpline(t_clean, x_clean[:, dim], k=5, s=1e-6)

                x_audio[:, dim] = spl(t_audio)
                v_audio[:, dim] = spl.derivative(n=1)(t_audio)
                a_audio[:, dim] = spl.derivative(n=2)(t_audio)

            g_vec = np.array([0.0, 0.0, -9.81])
            thrust_vec = a_audio - g_vec
            thrust_mag = np.linalg.norm(thrust_vec, axis=1, keepdims=True) # shape (max_step)
            nhat = thrust_vec / thrust_mag
            # v_mag = np.linalg.norm(v, axis=1, keepdims=True) # shape (max_step)
            # nhat = v / v_mag

            wbf_std = 35 # Hz
            wbf_base = self.rng.normal(wbf, wbf_std)

            # generate smooth continuous pitch drift
            # wbf_drift = self.generate_wbf_drift(t_audio, max_drift_hz=40.0, control_dt=0.08)
            wbf_drift = self.generate_wbf_drift(t_audio, drift_std_Hz=15.0)
            f_inst = wbf_base + wbf_drift

            omega_inst = 2.0 * np.pi * f_inst
            
            dt_audio = 1.0 / target_fs
            phi1 = (2.0 * self.rng.random() - 1.0) * np.pi # [-pi, pi)
            phase_fundamental = 2.0 * np.pi * np.cumsum(f_inst) * dt_audio + phi1

            F_mag = mass * thrust_mag.squeeze()
            F_scalar = F_mag * np.sin(phase_fundamental)
            dFdt_scalar = F_mag * omega_inst * np.cos(phase_fundamental)

            # at some point should consider adding second harmonic to F (but should refer to literature on separate orientation and phase locking!)
            dropoff = 1
            Delta_dB = [ -13/dropoff, -25/dropoff , -37/dropoff ] # approximate n-th harmonic change in dB from fundamental (Arthur 2014 Fig. 4)
            phases = [ np.pi / 2.0, 0.0 , 0.0] # sus, needs to be thoroughly studied
            # 2nd harmonic allegedly has max drag normal which is distinct from max thrust
            
            n_harmonics = min(n_harmonics, len(Delta_dB) + 1)
            for n in range(2, n_harmonics+1):
                phi_n = n * phi1 + phases[n-2] # phase-locked
                phase_n = n * (phase_fundamental - phi1) + phi_n

                An = np.power(10, Delta_dB[n-2]/20) / n
                Fn = An * F_mag * np.sin(phase_n)
                dFndt = An * F_mag * n * omega_inst * np.cos(phase_n)
                F_scalar += Fn
                dFdt_scalar += dFndt

            if monopole:
                formatted_segments.append({
                    't': t_audio,
                    'y': x_audio,
                    'v': v_audio,
                    'a': a_audio,
                    'Q': F_scalar / c0,
                    'dQdt': dFdt_scalar / c0,
                })
            else:
                F = F_scalar[:, np.newaxis] * nhat
                dFdt = dFdt_scalar[:, np.newaxis] * nhat

                formatted_segments.append({
                    't': t_audio,
                    'y': x_audio,
                    'v': v_audio,
                    'a': a_audio,
                    'F': F,
                    'dFdt': dFdt,
                })

        return formatted_segments

    def format_all_smooth_insects(self, trial:int, **kwargs):
        """
        Processes all insects in a trial and returns a clean list 
        containing only valid, successfully formatted tracks.
        """
        valid_insects = []
        for obj in self.obj_ids(trial):
            data_set = self.format_smooth_insect_data(trial, obj, **kwargs)
            for data in data_set:
                if data is not None:
                    valid_insects.append(data)
                
        return valid_insects

    def swarm_activity(self, swarm_data, dt=0.01):
        """
        Finds the time window with the maximum number of simultaneously active mosquitoes.

        Parameters:
        -----------
        swarm_data : list of dicts
            List of formatted sub-track dictionaries from mosquito data (each containing 't' key).
        dt : float, default 0.01 s
            Time resolution for counting active insects.
            
        Returns:
        --------
        summary : dict
            Contains 'max_mosquitoes', 'peak_time_start', 'peak_time_end', 't_grid', 'active_counts'
        """

        if not swarm_data:
            return {'max_mosquitoes': 0, 'peak_time_start': None, 'peak_time_end': None, 't_grid': None, 'active_counts': None}

        t_min = min(track['t'][0] for track in swarm_data)
        t_max = max(track['t'][-1] for track in swarm_data)
        t_grid = np.arange(t_min, t_max + dt, dt)

        active_counts = np.zeros(len(t_grid), dtype=int)

        for track in swarm_data:
            # use mask to accumulate which frames each track covers
            mask = (t_grid >= track['t'][0]) & (t_grid <= track['t'][-1])
            active_counts += mask

        max_count = np.max(active_counts)
        peak_indices = np.where(active_counts == max_count)[0]
        t_peak_start = t_grid[peak_indices[0]]
        t_peak_end = t_grid[peak_indices[-1]]

        return {
            'max_mosquitoes': max_count,
            'peak_time_start': t_peak_start,
            'peak_time_end': t_peak_end,
            't_grid': t_grid,
            'active_counts': active_counts,
        }

    def generate_test_insect(self, duration, wbf: float=864.0, n_harmonics=1, mass=1e-6, target_fs=44100, c0=343.0, t_start=0, monopole=False):
        """
        Generates test insect data at rest to obtain on-axis snapshot of SPL to compare near, far, and full solutions

        Returns:
        --------
        dict containing 't', 'y', 'v', 'a', 'F', 'dFdt'
        """

        t_audio = np.arange(t_start, t_start + duration, 1.0 / target_fs)

        # for now, generate static dipole
        # can always change to be moving by setting
        # velocity and calculating x = v t + x0
        x_audio = np.zeros((len(t_audio), 3))
        v_audio = np.zeros((len(t_audio), 3))
        a_audio = np.zeros((len(t_audio), 3))

        g_vec = np.array([0.0, 0.0, -9.81])
        thrust_vec = a_audio - g_vec
        thrust_mag = np.linalg.norm(thrust_vec, axis=1, keepdims=True) # shape (max_step)
        nhat = thrust_vec / thrust_mag

        omega_inst = 2.0 * np.pi * wbf

        phi1 = 0.0 # (2.0 * self.rng.random() - 1.0) * np.pi # [-pi, pi)
        phase_fundamental = 2.0 * np.pi * wbf * t_audio + phi1

        F_mag = mass * thrust_mag.squeeze()
        F_scalar = F_mag * np.sin(phase_fundamental)
        dFdt_scalar = F_mag * omega_inst * np.cos(phase_fundamental)

        # at some point should consider adding second harmonic to F (but should refer to literature on separate orientation and phase locking!)
        dropoff = 1
        Delta_dB = [ -13/dropoff, -25/dropoff , -37/dropoff ] # approximate n-th harmonic change in dB from fundamental (Arthur 2014 Fig. 4)
        phases = [ np.pi / 2.0, 0.0 , 0.0] # sus, needs to be thoroughly studied
        # 2nd harmonic allegedly has max drag normal which is distinct from max thrust
        
        n_harmonics = min(n_harmonics, len(Delta_dB) + 1)
        for n in range(2, n_harmonics+1):
            phi_n = n * phi1 + phases[n-2] # phase-locked
            phase_n = n * (phase_fundamental - phi1) + phi_n

            An = np.power(10, Delta_dB[n-2]/20) / n
            Fn = An * F_mag * np.sin(phase_n)
            dFndt = An * F_mag * n * omega_inst * np.cos(phase_n)
            F_scalar += Fn
            dFdt_scalar += dFndt

        if monopole:
            return {
                't': t_audio,
                'y': x_audio,
                'v': v_audio,
                'a': a_audio,
                'Q': F_scalar / c0,
                'dQdt': dFdt_scalar / c0,
            }
        
        else:
            F = F_scalar[:, np.newaxis] * nhat
            dFdt = dFdt_scalar[:, np.newaxis] * nhat

            return {
                't': t_audio,
                'y': x_audio,
                'v': v_audio,
                'a': a_audio,
                'F': F,
                'dFdt': dFdt,
            }

    def max_step(self, trial: int, objno: int):
        return len(self.x(trial, objno))

    def get(self, key: str, trial: int, objno: int):
        """Core data extraction method."""
        data = self.raw_data['exp'][self.trials[trial]]['tracks']['object'][self.objs[trial][objno]][key]
        if key == 'scalars':
            return self.raw_data['exp'][self.trials[trial]]['tracks']['object'][self.objs[trial][objno]]['scalars']
        else:
            return np.array(data).flatten() # flatten since data is in unnecessary shape (N,1)

    def __getattr__(self, name: str):
        """Dynamically routes method calls for any key in self.data_keys."""
        if name in self.data_keys:
            return lambda trial, objno: self.get(name, trial, objno)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")



if __name__ == "__main__":
    m = mat(fname)
    print(m.data_keys)
    print(m.scalar_keys)
    print(m.metadata_keys)

    # fix trial number
    tno = 4
    t_fs = 100

    t_start = None   # float or None
    duration = None  # float or None
    
    # print(m.max_obj[tno])
    # print(m.y(tno, 0))
    # print(m.frames(tno, 4))
    # print(m.raw_data['exp'][m.trials[0]]['metadata'].keys())

    # print(np.array(m.raw_data['exp'][m.trials[tno]]['tracks']['object'][m.objs[tno][50]]['scalars']['dynamic'][:,0]))
    # print(np.array(m.raw_data['exp'][m.trials[tno]]['tracks']['object'][m.objs[tno][0]]['scalars']['num_recording'][0,0]))
    # print(np.array(m.raw_data['exp'][m.trials[tno]]['tracks']['object'][m.objs[tno][0]]['scalars']['was_initiator']))

    # objno = 0
    # x,y,z = m.x(tno, objno), m.y(tno, objno), m.z(tno, objno)
    # x0,y0,z0 = m.x(tno, objno)[0], m.y(tno, objno)[0], m.z(tno, objno)[0]

    # objects = list(range(min(10, m.max_obj[tno])))
    # objects = list(range(m.max_obj[tno]))
    # objects = m.obj_ids(tno)
    # max_frames = max(m.max_step(tno, obj) for obj in objects)
    # max_frm = max(np.max(m.frames(tno,obj)) for obj in objects)
    # min_frm = min(np.min(m.frames(tno,obj)) for obj in objects)
    # print(min_frm, max_frm)
    # print(min(np.min(m.time(tno,obj)) for obj in objects), max(np.max(m.time(tno,obj)) for obj in objects))

    if len(swarm_data := m.format_all_smooth_insects(tno, target_fs=t_fs, t_start=t_start, duration=duration)) == 0:
        raise ValueError("No valid insect sub-tracks found for this trial.")

    max_activity = m.swarm_activity(swarm_data)
    print(max_activity.items())

    # construct global continuous time grid across the whole swarm recording
    data_t_min = min(insect['t'][0] for insect in swarm_data)
    data_t_max = max(insect['t'][-1] for insect in swarm_data)

    t_buffer = 10 # s
    t_start = max(data_t_min, max_activity['peak_time_start'] - t_buffer)
    duration = max_activity['peak_time_end'] - max_activity['peak_time_start'] + 2.0 * t_buffer

    anim_t_start = t_start if t_start is not None else data_t_min

    if duration is not None:
        anim_t_end = min(anim_t_start + duration, data_t_max)
    else:
        anim_t_end = data_t_max

    dt_anim = 1.0 / t_fs
    t_grid = np.arange(anim_t_start, anim_t_end + dt_anim, dt_anim)
    max_frames = len(t_grid)

    print(f"Animating {len(swarm_data)} sub-tracks from t = {anim_t_start:.2f} s to {anim_t_end:.2f} s ({max_frames} frames)...")

    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw={"projection": "3d"})

    N_insects = len(swarm_data)
    cmap = plt.get_cmap("jet")
    colors = (
        cmap(np.linspace(0, 1, N_insects))
        if len(swarm_data) > 1
        else [cmap(0.5)]
    )

    lines = [
        ax.plot([], [], [], color=colors[i], alpha=0.6, lw=1.5)[0]
        for i in range(len(swarm_data))
    ]

    p = ax.scatter(
        np.zeros(N_insects), 
        np.zeros(N_insects), 
        np.zeros(N_insects), 
        c=list(range(N_insects)), 
        cmap="jet", 
        s=10, 
        alpha=0.9
    )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    lim = 0.65
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(0.0, 1.25)

    def update(frame_idx):
        t_curr = t_grid[frame_idx]
        x_vals, y_vals, z_vals = [], [], []

        for i, insect in enumerate(swarm_data):
            t_start_i = insect['t'][0]
            t_end_i = insect['t'][-1]

            if t_start_i <= t_curr <= t_end_i:
                # Local index within this insect's dataset
                idx = int(round((t_curr - t_start_i) * t_fs))
                idx = min(idx, len(insect['t']) - 1)  # Safeguard index bound

                # trailing path
                st_idx = max(0, idx - 25)
                end_idx = idx + 1

                x_vals.append(insect['y'][idx, 0])
                y_vals.append(insect['y'][idx, 1])
                z_vals.append(insect['y'][idx, 2])

                lines[i].set_data(
                    insect['y'][st_idx:end_idx, 0],
                    insect['y'][st_idx:end_idx, 1],
                )
                lines[i].set_3d_properties(insect['y'][st_idx:end_idx, 2])
            else:
                x_vals.append(np.nan)
                y_vals.append(np.nan)
                z_vals.append(np.nan)

                lines[i].set_data([], [])
                lines[i].set_3d_properties([])
        
        p._offsets3d = (x_vals, y_vals, z_vals)
        ax.set_title(f"Global Time: t = {t_curr:.2f} s")
        
        return (p, *lines)


    ani = animation.FuncAnimation(
        fig=fig, 
        func=update, 
        frames=max_frames, 
        interval=dt_anim * 1e3, 
        blit=False
    )

    plt.show()

    # writer = animation.FFMpegWriter(
    #     fps=t_fs,
    #     bitrate=-1,
    #     extra_args=[
    #         '-vcodec', 'libx264',
    #         '-crf', '18',          # 18 = visually lossless, 23 = default
    #         '-pix_fmt', 'yuv420p'  # Ensures MP4 plays on QuickTime / Mac Preview
    #     ])
    # ani.save(f"{path}Poda_interpolated_track_{tno}.mp4", writer=writer, dpi=180)
