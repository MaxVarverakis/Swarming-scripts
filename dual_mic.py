import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import csd,hilbert
import copy

import Lowson_solver as ls
import Poda_data as pod

plt.rcParams['font.size'] = 16

def dipole_energy_ratio(p_above, p_below):
    """
    Computes the proportion of antisymmetry in the two signals
    """
    S = 0.5 * (p_above + p_below) # symmetrize
    D = 0.5 * (p_above - p_below) # antisymmetrize
    
    energy_S = np.sum(S**2)
    energy_D = np.sum(D**2)
    
    eta_D = energy_D / (energy_S + energy_D)
    return eta_D, S, D

def csd_phase(p_above, p_below, fs):
    """
    Computes the Cross-spectrial density which is used to reveal phase differences between signals across their frequencies.
    """
    f, P12 = csd(p_above, p_below, fs, nperseg=2048)
    phase_rad = np.angle(P12)
    return f, np.abs(P12), phase_rad

def time_window_correlations(p_above, p_below, window_samples=441):
    """
    Normalized cross-correlation actoss time:
        ( sum_{k in window} p_a[k] p_b[k] ) / ( sqrt( sum_k p_a[k]^2 ) sqrt( sum_k p_a[k]^2 ) )
    
    Monopole --> +1 ;
    Dipole   --> -1
    """
    kernel = np.ones(window_samples) / window_samples

    num = np.convolve(p_above * p_below, kernel, mode="same")
    den = np.sqrt(
        np.convolve(p_above**2, kernel, mode="same") * 
        np.convolve(p_below**2, kernel, mode="same")
    )

    return num / den

def instantaneous_phase_difference(p_above, p_below):
    """
    Uses the Hilbert transform of each microphone signal to extract instantaneous phase of each signal.
    
    Returns
    -------
    delta_phi : list
        Instantaneous phase differences between the two signals at each time
    """
    z_above = hilbert(p_above)
    z_below = hilbert(p_below)
    
    delta_phi = np.angle(z_above * np.conj(z_below))
    return delta_phi

if __name__ == "__main__":

    # global settings
    fname = "/Users/max/UCLA/Research/Codes/Data/Poda2024/Database S1/all_data.mat"
    path = "/Users/max/UCLA/Research/Codes"

    fs = 44100
    start_time = 0 # 950 # s
    target_duration = 1.0 # s
    n_harm = 4

    # d = 0.4
    d = 1.0
    # d = 2.0

    mic_positions = np.array([
        [0.0, 0.0,  d],
        [0.0, 0.0, -d]
    ]) # m
    labels = ["Above", "Below"]

    # initialize swarm data
    m = pod.mat(fname)
    # fix trial number
    tno = 4

    # monopole_data = m.format_all_smooth_insects(tno, n_harmonics=n_harm, t_start=start_time, duration=target_duration, monopole=True)
    # dipole_data = m.format_all_smooth_insects(tno, n_harmonics=n_harm, t_start=start_time, duration=target_duration, monopole=False)
    monopole_data = [ m.generate_test_insect(duration=target_duration, n_harmonics=n_harm, monopole=True) ]
    dipole_data = [ m.generate_test_insect(duration=target_duration, n_harmonics=n_harm, monopole=False) ]


    st = start_time + target_duration / 2
    t_end = 0.05 # s

    z = []
    for insect in dipole_data:
        if insect['t'][0] <= st <= insect['t'][-1]:
            z.append(np.interp(st, insect['t'], insect['y'][:, 2]))
    z_cm = np.mean(z)

    # z_cms = np.zeros(len(t_audio))
    # for i in range(len(z_cms)):
    #     z = []
    #     for insect in dipole_data:
    #                 if insect['t'][0] <= start_time <= insect['t'][-1]:
    #                         z.append(np.interp(start_time, insect['t'], insect['y'][:, 2]))
    #     z_cms[i] = np.mean(z)
    # cm_offset = np.array([0.0, 0.0, np.mean(z_cms)])

    cm_offset = np.array([0.0, 0.0, z_cm])
    mic_positions += cm_offset

    t_audio = {
        'monopole': [],
        'dipole': [],
    }
    # deep copy ensures `p` contains no mutable linkage to `t_audio`
    p = copy.deepcopy(t_audio)


    for mic in mic_positions:
        m_t_audio,m_p = ls.swarm_sound_monopole(mic, monopole_data)
        d_t_audio,d_p = ls.swarm_sound_dipole(mic, dipole_data)

        t_audio['monopole'].append(m_t_audio)
        p['monopole'].append(1e6 * m_p) # uPa
        
        t_audio['dipole'].append(d_t_audio)
        p['dipole'].append(1e6 * d_p) # uPa

    t_start_idx = np.argmin(abs(d_t_audio - st))
    t_end_idx = np.argmin(abs(d_t_audio - (st + t_end)))



    block = 2

    if block == 1:
        fig,ax = plt.subplots(figsize=(10,6), ncols=1, nrows=2, sharex=True, gridspec_kw={'hspace': 0.25})

        for idx in range(len(mic_positions)):
            ax[0].plot(t_audio['monopole'][idx][t_start_idx:t_end_idx+1] - t_audio['monopole'][idx][t_start_idx], p['monopole'][idx][t_start_idx:t_end_idx+1], lw=2, label=labels[idx])
            ax[1].plot(t_audio['dipole'][idx][t_start_idx:t_end_idx+1] - t_audio['dipole'][idx][t_start_idx], p['dipole'][idx][t_start_idx:t_end_idx+1], lw=2)

        ax[0].set_ylabel(r"Pressure [$\mu$Pa]")
        ax[1].set_ylabel(r"Pressure [$\mu$Pa]")
        ax[1].set_xlabel("Time [s]")

        ax[0].legend(ncols=2, loc='upper right')

        ax[0].set_title("Monopole sources")
        ax[1].set_title("Dipole sources")

        ax[0].grid(True)
        ax[1].grid(True)

        ax[0].margins(y=0.25) 

        plt.suptitle(f"Microphones {d:.1f} m from swarm CM", weight="bold", y=0.99)

        # plt.savefig(f"{path}/Figures/Poda_{tno+1}_dual_mic_{d:.2f}_m_{n_harm}_harmonics.png", dpi=800)

        plt.show()

    elif block == 2:
        p_m_above = p['monopole'][0][t_start_idx:t_end_idx+1]
        p_m_below = p['monopole'][1][t_start_idx:t_end_idx+1]

        p_d_above = p['dipole'][0][t_start_idx:t_end_idx+1]
        p_d_below = p['dipole'][1][t_start_idx:t_end_idx+1]

        eta_m, S_m, D_m = dipole_energy_ratio(p_m_above, p_m_below)
        eta_d, S_d, D_d = dipole_energy_ratio(p_d_above, p_d_below)

        tcorr_m = time_window_correlations(p_m_above, p_m_below)
        tcorr_d = time_window_correlations(p_d_above, p_d_below)

        # print(f"Monopole Dataset -> Dipole Energy Ratio eta_D: {eta_m:.4f} (Expected ~0.0)")
        # print(f"Dipole Dataset   -> Dipole Energy Ratio eta_D: {eta_d:.4f} (Expected ~1.0)")


        t_axis = t_audio['monopole'][0][t_start_idx:t_end_idx+1] - t_audio['monopole'][0][t_start_idx]

        tick_positions = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
        tick_labels = [r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$']

        fig, ax = plt.subplots(3, 2, figsize=(16, 8), gridspec_kw={'wspace': 0.15, 'hspace': 0.75}, sharey='row')

        # Top Left: Monopole Sum vs Difference
        ax[0, 0].plot(t_axis, S_m, label="Symmetric (Sum)", c="mediumblue")
        ax[0, 0].plot(t_axis, D_m, label="Anti-Symmetric (Diff)", c="tab:red", ls="-.")
        ax[0, 0].set_title(f"Monopole Decomposition ($\\eta_D = {eta_m:.2f}$)")
        ax[0, 0].set_ylabel(r"Pressure [$\mu$Pa]")
        ax[0, 0].legend(fontsize=12, ncols=2, loc="center", bbox_to_anchor=(0.5, 0.9), framealpha=0)
        ax[0, 0].grid(True, alpha=0.3)
        ax[0, 0].margins(y=0.2, x=0)
        ax[0, 0].set_xlabel("Time [s]")

        # Top Right: Dipole Sum vs Difference
        ax[0, 1].plot(t_axis, S_d, label="Symmetric (Sum)", c="mediumblue")
        ax[0, 1].plot(t_axis, D_d, label="Anti-Symmetric (Diff)", c="tab:red", ls="-.")
        ax[0, 1].set_title(f"Dipole Decomposition ($\\eta_D = {eta_d:.2f}$)")
        # ax[0, 1].set_ylabel(r"Pressure [$\mu$Pa]")
        ax[0, 1].legend(fontsize=12, ncols=2, loc="center", bbox_to_anchor=(0.5, 0.9), framealpha=0)
        ax[0, 1].grid(True, alpha=0.3)
        ax[0, 1].margins(y=0.2, x=0)
        ax[0, 1].set_xlabel("Time [s]")


        
        # Middle Left: Monopole time-windowed pressure correlations
        ax[1, 0].scatter(t_axis, tcorr_m, c="m", marker='o', s=5, label="Monopole")
        ax[1, 0].set_title(f"Monopole Correlations")
        ax[1, 0].set_ylabel(r"Correlation")
        ax[1, 0].grid(True, alpha=0.3)
        ax[1, 0].set_xlabel("Time [s]")
        ax[1, 0].margins(x=0)


        # Middle Right: Dipole time-windowed pressure correlations
        ax[1, 1].scatter(t_axis, tcorr_d, edgecolors="m", facecolors='none', marker='o', s=5, label="Dipole")
        ax[1, 1].set_title(f"Dipole Correlations")
        # ax[1, 1].set_ylabel(r"Correlation")
        ax[1, 1].grid(True, alpha=0.3)
        ax[1, 1].set_xlabel("Time [s]")
        ax[1, 1].margins(x=0)


        # Bottom Left: CSD Phase Spectrum (Monopole)
        f_m, P12_m, phase_m = csd_phase(p_m_above, p_m_below, fs)
        f_d, P12_d, phase_d = csd_phase(p_d_above, p_d_below, fs)
        ax[2, 0].scatter(f_m, phase_m, c="mediumblue", marker='o', s=18, label="Monopole")
        ax[2, 0].scatter(f_d, phase_d, edgecolors="tab:red", facecolors='none', marker='o', s=18, label="Dipole")
        ax[2, 0].set_xlim(0, 4000)
        ax[2, 0].set_ylim(-np.pi, np.pi)
        ax[2, 0].set_xlabel("Frequency [Hz]")
        ax[2, 0].set_ylabel(r"$\Delta\phi$ [rad]")
        ax[2, 0].set_title("(Cross-spectral density)")
        ax[2, 0].grid(True, alpha=0.3)
        ax[2, 0].set_yticks(tick_positions)
        ax[2, 0].set_yticklabels(tick_labels)
        ax[2, 0].set_ylim(top=3*np.pi/2)
        for idx,f in enumerate(864.0 * np.arange(1,5,1)):
            ax[2,0].axvline(f, ymax=0.83, c='k', ls='-', lw=2, label="_" if idx!=0 else "Harmonics")
        ax[2, 0].legend(ncols=3, loc="center", bbox_to_anchor=((0.5, 0.9)), framealpha=0, fontsize=13, columnspacing=1.5)

        # Bottom Right: Phase difference from Hilbert transform
        H_phase_m = instantaneous_phase_difference(p_m_above, p_m_below)
        H_phase_d = instantaneous_phase_difference(p_d_above, p_d_below)
        ax[2, 1].hist(H_phase_m, bins=25, orientation="horizontal", edgecolor="mediumblue", fill=False, density=True, label="Monopole", histtype='stepfilled', lw=2)
        ax[2, 1].hist(H_phase_d, bins=25, orientation="horizontal", edgecolor="tab:red", fill=False, density=True, label="Dipole", histtype='stepfilled', lw=1.5)
        ax[2, 1].set_ylim(-np.pi, np.pi)
        # ax[2, 1].set_ylabel(r"$\Delta\phi$ [rad]")
        ax[2, 1].set_title("(Hilbert transform)")
        ax[2, 1].grid(True, alpha=0.3)
        ax[2, 1].set_yticks(tick_positions)
        ax[2, 1].set_yticklabels(tick_labels)
        ax[2, 1].set_ylim(top=3*np.pi/2)
        ax[2, 1].set_xticklabels([])
        ax[2, 1].tick_params(bottom=False) 
        ax[2, 1].legend(ncols=2, loc="center", bbox_to_anchor=((0.5, 0.9)), framealpha=0)

        plt.suptitle(f"Microphones {d:.1f} m from swarm CM", weight="bold", y=0.99)

        # plt.savefig(f"{path}/Figures/Poda_{tno+1}_dual_mic_correlations_{d:.2f}_m.png", dpi=800)
        ax[2, 1].set_xlim(right=3)
        # plt.savefig(f"{path}/Figures/test_source_dual_mic_correlations_{d:.2f}_m_{n_harm}_harmonics.png", dpi=800)

        plt.show()
