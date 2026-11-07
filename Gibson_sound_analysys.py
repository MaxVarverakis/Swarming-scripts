import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.ticker as ticker
from matplotlib.animation import FuncAnimation
import librosa as lib
from scipy.signal import find_peaks

fs = 16
plt.rcParams['font.size'] = fs

sup2 = [
    r"$\it{An. coluzzii}$ (male)",
    r"$\it{An. gambiae\ s.\!s.}$ (male)",
    r"$\it{An. coluzzii}$ (female)",
    r"$\it{An. gambiae\ s.\!s.}$ (female)"
]

def spectrum(fname: str, S_idx: int, return_data=False, show=False, print_info=False, fft=False, extract_peaks=False, slice_t=-1):
    audio_data, sample_rate = lib.load(fname, sr=None)

    if print_info:
        print("Audio shape:", audio_data.shape)  # Flattens to mono by default
        print("Sample rate:", sample_rate)

    fig = plt.figure(figsize=(14,8))
    gs = fig.add_gridspec(
        nrows=2, 
        ncols=4 if fft else 3, 
        height_ratios=[0.05, 1], 
        width_ratios=[3, 1, 1, 1] if fft else [3, 1, 1], 
        hspace=0.02, 
        wspace=0.0,
        top=0.85,
        bottom=0.15
    )

    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1], sharey=ax1)
    ax3 = fig.add_subplot(gs[1, 2])
    if fft:
        ax4 = fig.add_subplot(gs[1, 3])
    cax = fig.add_subplot(gs[0, 0])

    # vmin=-250
    # vmax=-30
    vmin=-220
    vmax=0

    # 256
    multiplier = 1 # 20 # 1e2
    spectrum, freqs, t, _ = ax1.specgram(audio_data, NFFT=int(sample_rate/50 * multiplier), Fs=sample_rate, noverlap=128, cmap='jet', aspect='equal', vmin=vmin, vmax=vmax)
    spectrum = 10 * np.log10(spectrum)
    spectrum -= np.max(spectrum)
    im = ax1.imshow(
        spectrum, 
        origin='lower', 
        extent=[t[0], t[-1], freqs[0], freqs[-1]], 
        aspect='auto', 
        cmap='jet', 
        vmin=vmin, 
        vmax=vmax
    )

    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Frequency [kHz]')
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{y/1000:g}'))

    hHz = 3.7e3
    if S_idx in [1,2]:
        lHz = 5e2
    else:
        lHz = 2.5e2
    for ax in [ax1,ax2]:
        ax.axhline(hHz, ls='--', c='b', alpha=1.0, lw=1.2)
        ax.axhline(lHz, ls='-.', c='k', alpha=1.0, lw=1.2)
    ax2.text(-200, hHz-7.5e2, f"{hHz * 1e-3:.1f} kHz", color='b', fontsize=fs-3)
    ax2.text(-200, lHz+1.25e2, f"{lHz * 1e-3:.1f} kHz", color='k', fontsize=fs-3)

    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')
    cbar.set_label('Relative Intensity [dB]', labelpad=10)

    a2_lab_x = 1.
    a2_lab_y = -0.07

    if slice_t >= 0:
        t_target = slice_t # s
        t_idx = np.argmin(abs(t - t_target))
        power_dB = spectrum[:, t_idx]
        ax1.axvline(x=t[t_idx], ls='-', c='k', alpha=0.7, lw=1.2)
        ax2.set_title(f"Slice: t = {t[t_idx]:.2f} s", fontsize=fs-2, fontweight='bold')
        ax2.text(s="Power [dB, arb. ref.]", x=a2_lab_x, y=a2_lab_y, ha="center", va="top", transform=ax.transAxes)
    else:
        power_dB = np.mean(spectrum, axis=1)
        ax2.text(s="Mean Power [dB, arb. ref.]", x=a2_lab_x, y=a2_lab_y, ha="center", va="top", transform=ax.transAxes)

    ax2.plot(power_dB, freqs, color='crimson', lw=1.5)
    ax2.grid(True)
    ax2.tick_params(left=False, labelleft=False)
    ax2.tick_params(axis='x', labelsize=12)
    ax2.set_xlim(vmin, vmax)

    power_a3 = power_dB - max(power_dB)
    ax3.plot(power_a3, freqs, color='g', lw=1.5)
    ax3.grid(True)
    ax3.tick_params(left=False, labelleft=False, right=not fft, labelright=not fft)
    ax3.yaxis.set_label_position("right")
    ax3.set_ylabel('Frequency [kHz]')
    ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{y/1000:g}'))
    ax3.tick_params(axis='x', labelsize=12)
    ax3.set_xlim(np.min(power_a3[np.argmin(abs(freqs - hHz))]), 0)
    ax3.set_ylim(lHz, hHz)
    ax3.spines["top"].set_color("b")
    ax3.spines["top"].set_linewidth(1.5)
    ax3.spines["top"].set_linestyle("--")
    ax3.spines["bottom"].set_color("k")
    ax3.spines["bottom"].set_linewidth(1.5)
    ax3.spines["bottom"].set_linestyle("-.")
    ax3.spines["left"].set_linewidth(1.5)

    # f0_mask = (freqs>=lHz) & (freqs<=1e3)
    # f1_mask = (freqs>=1.5e3) & (freqs<=2e3) & (freqs<1770)
    # f_mark = freqs[f1_mask][np.argmax(power_a3[f1_mask])]
    # print(f_mark)
    # ax3.axhline(f_mark, c='k')

    if fft:
        window = np.hamming(len(audio_data))
        windowed_signal = audio_data * window
        rate_reduction = 1
        fft_result = np.fft.rfft(windowed_signal, n=int(sample_rate/rate_reduction))
        fftf = np.fft.rfftfreq(int(sample_rate/rate_reduction), d=1/sample_rate)
        pspec = np.abs(fft_result)

        idx_low  = np.argmin(abs(fftf - lHz))
        idx_high = np.argmin(abs(fftf - hHz))

        ax4.plot(pspec, fftf, color='mediumslateblue', lw=1)
        ax4.grid(True)
        ax4.tick_params(left=False, labelleft=False, right=True, labelright=True)
        ax4.yaxis.set_label_position("right")
        ax4.set_ylabel('Frequency [kHz]')
        ax4.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{y/1000:g}'))
        ax4.tick_params(axis='x', labelsize=12, colors='mediumslateblue')
        ax4.set_xlim(1.1 * np.max(pspec[idx_low:idx_high]), 0)
        ax4.set_ylim(lHz, hHz)
        ax4.spines["top"].set_color("b")
        ax4.spines["top"].set_linewidth(1.5)
        ax4.spines["top"].set_linestyle("--")
        ax4.spines["bottom"].set_color("k")
        ax4.spines["bottom"].set_linewidth(1.5)
        ax4.spines["bottom"].set_linestyle("-.")
        ax4.spines["left"].set_linewidth(1.5)
        # ax4.axhline(f_mark, c='k')

        if extract_peaks:
            df = fftf[1] - fftf[0]
            min_dist_pts = max(1, int(400.0 / df))

            peak_indices, _properties = find_peaks(pspec[idx_low:idx_high], prominence=0.05 * np.max(pspec[idx_low:idx_high]), distance=min_dist_pts)
            peak_frequencies = fftf[idx_low + peak_indices]
            peak_magnitudes = pspec[idx_low + peak_indices]

            # sort peaks from highest amplitude to lowest
            sorted_order = np.argsort(peak_magnitudes)[::-1]
            sorted_freqs = peak_frequencies[sorted_order]
            sorted_mags = peak_magnitudes[sorted_order]

            for idx, (freq, mag) in enumerate(zip(sorted_freqs, sorted_mags)):
                print(f"Rank {idx+1}: {freq*1e-3:.2f} kHz (Magnitude: {mag:.4f})\tHarmonicity: {freq/sorted_freqs[0]:.2f}")


    plt.suptitle(f"Swarm sound S{S_idx}, {sup2[S_idx-1]}")

    if show:
        plt.show()

    if return_data:
        return spectrum,freqs,t,im

def spec_animate(fname: str, S_idx:int, save_anim: str = '', show=False, print_info=False):
    audio_data, sample_rate = lib.load(fname, sr=None)
    
    if print_info:
        print("Audio shape:", audio_data.shape)  # Flattens to mono by default
        print("Sample rate:", sample_rate)

    fig = plt.figure(figsize=(10,8))
    gs = fig.add_gridspec(
        nrows=2, 
        ncols=2, 
        height_ratios=[0.05, 1], 
        width_ratios=[3, 1], 
        hspace=0.02, 
        wspace=0.0,
        top=0.85,
        bottom=0.15
    )

    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1], sharey=ax1)
    cax = fig.add_subplot(gs[0, 0])

    # vmin=-250
    # vmax=-30
    vmin=-220
    vmax=0

    spectrum, freqs, t, _ = ax1.specgram(audio_data, NFFT=int(sample_rate/50), Fs = sample_rate, noverlap=128, cmap='jet', aspect='equal', vmin=vmin, vmax=vmax)

    spectrum = 10 * np.log10(spectrum)
    spectrum -= np.max(spectrum)
    im = ax1.imshow(
        spectrum, 
        origin='lower', 
        extent=[t[0], t[-1], freqs[0], freqs[-1]], 
        aspect='auto', 
        cmap='jet', 
        vmin=vmin, 
        vmax=vmax
    )

    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Frequency [kHz]')
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{y/1000:g}'))

    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')
    cbar.set_label('Relative Intensity [dB]', labelpad=10)

    t_idx = 0
    power_dB = spectrum[:, t_idx]
    vline = ax1.axvline(x=t[t_idx], ls='-', c='k', alpha=0.7, lw=1.2)
    # time_text = ax2.set_title(f"Slice: t = {t[t_idx]:.2f} s", fontsize=fs-2, fontweight='bold')
    time_text = ax2.text(
        0.03, 0.01, f"", 
        transform=ax2.transAxes, 
        ha='left', va='bottom', 
        fontsize=fs-2,
        fontweight='bold',
        # backgroundcolor="w"
    )

    (line,) = ax2.plot(power_dB, freqs, color='crimson', lw=1.5)
    ax2.set_xlabel("Power [dB, arb. ref.]")
    ax2.grid(True)
    ax2.tick_params(left=False, labelleft=False)
    ax2.tick_params(axis='x', labelsize=12)

    ax2.set_xlim(vmin, vmax)

    plt.suptitle(f"Swarm sound S{S_idx}, {sup2[S_idx-1]}")

    def update(frame):
        t_current = t[frame]
        power_dB = spectrum[:,frame]

        vline.set_xdata([t_current, t_current])
        line.set_xdata(power_dB)
        time_text.set_text(f"t = {t_current:.2f} s")

        return vline, line, time_text

    dt = (t[1] - t[0]) * 1000 # in ms to match real time step for `interval` which expects ms
    
    anim = FuncAnimation(
        fig, 
        update, 
        frames=len(t), 
        interval=dt, 
        blit=True
    )

    if save_anim:
        anim.save(save_anim, writer='ffmpeg', fps=int(1000 / dt))

    if show:
        plt.show()

    return anim

def spectrum_analyze(fname: str, return_data=False, show=False, save_path='', print_info=False, add_trim=False, fft=False, extract_peaks=False, slice_t=-1, lHz = 5.0e2, hHz = 3.7e3):
    audio_data, sample_rate = lib.load(fname, sr=None)

    if print_info:
        print("Audio shape:", audio_data.shape)  # Flattens to mono by default
        print("Sample rate:", sample_rate)

    fig = plt.figure(figsize=(10 + 2*int(add_trim) + 2*int(fft),8))
    gs = fig.add_gridspec(
        nrows=2, 
        ncols=4 if fft & add_trim else 3 if add_trim else 2, 
        height_ratios=[0.05, 1], 
        width_ratios=[3, 1, 1, 1] if fft & add_trim else [3, 1, 1] if add_trim else [3,1], 
        hspace=0.02, 
        wspace=0.0,
        top=0.85,
        bottom=0.15
    )

    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1], sharey=ax1)
    if add_trim:
        ax3 = fig.add_subplot(gs[1, 2])
        if fft:
            ax4 = fig.add_subplot(gs[1, 3])
    
    cax = fig.add_subplot(gs[0, 0])

    # vmin=-250
    # vmax=-30
    # vmin=-220
    # vmax=0

    # 256
    multiplier = 1 # 20 # 1e2
    spectrum, freqs, t = mlab.specgram(
        audio_data, 
        NFFT=int(sample_rate / 50 * multiplier), 
        Fs=sample_rate, 
        noverlap=128
    )
    # spectrum, freqs, t, _ = ax1.specgram(audio_data, NFFT=int(sample_rate/50 * multiplier), Fs=sample_rate, noverlap=128, cmap='jet', aspect='equal') #, vmin=vmin, vmax=vmax)
    spectrum = 10 * np.log10(spectrum + 1e-16)
    spectrum -= np.max(spectrum)
    # f_mask = (freqs > lHz) & (freqs < hHz)
    # spectrum = spectrum[f_mask]
    # freqs = freqs[f_mask]
    im = ax1.imshow(
        spectrum, 
        origin='lower', 
        extent=[t[0], t[-1], freqs[0], freqs[-1]], 
        aspect='auto', 
        cmap='jet', 
        # vmin=vmin, 
        # vmax=vmax
    )

    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Frequency [kHz]')
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{y/1000:g}'))

    if add_trim:
        for ax in [ax1,ax2]:
            ax.axhline(hHz, ls='--', c='m', alpha=1.0, lw=1.2)
            ax.axhline(lHz, ls='-.', c='k', alpha=1.0, lw=1.2)
        ax2.text(-200, hHz+4.25e2, f"{hHz * 1e-3:.1f} kHz", color='m', fontsize=fs-3,
                bbox=dict(
                facecolor='white',     # Background color
                edgecolor='m',      # Border color
                boxstyle='round,pad=0.25', # Box shape and padding
                alpha=0.8               # Transparency (0 to 1)
        ))
        ax2.text(-200, lHz+4.25e2, f"{lHz * 1e-3:.1f} kHz", color='k', fontsize=fs-3,
                bbox=dict(
                facecolor='white',     # Background color
                edgecolor='k',      # Border color
                boxstyle='round,pad=0.25', # Box shape and padding
                alpha=0.8               # Transparency (0 to 1)
        ))

    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')
    cbar.set_label('Intensity [dB, arb. ref.]', labelpad=10)

    a2_lab_x = 1.
    a2_lab_y = -0.07

    if slice_t >= 0:
        t_target = slice_t # s
        t_idx = np.argmin(abs(t - t_target))
        power_dB = spectrum[:, t_idx]
        ax1.axvline(x=t[t_idx], ls='-', c='k', alpha=0.7, lw=1.2)
        ax2.set_title(f"Slice: t = {t[t_idx]:.2f} s", fontsize=fs-2, fontweight='bold')
        if add_trim:
            ax2.text(s="Power [dB, arb. ref.]", x=a2_lab_x, y=a2_lab_y, ha="center", va="top", transform=ax.transAxes)
    else:
        power_dB = np.mean(spectrum, axis=1)
        if add_trim:
            ax2.text(s="Mean [dB, arb. ref.]", x=a2_lab_x, y=a2_lab_y, ha="center", va="top", transform=ax.transAxes)
        else:
            ax2.set_title("Mean power")

    if not add_trim:
        ax2.set_xlabel("[dB, arb. ref.]")

    ax2.plot(power_dB, freqs, color='crimson', lw=1.5)
    ax2.grid(True)
    ax2.tick_params(left=False, labelleft=False)
    ax2.tick_params(axis='x', labelsize=12)
    # ax2.set_xlim(vmin, vmax)

    if add_trim:
        power_a3 = power_dB - max(power_dB)
        ax3.plot(power_a3, freqs, color='g', lw=1.5)
        ax3.grid(True)
        ax3.tick_params(left=False, labelleft=False, right=not fft, labelright=not fft)
        ax3.yaxis.set_label_position("right")
        ax3.set_ylabel('Frequency [kHz]')
        ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{y/1000:g}'))
        ax3.tick_params(axis='x', labelsize=12)
        ax3.set_xlim(np.min(power_a3[np.argmin(abs(freqs - hHz))]), 0)
        ax3.set_ylim(lHz, hHz)
        ax3.spines["top"].set_color("m")
        ax3.spines["top"].set_linewidth(1.5)
        ax3.spines["top"].set_linestyle("--")
        ax3.spines["bottom"].set_color("k")
        ax3.spines["bottom"].set_linewidth(1.5)
        ax3.spines["bottom"].set_linestyle("-.")
        ax3.spines["left"].set_linewidth(1.5)

    # f0_mask = (freqs>=lHz) & (freqs<=1e3)
    # f1_mask = (freqs>=1.5e3) & (freqs<=2e3) & (freqs<1770)
    # f_mark = freqs[f1_mask][np.argmax(power_a3[f1_mask])]
    # print(f_mark)
    # ax3.axhline(f_mark, c='k')

    if add_trim & fft:
        window = np.hamming(len(audio_data))
        windowed_signal = audio_data * window
        rate_reduction = 1
        fft_result = np.fft.rfft(windowed_signal, n=int(sample_rate/rate_reduction))
        fftf = np.fft.rfftfreq(int(sample_rate/rate_reduction), d=1/sample_rate)
        pspec = np.abs(fft_result)

        idx_low  = np.argmin(abs(fftf - lHz))
        idx_high = np.argmin(abs(fftf - hHz))

        ax4.plot(pspec, fftf, color='mediumslateblue', lw=1)
        ax4.grid(True)
        ax4.tick_params(left=False, labelleft=False, right=True, labelright=True)
        ax4.yaxis.set_label_position("right")
        ax4.set_ylabel('Frequency [kHz]')
        ax4.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{y/1000:g}'))
        ax4.tick_params(axis='x', labelsize=12, colors='mediumslateblue')
        ax4.set_xlim(1.1 * np.max(pspec[idx_low:idx_high]), 0)
        ax4.set_ylim(lHz, hHz)
        ax4.spines["top"].set_color("m")
        ax4.spines["top"].set_linewidth(1.5)
        ax4.spines["top"].set_linestyle("--")
        ax4.spines["bottom"].set_color("k")
        ax4.spines["bottom"].set_linewidth(1.5)
        ax4.spines["bottom"].set_linestyle("-.")
        ax4.spines["left"].set_linewidth(1.5)
        # ax4.axhline(f_mark, c='k')

        if add_trim & fft & extract_peaks:
            df = fftf[1] - fftf[0]
            min_dist_pts = max(1, int(400.0 / df))

            peak_indices, _properties = find_peaks(pspec[idx_low:idx_high], prominence=0.05 * np.max(pspec[idx_low:idx_high]), distance=min_dist_pts)
            peak_frequencies = fftf[idx_low + peak_indices]
            peak_magnitudes = pspec[idx_low + peak_indices]

            # sort peaks from highest amplitude to lowest
            sorted_order = np.argsort(peak_magnitudes)[::-1]
            sorted_freqs = peak_frequencies[sorted_order]
            sorted_mags = peak_magnitudes[sorted_order]

            for idx, (freq, mag) in enumerate(zip(sorted_freqs, sorted_mags)):
                print(f"Rank {idx+1}: {freq*1e-3:.2f} kHz (Magnitude: {mag:.4f})\tHarmonicity: {freq/sorted_freqs[0]:.2f}")


    # plt.suptitle(r"Microphone location (x,y,z)$_{\text{rel}}$ =" + f" ({mx},{my},{mz}) m")

    if save_path:
        plt.savefig(f"{save_path}.png", dpi=800)

    if show:
        plt.show()

    if return_data:
        return spectrum,freqs,t,im


if __name__ == "__main__":
    pre_path = "/Users/max/UCLA/Research/Codes"
    save_path = f"{pre_path}/Animations/"
    path = f"{pre_path}/Data/Gibson2021"

    for i in range(1,2):
        # i+=1
        s = f"{path}/sound-S{i}.mp3"
        
        spectrum(s, S_idx=i, show=True, fft=True, extract_peaks=True, slice_t=-1)
        # plt.savefig(f"{pre_path}/Figures/Gibson_spectrogram_mean_power_sound_S{i}.png", dpi=500)

        # save_name = f"{save_path}Gibson_spectrogram_sweep_sound_S{i}.mp4"
        # spec_animate(s, S_idx=i, save_anim=save_name, show=False)
        # spec_animate(s, S_idx=i, show=True)
