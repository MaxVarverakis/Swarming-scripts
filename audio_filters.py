import librosa as lib
import soundfile as sf
from scipy import signal

def filter(fname, cutoff_hz, btype, save_path='', custom_name='', order=5):
    """
    Applies a Butterworth filter to a float32 audio array.

    Parameters:
    -----------
    fname : str
        full path and filename of the audio file of interest
    cutoff_hz : float or length-2 sequence of floats
        critical frequency of the filter
    btype : str
        {'lowpass', 'highpass', 'bandpass', 'bandstop'}
    """

    audio_data, sample_rate = lib.load(fname, sr=None)

    sos = signal.butter(
        N=order, 
        Wn=cutoff_hz, 
        btype=btype, 
        analog=False, 
        fs=sample_rate, 
        output='sos'
    )

    filtered_data = signal.sosfiltfilt(sos, audio_data, axis=-1)

    if save_path:
        if 'band' in btype:
            sf.write(f"{save_path}/{btype}_{cutoff_freq[0]:.2e}_{cutoff_freq[1]:.2e}{custom_name}.wav", filtered_data, sample_rate)
        else:
            sf.write(f"{save_path}/{btype}_{cutoff_freq:.2e}{custom_name}.wav", filtered_data, sample_rate)

    
    # Filter along the time axis (axis=-1 handles both mono and stereo seamlessly)
    return filtered_data

if __name__ == "__main__":
    pre_path = "/Users/max/UCLA/Research/Codes"
    path = f"{pre_path}/Data/Gibson2021"


    i = 1
    s = f"{path}/sound-S{i}.mp3"

    cutoff_freq = [5e2, 3.7e3] # Hz

    btype = 'bandpass'
    # filtered_data = filter(s, cutoff_freq, btype=btype, save_path='', custom_name=f"-S{i}")
    # filtered_data = filter(s, cutoff_freq, btype=btype, save_path=f"{path}", custom_name=f"-S{i}")

