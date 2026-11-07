import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

plt.rcParams['font.size'] = 16

# def raw_noise(size: int):
#     return [i for i in range(size)], np.random.uniform(-15.0, 15.0, size)

target_fs = 44.1e3 # Hz
duration = 10 # s
# control_dt = 0.1 # s
t_control = [ 0.0 ]
while t_control[-1] < duration:
    control_step = np.random.uniform(0.1, 0.35)
    t_control.append(t_control[-1] + control_step)
t_control[-1] = duration

# t_control = np.arange(0.0, duration, control_dt)
# drift_control = np.random.uniform(-15.0, 15.0, len(t_control))

drift_control = np.random.normal(0.0, 15.0, len(t_control))

cs = CubicSpline(t_control, drift_control, bc_type='natural')

t_audio = np.arange(0.0, duration, 1.0 / target_fs)
smooth_drift = cs(t_audio)

# sigma = [ 5, 10, 50, 100 ]

fig = plt.figure(figsize=(14,8))

# plt.plot(t_control, drift_control)
plt.plot(t_audio, smooth_drift, c="tab:red", alpha=0.8)
# plt.scatter(t_control, drift_control, color='k')
plt.plot(t_control, drift_control, 'ko', markersize=3.25)


# rn2 = raw_noise(1000)
# plt.plot(rn2[0], rn2[1], c='k', alpha=0.3, label="Raw noise")
# for sig in sigma:
#     plt.plot(rn2[0], gaussian_filter1d(rn2[1], sigma=sig), label=fr"$\sigma = {sig}$")
# plt.legend(bbox_to_anchor=(.1,1), ncols=6)

plt.show()
