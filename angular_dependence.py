import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import matplotlib.colors as mcolors
import copy

import Lowson_solver as ls
import Poda_data as pod



plt.rcParams['font.size'] = 16

# global settings
fname = "/Users/max/UCLA/Research/Codes/Data/Poda2024/Database S1/all_data.mat"
path = "/Users/max/UCLA/Research/Codes"

test_source = False

save_fig = False

fs = 44100
start_time = 0 if test_source else 950 # s
target_duration = 0.05 # s
n_harm = 4

# method="percentile"
method="rms"
# method="max"

dB=False

# initialize swarm data
m = pod.mat(fname)
# fix trial number
tno = 4

if test_source:
    monopole_data = [ m.generate_test_insect(duration=target_duration, n_harmonics=n_harm, monopole=True) ]
    dipole_data = [ m.generate_test_insect(duration=target_duration, n_harmonics=n_harm, monopole=False) ]
else:
    monopole_data = m.format_all_smooth_insects(tno, n_harmonics=n_harm, t_start=start_time, duration=target_duration, monopole=True)
    dipole_data = m.format_all_smooth_insects(tno, n_harmonics=n_harm, t_start=start_time, duration=target_duration, monopole=False)




####################################################
# mic setup/field calculations
####################################################

r = 1.0
dtheta = np.pi/64
end_theta = 2*np.pi
thetas = np.arange(0, end_theta + dtheta - 0.01, dtheta)

if test_source:
    phis = [ 0.0 ]
else:
    dphi = np.pi / 3
    end_phi = 2*np.pi
    phis = np.arange(0, end_phi + dphi, dphi)


mic_positions = np.array([
    [r * np.array([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)]) for theta in thetas] for phi in phis
]) # m

view_mics = False
if view_mics:
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "3d"})
    positions = np.array([ np.array([mic_positions[i,j][0], mic_positions[i,j][1], mic_positions[i,j][2]]) for j in range(len(thetas)) for i in range(len(phis))])
    print(positions.shape)
    ax.scatter(positions[:,0], positions[:,1], positions[:,2])
    ax.set_xlabel("x[m]")
    ax.set_ylabel("y[m]")
    ax.set_zlabel("z[m]")
    plt.show()

st = start_time
t_end = target_duration # s

# center microphone array over the swarm
x = []
y = []
z = []
for insect in dipole_data:
    if insect['t'][0] <= st <= insect['t'][-1]:
        x.append(np.interp(st, insect['t'], insect['y'][:, 0]))
        y.append(np.interp(st, insect['t'], insect['y'][:, 1]))
        z.append(np.interp(st, insect['t'], insect['y'][:, 2]))
x_cm = np.mean(x)
y_cm = np.mean(y)
z_cm = np.mean(z)

cm_offset = np.array([x_cm, y_cm, z_cm])
mic_positions += cm_offset

t_audio = {
    'monopole': [],
    'dipole': [],
}
# deep copy ensures `p` contains no mutable linkage to `t_audio`
p = copy.deepcopy(t_audio)
pmag = copy.deepcopy(t_audio)

pmag['monopole'] = np.zeros((len(phis), len(thetas)))
pmag['dipole'] = np.zeros_like(pmag['monopole'])

print(f"Total calculations: {pmag['monopole'].size}")
for phi_idx in range(len(phis)):
    for theta_idx in range(len(thetas)):
        print(f"{phi_idx * len(thetas) + theta_idx + 1}/{pmag['monopole'].size}")
        m_t_audio,p_m = ls.swarm_sound_monopole(mic_positions[phi_idx, theta_idx], monopole_data)
        d_t_audio,p_d = ls.swarm_sound_dipole(mic_positions[phi_idx, theta_idx], dipole_data)

        p_m *= 1e6 # uPa
        p_d *= 1e6 # uPa

        t_start_idx = np.argmin(abs(d_t_audio - st))
        t_end_idx = np.argmin(abs(d_t_audio - (st + t_end)))
        
        p_m = p_m[t_start_idx:t_end_idx+1]
        p_d = p_d[t_start_idx:t_end_idx+1]
        m_t_audio = m_t_audio[t_start_idx:t_end_idx+1]
        d_t_audio = d_t_audio[t_start_idx:t_end_idx+1]

        t_audio['monopole'].append(m_t_audio)
        t_audio['dipole'  ].append(d_t_audio)
        
        p['monopole'].append(p_m)
        p['dipole'  ].append(p_d)
        
        if method == "max":
            pmag['monopole'][phi_idx,theta_idx] = np.max(abs(p_m))
            pmag['dipole'  ][phi_idx,theta_idx] = np.max(abs(p_d))
        
        elif method == "rms":
            if dB:
                p_ref = 20.0  # uPa reference in air
                p_m_rms = np.sqrt(np.mean(p_m**2))
                p_d_rms = np.sqrt(np.mean(p_d**2))
                pmag['monopole'][phi_idx, theta_idx] = 20 * np.log10(p_m_rms / p_ref)
                pmag['dipole'  ][phi_idx, theta_idx] = 20 * np.log10(p_d_rms / p_ref)
            else:
                pmag['monopole'][phi_idx, theta_idx] = np.sqrt(np.mean(p_m**2))
                pmag['dipole'  ][phi_idx, theta_idx] = np.sqrt(np.mean(p_d**2))
        
        elif method == "percentile":
            pmag['monopole'][phi_idx,theta_idx] = np.percentile(abs(p_m), 95)
            pmag['dipole'  ][phi_idx,theta_idx] = np.percentile(abs(p_d), 95)

####################################################
# plotting stuff
####################################################


# tick_positions = angles = np.radians(np.arange(0, 405, 45))
tick_positions = angles = np.radians(np.arange(0, 360, 45))
tick_labels = ['', '45°', '90°', '135°', '180°', '225°', '270°', '315°']
# tick_labels = [r'$0$', r'$\pi\,/\,4$', r'$\pi\,/\,2$', r'$3\pi\,/\,4$', r'$\pi$', r'$5\pi\,/\,4$', r'$3\pi\,/\,2$', r'$7\pi\,/\,4$', '']
# tick_labels = [r'$0$', r'$\pi\,/\,4$', r'$\pi\,/\,2$', r'$3\pi\,/\,4$', r'$\pi$']
# tick_labels = [r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$']

fig,ax = plt.subplots(figsize=(8 if test_source else 9, 8), subplot_kw={'projection': 'polar'})

ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

# ax.set_yticklabels([])
# uPa_color = "k"
# ax.text(
#         0.1, 0.2, r"Amplitude [$\mathregular{\mu}$Pa]",
#         transform=ax.transAxes,
#         ha='center', va='center', fontsize=14, fontweight='normal',
#         color=uPa_color,
#     )



if test_source:
    plt.plot(thetas, pmag['monopole'][0], c="#0032E8", label="Monopole", lw=2)
    plt.plot(thetas, pmag['dipole'  ][0], c="#E00D0D", label="Dipole", ls='--', lw=2)
else:
    # base_cmap = plt.get_cmap("hsv")
    # hsv_colors = base_cmap(np.linspace(0,1,256))
    # hsv_colors[:,:3] *= 0.8
    # cmap = mcolors.ListedColormap(hsv_colors)
    cmap = plt.get_cmap("Greys")
    colors = cmap(np.linspace(0, 1, len(phis)))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(np.degrees(phis)), vmax=max(np.degrees(phis))))
    
    for idx in range(len(phis)):
        ax.plot(thetas, pmag['monopole'][idx], c=colors[idx], label="Monopole" if idx==0 else None, lw=2)
        ax.plot(thetas, pmag['dipole'  ][idx], c=colors[idx], label="Dipole" if idx==0 else None, ls='--', lw=2)

leg = plt.legend(bbox_to_anchor=(1.15, 1.15))
if not test_source:
    for legline in leg.get_lines():
        legline.set_color("k")

if test_source:
    plt.text(
        0.5, 1.05,
        "3D Polar angle [rad]",
        transform=ax.transAxes,
        ha='center',
        va='center',
        )
    plt.ylim(bottom=0)
else:
    theta_text = plt.text(
        0.5, 1.035,
        r"$\theta$",
        transform=ax.transAxes,
        ha='center',
        va='center',
        fontsize=20
        )
    theta_text.set_usetex(False)

    cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label("Azimuthal angle", labelpad=12, fontsize=20, usetex=False)
    # cbar.set_label(r"$\phi$", labelpad=12, fontsize=20, usetex=False)
    cbar.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}°"))

    cbar.set_ticks(np.degrees(phis))
    # cbar.set_ticklabels([r'$0$', r'$\frac{\pi}{2}$', '', r'$2\pi$'])

ax.xaxis.set_label_position('top')

# plt.ylabel(r"Sound wave amplitude [$\mathregular{\mu}$Pa]")
plt.xlim(min(thetas), end_theta)

plt.xticks(tick_positions, tick_labels, fontsize=16)
ax.xaxis.set_minor_locator(AutoMinorLocator(4))
ax.tick_params(axis='x', which='minor', length=4, width=1)
ax.tick_params(axis='x', which='major', length=8, width=1)

# spokes
ax.grid(axis='x', alpha=0.3)

ax.set_rlabel_position(0) # deg
uPa_color = "gray"
ax.grid(axis='y', alpha=0.3, color=uPa_color)
ax.tick_params(axis='y', labelcolor=uPa_color, labelsize=10)

def r_formatter(val, _):
    ticks = ax.get_yticks()
    # Check if this tick corresponds to the second-to-last tick value
    if len(ticks) > 1 and val == ticks[0]:
        if method=="rms" and dB:
            return rf"{val:g} [dB]"
        else:
            return rf"{val:g} [$\mathregular{{\mu}}$Pa]"
    return f"{val:g}"

ax.yaxis.set_major_formatter(ticker.FuncFormatter(r_formatter))

ax.text(
        0.5, 1.2, f"{r:.2f} m from swarm CM",
        transform=ax.transAxes,
        ha='center', va='center',
        # fontsize=16,
        fontweight='bold',
    )

ax.annotate(
    "",
    xy=(0.5, 0.55),      # Arrow head position (axes fraction)
    xytext=(0.5, 0.45),  # Arrow tail position (axes fraction)
    xycoords='axes fraction',
    arrowprops=dict(
        arrowstyle="->",
        color='k',
        lw=1.75,
        mutation_scale=12  # Controls arrow head size
    )
)

ax.text(
        0.52, 0.495, r"$\hat{z}$",
        transform=ax.transAxes,
        ha='center', va='center',
        fontsize=18,
        # fontweight='bold',
    )

if save_fig:
    if test_source:
        plt.savefig(f"{path}/Figures/test_source_angular_dependence_{r:.2f}_m_{target_duration:.2f}_s_{method}.png", dpi=800)
    else:
        plt.savefig(f"{path}/Figures/Poda_{tno+1}_angular_dependence_{r:.2f}_m_{target_duration:.2f}_s_{method}.png", dpi=800)

plt.show()

