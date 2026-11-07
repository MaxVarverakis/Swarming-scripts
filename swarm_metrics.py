import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

fs = 16

pre_path = "/Users/max/UCLA/Research/Codes"
path = f"{pre_path}/Data/JF2026_Fig2_Gamma_Eta_sweep_open_BC/"

gammas = [ 1.0, 2.0, 2.5, 3.0 ]
etas = [ 0.0, 0.1, 1.0 ]
rcid = [ (0,0), (0,1), (1,0), (1,1) ]
colors = [ "#00cd6c" ,"#af58ba", "#f28522" ]
share = True
fig, ax = plt.subplots(nrows=2, ncols=2, sharex=share, sharey=share, figsize=(7, 7), gridspec_kw={'wspace': 0.1, 'hspace': 0.15})

# time-average density stuff
# gammas = [ 0.0, 1.0, 2.0 ]
# etas = [ 0.1 ]
# hist_colors = ["#0000FF", "#FF6F00", "#00C76A"]
# fig = plt.figure(figsize=(6,6))


for gid,gamma in enumerate(gammas):
    for eid,eta in enumerate(etas):
        end_path = f"_data_g_{gamma:.1f}_eta_{eta:.1f}.txt"
        data_path = path + "swarm" + end_path
        meta_path = path + "meta"  + end_path

        df = pd.read_csv(data_path, header=0)
        md = pd.read_csv(meta_path, header=0)

        if gid + eid == 0:
            N = md.N[0]
            step_count = int(df.index.stop / N)
            steps = np.arange(step_count, step=1)
            step_start_idx = np.argmin(abs(steps - int(1e2)))
            late_steps = steps[step_start_idx:]
            ls = late_steps
            ls2 = late_steps**2

            # make both start at <r^2> = 1e3 for step = 1e2
            ls = ls / ls[0] * 1e3
            ls2 = ls2 / ls2[0] * 1e3

        # cm = np.zeros((step_count, 2))
        Rs = np.zeros(step_count)

        x = df.x.to_numpy().reshape(step_count, N)
        y = df.y.to_numpy().reshape(step_count, N)
        # check to make sure resize was done correctly
        # print(np.all(df.x[::N].to_numpy() == x[:,0]))

        cm_x = np.mean(x, axis=1, keepdims=True)
        cm_y = np.mean(y, axis=1, keepdims=True)
        Rs   = np.mean((x - cm_x)**2 + (y - cm_y)**2, axis=1) # raidal variance: <r^2>

        # time-average density stuff
        # D = np.sqrt((x - cm_x)**2 + (y - cm_y)**2).flatten()
        # b_edges = np.linspace(0, 50, 51) # adjust third arg to change bin width
        # b_width = b_edges[1] - b_edges[0]
        # b_cents = b_edges[1:] - b_width / 2
        # areas   = np.pi * (b_edges[1:]**2 - b_edges[:-1]**2)
        # counts  = np.histogram(D, bins=b_edges)[0]
        # rho     = counts / (areas * step_count)
        # plt.plot(b_cents, rho, c=hist_colors[gid], label=fr"$\gamma = $ {gamma:.0f}")

        p = ax[rcid[gid]]
        p.loglog(steps, Rs, c = colors[eid], lw=2, label=fr'$\eta = $ {md.eta[0]:.1f}')

        # looks like these are set to start at a pre-selected y-value?
        p.loglog(late_steps, ls, c='r', ls=':', lw=2)
        p.loglog(late_steps, ls2, c='k', ls=':', lw=2)

        p.set_title(fr'$\gamma = $ {md.gamma[0]:.1f}', fontsize = fs)

        p.xaxis.set_major_locator(ticker.FixedLocator([1, 10, 100, 1000, 10000, 100000]))
        p.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
        
        p.yaxis.set_major_locator(ticker.FixedLocator([10, 1000, 100000, 10000000, 1000000000]))
        p.yaxis.set_major_formatter(ticker.LogFormatterMathtext())

        p.tick_params(axis='both', labelsize=fs-3)

        # p.set_xlabel("Time steps", fontsize = fs-3)
        # p.set_ylabel("Radial Variance", fontsize = fs-3)

        p.set_xlim(left=1) #, right=2e5)
        p.set_ylim(bottom=10)

for i in [0,1]:
    ax[1,i].set_xlabel("Time steps", fontsize = fs-3)
    ax[i,0].set_ylabel("Radial Variance", fontsize = fs-3)
ax[0,0].legend(fontsize = fs-2)

# plt.savefig(f"{pre_path}/Figures/gamma_eta_sweep.png", dpi = 500)

# time-average density stuff
# plt.ylim(0.0, 0.7)
# plt.xlim(0, md.L[0] / 2)

# plt.xlabel("Distance from center", fontsize=fs-2)
# plt.ylabel("Density [insects/area]", fontsize=fs-2)
# plt.xticks([0, md.L[0]/4, md.L[0]/2], labels=["0", r"$\frac{L}{4}$", r"$\frac{L}{2}$"], fontsize=fs)
# plt.yticks(fontsize=fs-3)
# plt.legend(fontsize=fs-2)

# plt.savefig(f"{pre_path}/Figures/time_avg_density_profiles.png", dpi = 500)

plt.show()
