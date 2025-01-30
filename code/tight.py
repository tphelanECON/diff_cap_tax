"""
Figures for the case of tight collateral constraints.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# following added to meet IER requirements:
plt.rcParams["font.family"] = "Times New Roman"
import classes
import parameters

alpha = parameters.alpha
sigma = parameters.sigma
rhoS = parameters.rhoS
rhoD = parameters.rhoD
delta = parameters.delta
psi_ratio = parameters.psi_ratio
phigrid = parameters.phigrid
rho = parameters.rho

norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.Blues)
show = 0

"""
iotabar = 1
"""

print("Benchmark case, tight")
iotabar = 1.0
iotaround = np.round(iotabar, 2)

X, Y = {}, {}
for n, psi in enumerate(psi_ratio):
    print('$\psi$ = {0}'.format(psi))
    Y[n] = {}
    Y[n]['tauPi'] = np.zeros(len(phigrid))
    Y[n]['S'] = np.zeros(len(phigrid))
    Y[n]['Pi'] = np.zeros(len(phigrid))
    Y[n]['x'] = np.zeros(len(phigrid))
    Y[n]['mu_c'] = np.zeros(len(phigrid))
    Y[n]['sig_c'] = np.zeros(len(phigrid))
    Y[n]['omegabar'] = np.zeros(len(phigrid))
    Y[n]['omegabar_d'] = np.zeros(len(phigrid))
    Y[n]['check1'] = np.zeros(len(phigrid))
    Y[n]['check2'] = np.zeros(len(phigrid))
    # wedges and cost of borrowing:
    Y[n]['nu_B'] = np.zeros(len(phigrid))
    Y[n]['nu_K'] = np.zeros(len(phigrid))
    Y[n]['r_b'] = np.zeros(len(phigrid))
    for i in range(len(phigrid)):
        if i % 50 == 0:
            print(i)
        X[n] = classes.captax(alpha=alpha, rhoD=rhoD, rhoS=rhoS, sigma=sigma, delta=delta, psi=psi, iota=iotabar*phigrid[i])
        # quantities pertaining to efficient allocation:
        Y[n]['S'][i] = X[n].S_hat(X[n].phigrid[i])
        Y[n]['omegabar'][i] = np.sqrt(X[n].rho)*X[n].phigrid[i]*X[n].sigma/(X[n].rho*X[n].iota)
        Y[n]['x'][i] = X[n].x(Y[n]['S'][i], Y[n]['omegabar'][i])
        Y[n]['Pi'][i] = X[n].rhoS + Y[n]['S'][i]*np.sqrt(X[n].rho)*X[n].phigrid[i]*X[n].sigma
        Y[n]['mu_c'][i] = X[n].mu_c(Y[n]['S'][i], Y[n]['omegabar'][i])
        Y[n]['sig_c'][i] = X[n].sig_c(Y[n]['S'][i], Y[n]['omegabar'][i])
        # equilibrium allocations:
        Y[n]['tauPi'][i] = 1 - X[n].phigrid[i]
        Y[n]['omegabar_d'][i] = X[n].omegabar_d(Y[n]['S'][i],Y[n]['omegabar'][i])
        Y[n]['nu_B'][i] = X[n].nu_B(Y[n]['S'][i], Y[n]['omegabar'][i])
        Y[n]['nu_K'][i] = X[n].nu_K(Y[n]['S'][i], Y[n]['omegabar'][i], Y[n]['Pi'][i])
        Y[n]['r_b'][i] = X[n].r_b(Y[n]['Pi'][i], X[n].phigrid[i])
        # final checks on sufficiency:
        Y[n]['check1'][i] = X[n].check1(Y[n]['S'][i], Y[n]['omegabar'][i])
        Y[n]['check2'][i] = X[n].check2(Y[n]['S'][i], Y[n]['omegabar'][i], Y[n]['x'][i])

"""
Check that assumptions in Appendix A are satisfied (these ensure that there is no arbitrage 
opportunity and that the contracting problem is well-defined).
"""

print("Assumptions satisfied?")
for n, psi in enumerate(psi_ratio):
    print('$\psi$ = {0}: {1}'.format(psi, Y[n]['check1'].all()*Y[n]['check2'].all()))

"""
Figures
"""

"""
Wedge on risk-free asset and efficient cost of borrowing
"""

fig, ax = plt.subplots()
for n, psi in enumerate(psi_ratio):
    ax.plot(X[n].phigrid, 100*Y[n]['nu_B'], label=r'$\psi$ = {0}'.format(np.round(psi, 3)),
            c=cmap.to_rgba((n+1)/len(psi_ratio)), linewidth=2.0)
ax.legend(loc='upper left')
plt.ylabel("Percent (%)", fontsize=18)
plt.xlabel("Agency friction parameter $\phi$", fontsize=18)
ax.set_title(r'Savings wedge on entrepreneurs $\nu_B$', fontsize=18)
destin = '../main/figures/nu_B{0}.png'.format(iotaround)
plt.savefig(destin, format='png', dpi=1000)
if show == 1:
    plt.show()
plt.close('all')

fig, ax = plt.subplots()
for n, psi in enumerate(psi_ratio):
    ax.plot(X[n].phigrid, 100*Y[n]['r_b'], label=r'$\psi$ = {0}'.format(np.round(psi, 3)),
            c=cmap.to_rgba((n+1)/len(psi_ratio)), linewidth=2.0)
ax.legend(loc='lower left')
plt.ylabel("Percent (%)", fontsize=18)
plt.xlabel("Agency friction parameter $\phi$", fontsize=18)
ax.set_title(r'Efficient cost of borrowing $r_{b}$', fontsize=18)
destin = '../main/figures/r_b{0}.png'.format(iotaround)
plt.savefig(destin, format='png', dpi=1000)
if show == 1:
    plt.show()
plt.close('all')

"""
Drift and diffusion of consumption growth
"""

fig, ax = plt.subplots()
for n, psi in enumerate(psi_ratio):
    ax.plot(X[n].phigrid, 100*Y[n]['mu_c'], label=r'$\psi$ = {0}'.format(np.round(psi, 3)),
            c=cmap.to_rgba((n+1)/len(psi_ratio)), linewidth=2.0)
ax.legend(loc='upper left')
plt.ylabel("Percent (%)", fontsize=18)
plt.xlabel("Agency friction parameter $\phi$", fontsize=18)
ax.set_title(r'Mean consumption growth $\mu_c$', fontsize=18)
destin = '../main/figures/mu_c{0}.png'.format(iotabar)
plt.savefig(destin, format='png', dpi=1000)
if show == 1:
    plt.show()
plt.close('all')

fig, ax = plt.subplots()
for n, psi in enumerate(psi_ratio):
    ax.plot(X[n].phigrid, Y[n]['sig_c'], label=r'$\psi$ = {0}'.format(np.round(psi, 3)),
            c=cmap.to_rgba((n+1)/len(psi_ratio)), linewidth=2.0)
ax.legend(loc='upper left')
#plt.ylabel("Percent (%)", fontsize=18)
plt.xlabel("Agency friction parameter $\phi$", fontsize=18)
ax.set_title(r'Volatility of consumption growth $\sigma_c$', fontsize=18)
destin = '../main/figures/sig_c{0}.png'.format(iotabar)
plt.savefig(destin, format='png', dpi=1000)
if show == 1:
    plt.show()
plt.close('all')
