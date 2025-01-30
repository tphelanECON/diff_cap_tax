"""
Figures for the case of relaxed collateral constraints. Relative to tight.py, this script also
contains expressions for revenue and transfers. Transfers in the implementation with untaxed
workers and common human wealth when no-absconding constraint holds as an inequality:

    h_W = h_E = (Pi_hat  + delta)(1/alpha-1)K/rho
    eta_W = kappa_E*np.exp(-x**2/2) - (Pi_hat  + delta)(1/alpha-1)/rho
    eta_E = kappa_E - (Pi_hat  + delta)(1/alpha-1)/rho

These expressions follow from the proofs of the main decentralization in the appendix.
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

print("Benchmark case, relaxed")
iotabar = np.sqrt(rho)*sigma/(rho*np.exp(1/2))
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
    # revenue in benchmark implementation:
    Y[n]['revenue_benchmark'] = np.zeros(len(phigrid))
    # wealth of newborns in benchmark implementation:
    Y[n]['kappaE'] = np.zeros(len(phigrid))
    Y[n]['kappaE2'] = np.zeros(len(phigrid))
    Y[n]['kappaW'] = np.zeros(len(phigrid))
    Y[n]['etaE'] = np.zeros(len(phigrid))
    Y[n]['etaW'] = np.zeros(len(phigrid))
    for i in range(len(phigrid)):
        if i % 100 == 0:
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
        Y[n]['omegabar_d'][i] = X[n].omegabar_d(Y[n]['S'][i], Y[n]['omegabar'][i])
        Y[n]['nu_B'][i] = X[n].nu_B(Y[n]['S'][i], Y[n]['omegabar'][i])
        Y[n]['nu_K'][i] = X[n].nu_K(Y[n]['S'][i], Y[n]['omegabar'][i], Y[n]['Pi'][i])
        Y[n]['r_b'][i] = X[n].r_b(Y[n]['Pi'][i], X[n].phigrid[i])
        # additional quantities only plotted for relaxed case:
        Y[n]['revenue_benchmark'][i] = X[n].revenue_benchmark(Y[n]['Pi'][i], X[n].phigrid[i], Y[n]['omegabar'][i])
        # transfers to newborns in benchmark implementation (assuming no-absconding constraint is strict):
        num = (Y[n]['Pi'][i]/X[n].alpha + X[n].delta*(1/X[n].alpha-1))/X[n].rho
        denom = 1 - X[n].psi + X[n].psi*np.exp(-Y[n]['x'][i]**2/2)
        Y[n]['kappaE'][i] = num/denom
        Y[n]['kappaE2'][i] = X[n].phigrid[i]*X[n].sigma/(np.sqrt(X[n].rho)*Y[n]['x'][i]*(1-X[n].psi))
        Y[n]['kappaW'][i] = Y[n]['kappaE'][i]*np.exp(-Y[n]['x'][i]**2/2)
        Y[n]['etaE'][i] = Y[n]['kappaE'][i] - (Y[n]['Pi'][i] + X[n].delta)*(1/X[n].alpha-1)/X[n].rho
        Y[n]['etaW'][i] = Y[n]['kappaE'][i]*np.exp(-Y[n]['x'][i]**2/2) - (Y[n]['Pi'][i] + X[n].delta)*(1/X[n].alpha-1)/X[n].rho
        # final checks on sufficiency:
        Y[n]['check1'][i] = X[n].check1(Y[n]['S'][i], Y[n]['omegabar'][i])
        Y[n]['check2'][i] = X[n].check2(Y[n]['S'][i], Y[n]['omegabar'][i], Y[n]['x'][i])

"""
Check that the assumptions in Appendix A are satisfied.
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
Revenue in implementation with untaxed workers and common human wealth
"""

fig,ax = plt.subplots()
for n,psi in enumerate(psi_ratio):
    ax.plot(X[n].phigrid, 100*Y[n]['revenue_benchmark'], label=r'$\psi$ = {0}'.format(np.round(psi, 3)),
            c=cmap.to_rgba((n+1)/len(psi_ratio)), linewidth=2.0)
ax.legend(loc='upper left')
plt.ylabel("Percent (%)", fontsize=18)
plt.xlabel("Agency friction parameter $\phi$", fontsize=18)
ax.set_title(r'Average tax revenue as fraction of income', fontsize=18)
destin = '../main/figures/revenue_benchmark{0}.png'.format(iotaround)
plt.savefig(destin, format='png', dpi=1000)
if show == 1:
    plt.show()
plt.close('all')

"""
Ratio of initial wealth
"""

fig, ax = plt.subplots()
for n, psi in enumerate(psi_ratio):
    ax.plot(X[n].phigrid, Y[n]['etaE']/Y[n]['etaW'], label=r'$\psi$ = {0}'.format(np.round(psi, 3)),
            c=cmap.to_rgba((n+1)/len(psi_ratio)), linewidth=2.0)
ax.legend(loc='upper left')
plt.xlabel("Agency friction parameter $\phi$", fontsize=18)
ax.set_title(r'Ratio of initial wealth $\eta_E/\eta_W$', fontsize=18)
destin = '../main/figures/eta_rat{0}.png'.format(iotaround)
plt.savefig(destin, format='png', dpi=1000)
if show == 1:
    plt.show()
plt.close('all')
