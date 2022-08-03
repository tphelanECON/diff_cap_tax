"""
Extensions. This script computes the taxes when there is private risk-sharing.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.sparse as sp
import classes, parameters
import scipy.optimize as scopt

#general parameters fixed throughout
alpha = parameters.alpha
sigma = parameters.sigma
rhoS = parameters.rhoS
rhoD = parameters.rhoD
delta = parameters.delta
psi_ratio = parameters.psi_ratio
phigrid = parameters.phigrid
rho = parameters.rho
phimin, phimax = parameters.phimin, parameters.phimax

#loose value of collateral constraints
iotabar = np.sqrt(rho)*sigma/(rho*np.exp(1/2))

X, Y = {}, {}
for n,psi in enumerate(psi_ratio):
    Y[n] = {}
    Y[n]['Pi'], Y[n]['tauPi'] = np.zeros(len(phigrid)), np.zeros(len(phigrid))
    Y[n]['taus'], Y[n]['tausW'] = np.zeros(len(phigrid)), np.zeros(len(phigrid))
    Y[n]['taus2'], Y[n]['tausW2'] = np.zeros(len(phigrid)), np.zeros(len(phigrid))
    Y[n]['r'], Y[n]['r2'] = np.zeros(len(phigrid)), np.zeros(len(phigrid))
    Y[n]['S'], Y[n]['rE'] = np.zeros(len(phigrid)), np.zeros(len(phigrid))
    Y[n]['mu_c'] = np.zeros(len(phigrid))
    Y[n]['sig_c'] = np.zeros(len(phigrid))
    Y[n]['x'] = np.zeros(len(phigrid))
    Y[n]['omegahat'] = np.zeros(len(phigrid))
    Y[n]['leverage'] = np.zeros(len(phigrid))
    for i in range(len(phigrid)):
        print(i)
        X[n] = classes.captax_general(alpha=alpha,phimin=phimin,phimax=phimax,rhoD=rhoD,rhoS=rhoS,sigma=sigma,delta=delta,psi=psi,iota=iotabar*phigrid[i])
        Y[n]['S'][i] = X[n].S_hat(X[n].phigrid[i],X[n].sigma)
        Y[n]['Pi'][i] = X[n].rhoS + Y[n]['S'][i]*np.sqrt(X[n].rho)*X[n].phigrid[i]*X[n].sigma
        Y[n]['tauPi'][i] = 1 - X[n].phigrid[i]
        Y[n]['taus2'][i] = X[n].taus2(Y[n]['Pi'][i],X[n].phigrid[i],X[n].sigma)
        Y[n]['tausW2'][i] = X[n].tausW2(Y[n]['Pi'][i],X[n].phigrid[i],X[n].sigma)
        Y[n]['r2'][i] = X[n].r2(Y[n]['Pi'][i],X[n].phigrid[i],X[n].sigma)
        omegabar = np.sqrt(X[n].rho)*X[n].phigrid[i]*X[n].sigma/(X[n].rho*X[n].iota)
        Y[n]['x'][i] = X[n].x(Y[n]['S'][i],omegabar)
        Y[n]['mu_c'][i] = X[n].mu_c(Y[n]['S'][i],omegabar)
        Y[n]['sig_c'][i] = X[n].sig_c(Y[n]['S'][i],omegabar)
        Y[n]['rE'][i] = X[n].rho*(1 - Y[n]['x'][i]**2 + Y[n]['mu_c'][i]/X[n].rho)
        Y[n]['omegahat'][i] = X[n].omegahat(Y[n]['S'][i],omegabar)
        Y[n]['leverage'][i] = Y[n]['sig_c'][i]/(X[n].phigrid[i]*X[n].sigma)

norm = matplotlib.colors.Normalize(vmin=-psi_ratio[-1]**2/2, vmax=psi_ratio[-1]**2)
cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.Blues)

fig,ax = plt.subplots()
for n,psi in enumerate(psi_ratio):
    ax.plot(X[n].phigrid,100*Y[n]['taus2'],label=r'$\psi$ = {0}'.format(np.round(psi,3)),c=cmap.to_rgba(psi**2),linewidth=1)
ax.legend(loc='upper left')
plt.ylabel("Tax (%)")
plt.xlabel("$\phi$")
ax.set_title(r'Savings tax on entrepreneurs (with private equity)', fontsize=13)
destin = '../main/figures/taxes_entrepreneurs_pe.eps'.format(iotabar)
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

fig,ax = plt.subplots()
for n,psi in enumerate(psi_ratio):
    ax.plot(X[n].phigrid,100*Y[n]['tausW2'],label=r'$\psi$ = {0}'.format(np.round(psi,3)),c=cmap.to_rgba(psi**2),linewidth=1)
ax.legend(loc='lower left')
plt.ylabel("Tax (%)")
plt.xlabel("$\phi$")
ax.set_title(r'Savings tax on workers (with private equity)', fontsize=13)
destin = '../main/figures/taxes_workers_pe.eps'.format(psi_ratio.index(psi))
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

fig,ax = plt.subplots()
for n,psi in enumerate(psi_ratio):
    ax.plot(X[n].phigrid, 100*Y[n]['r2'], label=r'$\psi$ = {0}'.format(np.round(psi,3)),c=cmap.to_rgba(psi**2),linewidth=1)
ax.legend(loc='lower left')
plt.ylabel("Percent (%)")
plt.xlabel("$\phi$")
ax.set_title("Interest rate (with private equity)", fontsize=13)
destin = '../main/figures/r_pe.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()
