"""
This script computes the taxes when there is private risk-sharing.

Suffix convention: no qualifier if collateral constraints at relaxed value, if
collateral constraints tight, 1.0 suffix used.
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

norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.Blues)

print("Private equity, relaxed")
iotabar = np.sqrt(rho)*sigma/(rho*np.exp(1/2))

X, Y = {}, {}
for n,psi in enumerate(psi_ratio):
    print('$\psi$ = {0}'.format(psi))
    Y[n] = {}
    Y[n]['Pi'], Y[n]['tauPi'] = np.zeros(len(phigrid)), np.zeros(len(phigrid))
    Y[n]['taus_pe'], Y[n]['tausW_pe'] = np.zeros(len(phigrid)), np.zeros(len(phigrid))
    Y[n]['r_pe'], Y[n]['rE'] = np.zeros(len(phigrid)), np.zeros(len(phigrid))
    Y[n]['S'], Y[n]['x'] = np.zeros(len(phigrid)), np.zeros(len(phigrid))
    Y[n]['mu_c'], Y[n]['sig_c'] = np.zeros(len(phigrid)), np.zeros(len(phigrid))
    Y[n]['omegahat'], Y[n]['omegabar'] = np.zeros(len(phigrid)), np.zeros(len(phigrid))
    Y[n]['check1'], Y[n]['check2'] = np.zeros(len(phigrid)), np.zeros(len(phigrid))
    Y[n]['leverage'] = np.zeros(len(phigrid))
    for i in range(len(phigrid)):
        if i % 25 == 0:
            print(i)
        X[n] = classes.captax(alpha=alpha,rhoD=rhoD,rhoS=rhoS,sigma=sigma,delta=delta,psi=psi,iota=iotabar*phigrid[i])
        Y[n]['S'][i] = X[n].S_hat(X[n].phigrid[i])
        Y[n]['omegabar'][i] = np.sqrt(X[n].rho)*X[n].phigrid[i]*X[n].sigma/(X[n].rho*X[n].iota)
        Y[n]['x'][i] = X[n].x(Y[n]['S'][i],Y[n]['omegabar'][i])
        Y[n]['Pi'][i] = X[n].rhoS + Y[n]['S'][i]*np.sqrt(X[n].rho)*X[n].phigrid[i]*X[n].sigma
        Y[n]['tauPi'][i] = 1 - X[n].phigrid[i]
        Y[n]['taus_pe'][i] = X[n].taus_pe(Y[n]['Pi'][i],X[n].phigrid[i])
        Y[n]['tausW_pe'][i] = X[n].tausW_pe(Y[n]['Pi'][i],X[n].phigrid[i])
        Y[n]['r_pe'][i] = X[n].r_pe(Y[n]['Pi'][i],X[n].phigrid[i])
        Y[n]['mu_c'][i] = X[n].mu_c(Y[n]['S'][i],Y[n]['omegabar'][i])
        Y[n]['sig_c'][i] = X[n].sig_c(Y[n]['S'][i],Y[n]['omegabar'][i])
        Y[n]['rE'][i] = X[n].rho*(1 - Y[n]['x'][i]**2 + Y[n]['mu_c'][i]/X[n].rho)
        Y[n]['omegahat'][i] = X[n].omegahat(Y[n]['S'][i],Y[n]['omegabar'][i])
        Y[n]['check1'][i] = X[n].check1(Y[n]['S'][i],Y[n]['omegabar'][i])
        Y[n]['check2'][i] = X[n].check2(Y[n]['S'][i],Y[n]['omegabar'][i], Y[n]['x'][i])
        Y[n]['leverage'][i] = Y[n]['sig_c'][i]/(X[n].phigrid[i]*X[n].sigma)

"""
Check that the assumptions in Appendix A.2 are satisfied (these ensure that
there is no arbitrage opportunity and that the contracting problem is well-defined)
"""

for n,psi in enumerate(psi_ratio):
    print("Assumptions satisfied for " + '$\psi$ = {0}'.format(psi),"?")
    print(Y[n]['check1'].all()*Y[n]['check2'].all())

"""
Figures
"""

fig,ax = plt.subplots()
for n,psi in enumerate(psi_ratio):
    ax.plot(X[n].phigrid,100*Y[n]['taus_pe'],label=r'$\psi$ = {0}'.format(np.round(psi,3)),c=cmap.to_rgba((n+1)/len(psi_ratio)),linewidth=1.25)
ax.legend(loc='upper left')
plt.ylabel("Tax (%)")
plt.xlabel("$\phi$")
ax.set_title(r'Savings tax on entrepreneurs with private equity', fontsize=13)
destin = '../main/figures/taxes_entrepreneurs_pe.eps'
plt.savefig(destin, format='eps', dpi=1000)
#plt.show()

fig,ax = plt.subplots()
for n,psi in enumerate(psi_ratio):
    ax.plot(X[n].phigrid,100*Y[n]['tausW_pe'],label=r'$\psi$ = {0}'.format(np.round(psi,3)),c=cmap.to_rgba((n+1)/len(psi_ratio)),linewidth=1.25)
ax.legend(loc='lower left')
plt.ylabel("Tax (%)")
plt.xlabel("$\phi$")
ax.set_title(r'Savings tax on workers with private equity', fontsize=13)
destin = '../main/figures/taxes_workers_pe.eps'
plt.savefig(destin, format='eps', dpi=1000)
#plt.show()

fig,ax = plt.subplots()
for n,psi in enumerate(psi_ratio):
    ax.plot(X[n].phigrid, 100*Y[n]['r_pe'], label=r'$\psi$ = {0}'.format(np.round(psi,3)),c=cmap.to_rgba((n+1)/len(psi_ratio)),linewidth=1.25)
ax.legend(loc='lower left')
plt.ylabel("Percent (%)")
plt.xlabel("$\phi$")
ax.set_title("Interest rate with private equity", fontsize=13)
destin = '../main/figures/r_pe.eps'
plt.savefig(destin, format='eps', dpi=1000)
#plt.show()

fig,ax = plt.subplots()
for n,psi in enumerate(psi_ratio):
    ax.plot(X[n].phigrid,100*(Y[n]['Pi'] - Y[n]['r_pe']),label=r'$\psi$ = {0}'.format(np.round(psi,3)),c=cmap.to_rgba((n+1)/len(psi_ratio)),linewidth=1.25)
ax.legend(loc='upper left')
plt.ylabel("Percent (%)")
plt.xlabel("$\phi$")
ax.set_title("Excess return $\hat{\Pi} - r$ with private equity", fontsize=13)
destin = '../main/figures/excess_pe.eps'
plt.savefig(destin, format='eps', dpi=1000)
#plt.show()

"""
Tighter collateral constraints
"""

print("Private equity, tight")
iotabar = 1.0

X, Y = {}, {}
for n,psi in enumerate(psi_ratio):
    print('$psi$ = {0}'.format(psi))
    Y[n] = {}
    Y[n]['Pi'], Y[n]['tauPi'] = np.zeros(len(phigrid)), np.zeros(len(phigrid))
    Y[n]['taus'], Y[n]['tausW'] = np.zeros(len(phigrid)), np.zeros(len(phigrid))
    Y[n]['taus_pe'], Y[n]['tausW_pe'] = np.zeros(len(phigrid)), np.zeros(len(phigrid))
    Y[n]['r'], Y[n]['r_pe'] = np.zeros(len(phigrid)), np.zeros(len(phigrid))
    Y[n]['S'], Y[n]['rE'] = np.zeros(len(phigrid)), np.zeros(len(phigrid))
    Y[n]['mu_c'], Y[n]['sig_c'] = np.zeros(len(phigrid)), np.zeros(len(phigrid))
    Y[n]['x'] = np.zeros(len(phigrid))
    Y[n]['omegahat'] = np.zeros(len(phigrid))
    Y[n]['leverage'] = np.zeros(len(phigrid))
    for i in range(len(phigrid)):
        if i % 25 == 0:
            print(i)
        X[n] = classes.captax(alpha=alpha,rhoD=rhoD,rhoS=rhoS,sigma=sigma,delta=delta,psi=psi,iota=iotabar*phigrid[i])
        Y[n]['S'][i] = X[n].S_hat(X[n].phigrid[i])
        Y[n]['Pi'][i] = X[n].rhoS + Y[n]['S'][i]*np.sqrt(X[n].rho)*X[n].phigrid[i]*X[n].sigma
        Y[n]['tauPi'][i] = 1 - X[n].phigrid[i]
        Y[n]['taus_pe'][i] = X[n].taus_pe(Y[n]['Pi'][i],X[n].phigrid[i])
        Y[n]['tausW_pe'][i] = X[n].tausW_pe(Y[n]['Pi'][i],X[n].phigrid[i])
        Y[n]['r_pe'][i] = X[n].r_pe(Y[n]['Pi'][i],X[n].phigrid[i])
        omegabar = np.sqrt(X[n].rho)*X[n].phigrid[i]*X[n].sigma/(X[n].rho*X[n].iota)
        Y[n]['x'][i] = X[n].x(Y[n]['S'][i],omegabar)
        Y[n]['mu_c'][i] = X[n].mu_c(Y[n]['S'][i],omegabar)
        Y[n]['sig_c'][i] = X[n].sig_c(Y[n]['S'][i],omegabar)
        Y[n]['rE'][i] = X[n].rho*(1 - Y[n]['x'][i]**2 + Y[n]['mu_c'][i]/X[n].rho)
        Y[n]['omegahat'][i] = X[n].omegahat(Y[n]['S'][i],omegabar)
        Y[n]['leverage'][i] = Y[n]['sig_c'][i]/(X[n].phigrid[i]*X[n].sigma)

fig,ax = plt.subplots()
for n,psi in enumerate(psi_ratio):
    ax.plot(X[n].phigrid,100*Y[n]['taus_pe'],label=r'$\psi$ = {0}'.format(np.round(psi,3)),c=cmap.to_rgba((n+1)/len(psi_ratio)),linewidth=1.25)
ax.legend(loc='upper left')
plt.ylabel("Tax (%)")
plt.xlabel("$\phi$")
ax.set_title(r'Savings tax on entrepreneurs with private equity', fontsize=13)
destin = '../main/figures/taxes_entrepreneurs_pe{0}.eps'.format(iotabar)
plt.savefig(destin, format='eps', dpi=1000)
#plt.show()

fig,ax = plt.subplots()
for n,psi in enumerate(psi_ratio):
    ax.plot(X[n].phigrid,100*Y[n]['tausW_pe'],label=r'$\psi$ = {0}'.format(np.round(psi,3)),c=cmap.to_rgba((n+1)/len(psi_ratio)),linewidth=1.25)
ax.legend(loc='lower left')
plt.ylabel("Tax (%)")
plt.xlabel("$\phi$")
ax.set_title(r'Savings tax on workers with private equity', fontsize=13)
destin = '../main/figures/taxes_workers_pe{0}.eps'.format(iotabar)
plt.savefig(destin, format='eps', dpi=1000)
#plt.show()

fig,ax = plt.subplots()
for n,psi in enumerate(psi_ratio):
    ax.plot(X[n].phigrid, 100*Y[n]['r_pe'], label=r'$\psi$ = {0}'.format(np.round(psi,3)),c=cmap.to_rgba((n+1)/len(psi_ratio)),linewidth=1.25)
ax.legend(loc='lower left')
plt.ylabel("Percent (%)")
plt.xlabel("$\phi$")
ax.set_title("Interest rate with private equity", fontsize=13)
destin = '../main/figures/r_pe{0}.eps'.format(iotabar)
plt.savefig(destin, format='eps', dpi=1000)
#plt.show()

fig,ax = plt.subplots()
for n,psi in enumerate(psi_ratio):
    ax.plot(X[n].phigrid,100*(Y[n]['Pi'] - Y[n]['r_pe']),label=r'$\psi$ = {0}'.format(np.round(psi,3)),c=cmap.to_rgba((n+1)/len(psi_ratio)),linewidth=1.25)
ax.legend(loc='upper left')
plt.ylabel("Percent (%)")
plt.xlabel("$\phi$")
ax.set_title("Excess return $\hat{\Pi} - r$ with private equity", fontsize=13)
destin = '../main/figures/excess_pe{0}.eps'.format(iotabar)
plt.savefig(destin, format='eps', dpi=1000)
