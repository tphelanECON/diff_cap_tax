"""
parameters used for the main text.

What happens if we send rhoD to zero?
"""


import classes

alpha, sigma, rhoS, rhoD, delta = 0.33, 0.2, 0.04, 1/50, 0.05
rho, phimin, phimax = rhoS + rhoD, 0.001, 1.0

#psi_ratio = [1/2, 3/4, 5/6, 7/8]
#psi_ratio = [1/2, 0.75, 0.84, 0.885]
psi_ratio = [0.66, 4/5, 0.85, 0.885]
phigrid = classes.captax(phimin=phimin,phimax=phimax).phigrid
