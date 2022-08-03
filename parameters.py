"""
parameters used for the main text.
"""


import classes

alpha, sigma, rhoS, rhoD, delta = 0.33, 0.2, 0.035, 0.025, 0.05
rho, phimin, phimax = rhoS + rhoD, 0.001, 1.0

psi_ratio = [1/2, 3/4, 5/6, 7/8]
phigrid = classes.captax_general(phimin=phimin,phimax=phimax).phigrid
