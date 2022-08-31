"""
Parameters used for the main text
"""

import classes

alpha, sigma, rhoS, rhoD, delta = 0.33, 0.2, 0.04, 0.02, 0.06
rho = rhoS + rhoD

psi_ratio = [0.66, 4/5, 0.85, 0.885]
phigrid = classes.captax().phigrid
