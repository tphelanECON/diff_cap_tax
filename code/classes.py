"""
Class constructor for "On the Optimality of Differential Asset Taxation."

Author: Thomas Phelan.
Email: tom.phelan@clev.frb.org.

First version: September 2022.
This version: December 2024.

The notation adopted in this code coincides with that given in the "guide_to_code"
document found at the github page: https://github.com/tphelanECON/diff_cap_tax.

List of methods:
    Principal-agent setting:
    * xbar(omegabar): \overline{x} defined in appendix.
    * xbarbar(omegabar): \overline{\overline{x}} defined in appendix.
    * g(S, omegabar, x): function g defined in the guide to code.
    * h(S, omegabar, x): function h defined in the guide to code.
    * x(S, omegabar): x function defined in main text.
    * cbar(S, omegabar): consumption policy function.
    * mu_c(S, omegabar): mean growth in consumption.
    * sig_c(S, omegabar): coefficient of diffusion term for consumption.
    * S(Pi,phi): variable defined in PA setting.
    * nu_B(S, omegabar): wedge on risk-free asset (given in Lemma A.10).
    * nu_K(S, omegabar): wedge on risky asset (given in Lemma A.10).

    Stationary efficient allocations:
    * f(S, omegabar,phi): function defining stationary form of the goods resource constraint.
    * S_hat(phi): efficient S (root of above function f).
    * Pi_hat(phi): efficient MPK.

    Decentralization:
    * omegabar_d(S, omegabar): constant appearing in collateral constraint.
    * r_b(Pi,phi): efficient cost of borrowing.
    * revenue_benchmark(Pi, phi, omegabar): total revenue raised per income in benchmark decentralization.

    Miscellaneous:
    * check1(S, omegabar): verification of Assumption A.1.
    * check2(S, omegabar, x): verification of Assumption A.3.
"""

import numpy as np
import scipy.optimize as scopt

class captax(object):
    def __init__(self, alpha=0.33, rhoS=0.04, rhoD=0.02, sigma=0.2, delta=0.05, psi=0.0, iota=1.0):
        self.rhoD, self.rhoS, self.rho = rhoD, rhoS, rhoD + rhoS
        self.sigma, self.iota, self.omega = sigma, iota, 1/(iota*self.rho)
        self.alpha, self.delta, self.psi = alpha, delta, psi
        self.phigrid = np.linspace(10**-3, 1-10**-3, 500)

    """
    Partial equilibrium setting
    """

    """
    xbar: max x for which no-absconding constraint holds as strict inequality given inverse-Euler.
    xbarbar: max x for which consumption growth is below rate of discount rho.
    """

    def xbar(self, omegabar):
        return scopt.brentq(lambda x: x*np.exp(x**2/2)-omegabar, 0, 10)

    def xbarbar(self, omegabar):
        return scopt.brentq(lambda x: x*np.exp(x**2/2-1)-omegabar, 0, 10)

    """
    Solve principal's problem. 
        * the g and h functions defined in guide to code for convenience. 
        They are defined s.t. the objective of the principal is g times h. 
        * the c function given in proof of Proposition 2.1 and in the statement of Corollary A.7 of 2024 version.
    
    S variable defined in Section 2 (recall tau_I = -rho_D). 
    Expressions for cbar, mu_c, sig_c and wedges given in Appendix A.  
    """

    def g(self, S, omegabar, x):
        return (S*x-1)*np.exp(x**2/2)/self.rho

    def h(self, S, omegabar, x):
        ind = x < self.xbar(omegabar)
        return ind + (1-ind)*(omegabar/x)*np.exp(-x**2/2)/(1 + np.log(omegabar/x) - x**2/2)

    def x(self, S, omegabar):
        xgrid = np.linspace(10**-5, self.xbarbar(omegabar)-10**-5, 5000)
        maximand = self.g(S, omegabar, xgrid)*self.h(S, omegabar, xgrid)
        return xgrid[np.argmin(-maximand)]

    def cbar(self, S, omegabar):
        return np.minimum(np.exp(self.x(S, omegabar)**2/2), omegabar/self.x(S, omegabar))

    def mu_c(self, S, omegabar):
        return self.rho*np.maximum(self.x(S, omegabar)**2/2 - np.log(omegabar/self.x(S, omegabar)), 0)

    def sig_c(self, S, omegabar):
        return np.sqrt(self.rho)*self.x(S, omegabar)

    def S(self, Pi, phi):
        return (Pi - self.rhoS)/(np.sqrt(self.rho)*phi*self.sigma)

    def nu_B(self, S, omegabar):
        return self.rho*self.x(S, omegabar)**2 - self.mu_c(S, omegabar)

    def nu_K(self, S, omegabar, Pi):
        return Pi - self.rhoS + self.rho*self.x(S, omegabar)**2 - np.sqrt(self.rho)*self.sigma*self.x(S, omegabar) - self.mu_c(S, omegabar)

    """
    Characterization of stationary efficient allocations.
    """
    def f(self, S, omegabar, phi):
        scale = self.rhoD/(self.rhoD - self.mu_c(S, omegabar))
        C = self.cbar(S, omegabar)*scale
        K = self.cbar(S, omegabar)*self.x(S, omegabar)*scale/(np.sqrt(self.rho)*phi*self.sigma)
        lhs = (1-self.psi)*C + self.psi
        rhs = ((S*np.sqrt(self.rho)*phi*self.sigma + self.rhoS)/self.alpha + (1/self.alpha-1)*self.delta)*(1-self.psi)*K
        return lhs - rhs

    def S_hat(self, phi):
        omegabar = np.sqrt(self.rho)*phi*self.sigma/(self.rho*self.iota)
        # only consider S < Shigh, the value s.t. mu_c < rho_D to ensure finite C and K.
        Shigh = scopt.brentq(lambda S: self.rhoD - self.mu_c(S, omegabar), 10**-6, 1/self.xbarbar(omegabar)-10**-3)
        low, high = 10**-8, Shigh - 10**-3
        return scopt.brentq(lambda S: self.f(S, omegabar, phi), low, high)

    def Pi_hat(self, phi):
        return self.rhoS + self.S_hat(phi)*np.sqrt(self.rho)*phi*self.sigma

    """
    Quantities pertaining to decentralization.
    
    omegabar_d is constant in collateral constraint. NOT to be confused with omegabar in PA setting. 
     
    For cost of borrowing in implementation recall   
    
        r_b = rho_S - nu^B + nu^K 
            = self.rho_S - self.rho*self.x(S, omegabar)**2 + self.mu_c(S, omegabar) + Pi - self.rhoS 
            + self.rho*self.x(S, omegabar)**2 - np.sqrt(self.rho)*self.sigma*self.x(S, omegabar) - self.mu_c(S, omegabar)
            = Pi - np.sqrt(self.rho)*self.sigma*self.x(S, omegabar) 
    """

    def omegabar_d(self, S, omegabar):
        return np.exp((self.mu_c(S, omegabar) - self.sig_c(S, omegabar)**2/2)/self.rho)/self.iota

    def r_b(self, Pi, phi):
        omegabar = np.sqrt(self.rho)*phi*self.sigma/(self.rho*self.iota)
        return Pi - np.sqrt(self.rho)*self.sigma*self.x(self.S(Pi, phi), omegabar)

    # total revenue raised in benchmark decentralization as fraction of income
    # NOTE: only valid when collateral constraint does not bind (only case plotted in paper)
    def revenue_benchmark(self, Pi, phi, omegabar):
        return self.x(self.S(Pi, phi), omegabar)**2/(2*self.x(self.S(Pi, phi), omegabar)**2+1)

    """
    Two checks corresponding to the assumptions in A.1 and A.3 of the appendix
    """

    def check1(self, S, omegabar):
        return S*self.xbarbar(omegabar) < 1

    def check2(self, S, omegabar, x):
        vbar = self.g(S, omegabar, x)*self.h(S, omegabar, x)
        return S*omegabar + self.rho*(1 + S**(-2))*vbar < 0
