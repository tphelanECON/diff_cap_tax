"""
class constructors for "On the Optimality of Differential Asset Taxation"
"""

import numpy as np
import scipy.optimize as scopt

class captax(object):
    def __init__(self,alpha=0.33,rhoS=0.04,rhoD=0.02,sigma=0.2,delta=0.05,psi=0.0,iota=1.0):
        self.rhoD, self.rhoS, self.rho = rhoD, rhoS, rhoD + rhoS
        self.sigma, self.iota, self.omega = sigma, iota, 1/(iota*self.rho)
        self.alpha, self.delta, self.psi = alpha, delta, psi
        self.phigrid = np.linspace(10**-4,1-10**-4,500)

    def xbar(self,omegabar):
        return scopt.brentq(lambda x: x*np.exp(x**2/2)-omegabar,0,10)

    def xbarbar(self,omegabar):
        return scopt.brentq(lambda x: x*np.exp(x**2/2-1)-omegabar,0,10)

    def g(self,S,omegabar,x):
        return (S*x-1)*np.exp(x**2/2)/self.rho

    def h(self,S,omegabar,x):
        ind = x < self.xbar(omegabar)
        return ind + (1-ind)*(omegabar/x)*np.exp(-x**2/2)/(1 + np.log(omegabar/x) - x**2/2)

    def x(self,S,omegabar):
        xgrid = np.linspace(10**-5,self.xbarbar(omegabar)-10**-5,2500)
        maximand = self.g(S,omegabar,xgrid)*self.h(S,omegabar,xgrid)
        return xgrid[np.argmin(-maximand)]

    def c(self,S,omegabar):
        return np.minimum(np.exp(self.x(S,omegabar)**2/2), omegabar/self.x(S,omegabar))

    def mu_c(self,S,omegabar):
        return self.rho*np.maximum(-np.log(omegabar/self.x(S,omegabar)) + self.x(S,omegabar)**2/2, 0)

    def sig_c(self,S,omegabar):
        return np.sqrt(self.rho)*self.x(S,omegabar)

    def omegahat(self,S,omegabar):
        return np.exp((self.mu_c(S,omegabar) - self.sig_c(S,omegabar)**2/2)/self.rho)/self.iota

    def S(self,Pi,phi):
        return (Pi - self.rhoS)/(np.sqrt(self.rho)*phi*self.sigma)

    def f(self,S,omegabar,phi):
        scale = self.rhoD/(self.rhoD - self.mu_c(S,omegabar))
        LHS = self.alpha*np.sqrt(self.rho)*phi*self.sigma*((1-self.psi)*self.c(S,omegabar)*scale + self.psi)
        RHS = (self.rhoS + S*np.sqrt(self.rho)*phi*self.sigma + (1-self.alpha)*self.delta)*(1-self.psi)*self.c(S,omegabar)*self.x(S,omegabar)*scale
        return LHS - RHS

    def S_hat(self,phi):
        omegabar = np.sqrt(self.rho)*phi*self.sigma/(self.rho*self.iota)
        Shigh = scopt.brentq(lambda S: self.rhoD - self.mu_c(S,omegabar), 10**-6, 1/self.xbarbar(omegabar)-10**-3)
        low, high = 10**-8, Shigh - 10**-3
        F = lambda S: self.f(S,omegabar,phi)
        return scopt.brentq(F,low, high)

    def Pi_hat(self,phi):
        return self.rhoS + self.S_hat(phi)*np.sqrt(self.rho)*phi*self.sigma

    #interest rate in benchmark case (no private risk-sharing)
    def r(self,Pi,phi):
        omegabar = np.sqrt(self.rho)*phi*self.sigma/(self.rho*self.iota)
        return Pi - np.sqrt(self.rho)*self.sigma*self.x(self.S(Pi,phi),omegabar)

    #interest rate with private risk-sharing (note presenc of phi in expression)
    def r_pe(self,Pi,phi):
        omegabar = np.sqrt(self.rho)*phi*self.sigma/(self.rho*self.iota)
        return Pi - np.sqrt(self.rho)*self.sigma*phi*self.x(self.S(Pi,phi),omegabar)

    #entrepreneur taxes in benchmark case (no private risk-sharing)
    def taus(self,Pi,phi):
        omegabar = np.sqrt(self.rho)*phi*self.sigma/(self.rho*self.iota)
        S = self.S(Pi,phi)
        return 1 - self.rho*(1 - self.x(S,omegabar)**2 + self.mu_c(S,omegabar)/self.rho)/(self.r(Pi,phi) + self.rhoD)

    #worker taxes in benchmark case (no private risk-sharing)
    def tausW(self,Pi,phi):
        return 1 - self.rho/(self.r(Pi,phi) + self.rhoD)

    #entrepreneur taxes with private risk-sharing
    def taus_pe(self,Pi,phi):
        omegabar = np.sqrt(self.rho)*phi*self.sigma/(self.rho*self.iota)
        S = self.S(Pi,phi)
        return 1 - self.rho*(1 - self.x(S,omegabar)**2 + self.mu_c(S,omegabar)/self.rho)/(self.r_pe(Pi,phi) + self.rhoD)

    #worker taxes with private risk-sharing
    def tausW_pe(self,Pi,phi):
        return 1 - self.rho/(self.r_pe(Pi,phi) + self.rhoD)

    #wedges on the risk-free and risky assets
    def nu_B(self,S,omegabar):
        return self.rho*self.x(S,omegabar)**2

    def nu_K(self,S,omegabar,Pi):
        return Pi - self.rhoS + self.rho*self.x(S,omegabar)**2 - np.sqrt(self.rho)*self.sigma*self.x(S,omegabar)

    #two checks corresponding to the assumptions in A.1 and A.2 of the appendix
    def check1(self,S,omegabar):
        return S*self.xbarbar(omegabar) < 1

    def check2(self,S,omegabar,x):
        vbar = self.g(S,omegabar,x)*self.h(S,omegabar,x)
        return S*omegabar + self.rho*(1 + S**(-2))*vbar < 0
