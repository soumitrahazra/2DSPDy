from numpy import *
import matplotlib.pyplot as plt
params = {'axes.labelsize': 24,
          'axes.labelweight': 'bold',
          'font.size': 24,
          'legend.fontsize': 24,
          'xtick.labelsize': 24,
          'ytick.labelsize': 24,
          'lines.markersize':12,
          'text.usetex': True,
          'axes.spines.right':False,
          'axes.spines.top':False}
plt.rcParams.update(params)
import matplotlib.cm as cmap
from scipy.integrate import *

def bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0,
                     mux=0.0, muy=0.0, sigmaxy=0.0):
    """
    Bivariate Gaussian distribution for equal shape *X*, *Y*.
    See `bivariate normal
    <http://mathworld.wolfram.com/BivariateNormalDistribution.html>`_
    at mathworld.
    """
    Xmu = X-mux
    Ymu = Y-muy

    rho = sigmaxy/(sigmax*sigmay)
    z = Xmu**2/sigmax**2 + Ymu**2/sigmay**2 - 2*rho*Xmu*Ymu/(sigmax*sigmay)
    denom = 2*np.pi*sigmax*sigmay*np.sqrt(1-rho**2)
    return np.exp(-z/(2*(1-rho**2))) / denom


from mpl_toolkits.axes_grid1 import make_axes_locatable

tr,brf,g01,g02,g11,h11,g21,h21,g22,h22,bfe,bre=(loadtxt('mbu'))
# tr -time in diffusive units
# brf the mean density of the nonaxisymmetric radial magnetic field flux
# bfe - the same for the toroidal magnetic field
# bre - the same for the axisymmetric toroidal magnetic field
# g01, g02, g11,h11,g21,h21,g22,h22 dipole, quadrupole, equatorial dipole,quadrupole

Nc=len(tr)
tm=tr[-1]*62/Nc
time=linspace(0,tm*Nc,Nc)
#Nc1=len(ts)
#tm1=ts[-1]*53/Nc
#time1=linspace(0,tm1*Nc1,Nc1)
sintt=bre-(max(bre)-min(bre[15000:]))*abs(sin((time)/3.25))+min(bre[15000:])
plt.figure()
lin0=plt.plot(time,bfe,'k',time,brf,'r')
              #time,(max(bre)-min(bre[15000:]))*abs(sin((time)/3.25))+min(bre[15000:]),'b',lw=2) #,time,bre,'r')
plt.legend(lin0,['$\\overline{B_{\\phi}}$','$\\int \left|B_r\\right| dS$'], loc='auto', frameon=False)
#plt.legend(lin0,['sin','axial','equa'], loc='auto', frameon=False)
#plt.xlim(0,150)
plt.show()
plt.figure()
lin0=plt.plot(time,sintt,'k')
plt.show()


plt.figure()
lin0=plt.plot(time,abs(g11**2+h11**2),'k',lw=2) #,time,bre,'r')
#plt.legend(lin0,['sin','axial','equa'], loc='auto', frameon=False)
#plt.xlim(0,150)
plt.show()

plt.figure()
lin0=plt.plot(time,max(g01)*sin((time)/3.2-1),'b',time,(g01),'k',
              time,sqrt(g11**2+h11**2),'r') #,time,bre,'r')
#plt.legend(lin0,['sin','axial','equa'], loc='auto', frameon=False)
plt.xlim(75,115)
plt.show()
plt.figure()
lin0=plt.plot(time,g01-max(g01)*sin((time)/3.191-1),'k') #,time,bre,'r')
#plt.legend(lin0,['axial-sin'], loc='auto', frameon=False)
#plt.xlim(75,115)
plt.show()
sint=g01-max(g01)*sin((time)/3.2-1)



#plt.figure()
#lin0=plt.plot(time,g01s,'r',time,g01,'b')
#plt.legend(lin0,['dip-1d','dip-2d'], loc='auto', frameon=False)
#plt.show()

parity=(g02**2-g01**2)/(g01**2+g02**2)

from scipy.integrate import simps
import pywt


scales=arange(30,Nc//8)
#scales1=arange(30,Nc1//6)
coef0,freqs0=pywt.cwt(abs(g01),scales,'morl')
pow0=simps(abs(coef0),time,axis=1)
coef1,freqs1=pywt.cwt(bre,scales,'morl')
pow1=simps(abs(coef1),time,axis=1)
coef2,freqs2=pywt.cwt(brf,scales,'morl')
pow2=simps(abs(coef2),time,axis=1)

plt.figure()
lin0=plt.plot(tm/freqs0, pow1/(simps(pow1[:-1500],tm/freqs0[:-1500])),'k',tm/freqs2, pow2/(simps(pow2[:-1500],tm/freqs2[:-1500])),'r')
              #tm/freqs0[:200],(tm/freqs0[:200])**(2.5)/(tm/freqs0[-1])/10,'k--',
              #tm/freqs0[:200],(tm/freqs0[:200])**(1)/(tm/freqs0[1000]),'r--',lw=2)
#plt.legend(lin0,['$\\overline{B_{\\phi}}$','$\\int \left|B_r\right| dS$'], loc='auto', frameon=False)
plt.xlim(.5,20)
plt.show()
plt.figure()
lin0=plt.loglog(tm/freqs0, pow2/max(pow2),'k',
              tm/freqs0[:600],(tm/freqs0[:600])**(0.8)/(tm/freqs0[2000]),lw=2)
plt.xlim(0,20)
plt.show()

import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker

xa,ya=meshgrid(arange(Nc)*tm,tm/freqs0)

coef2=coef2*max(pow2)
##
plt.figure()
CS=plt.pcolormesh(xa[::16,::16],ya[::16,::16],(coef2[::16,::16]),
                  norm=colors.SymLogNorm(linthresh=.01, linscale=1,
        vmin=-1,vmax=1),cmap=cmap.jet) 
#CS=plt.contourf(xa,ya,abs(coef0),locator=ticker.LogLocator(),cmap=cmap.jet)
#plt.yscale("log")
plt.ylim(.4,12)
plt.xlim(75,140)
plt.yticks([1,5,10])
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "3%", pad="1%")
CB=plt.colorbar(CS, cax=cax,ticks=[-1,-0.1,0,0.1,1],format='%.1f')
#CB.set_label('$U\\phi$,[M/S]',fontsize=22)
plt.tight_layout()
plt.show()

epi=[3,7,8,8,9,9,14,17,21,22,28,35,35,41,47,54,55,65,68,75,75,80,85,87,87,95,101,123,130,150,166,195]
