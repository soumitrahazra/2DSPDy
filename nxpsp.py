#import sys
fim='/home/va/work/kpvt/'
#sys.path.append(fim);

from numpy import *
from numpy.fft import *
from scipy import *
from scipy.interpolate import *
from pylab import *
from astropy.io import fits
from scipy.integrate import *
    
import matplotlib.cm as cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.utils.data import get_pkg_data_filename
import matplotlib.pyplot as plt
params = {'axes.labelsize': 24,
          'font.size': 24,
          'legend.fontsize': 30,
          'xtick.labelsize': 30,
          'ytick.labelsize': 30,
          'savefig.dpi':32,
          'text.usetex': True}
plt.rcParams.update(params)
from pyshtools.expand import SHExpandDH
from pyshtools.spectralanalysis import spectrum
from pyshtools.expand import MakeGridDH


CR=range(1645,2135)
LM=len(CR)
time=linspace(1976.62670,2013.21780,LM)
NT=180
Nph=360
Nph0=180
Nt0=64
Rs=6.96e10
mu=linspace(-1,1,NT)
phi=arange(Nph)*2*pi/Nph
mu0=cos(arange(NT)*pi/(NT))

brf=zeros((LM,Nph//2+1))
brfm=zeros((LM,Nph//2+1))
modes=arange(Nph//2+1)
modes[0]=1
netf=zeros(LM)
grsf=zeros(LM)
br0=zeros((LM,NT));
br1=zeros((LM,NT))+1j*zeros((LM,NT));
br2=zeros((LM,NT))+1j*zeros((LM,NT));

c1=sum(ones(Nph)*simps(ones(NT),mu))*2*pi/Nph
c2=sum(ones(Nph)*simps(ones(NT),arccos(mu0)))*2*pi/Nph
c3=sum(ones(Nph)*simps(ones(NT),(mu0)))*2*pi/Nph
bp=zeros((len(CR),NT//2));bpm=zeros((len(CR),NT//2));
bpn=zeros((len(CR),NT//2));bpm1=zeros((len(CR),NT//2));
g01=zeros(len(CR))
g11=zeros(len(CR))
h11=zeros(len(CR))
g12=zeros(len(CR))
h12=zeros(len(CR))
g22=zeros(len(CR))
h22=zeros(len(CR))
g21=zeros(len(CR))
h21=zeros(len(CR))
kpf=zeros(len(CR));kpfm=zeros(len(CR));


#get_pkg_data_filename(fim+'m'+str(CR[i0])+'f.fits')
i0=0
for i in CR:
    image_data = fits.getdata(fim+'m'+str(CR[i0])+'f.fits', ext=0)
    imf=sum(simps(image_data,mu,axis=0))*2*pi/Nph
    image_n =image_data -imf/c1
    kpf[i0]=simps(sum((image_n)**2,axis=1)*2*pi/Nph,mu)/4./pi
    kpfm[i0]=simps((mean(image_n,axis=1)**2),mu)/2
    
    f = interp2d(phi,mu,image_n)
    Z1 = f(phi,mu0); imf=simps(sum(Z1,axis=1)*2*pi/Nph,mu0)
    Z2=(Z1-imf/c3)[-1::-1,:] #mean(Z2) #imf#/4./pi
    cilm = SHExpandDH(Z2, sampling=2);
    coef0 = cilm.copy();coef0[:, :, 0] = 0.
    coef1 = cilm.copy();coef1[:, :, 1:] = 0.
    coefm = cilm.copy();coefm[:, :, 0] = 0.;coefm[:, :, 2:] = 0.

    g01[i0]=cilm[0,1,0]
    g11[i0]=cilm[0,1,1];h11[i0]=cilm[1,1,1];
    g21[i0]=cilm[0,2,1];h21[i0]=cilm[1,2,1];
    g12[i0]=cilm[0,1,2];h12[i0]=cilm[1,1,2];    
    g22[i0]=cilm[0,2,2];h22[i0]=cilm[1,2,2];    
    
    power = spectrum(cilm,unit='per_lm');bp[i0,:]=power;
    power = spectrum(coef0,unit='per_lm');bpn[i0,:]=power;
    power = spectrum(coef1,unit='per_lm');bpm[i0,:]=power;
    power = spectrum(coefm,unit='per_lm');bpm1[i0,:]=power;
    br0[i0,:]=sum(image_n,axis=1)/Nph

    image_f=rfft(image_n,axis=1)/(Nph)
    brfff=real(simps(image_f*conj(image_f),mu,axis=0))/2.
    brf[i0,:]=brfff[:]# *2*pi*Rs**2
    print(i)
    i0=i0+1


naxr=sum(brf[:,1:],axis=1)/sum(brf[:,:],axis=1)
naxrl=sum(brf[:,1:11],axis=1)/sum(brf[:,:11],axis=1)
plt.figure()
plt.plot(time,naxr,'k',time,naxrl,'k--')
plt.show()
Rs=6.96e10

dip0=simps((br0[:,:]-br0[:,-1::-1])**2,mu,axis=1)
qup0=simps((br0[:,:]+br0[:,-1::-1])**2,mu,axis=1)

plt.figure()
plt.plot(time,(qup0-dip0)/(dip0+qup0),'k--',time,naxrl,'k')
plt.ylabel('P$_X$, P$_E$')
plt.show()

plt.figure()
lin=plt.plot(arange(len(power)-1)+1,sqrt((mean(bp,axis=0)[1:])),'k',
         #arange(len(power)-1)+1,sqrt((mean(bpm,axis=0)[1:])),'b',
         arange(len(power)),sqrt(mean(brf,axis=0)[:90])/sqrt(2*pi),'k--',
         arange(len(power)-1)+1,sqrt((mean(bph,axis=0)[1:])),'r',
         #arange(len(power)-1)+1,sqrt((mean(bpm,axis=0)[1:])),'b',
         arange(len(power)),sqrt(mean(brfh,axis=0)[:90])/sqrt(2*pi),'r--')
plt.legend(lin, ['$\\ell$, KPO-SOLIS','m','$\\ell$, HMI','m' ],
           loc='auto', frameon=False)
plt.xscale('log')
plt.yscale('log')
plt.ylabel('$\\overline{\\langle {B}_r\\rangle}$, [G]') 
plt.xlabel('$\\ell$, m') 
plt.show()    
savetxt('splkp', [arange(len(power)-1)+1,sqrt((mean(bp,axis=0)[1:]))])
savetxt('spmkp', [arange(len(power)),sqrt(mean(brf,axis=0)[:90])/sqrt(2*pi)])
