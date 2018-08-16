#import sys45
fim='/home/va/work/kpvt/hmi/'
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

CR=range(2097,2202)
i0=0
image_data = nan_to_num(fits.getdata(fim+'m'+str(CR[i0])+'f.fits', ext=0))

#CR=append(append(range(1626,1640),range(1645,1854)),range(1855,2008))
LM=len(CR)
timeh=linspace(2010.+139/365.25,2018+81./365.25,LM)
NT=180
Nph=360
Nph0=180
Nt0=64
Rs=6.96e10
mu=linspace(-1,1,NT)
phi=arange(Nph)*2*pi/Nph
mu0=cos(arange(NT)*pi/(NT))

brfh=zeros((LM,Nph//2+1))
brfmh=zeros((LM,Nph//2+1))
modes=arange(Nph//2+1)
modes[0]=1
netf=zeros(LM)
grsf=zeros(LM)
br0h=zeros((LM,NT));
br1h=zeros((LM,NT))+1j*zeros((LM,NT));
br2h=zeros((LM,NT))+1j*zeros((LM,NT));

c1=sum(ones(Nph)*simps(ones(NT),mu))*2*pi/Nph
c2=sum(ones(Nph)*simps(ones(NT),arccos(mu0)))*2*pi/Nph
c3=sum(ones(Nph)*simps(ones(NT),(mu0)))*2*pi/Nph
bph=zeros((len(CR),NT//2));bpmh=zeros((len(CR),NT//2));
bpnh=zeros((len(CR),NT//2));bpm1h=zeros((len(CR),NT//2));
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
    image_data = nan_to_num(fits.getdata(fim+'m'+str(CR[i0])+'f.fits', ext=0))
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
    
    power = spectrum(cilm,unit='per_lm');bph[i0,:]=power;
    power = spectrum(coef0,unit='per_lm');bpnh[i0,:]=power;
    power = spectrum(coef1,unit='per_lm');bpmh[i0,:]=power;
    power = spectrum(coefm,unit='per_lm');bpm1h[i0,:]=power;
    br0h[i0,:]=sum(image_n,axis=1)/Nph

    image_f=rfft(image_n,axis=1)/(Nph)
    brfff=real(simps(image_f*conj(image_f),mu,axis=0))/2.
    brfh[i0,:]=brfff[:]# *2*pi*Rs**2
    print(i)
    i0=i0+1


naxrh=sum(brfh[:,1:],axis=1)/sum(brfh[:,:],axis=1)
naxrlh=sum(brfh[:,1:11],axis=1)/sum(brfh[:,:11],axis=1)
plt.figure()
plt.plot(timeh,naxrh,'k',timeh,naxrlh,'k--')
plt.show()
Rs=6.96e10
plt.figure()
lin=plt.plot(time,sqrt(sum(brf[:,:],axis=1))/sqrt(sqrt(2*pi)),'k',
             time,sqrt(sum(brf[:,:11],axis=1))/sqrt(sqrt(2*pi)),'b',
             time,sqrt(brf[:,0])/sqrt(sqrt(2*pi)),'k--')
lin0=plt.plot(timeh,sqrt(sum(brfh[:,:],axis=1))/sqrt(sqrt(2*pi)),'r',
             timeh,sqrt(sum(brfh[:,:11],axis=1))/sqrt(sqrt(2*pi)),'g',
             timeh,sqrt(brfh[:,0])/sqrt(sqrt(2*pi)),'r--')
#         time,sqrt(mdi)/sqrt(2*pi),'r')#         time,sqrt(kpf),'b',time,sqrt(kpfm),'g')
#plt.yscale('log')
plt.legend(lin, ['${\\langle {B}_{r}\\rangle}$','${\\langle {B}_{r}\\rangle}^{(m\\le 11)}$',
                 '${\\left|\\bar{B}_{r}\\right|}$'],
           loc='auto', frameon=False)
plt.ylabel('[G]')
plt.xlabel('[YR]')
plt.show()

dip0h=simps((br0h[:,:]-br0h[:,-1::-1])**2,mu,axis=1)
qup0h=simps((br0h[:,:]+br0h[:,-1::-1])**2,mu,axis=1)


plt.figure()
lin=plt.plot(time,(qup0-dip0)/(dip0+qup0),'k--',time,naxrl,'k')
lin0=plt.plot(timeh,(qup0h-dip0h)/(dip0h+qup0h),'r--',timeh,naxrlh,'r')
plt.ylabel('P$_X$, P$_E$')
plt.show()

plt.figure()
plt.plot(arange(len(power)-1)+1,sqrt((mean(bph,axis=0)[1:])),'k',
         #arange(len(power)-1)+1,sqrt((mean(bpm,axis=0)[1:])),'b',
         arange(len(power)),sqrt(mean(brfh,axis=0)[:90])/sqrt(2*pi),'r')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('$\\overline{\\langle {B}_r\\rangle}$') 
plt.xlabel('$\\ell$, m') 
plt.show()    

#savetxt('splmdi', [arange(len(power)-1)+1,sqrt((mean(bp,axis=0)[1:]))])
#savetxt('spmmdi', [arange(len(power)),sqrt(mean(brf,axis=0)[:90])/sqrt(2*pi)])

savetxt('splhm', [arange(len(power)-1)+1,sqrt((mean(bph,axis=0)[1:]))])
savetxt('spmhm', [arange(len(power)),sqrt(mean(brfh,axis=0)[:90])/sqrt(2*pi)])
