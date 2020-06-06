from scipy.linalg import *
from numpy.random import *
from func_dif import *
#from scipy.special import *
#from plac import *
import matplotlib.pyplot as plt
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

params = {'axes.labelsize': 24,
          'font.size': 24,
          'legend.fontsize': 24,
          'xtick.labelsize': 24,
          'ytick.labelsize': 24,
          'text.usetex': True}
plt.rcParams.update(params)
import matplotlib.cm as cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import cartopy.crs as ccrs
fia='/home/va/work/talk/nordita/ani/mod/'

set_printoptions(precision=16)
from pyshtools import *
from pyshtools.expand import SHGLQ,SHExpandGLQ,MakeGridGLQ,SHMultiply
from pyshtools.shio import SHCilmToCindex,SHCindexToCilm,SHrtoc,SHctor

from axm import *
from initrd import *
Nph=(2*Nlr+1)
phi=arange(Nph)*2*pi/Nph

zero=xch[-1::-1]

def bfah(aa,ba,hels,S,T,htn,phi0,alpf):
    sd=sqrt(1-xch**2)
    beta=zeros((Nch,Nph))
    nnl=zeros((Nch,Nph))
    alfs0=zeros((Nch,Nph))
    alfs=zeros((Nch,Nph))
    alfs1=zeros((Nch,Nph))
    alfs2=zeros((Nch,Nph))
    alft0=zeros((Nch,Nph))
    alft1=zeros((Nch,Nph))
    alft2=zeros((Nch,Nph))
    but1=zeros((Nch,Nph))
    but2=zeros((Nch,Nph))
    bus3=zeros((Nch,Nph))
    bus1=zeros((Nch,Nph))
    bus2=zeros((Nch,Nph))
  
    
    brc=-dot(ms0,S)
    bfc=dot(mst,T)+1j*dot(msf,S)
    bpc=1j*dot(msf,T)-dot(mst,S)
    apc=1j*dot(msf,S);afc=dot(mst,S)    

    cilm=zeros((2,Nlr+1,Nmr+1))
    cilm[0,nk,mk]=real(brc);
    cilm[1,nk,mk]=imag(brc) 
    brcr=SHctor(cilm)
    br = MakeGridGLQ(brcr,zero,norm=4)[-1::-1,:]

    cilm=zeros((2,Nlr+1,Nmr+1))
    cilm[0,nk,mk]=real(T);
    cilm[1,nk,mk]=imag(T) 
    arcr=SHctor(cilm)
    ar = MakeGridGLQ(arcr,zero,norm=4)[-1::-1,:]

    cilm=zeros((2,Nlr+1,Nmr+1))
    cilm[0,nk,mk]=real(bfc);
    cilm[1,nk,mk]=imag(bfc) 
    bfcr=SHctor(cilm)
    bf = MakeGridGLQ(bfcr,zero,norm=4)[-1::-1,:] 

    cilm=zeros((2,Nlr+1,Nmr+1))
    cilm[0,nk,mk]=real(afc);
    cilm[1,nk,mk]=imag(afc) 
    afcr=SHctor(cilm)
    af = MakeGridGLQ(afcr,zero,norm=4)[-1::-1,:] 

    cilm=zeros((2,Nlr+1,Nmr+1))
    cilm[0,nk,mk]=real(bpc);
    cilm[1,nk,mk]=imag(bpc) 
    bpcr=SHctor(cilm)
    bp = MakeGridGLQ(bpcr,zero,norm=4)[-1::-1,:]

    cilm=zeros((2,Nlr+1,Nmr+1))
    cilm[0,nk,mk]=real(apc);
    cilm[1,nk,mk]=imag(apc) 
    apcr=SHctor(cilm)
    ap = MakeGridGLQ(apcr,zero,norm=4)[-1::-1,:]  
    ABn=(ar*(br+tensordot(dot(df1r,aa),ones(Nph),0))
    +ap*bp+af*(bf+tensordot(b0,ones(Nph),0))
    +bf* tensordot(aa,ones(Nph),0)
    +br* tensordot(dot(mabp,b0),ones(Nph),0))
    
    cilm=zeros((2,Nlr+1,Nmr+1))
    cilm[0,nk,mk]=real(htn);
    cilm[1,nk,mk]=imag(htn) 
    hnaxc=SHctor(cilm)
    hnax = MakeGridGLQ(hnaxc,zero,norm=4)[-1::-1,:] 

    bmax=max(sqrt(br[:,:]**2+bp[:,:]**2+(bf[:,:]+tensordot(ba,ones(Nph),0))**2).flatten())
    br0=dot(df1r,aa)
    brmax=max(sqrt((br[:,:]+tensordot(br0,ones(Nph),0))**2).flatten())
    bnaxi= sum(sqrt(br[:,:]**2+bp[:,:]**2+bf[:,:]**2),axis=1)/Nph       
    for j in range(Nch):
        beta[j,:]=sqrt(br[j,:]**2+(bf[j,:]+ba[j])**2+bp[j,:]**2)
        nnl[j,:]=1.#/(1+beta[j,:]**2)

    if cbu[1] > 0:
        tetm=arccos(xch[argmax(abs(ba)[:Nch//2])])
        Cbu=cbu*exp(-((tetxh-tetm)/dtet)**2)
        Cbu[Nch//2:]=0.
    elif cbu[-2] > 0:
        tetm=arccos(xch[Nch//2+argmax(abs(ba)[Nch//2:])])
        Cbu=cbu*exp(-((tetxh-tetm)/dtet)**2)
        Cbu[:Nch//2]=0.
    else:
        Cbu=zeros(Nch)
    for j in range(Nph):
        alfs[:,j]=(Cmu*alff*alpf[j]*nnl[:,j]+Ch*(hels+hnax[:,j]))*(br[:,j]-dot(df1r,aa))
        alfs1[:,j]=dot(DD1,(Cmu*alff*alpf[j]*nnl[:,j]+Ch*(hels+hnax[:,j]))*sd*(bp[:,j]+aa))
        alfs2[:,j]=(Cmu*alff*nnl[:,j]*alpf[j]+Ch*(hels+hnax[:,j]))*(bf[:,j]+ba)/sqrt(1-xch**2)
        alft1[:,j]=dot(DD1,(Cmu*alff*alpf[j]*nnl[:,j]+Ch*(hels+hnax[:,j]))*(bf[:,j]+ba)*sqrt(1-xch**2))#/sqrt(1-xch**2)
        alft2[:,j]=(Cmu*alff*alpf[j]*nnl[:,j]+Ch*(hels+hnax[:,j]))*(bp[:,j]+aa)/sqrt(1-xch**2)
        alft0[:,j]=(Cmu*alff*alpf[j]*nnl[:,j]+Ch*(hels+hnax[:,j]))*bf[:,j]
        
        alfs0[:,j]=(dot(DD3,(Cmu*alff*alpf[j]*nnl[:,j]
                +Ch*(hels+hnax[:,j]))*(br[:,j]))/sqrt(1-xch**2)+
             (Cmu*alff*alpf[j]*nnl[:,j]+Ch*(hels+hnax[:,j]))*bp[:,j])
        but1[:,j]=Cbu*vbu(beta[:,j])*exp(-100*sin((phi[j]-phi0)/2)**2)*(aa+bp[:,j])/sqrt(1-xch**2)
        but2[:,j]=dot(DD1,Cbu*vbu(beta[:,j])*(ba+bf[:,j])*sqrt(1-xch**2))*exp(-100*sin((phi[j]-phi0)/2)**2)
        bus1[:,j]=Cbu*vbu(beta[:,j])*exp(-100*sin((phi[j]-phi0)/2)**2)*(bf[:,j]+ba)/sqrt(1-xch**2)
        bus2[:,j]=dot(DD1,Cbu*vbu(beta[:,j])*(aa+bp[:,j])*sqrt(1-xch**2))*exp(-100*sin((phi[j]-phi0)/2)**2)
        bus3[:,j]=vbu(beta[:,j])*(1.+Cbu*exp(-100*sin((phi[j]-phi0)/2)**2))#*(0*bf[:,j]+ba)
        
    cilm = SHExpandGLQ(alfs[-1::-1,:],wch,zero, norm=4)
    ccilm=SHrtoc(cilm)
    salf0=(ccilm[0,nk,mk]+1j*ccilm[1,nk,mk]);

    cilm = SHExpandGLQ(alfs1[-1::-1,:],wch,zero, norm=4)
    ccilm1=SHrtoc(cilm)
    cilm = SHExpandGLQ(alfs2[-1::-1,:],wch,zero, norm=4)
    ccilm2=SHrtoc(cilm)
    salf1=((ccilm1[0,nk,mk]+1j*ccilm1[1,nk,mk])
    -1j*mk*(ccilm2[0,nk,mk]+1j*ccilm2[1,nk,mk]));
    
    cilm = SHExpandGLQ(alft1[-1::-1,:],wch,zero, norm=4)
    ccilm1=SHrtoc(cilm)
    cilm = SHExpandGLQ(alft2[-1::-1,:],wch,zero, norm=4)
    ccilm2=SHrtoc(cilm)
    talf=((ccilm1[0,nk,mk]+1j*ccilm1[1,nk,mk])
    +1j*mk*(ccilm2[0,nk,mk]+1j*ccilm2[1,nk,mk]));

    cilm = SHExpandGLQ(but1[-1::-1,:],wch,zero, norm=4)
    ccilm1=SHrtoc(cilm)
    cilm = SHExpandGLQ(but2[-1::-1,:],wch,zero, norm=4)
    ccilm2=SHrtoc(cilm)
    buot=((ccilm2[0,nk,mk]+1j*ccilm2[1,nk,mk])
    +1j*mk*(ccilm1[0,nk,mk]+1j*ccilm1[1,nk,mk]));

    cilm = SHExpandGLQ(bus1[-1::-1,:],wch,zero, norm=4)
    ccilm1=SHrtoc(cilm)
    cilm = SHExpandGLQ(bus2[-1::-1,:],wch,zero, norm=4)
    ccilm2=SHrtoc(cilm)
    buos=((ccilm2[0,nk,mk]+1j*ccilm2[1,nk,mk])
    -1j*mk*(ccilm1[0,nk,mk]+1j*ccilm1[1,nk,mk]));
    

    cilm = SHExpandGLQ(ABn[-1::-1,:],wch,zero, norm=4)
    ccilm=SHrtoc(cilm)
    AB=(ccilm[0,nk,mk]+1j*ccilm[1,nk,mk]);

    abm=mean(ABn,axis=1)
    soura=mean(alft0,axis=1)
    sourb=mean(alfs0,axis=1)
    soubu=mean(bus3,axis=1)
    return salf0+dot(MB_1,salf1),dot(MB_1,talf),1./(1+mean(beta,axis=1)**2),mean(beta,axis=1),AB,abm,bmax,brmax,soura,sourb,dot(MB_1,buos),dot(MB_1,buot),bnaxi,soubu


etas=0.01
Rw=1000
Rbu=50.
Ch=1
Rm=1.e6
Ra=1.
Ra2=Ra
tau=(xch[1]-xch[0])/12.
dtet=3.8*pi/180.
tetxh=arccos(xch)

b0=zeros(Nch) #(xch+xch**2)*(1-xch**2)**(0.5)
a0=0.01*sqrt(1-xch**2)*(1+0.001*xch) #(1.-.1*xch*sqrt(3.*pi/4.))*
#a0r=dot(mbp,b0)
b0r=dot(df1r,a0)
hel=zeros(Nch)
hels=zeros(Nch)

#print sum((a0*b0*wch))#,sum((a0r*b0r*wch))

ff0=ones(Nch)  
dfr=Rw*dot(diag(ff0),D1)
A=d2f
C=dfr
D=d2f

alff=xch
Cmu=ones(Nch)
#alpf[:]=0.1*sin(phi)


from numpy.fft import rfft,irfft
Nc=6000
#dyn=zeros((Nc,Nch*2))
#omeg=zeros((Nc,Nch*2))
sig=.5
si_g=1.-sig
Un=eye(Nch,Nch)
DD=Un*(1+tau)-tau*d2f*si_g;D_D=Un+tau*d2f*sig;
D2_n=d2f 
DDa=Un*(1+tau)-tau*D2_n*si_g; D_Da=Un+tau*D2_n*sig; 
D1a=tau*dfr
D2fh_=0.1*tau*d2f
MH=inv(Un*(1+tau/Rm)-si_g*D2fh_)
M_H=dot(MH,Un*(1+sig*D2fh_))
DD_a=inv(DDa)
unit=eye(NN)+tau*MB2*si_g-1j*abs(Rw)*tau*MRT*si_g
unis=eye(NN)+tau*MB2*si_g-1j*abs(Rw)*tau*dot(MB_1,MRS)*si_g

uni_t=inv((eye(NN)*(1+1.5*tau)-tau*MB2*sig+sig*1j*abs(Rw)*tau*MRT))
uni_s=inv((eye(NN)*(1+1.5*tau)-tau*MB2*sig+sig*1j*abs(Rw)*tau*dot(MB_1,MRS)))
unih=eye(NN)+0.1*tau*MB2*si_g
uni_h=inv(eye(NN)*(1+tau/Rm)-0.1*tau*MB2*sig)


salf=zeros(NN)+1j*zeros(NN)
talf=zeros(NN)+1j*zeros(NN)

S0=zeros(NN)+1j*zeros(NN)
T0=zeros(NN)+1j*zeros(NN)
AB=zeros(NN)+1j*zeros(NN)
heln=zeros(NN)+1j*zeros(NN)

S0[((mk==1)) & (nk ==2)]=0.0
#T0[((mk==1) | (mk ==-1) | (mk==2) | (mk ==-2)) & (nk ==1)]=0.0001
#T0[((mk==1) | (mk ==-1) | (mk==2) | (mk ==-2)) & (nk ==2)]=0.0001
tn=T0;sn=S0;hn=heln
t_n=tn;s_n=sn;h_n=hn
phi0=pi/2.
cbu=zeros(Nch)


#salf,talf,bet,beta,AB,abm,bmax,brmax,sour0,sour1,buos,buot,bnaxi,soubu=bfah(a0,b0,hels,sn,tn,heln,phi0) 
i=0
#a0b=a0;b0b=b0;hel0=hel;tn0=tn;sn0=sn;hn0=hn;
#savetxt('mfm0',asarray([a0b,b0b,hel0,abm]))
#savetxt('nxRm0',real(asarray([tn0,sn0,hn0,AB])))
#savetxt('nxIm0',imag(asarray([tn0,sn0,hn0,AB])))
#tn0,sn0,hn0,AB=loadtxt('nxRm0')+1j*loadtxt('nxIm0')
#a0b,b0b,hel0,abm=loadtxt('mfm0')/1000.
alpf=ones((Nph))
from numpy.fft import rfft, irfft
#a0=a0b/1000.;b0=b0b/1000.;hel=hel0/1000.;
#tn=tn0;sn=sn0;hn=hn0;t_n=tn0;s_n=sn0;h_n=hn0;
salf,talf,bet,beta,AB,abm,bmax,brmax,sour0,sour1,buos,buot,bnaxi,soubu=bfah(a0,b0,hels,sn,tn,heln,phi0,alpf)
si0=zeros((Nc,NN))+1j*zeros((Nc,NN))
ti0=zeros((Nc,NN))+1j*zeros((Nc,NN))
htn=zeros((Nc,NN))+1j*zeros((Nc,NN))
abn=zeros((Nc,NN))+1j*zeros((Nc,NN))
at0=zeros((Nc,Nch))
bt0=zeros((Nc,Nch))
abM=zeros((Nc,Nch))

yal=zeros(Nph)+1j*zeros(Nph)


be=zeros((Nc))
bm=zeros((Nc));bmr=zeros((Nc));
bma=zeros((Nc));
brma=zeros((Nc));
phasa=zeros((Nc));
phasa2=zeros((Nc));

g01=zeros((Nc))
g02=zeros((Nc))
g11=zeros((Nc))
h11=zeros((Nc))
g21=zeros((Nc))
h21=zeros((Nc))
g22=zeros((Nc))
h22=zeros((Nc))

br0=zeros((Nc,Nch))
hl0=zeros((Nc,Nch))
bnx=zeros((Nc,Nch))
jran=cumsum(250*(abs(ones(Nc)))).astype(int);#cumsum((300*(1+abs(randn(Nc))))).astype(int);
jr=cumsum((6*(1+abs(randn(Nc))))).astype(int) #cumsum(12*ones(Nc+1)).astype(int) ; 
alpha=zeros((Nc))
al0=zeros((Nc,Nch))
Cmu=ones(Nch)
#Calpha=1.;alpf=Calpha*zeros(Nph)
#Ca0=1+0.25*sqrt(Nph)*randn(Nph);
#yal=rfft(Ca0)/Nph;yal[Nph//8:]=0;Ca0=irfft(yal)*Nph;
#
Calpha=1#mean(Ca0)
alpf=Calpha*ones(Nph)
Ca0=alpf
cbu=zeros(Nch)
cbun=zeros((Nc,Nch))
phi00=zeros(Nc)
caf=zeros((Nc,Nph))
##
naxr=zeros((Nc,Nph//2+1))
naxf=zeros((Nc,Nph//2+1))
naxl=zeros((Nc,Nlr))
axl=zeros((Nc,Nlr))
mono=zeros((Nc))
sar=zeros((Nc))

#br1=zeros((Nc,Nch))+1j*zeros((Nc,Nch))
#br2=zeros((Nc,Nch))+1j*zeros((Nc,Nch))

#bf1=zeros((Nc,Nch))+1j*zeros((Nc,Nch))
#bf2=zeros((Nc,Nch))+1j*zeros((Nc,Nch))

#cafn=caf
#phin=phi00#zeros(Nc)
#cbunn=cbun
#if i/jran-fix(i/jran) == 0.0:

# In[12]:
ij=0
ij1=0
xi_h=0.;tfluc=0
phi0=pi
cbu=zeros(Nch);
taucfl=tau*6.*2*16
C_alf0=0.; C_alf01=0.; C_alf02=0.;C_alf03=0.;
for i in range(Nc):
    #C_alf3=(C_alf03+3*tau/taucfl*(0.5*randn(1))*sqrt(2.*taucfl/(tau)))/(1.+3*tau/taucfl) 
    #C_alf2=(C_alf02+3*tau*C_alf3/taucfl)/(1.+3*tau/taucfl)
    #C_alf1=(C_alf01+3*tau*C_alf2/taucfl)/(1.+3*tau/taucfl)
    #C_alf=(C_alf0+3.*tau*C_alf1/taucfl)/(1.+3*tau/taucfl)
    #C_alf0=C_alf; C_alf01=C_alf1; C_alf02=C_alf2;C_alf03=C_alf3
    #Calpha=1+C_alf
    
    #Calpha=c08[i]
    #alpf=caf[i,:]
    #phi0=phi00[i] 
    #cbu=cbun[i,:]
    #     if i == jran[ij]:
    #         #Cmu=1+dot(FMc3,0.1*sqrt(Nch)*randn(Nch));
    #         Ca0=1+0.2*sqrt(Nph)*randn(Nph);
    #         yal=rfft(Ca0)/Nph;yal[8:]=0;
    #         Ca0=irfft(yal)*Nph;
    #         Ca0=append(Ca0,(Ca0[0]+Ca0[-1])/2.)
    #         Calpha=mean(Ca0)
    #         alpf=Ca0
    #         ij=ij+1
    if i==jr[ij1]:
        tfluc=0.
        phi0=pi*randn(1) 
        cbu=zeros(Nch)
        ij1=ij1+1
    if (tfluc<4 and i<jr[ij1+1]):
        if int(phi0*5.) % 2:
            cbu[:Nch//2]=1.5*exp(tfluc)
            tfluc=tfluc+1.
        else: 
            cbu[Nch//2:]=1.5*exp(tfluc)
            tfluc=tfluc+1.
    else:
        cbu=zeros(Nch)
    # #    #
    # #    if i==5:
    # #        tfluc=0.
    # #        phi0=pi 
    # #        cbu=zeros(Nch)
    # #    if (tfluc<4 and i<11):
    # #        cbu[:Nch//2]=1.6*exp(tfluc)
    # #        tfluc=tfluc+1.
    # #    if i==11:cbu=zeros(Nch)
    # #    
    # #    if i==25:
    # #        tfluc=0.
    # #        phi0=pi*3./2. 
    # #        cbu=zeros(Nch)
    # #    if (tfluc<4 and i<31):
    # #        cbu[Nch//2:]=1.6*exp(tfluc)
    # #        tfluc=tfluc+1.
    # #    if i>31:cbu=zeros(Nch)
    phi00[i]=phi0 
    cbun[i,:]=cbu
    caf[i,:]=Ca0
    alpha[i]=Calpha
    #for i in range(2):
    #bet=1./(1+abs(b0)**2)
    D1b=tau*diag(Calpha*Cmu*alff*bet+Ch*hels);
    Da2=dot(DD1,dot(D1b,df1r))+D1b#+dot(diag(sqrt(1-xch**2)*xch),dot(D1b,df1r))
    M_1=inv(bmat([[DDa,-Ra*D1b*si_g],[-D1a*si_g-si_g*Ra2*Da2,DD]])) #+si_g*D1bu
    M1=bmat([[D_Da,Ra*D1b*sig],[D1a*sig+sig*Ra2*Da2,D_D]]) #-sig*D1bu
    MX=dot(M_1,M1)
    solut=asarray(dot(MX,append(a0,b0))).flatten()
    a0=dot(FMb,solut[:Nch]-Rbu*tau*dot(DD_a,vbu(b0)*a0)+tau*Ra2*dot(DD_a,sour0)-Rbu*tau*dot(DD_a,soubu*a0))
    b0=dot(FMb,solut[Nch:]-Rbu*tau*dot(DD_a,vbu(b0)*b0)+tau*Ra2*dot(DD_a,sour1)-Rbu*tau*dot(DD_a,soubu*b0)) #floss*b0+
    #g10[i]=-cnm(1)*sum(ma_0[0,:]*a0);g20[i]=-cnm(2)*sum(ma_0[1,:]*a0);

    ab=dot(df1r,a0)*dot(mabp,b0)+a0*b0
    he_l=dot(M_H,hel)+tau*dot(MH,ab+abm)/Rm  
    hels=he_l-ab-abm
    #alpha[i]=Calpha#1.+(Ca0[0]+Ca0[1])/sqrt(2.)
    #al0[i,:]=Calpha*Cmu*alff*bet+Ch*hels
    
    #
    tn=dot(uni_t,dot(unit,t_n)+Rw*tau*dot(ms1,s_n)+Ra2*tau*salf-Rbu*buot*tau)
    sn=dot(uni_s,dot(unis,s_n)+Ra2*tau*talf+Rbu*buos*tau)
    hn=dot(uni_h,dot(unih,h_n)+AB*tau/Rm) #+tau*xih*h_n)
    t_n=tn;s_n=sn;h_n=hn;heln=hn-AB
    salf,talf,bet,beta,AB,abm,bmax,brmax,sour0,sour1,buos,buot,bnaxi,soubu=bfah(a0,b0,hels,sn,tn,heln,phi0,alpf) 
    
    #if i % 4:
    hl0[i,:]=hels
    br0t=dot(df1r,a0)
    br0[i,:]=br0t
    at0[i,:]=a0
    bt0[i,:]=b0
    bm[i]=max(abs(b0));
    bmr[i]=max(abs(br0t));
    ti0[i,:]=tn
    si0[i,:]=sn
    brc=-dot(ms0,sn)
    cilm=zeros((2,Nlr+1,Nmr+1))
    cilm[0,nk,mk]=real(brc);
    cilm[1,nk,mk]=imag(brc) 
    brcr=SHctor(cilm)
    br = MakeGridGLQ(brcr,zero,norm=4)[-1::-1,:]+tensordot(dot(df1r,a0),ones(Nph),0)
    cilm = SHExpandGLQ(br[-1::-1,:],wch,zero, norm=4)
    g01[i]=cilm[0,1,0];g02[i]=cilm[0,2,0];
    g11[i]=cilm[0,1,1];h11[i]=cilm[1,1,1]
    g21[i]=cilm[0,2,1];h21[i]=cilm[1,2,1]
    g22[i]=cilm[0,2,2];h21[i]=cilm[1,2,2]
    sar[i]=sum(sum(abs(br[:,:Nph//2]),axis=1)*wch)*2*pi/Nph
    
    abn[i,:]=AB
    abM[i,:]=abm
    ABM[i,:]=ab
    htn[i,:]=heln
    bnx[i,:]=bnaxi
    bma[i]=bmax;
    brma[i]=brmax;
    #bmo=brmod(a0,sn);
    #br1[i,:]=bmo[:,1];br2[i,:]=bmo[:,2];
    #naxr[i,:]=dot(wch,abs(bmo[:,:])**2)/2.
    #bmo=bfmod(b0,sn,tn);
    #bf1[i,:]=bmo[:,1];bf2[i,:]=bmo[:,2];
    #axf[i,:]=dot(wch,abs(bmo[:,:])**2)/2.
    #axl[i,:],axl[i,:]=spml(a0,sn);
    #mono[i]=sum(sum(brfield(a0,sn),axis=1)*wch)*2*pi/Nph
    if remainder(i,1000)==0:
        print(i,mean(Ca0))
    
timi=tau*arange(Nc)

# In[13]:

plt.figure()
lin=plt.plot(dot(abs(bt0),wch)/2.,'k',
             dot(abs(bnx),wch)*pi,'k--')
plt.legend(lin, ['$B_{\\phi}$','$\\tilde{B}$'], loc='auto', frameon=False)
plt.show()

xa,ya=meshgrid(phi*180/pi,90-arccos(xch)*180/pi)

sn=si0[-1960,:]
tn=ti0[-1960,:]
a0=at0[-1960,:]
brc=-dot(ms0,sn)
bfc=dot(mst,tn)+1j*dot(msf,sn)
cilm1=zeros((2,Nlr+1,Nmr+1))
cilm1[0,nk,mk]=real(brc);
cilm1[1,nk,mk]=imag(brc) 
brcr=SHctor(cilm1)
br = MakeGridGLQ(brcr,zero,norm=4)[-1::-1,:]#+tensordot(dot(df1r,a0),ones(Nph),0)
cilm=zeros((2,Nlr+1,Nmr+1))
cilm[0,nk,mk]=real(bfc);
cilm[1,nk,mk]=imag(bfc) 
bfcr=SHctor(cilm)
bf= MakeGridGLQ(bfcr,zero,norm=4)[-1::-1,:]#+tensordot(b0,ones(Nph),0)

levs=linspace(-1.,1.,15)

plt.figure()
CS2=plt.pcolor(xa,ya,br,norm=colors.SymLogNorm(linthresh=.05, linscale=1,
            vmin=-.1,vmax=.1),cmap=cmap.seismic)
#cmap=cmap.bwr,vmin=-.1,vmax=.1)
plt.contour(xa,ya,bf[:,:],levs,colors='k')
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "3%", pad="3%")
CB=plt.colorbar(CS2, cax=cax,ticks=[-.1,0,.1],format='%.1f')
CB.set_label('B$_{r}$',fontsize=30)
plt.tight_layout()
plt.show()

# In[13]:
import cartopy.crs as ccrs

import matplotlib.colors as colors
#from matplotlib.mlab import bivariate_normal
from cartopy.util import  add_cyclic_point
bffc = add_cyclic_point(bf)
brfc = add_cyclic_point(br)
phin=arange(Nph+1)*2*pi/Nph

xn,yn=meshgrid(phin*180/pi,90-arccos(xch)*180/pi)

# In[12]:

plt.figure()
ax = plt.axes(projection=ccrs.Mollweide(central_longitude=0))
#ax = plt.axes(projection=ccrs.Orthographic(central_latitude=20,central_longitude=180.0-(phi[1]-phi[0])*180/pi/2.))
#ax.contour(xn,yn, bffc,levs,colors='k',transform=ccrs.PlateCarree())
#CS=ax.pcolormesh(xn,yn,brfc,norm=colors.SymLogNorm(linthresh=.001, linscale=100,
#            vmin=-.1,vmax=.1),cmap=cmap.seismic)
                # vmin=-.1,vmax=.1,cmap=cmap.sesmic, transform=ccrs.PlateCarree())
#CS=ax.pcolor(xn,yn,brfc,norm=colors.SymLogNorm(linthresh=.01, linscale=1,
#            vmin=-.1,vmax=.1),cmap=cmap.seismic, transform=ccrs.PlateCarree())
CS=ax.pcolormesh(xn[::2,:],yn[::2,:],brfc[::2,:],norm=colors.SymLogNorm(linthresh=0.05, linscale=1.,vmin=-0.2,vmax=0.2),cmap=cmap.seismic, transform=ccrs.PlateCarree())
CB=plt.colorbar(CS,orientation='horizontal',fraction=0.05, pad=0.02,ticks=[-.1,0,.1],format='%.1f')#, cax=cax)
CB.set_label('$\\tilde{Br}$',fontsize=24)
plt.tight_layout() 
plt.show()

plt.figure()
plt.plot(xch,dot(df1r,a0),'k')
plt.xlabel('cos$\\theta$')
plt.ylabel('$\\overline{B}r$')
plt.show()
# In[13]:

bfc=abn[-1960,:]
cilm=zeros((2,Nlr+1,Nmr+1))
cilm[0,nk,mk]=real(bfc);
cilm[1,nk,mk]=imag(bfc) 
bfcr=SHctor(cilm)
bf= MakeGridGLQ(bfcr,zero,norm=4)[-1::-1,:]#+tensordot(ABM[990,:],ones(Nph),0)

levs=linspace(-1.,1.,15)

bffn = add_cyclic_point(bf)
plt.figure()
ax = plt.axes(projection=ccrs.Mollweide(central_longitude=0))
CS=ax.pcolormesh(xn[::2,:],yn[::2,:],bffn[::2,:],norm=colors.SymLogNorm(linthresh=0.001, linscale=1.,vmin=-0.1,vmax=0.1),cmap=cmap.seismic, transform=ccrs.PlateCarree())
CB=plt.colorbar(CS,orientation='horizontal',fraction=0.05, pad=0.02,ticks=[-.01,0,.01],format='%.2f')#, cax=cax)
CB.set_label('$\\tilde{\chi}$',fontsize=24)
plt.tight_layout() 
plt.show()
plt.figure()
lin=plt.plot(xch,hl0[-1960,:],'k',xch,abM[1960,:],'k--',xch,10*abM[1960,:],'r')
plt.xlabel('cos$\\theta$')
plt.legend(lin, ['$\\overline{\\chi}$','$\\overline{A}\cdot\\overline{B}$',
                 '10$\\tilde{A}\cdot\\tilde{B}$'], loc='auto', frameon=False)
#plt.yaxis(ticks=[-.01,0,.01],format='%.2f')
plt.show()

# In[14]:

tr=timi
xt=zeros((Nc,Nch),float);
yt=zeros((Nc,Nch),float);
brrt=zeros((Nc,Nch),float);
hS=zeros(Nc)
for i in range(Nc):
    yt[i,:]=(pi/2.-arccos(xch[:]))*180./pi;
    xt[i,:]=tr[i];
    brrt[i,:]=dot(df1r,at0[i,:])
    hS[i]=sum(abM[i,:Nch//2]*wch[:Nch//2])
# In[14]:
import matplotlib.colors as colors
    
indx=where(((xt[:,0] >= 0.5) & (xt[:,0] <= 2)))[0]
xto=xt[indx,:]
yto=yt[indx,:]
bct=bt0[indx,:] 
brt=brrt[indx,:]
hrt=abM[indx,:]

levh=-max(abs(hrt).flatten())/2.+max(abs(hrt).flatten())*arange(12)/11.
levb=-max(abs(bct).flatten())/2.+max(abs(bct).flatten())*arange(12)/11.
leva=-max(abs(brt).flatten())/2.+max(abs(brt).flatten())*arange(12)/11.
#levb2=-max(abs(b2).flatten())/2.+max(abs(b2).flatten())*arange(12)/11.
 

plt.figure(figsize=[8,4])
CS1=plt.contour(xt[::4,:],yt[::4,:],bt0[::4,:], levb,colors='k')
CS2=plt.pcolor(xt[::4,:], yt[::4,:], abM[::4,:],norm=colors.SymLogNorm(linthresh=.0001, linscale=100,
            vmin=-.01,vmax=.01),cmap=cmap.seismic)
#№,cmap=cmap.seismic,vmin=-.1,vmax=.1)
#plt.xlabel('R$^2/\\eta_T$')
plt.ylabel('LATITUDE')
plt.ylim( (-90, 90) )
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "3%", pad="3%")
CB=plt.colorbar(CS2, cax=cax,ticks=[-0.01,0,0.01],format='%.2f')
CB.set_label('$\\tilde{a}\\tilde{b}$',fontsize=30)
subplots_adjust(hspace=0,bottom=0.1, left=0.1, right=0.84, top=0.95)
plt.show()
# In[15]:
abdr=zeros((Nc,Nch),float);
aba=zeros((Nc,Nch),float);
abbu=zeros((Nc,Nch),float);
ABM=zeros((Nc,Nch),float);
for i in range(0,Nc):
    a0=at0[i,:];b0=bt0[i,:]
    tn=ti0[i,:];sn=si0[i,:]
    hn=htn[i,:]
    AB=abn[i,:]
    abm=abM[i,:]
    br0=dot(df1r,a0)
    sr0=dot(ma0b,dot(m_00c,br0))
    ar0=dot(mabp,b0)
    cr0=dot(ma0b,(mode*(mode+1))*dot(inv(ma0b),ar0))

    brc=-dot(ms0,sn)
    bfc=dot(mst,tn)+1j*dot(msf,sn)
    bpc=1j*dot(msf,tn)-dot(mst,sn)
    apc=1j*dot(msf,sn);afc=dot(mst,sn)    

    cilm=zeros((2,Nlr+1,Nmr+1))
    cilm[0,nk,mk]=real(brc);
    cilm[1,nk,mk]=imag(brc) 
    brcr=SHctor(cilm)
    br = MakeGridGLQ(brcr,zero,norm=4)[-1::-1,:]+tensordot(br0,ones(Nph),0)
    brn = MakeGridGLQ(brcr,zero,norm=4)[-1::-1,:]
    cilm=zeros((2,Nlr+1,Nmr+1))
    cilm[0,nk,mk]=real(bfc);
    cilm[1,nk,mk]=imag(bfc) 
    bfcr=SHctor(cilm)
    bf= MakeGridGLQ(bfcr,zero,norm=4)[-1::-1,:]+tensordot(b0,ones(Nph),0)
    bfm= MakeGridGLQ(bfcr,zero,norm=4)[-1::-1,:]
    
    cilm=zeros((2,Nlr+1,Nmr+1))
    cilm[0,nk,mk]=real(tn);
    cilm[1,nk,mk]=imag(tn) 
    arc=SHctor(cilm)
    ar = MakeGridGLQ(arc,zero,norm=4)[-1::-1,:]+tensordot(ar0,ones(Nph),0)
    arm = MakeGridGLQ(arc,zero,norm=4)[-1::-1,:]
    
    curb=-dot(ms0,tn);
    cilm[0,nk,mk]=real(curb);
    cilm[1,nk,mk]=imag(curb); 
    curlc=SHctor(cilm)
    curlr = MakeGridGLQ(curlc,zero,norm=4)[-1::-1,:]#+tensordot(cr0,ones(Nph),0)
       
    cilm=zeros((2,Nlr+1,Nmr+1))
    cilm[0,nk,mk]=real(apc);cilm[1,nk,mk]=imag(apc) 
    rcilm=SHctor(cilm)
    apr=(rcilm)
    ap = MakeGridGLQ(apr,zero,norm=4)[-1::-1,:] 
    cilm=zeros((2,Nlr+1,Nmr+1))
    cilm[0,nk,mk]=real(bpc);cilm[1,nk,mk]=imag(bpc) 
    bpcr=SHctor(cilm)
    bp = MakeGridGLQ(bpcr,zero,norm=4)[-1::-1,:]
    
    
    cilm=zeros((2,Nlr+1,Nmr+1))
    cilm[0,nk,mk]=real(afc);cilm[1,nk,mk]=imag(afc) 
    afcr=SHctor(cilm)
    af = MakeGridGLQ(afcr,zero,norm=4)[-1::-1,:]+tensordot(a0,ones(Nph),0)
    afm = MakeGridGLQ(afcr,zero,norm=4)[-1::-1,:]

    cilm=zeros((2,Nlr+1,Nmr+1))
    cilm[0,nk,mk]=real(hn);
    cilm[1,nk,mk]=imag(hn) 
    hcr=SHctor(cilm)
    hss = MakeGridGLQ(hcr,zero,norm=4)[-1::-1,:]#+tensordot(ar0,ones(Nph),0)
    
       
    hln=tensordot(ar0*br0+a0*b0,ones(Nph),0)#+ap*bp
    hlm=arm*brn+afm*bfm+ap*bp
    hlu=-brn*afm*Rw*tensordot(1.-xch**2/4.,ones(Nph),0)
    hla=(bp*afm-ap*bfm)*tensordot(xch,ones(Nph),0)
    hlb=Rbu*(ap*bp-afm*bfm)
    hc=br*curlr
    abdr[i,:]=sum(hlu,axis=1)*2*pi/Nph
    aba[i,:]=sum(hla,axis=1)*2*pi/Nph
    abbu[i,:]=sum(hlb,axis=1)*2*pi/Nph
    ABM[i,:]=dot(df1r,a0)*dot(mabp,b0)+a0*b0
    if remainder(i,1000)==0:
        print(i)
# In[16]:

plt.figure(figsize=[8,4])
CS1=plt.contour(xt[::4,:],yt[::4,:],bt0[::4,:], levb,colors='k')
CS2=plt.pcolor(xt[::4,:], yt[::4,:], (hl0)[::4,:],norm=colors.SymLogNorm(linthresh=.01, linscale=1,
            vmin=-.5,vmax=.5),cmap=cmap.seismic)
#№,cmap=cmap.seismic,vmin=-.1,vmax=.1)
#plt.xlabel('R$^2/\\eta_T$')
plt.ylabel('LATITUDE')
plt.ylim( (-90, 90) )
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "3%", pad="3%")
CB=plt.colorbar(CS2, cax=cax,ticks=[-0.1,0,0.1],format='%.2f')
CB.set_label('$\\bar{\chi}$',fontsize=30)
subplots_adjust(hspace=0,bottom=0.1, left=0.1, right=0.84, top=0.95)
plt.show()


plt.figure(figsize=[8,4])
#CS1=plt.contour(xt[::4,:],yt[::4,:],bt0[::4,:], levb,colors='k')
CS2=plt.pcolor(xt[::4,:], yt[::4,:], (abM)[::4,:],norm=colors.SymLogNorm(linthresh=.0001, linscale=100,
            vmin=-.01,vmax=.01),cmap=cmap.seismic)
#№,cmap=cmap.seismic,vmin=-.1,vmax=.1)
#plt.xlabel('R$^2/\\eta_T$')
plt.ylabel('LATITUDE')
plt.ylim( (-90, 90) )
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "3%", pad="3%")
CB=plt.colorbar(CS2, cax=cax,ticks=[-0.01,0,0.01],format='%.2f')
CB.set_label('$\\tilde{\chi}$',fontsize=30)
subplots_adjust(hspace=0,bottom=0.1, left=0.1, right=0.84, top=0.95)
plt.show()

plt.figure(figsize=[8,4])
#CS1=plt.contour(xt[::4,:],yt[::4,:],bt0[::4,:], levb,colors='k')
CS2=plt.pcolor(xt[::4,:], yt[::4,:], (abdr)[::4,:],norm=colors.SymLogNorm(linthresh=.05, linscale=10,
            vmin=-.2,vmax=.2),cmap=cmap.seismic)
#№,cmap=cmap.seismic,vmin=-.1,vmax=.1)
#plt.xlabel('R$^2/\\eta_T$')
plt.ylabel('LATITUDE')
plt.ylim( (-90, 90) )
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "3%", pad="3%")
CB=plt.colorbar(CS2, cax=cax,ticks=[-0.1,-0.01,0.01,0.1],format='%.2f')
CB.set_label('F$_{\\Omega}$',fontsize=30)
subplots_adjust(hspace=0,bottom=0.1, left=0.1, right=0.84, top=0.95)
plt.show()


plt.figure(figsize=[8,4])
#CS1=plt.contour(xt[::4,:],yt[::4,:],bt0[::4,:], levb,colors='k')
CS2=plt.pcolor(xt[::4,:], yt[::4,:], abbu[::4,:],norm=colors.SymLogNorm(linthresh=.01, linscale=10,
            vmin=-.05,vmax=.05),cmap=cmap.seismic)
#№,cmap=cmap.seismic,vmin=-.1,vmax=.1)
#plt.xlabel('R$^2/\\eta_T$')
plt.ylabel('LATITUDE')
plt.ylim( (-90, 90) )
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "3%", pad="3%")
CB=plt.colorbar(CS2, cax=cax,ticks=[-0.05,-0.01,0.01,0.05],format='%.2f')
CB.set_label('F$_{\\beta}$',fontsize=30)
subplots_adjust(hspace=0,bottom=0.1, left=0.1, right=0.84, top=0.95)
plt.show()
plt.figure(figsize=[8,4])
#CS1=plt.contour(xt[::4,:],yt[::4,:],bt0[::4,:], levb,colors='k')
CS2=plt.pcolor(xt[::4,:], yt[::4,:], aba[::4,:]*100,norm=colors.SymLogNorm(linthresh=.05, linscale=10,
            vmin=-.2,vmax=.2),cmap=cmap.seismic)
#№,cmap=cmap.seismic,vmin=-.1,vmax=.1)
#plt.xlabel('R$^2/\\eta_T$')
plt.ylabel('LATITUDE')
plt.ylim( (-90, 90) )
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "3%", pad="3%")
CB=plt.colorbar(CS2, cax=cax,ticks=[-0.1,-0.01,0.01,0.1],format='%.2f')
CB.set_label('100F$_{\\alpha}$',fontsize=30)
subplots_adjust(hspace=0,bottom=0.1, left=0.1, right=0.84, top=0.95)
plt.show()


flalf=at0*at0*tensordot(ones(Nc),xch,0)

plt.figure(figsize=[8,4])
#CS1=plt.contour(xt[::4,:],yt[::4,:],bt0[::4,:], levb,colors='k')
CS2=plt.pcolor(xt[::4,:], yt[::4,:], 100*(flalf+aba)[::4,:],norm=colors.SymLogNorm(linthresh=.03, linscale=10,
            vmin=-.2,vmax=.2),cmap=cmap.seismic)
#№,cmap=cmap.seismic,vmin=-.1,vmax=.1)
#plt.xlabel('R$^2/\\eta_T$')
plt.ylabel('LATITUDE')
plt.ylim( (-90, 90) )
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "3%", pad="3%")
CB=plt.colorbar(CS2, cax=cax,ticks=[-0.1,-0.01,0,0.01,0.1],format='%.2f')
CB.set_label('-${\\alpha}\\overline{B}_{\\phi}\\overline{A}_{\\theta}$',fontsize=30)
subplots_adjust(hspace=0,bottom=0.1, left=0.1, right=0.84, top=0.95)
plt.show()
