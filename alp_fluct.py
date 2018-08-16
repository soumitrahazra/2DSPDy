from scipy.linalg import *
from numpy.random import *
from func_dif import *
from scipy.special import *
from plac import *
params = {'axes.labelsize': 28,
          'font.size': 28,
          'legend.fontsize': 28,
          'xtick.labelsize': 28,
          'ytick.labelsize': 28,
          'text.usetex': True}
plt.rcParams.update(params)
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
fia='/home/va/work/pap/stars/sth/ani/'

set_printoptions(precision=16)
from axmat import *
from init00mc import *



def bspm(b0,T,S):
    Nmax=Nn
    sfk=zeros((Nch,Nph//2+1))+1j*zeros((Nch,Nph//2+1))
    sp=zeros(Nm)
    for ik in range(Nm):
        sfk[:,ik]=sum(MFT[:,(mk==ik) & (nk >=ik) & (nk <= Nmax)]*
        T[(mk==ik) & (nk >=ik) & (nk <= Nmax) ]+1j*MFS[:,(mk==ik) & (nk >=ik) & (nk <= Nmax)]*
        S[(mk==ik) & (nk >=ik) & (nk <= Nmax)],axis=1)
    sp[0]=sum(abs(b0)*wch)
    for j in range(1,Nm):
        sp[j]=sum(abs(sfk[:,j])*wch)
    return sp   
def spm(a0,S):
    Nmax=Nn
    sfk=zeros((Nch,Nph//2+1))+1j*zeros((Nch,Nph//2+1))
    sp=zeros(Nm)
    for ik in range(Nm):
        sfk[:,ik]=sum((SS0[:,((mk==ik) & (nk >=ik) & (nk <= Nmax))]
            *S[((mk==ik) & (nk >=ik) & (nk <= Nmax))]),axis=1)
    brm0=-dot(df1r,a0)
    sp[0]=sum(abs(brm0)*wch)
    for j in range(1,Nm):
        sp[j]=sum(abs(sfk[:,j])*wch)
    return sp   
def spml(a0,S):
    Nmax=Nn
    br=zeros((Nch,Nph),dtype=float64)
    sfk=zeros((Nch,Nph//2+1))+1j*zeros((Nch,Nph//2+1))
    sp0=zeros(Nmax); sp=zeros(Nmax)
    brfc=zeros(NN)+1j*zeros(NN)
    #for i in range(Nch):
    for ik in range(Nm):
        sfk[:,ik]=sum((SS0[:,((mk==ik) & (nk >=ik) & (nk <= Nmax))]
            *S[((mk==ik) & (nk >=ik) & (nk <= Nmax))]),axis=1)
    br[:,:]=-irfft(sfk,axis=1)*Nph   
    brft=rfft(br,axis=1)/Nph
    for i in range(NN//2):
        brfc[i]=sum(MM0[:,i]*(brft[:,mk[i]])*wch[:])
    brfc[NN//2:]=conjugate(brfc[:NN//2])
    brm0=dot(inv(M0C),-dot(df1r,a0))
    for j in range(Nmax):
        sp0[j]=sum((abs(brfc[(nk==j)]*conj(brfc[(nk==j)]))))/((2.*j+1)) #sqrt(len(brfc[(nk==j)]))
        sp[j]=(brm0[j]**2/(2.*j+1))#+sum(sqrt(abs(brfc[(nk==j)]*conj(brfc[(nk==j)]))))/sqrt((2.*j+1))#/sqrt(len(brfc[(nk==j)]))
    return sp0,sp   

def afield(S):
    Nmax=Nn
    sfk=zeros((Nch,Nph//2+1))+1j*zeros((Nch,Nph//2+1))
    #for i in range(Nch):
    for ik in range(Nm):
        sfk[:,ik]=sum((MM0[:,((mk==ik) & (nk >=ik) & (nk <= Nmax))]
            *S[((mk==ik) & (nk >=ik) & (nk <= Nmax))]),axis=1)  
    return irfft(sfk,axis=1)*Nph

def enhel(S):
    Nmax=Nn
    en=zeros(Nm)
    sfk=zeros((Nch,Nph//2+1))+1j*zeros((Nch,Nph//2+1))
    #for i in range(Nch):
    for ik in range(Nm):
        sfk[:,ik]=sum((MM0[:,((mk==ik) & (nk >=ik) & (nk <= Nmax))]
            *S[((mk==ik) & (nk >=ik) & (nk <= Nmax))]),axis=1)#+dot(df1r,a0)
        en[ik]=sum(abs(sfk)[:,ik]*wch,axis=0)
    return  en  

def enfield(S):
    Nmax=Nn
    en=zeros(Nm)
    sfk=zeros((Nch,Nph//2+1))+1j*zeros((Nch,Nph//2+1))
    #for i in range(Nch):
    for ik in range(Nm):
        sfk[:,ik]=sum((SS0[:,((mk==ik) & (nk >=ik) & (nk <= Nmax))]
            *S[((mk==ik) & (nk >=ik) & (nk <= Nmax))]),axis=1)#+dot(df1r,a0)
        en[ik]=sum(abs(sfk)[:,ik]*wch,axis=0)
    return  en  

def brmod(a0,S):
    Nmax=Nn
    br=zeros((Nch,Nph),dtype=float64)
    sfk=zeros((Nch,Nph//2+1))+1j*zeros((Nch,Nph//2+1))
    #for i in range(Nch):
    for ik in range(Nm):
        sfk[:,ik]=sum((SS0[:,((mk==ik) & (nk >=ik) & (nk <= Nmax))]
            *S[((mk==ik) & (nk >=ik) & (nk <= Nmax))]),axis=1)#+dot(df1r,a0)
    for ik in range(Nph):
        br[:,ik]=(-irfft(sfk,axis=1)[:,ik]*Nph-dot(df1r,a0))
    brm=rfft(br,axis=1)/Nph   
    return brm   

def bfmod(b0,S,T):
    Nmax=Nn
    bf=zeros((Nch,Nph),dtype=float64)
    sfk=zeros((Nch,Nph//2+1))+1j*zeros((Nch,Nph//2+1))
    #for i in range(Nch):
    for ik in range(Nm):
        sfk[:,ik]=sum(MFT[:,(mk==ik) & (nk >=ik) & (nk <= Nmax)]*
        T[(mk==ik) & (nk >=ik) & (nk <= Nmax) ]+1j*MFS[:,(mk==ik) & (nk >=ik) & (nk <= Nmax)]*
        S[(mk==ik) & (nk >=ik) & (nk <= Nmax)],axis=1)#+b
    for ik in range(Nph):
        bf[:,ik]=irfft(sfk,axis=1)[:,ik]*Nph+b0
    bfm=rfft(bf,axis=1)/Nph   
    return bfm   

def brfield(a0,S):
    Nmax=Nn
    br=zeros((Nch,Nph),dtype=float64)
    sfk=zeros((Nch,Nph//2+1))+1j*zeros((Nch,Nph//2+1))
    #for i in range(Nch):
    for ik in range(Nm):
        sfk[:,ik]=sum((SS0[:,((mk==ik) & (nk >=ik) & (nk <= Nmax))]
            *S[((mk==ik) & (nk >=ik) & (nk <= Nmax))]),axis=1)#+dot(df1r,a0)
    for ik in range(Nph):
        br[:,ik]=-irfft(sfk,axis=1)[:,ik]*Nph-dot(df1r,a0)
    return br   
def bfield(ba,T,S):
    Nmax=Nn
    bf=zeros((Nch,Nph),dtype=float64)
    bfn=zeros((Nch,Nph),dtype=float64)
    sfk=zeros((Nch,Nph//2+1))+1j*zeros((Nch,Nph//2+1))
    for ik in range(Nm):
        sfk[:,ik]=sum(MFT[:,(mk==ik) & (nk >=ik) & (nk <= Nmax)]*
        T[(mk==ik) & (nk >=ik) & (nk <= Nmax) ]+1j*MFS[:,(mk==ik) & (nk >=ik) & (nk <= Nmax)]*
        S[(mk==ik) & (nk >=ik) & (nk <= Nmax)],axis=1)#+b
    bfn[:,:]=irfft(sfk,axis=1)*Nph
    for ik in range(Nph):
        bf[:,ik]=bfn[:,ik]+ba
    return bf   

def brd(a,S):
    Nmax=Nn
    br=zeros((Nch,Nph),dtype=float64)
    brn=zeros((Nch,Nph),dtype=float64)
    sfk=zeros((Nch,Nph//2+1))+1j*zeros((Nch,Nph//2+1))
    sfk[:,1]=sum((SS0[:,((mk==1) & (nk <= 1))]*S[((mk==1) & (nk <=1))]),axis=1)
    brn[:,:]=-irfft(sfk,axis=1)*Nph
    for ik in range(Nph):
        br[:,ik]=brn[:,ik]-dot(FMd,dot(df1r,a))
    return br   
def brq(a,S):
    Nmax=Nn
    br=zeros((Nch,Nph),dtype=float64)
    brn=zeros((Nch,Nph),dtype=float64)
    sfk=zeros((Nch,Nph//2+1))+1j*zeros((Nch,Nph//2+1))
    sfk[:,1]=sum((SS0[:,((mk == 1) & (nk == 2))]*S[((mk==1) & (nk ==2))]),axis=1)
    sfk[:,2]=sum((SS0[:,((mk == 2) & (nk == 2))]*S[((mk==2) & (nk ==2))]),axis=1)
    brn[:,:]=-irfft(sfk,axis=1)*Nph
    for ik in range(Nph):
        br[:,ik]=brn[:,ik]-dot(FMq,dot(df1r,a))
    return br   



def btfield(T,S):
    Nmax=Nn
    bth=zeros((Nch,Nph),dtype=float64)
    sfk=zeros((Nch,Nph//2+1))+1j*zeros((Nch,Nph//2+1))
    for ik in range(Nm):
        sfk[:,ik]=sum(1j*MFS[:,((mk==ik) & (nk >=ik) & (nk <= Nmax))]*
        T[(mk==ik) & (nk >=ik) & (nk <= Nmax)]-MFT[:,(mk==ik) & (nk >=ik) & (nk <= Nmax)]*
        S[(mk==ik) & (nk >=ik) & (nk <= Nmax)] ,axis=1)
    bth[:,:]=irfft(sfk,axis=1)*Nph
    return bth   

#    for ik in range(Nm):
#        sfk[ik]=sum((MFT[i,:]*T)[(mk==ik) & (nk >=ik) & (nk <= Nmax)])
#    bf[i,:]=irfft(sfk)*Nph
#    for ik in range(Nm):
#        sfk[ik]=sum(-1j*(MFS[i,:]*T)[(mk==ik) & (nk >=ik) & (nk <= Nmax)])
#    bth[i,:]=irfft(sfk)*Nph

def bfah(aa,ba,hels,S,T,htn,phi0):
    Nmax=Nn
    #cbu=zeros(Nch)
    br=zeros((Nch,Nph),dtype=float64)
    bth=zeros((Nch,Nph),dtype=float64)
    bf=zeros((Nch,Nph),dtype=float64)
    ar=zeros((Nch,Nph),dtype=float64)
    ath=zeros((Nch,Nph),dtype=float64)
    af=zeros((Nch,Nph),dtype=float64)
    ABn=zeros((Nch,Nph),dtype=float64)

    nnl=zeros((Nch,Nph),dtype=float64)
    beta=zeros((Nch,Nph),dtype=float64)

    alfs=zeros((Nch,Nph),dtype=float64)
    alfs1=zeros((Nch,Nph),dtype=float64)
    alfs2=zeros((Nch,Nph),dtype=float64)
    alft1=zeros((Nch,Nph),dtype=float64)
    alft0=zeros((Nch,Nph),dtype=float64)
    alfs0=zeros((Nch,Nph),dtype=float64)

    alft2=zeros((Nch,Nph),dtype=float64)
    salf0=zeros(NN)+1j*zeros(NN)
    salf1=zeros(NN)+1j*zeros(NN)
    sfk=zeros((Nch,Nph//2+1))+1j*zeros((Nch,Nph//2+1))
    sd=sqrt(1-xch**2)
    but1=zeros((Nch,Nph),dtype=float64)
    but2=zeros((Nch,Nph),dtype=float64)
    bus1=zeros((Nch,Nph),dtype=float64)
    bus2=zeros((Nch,Nph),dtype=float64)
    bus3=zeros((Nch,Nph),dtype=float64)
    buos=zeros(NN)+1j*zeros(NN)
    buot=zeros(NN)+1j*zeros(NN)

    #for i in range(Nch):
    for ik in range(Nm):
        sfk[:,ik]=sum((SS0[:,((mk==ik) & (nk >=ik) & (nk <= Nmax))]
            *S[((mk==ik) & (nk >=ik) & (nk <= Nmax))]),axis=1)  
    br[:,:]=-irfft(sfk,axis=1)*Nph  
    for ik in range(Nm):
        sfk[:,ik]=sum(MFT[:,(mk==ik) & (nk >=ik) & (nk <= Nmax)]*
        T[(mk==ik) & (nk >=ik) & (nk <= Nmax) ]+1j*MFS[:,(mk==ik) & (nk >=ik) & (nk <= Nmax)]*
        S[(mk==ik) & (nk >=ik) & (nk <= Nmax)],axis=1)
    bf[:,:]=irfft(sfk,axis=1)*Nph
    for ik in range(Nm):
        sfk[:,ik]=sum(1j*MFS[:,((mk==ik) & (nk >=ik) & (nk <= Nmax))]*
        T[(mk==ik) & (nk >=ik) & (nk <= Nmax)]-MFT[:,(mk==ik) & (nk >=ik) & (nk <= Nmax)]*
        S[(mk==ik) & (nk >=ik) & (nk <= Nmax)] ,axis=1)
    bth[:,:]=irfft(sfk,axis=1)*Nph
    for ik in range(Nm):
        sfk[:,ik]=sum((MM0[:,((mk==ik) & (nk >=ik) & (nk <= Nmax))]
            *T[((mk==ik) & (nk >=ik) & (nk <= Nmax))]),axis=1)  
    ar[:,:]=-irfft(sfk,axis=1)*Nph  
    for ik in range(Nm):
        sfk[:,ik]=sum((1j*MFS[:,((mk==ik) & (nk >=ik) & (nk <= Nmax))]
            *S[((mk==ik) & (nk >=ik) & (nk <= Nmax))]),axis=1)  
    ath[:,:]=-irfft(sfk,axis=1)*Nph  
    for ik in range(Nm):
        sfk[:,ik]=sum((MFT[:,((mk==ik) & (nk >=ik) & (nk <= Nmax))]
            *S[((mk==ik) & (nk >=ik) & (nk <= Nmax))]),axis=1)  
    af[:,:]=-irfft(sfk,axis=1)*Nph  
    for j in range(Nph):
        ABn[:,j]=(ar[:,j]*br[:,j]+bth[:,j]*ath[:,j]+af[:,j]*bf[:,j]
        +dot(df1r,a0)*ar[:,j]+br[:,j]*dot(mabp,b0)+a0*bf[:,j]+b0*af[:,j])   
    for ik in range(Nm):
        sfk[:,ik]=sum((MM0[:,((mk==ik) & (nk >=ik) & (nk <= Nmax))]
            *htn[((mk==ik) & (nk >=ik) & (nk <= Nmax))]),axis=1)  
    hnax=-irfft(sfk,axis=1)*Nph  
    bmax=max(sqrt(br[:,:]**2+bth[:,:]**2+(bf[:,:]+tensordot(ba,ones(Nph),0))**2).flatten())
    br0=-dot(df1r,aa)
    brmax=max(sqrt((br[:,:]+tensordot(br0,ones(Nph),0))**2).flatten())
    bnaxi= sum(sqrt(br[:,:]**2+bth[:,:]**2+bf[:,:]**2),axis=1)/Nph       
    for j in range(Nch):
        beta[j,:]=sqrt(br[j,:]**2+(bf[j,:]+ba[j])**2+bth[j,:]**2)
        nnl[j,:]=1./(1+beta[j,:]**2)
    for j in range(Nph):
        alfs[:,j]=(Cmu*alff*alpf[j]*nnl[:,j]+Ch*(hels+hnax[:,j]))*(br[:,j]-dot(df1r,aa))
        alfs1[:,j]=dot(DD1,(Cmu*alff*alpf[j]*nnl[:,j]+Ch*(hels+hnax[:,j]))*sd*(bth[:,j]+aa))
        alfs2[:,j]=(Cmu*alff*nnl[:,j]*alpf[j]+Ch*(hels+hnax[:,j]))*(bf[:,j]+ba)/sqrt(1-xch**2)
        alft1[:,j]=dot(DD1,(Cmu*alff*alpf[j]*nnl[:,j]+Ch*(hels+hnax[:,j]))*(bf[:,j]+ba)*sqrt(1-xch**2))#/sqrt(1-xch**2)
        alft2[:,j]=(Cmu*alff*alpf[j]*nnl[:,j]+Ch*(hels+hnax[:,j]))*(bth[:,j]+aa)/sqrt(1-xch**2)
        alft0[:,j]=(Cmu*alff*alpf[j]*nnl[:,j]+Ch*(hels+hnax[:,j]))*bf[:,j]
        
        #*sqrt(1-xch**2))#/sqrt(1-xch**2)
        alfs0[:,j]=(dot(DD3,(Cmu*alff*alpf[j]*nnl[:,j]
                +Ch*(hels+hnax[:,j]))*(br[:,j]))/sqrt(1-xch**2)+
             (Cmu*alff*alpf[j]*nnl[:,j]+Ch*(hels+hnax[:,j]))*bth[:,j])
        but1[:,j]=cbu*vbu(ba)*exp(-50*sin((phi[j]-phi0)/2)**2)*(aa+bth[:,j])/sqrt(1-xch**2)
        but2[:,j]=dot(DD1,cbu*vbu(ba)*(ba+bf[:,j])*sqrt(1-xch**2))*exp(-50*sin((phi[j]-phi0)/2)**2)
        bus1[:,j]=cbu*vbu(ba)*exp(-50*sin((phi[j]-phi0)/2)**2)*(bf[:,j]+ba)/sqrt(1-xch**2)
        bus2[:,j]=dot(DD1,cbu*vbu(ba)*(aa+bth[:,j])*sqrt(1-xch**2))*exp(-50*sin((phi[j]-phi0)/2)**2)
        bus3[:,j]=vbu(ba)*(1.+cbu*exp(-50*sin((phi[j]-phi0)/2)**2))#*(0*bf[:,j]+ba)
        
    alsf=rfft(alfs,axis=1)/Nph
    alsf1=rfft(alfs1,axis=1)/Nph
    alsf2=rfft(alfs2,axis=1)/Nph
    altf1=rfft(alft1,axis=1)/Nph
    altf2=rfft(alft2,axis=1)/Nph
    but1f=rfft(but1,axis=1)/Nph;but2f=rfft(but2,axis=1)/Nph
    bus1f=rfft(bus1,axis=1)/Nph;bus2f=rfft(bus2,axis=1)/Nph
    
    abb=rfft(ABn,axis=1)/Nph
    abm=mean(ABn,axis=1)
    soura=mean(alft0,axis=1)
    sourb=mean(alfs0,axis=1)
    soubu=mean(bus3,axis=1)
    
    for i in range(NN//2):
        salf0[i]=sum(MM0[:,i]*(alsf[:,mk[i]])*wch[:])
        salf1[i]=sum(MM0[:,i]*(alsf1[:,mk[i]]-1j*mk[i]*alsf2[:,mk[i]])*wch[:])
        talf[i]=sum(MM0[:,i]*(altf1[:,mk[i]]+1j*mk[i]*altf2[:,mk[i]])*wch[:])
        AB[i]=sum(MM0[:,i]*(abb[:,mk[i]])*wch[:])
        buos[i]=sum(MM0[:,i]*(bus2f[:,mk[i]]-1j*mk[i]*bus1f[:,mk[i]])*wch[:])
        buot[i]=sum(MM0[:,i]*(but2f[:,mk[i]]+1j*mk[i]*but1f[:,mk[i]])*wch[:])

    salf0[NN//2:]=conjugate(salf0[:NN//2])
    salf1[NN//2:]=conjugate(salf1[:NN//2])
    talf[NN//2:]=conjugate(talf[:NN//2])
    AB[NN//2:]=conjugate(AB[:NN//2])
    return salf0+dot(MB_1,salf1),dot(MB_1,talf),1./(1+mean(beta,axis=1)**2),mean(beta,axis=1),AB,abm,bmax,brmax,soura,sourb,dot(MB_1,buos),dot(MB_1,buot),bnaxi,soubu





#print sum(vLpnm(3,0,xch)*vLpnm(3,0,xch)*wch)*sum(cos(phi)**2+sin(phi)**2)*2*pi/Nph#/(2.*2.+1.)

#aphi=loadtxt('/home/va/phi.txt')

Rw=1000
Rbu=50.
Ch=0.
Rm=1.e6
Ra=1
Ra2=Ra
tau=(xch[1]-xch[0])/16.
b0=zeros(Nch) #(xch+xch**2)*(1-xch**2)**(0.5)
a0=0.001*sqrt(1-xch**2)*(1+0.01*xch) #(1.-.1*xch*sqrt(3.*pi/4.))*
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
alpf=ones((Nph))
#alpf[:]=0.1*sin(phi)



Nc=100000
#dyn=zeros((Nc,Nch*2))
#omeg=zeros((Nc,Nch*2))

sig=.5
si_g=1.-sig
Un=eye(Nch,Nch)
DD=Un*(1+3*tau)-tau*d2f*si_g;D_D=Un+tau*d2f*sig;
D2_n=d2f 
DDa=Un*(1+3*tau)-tau*D2_n*si_g; D_Da=Un+tau*D2_n*sig; 
D1a=tau*dfr
D2fh_=0.05*tau*d2f
MH=inv(Un*(1+tau/Rm)-si_g*D2fh_)
M_H=dot(MH,Un*(1+sig*D2fh_))
DD_a=inv(DDa)
unit=eye(NN)+tau*MB1*si_g-1j*abs(Rw)*tau*MRT*si_g
unis=eye(NN)+tau*MB1*si_g-1j*abs(Rw)*tau*dot(MB_1,MRS)*si_g

uni_t=inv(eye(NN)*(1+3*tau)-tau*MB1*sig+sig*1j*abs(Rw)*tau*MRT)
uni_s=inv(eye(NN)*(1+3*tau)-tau*MB1*sig+sig*1j*abs(Rw)*tau*dot(MB_1,MRS))
unih=eye(NN)+0.5*tau*MB1*si_g
uni_h=inv(eye(NN)-0.5*tau*MB1*sig)


salf=zeros(NN)+1j*zeros(NN)
talf=zeros(NN)+1j*zeros(NN)

S0=zeros(NN)+1j*zeros(NN)
T0=zeros(NN)+1j*zeros(NN)
AB=zeros(NN)+1j*zeros(NN)
heln=zeros(NN)+1j*zeros(NN)

S0[((mk==1) | (mk ==-1) | (mk==2) | (mk ==-2)) & (nk ==1)]=0.0001
S0[((mk==1) | (mk ==-1) | (mk==2) | (mk ==-2)) & (nk ==2)]=0.0001
#T0[((mk==1) | (mk ==-1) | (mk==2) | (mk ==-2)) & (nk ==1)]=0.0001
#T0[((mk==1) | (mk ==-1) | (mk==2) | (mk ==-2)) & (nk ==2)]=0.0001
tn=T0;sn=S0;hn=heln
t_n=tn;s_n=sn;h_n=hn
phi0=pi/2.
cbu=zeros(Nch)
#salf,talf,bet,beta,AB,abm,bmax,sour0,sour1=bfah(a0,b0,hels,sn,tn,heln) 
i=0
#a0b=a0;b0b=b0;hel0=hel;tn0=tn;sn0=sn;hn0=hn;
#savetxt('mf',asarray([a0b,b0b,hel0]))
#savetxt('nxR',real(asarray([tn0,sn0,hn0])))
#savetxt('nxI',imag(asarray([tn0,sn0,hn0])))
#tn0,sn0,hn0=loadtxt('nxR')+1j*loadtxt('nxI')
#a0b,b0b,hel0=loadtxt('mf')
#
a0=a0b;b0=b0b;hel=hel0;tn=tn0;sn=sn0;hn=hn0;t_n=tn0;s_n=sn0;h_n=hn0;
salf,talf,bet,beta,AB,abm,bmax,brmax,sour0,sour1,buos,buot,bnaxi,soubu=bfah(a0,b0,hels,sn,tn,heln,phi0)
si0=zeros((Nc//4,NN))+1j*zeros((Nc//4,NN))
ti0=zeros((Nc//4,NN))+1j*zeros((Nc//4,NN))
htn=zeros((Nc//4,NN))+1j*zeros((Nc//4,NN))
abn=zeros((Nc//4,NN))+1j*zeros((Nc//4,NN))
at0=zeros((Nc//4,Nch))
bt0=zeros((Nc//4,Nch))

yal=zeros(Nph)+1j*zeros(Nph)


be=zeros((Nc//4))
bm=zeros((Nc//4));bmr=zeros((Nc//4));
bma=zeros((Nc//4));
brma=zeros((Nc//4));
phasa=zeros((Nc//4));
phasa2=zeros((Nc//4));

br0=zeros((Nc//4,Nch))
hl0=zeros((Nc//4,Nch))
bnx=zeros((Nc//4,Nch))
jran=cumsum((16*(1+abs(randn(Nc))))).astype(int); 
jr=cumsum(8*ones(Nc)).astype(int) #(3*(1+abs(randn(Nc)))); 
alpha=zeros((Nc))
al0=zeros((Nc,Nch))
Cmu=ones(Nch)
Calpha=1.;alpf=Calpha*zeros(Nph)
Ca0=1+0.25*sqrt(Nph)*randn(Nph);
yal=rfft(Ca0)/Nph;yal[Nph//8:]=0;Ca0=irfft(yal)*Nph;

alpf=0*Ca0
Calpha=1#mean(Ca0)
floss=zeros(Nch)
cbu=zeros(Nch)
cbun=zeros((Nc,Nch))
phi00=zeros(Nc)
caf=zeros((Nc,Nph))
##
naxr=zeros((Nc//4,Nph//2+1))
naxf=zeros((Nc//4,Nph//2+1))
naxl=zeros((Nc//4,Nn))
axl=zeros((Nc//4,Nn))
mono=zeros((Nc//4))

#br1=zeros((Nc,Nch))+1j*zeros((Nc,Nch))
#br2=zeros((Nc,Nch))+1j*zeros((Nc,Nch))

#bf1=zeros((Nc,Nch))+1j*zeros((Nc,Nch))
#bf2=zeros((Nc,Nch))+1j*zeros((Nc,Nch))

#cafn=caf
#phin=phi00#zeros(Nc)
#cbunn=cbun
#if i/jran-fix(i/jran) == 0.0:
ij=0
ij1=0
xi_h=0.;tfluc=0
for i in range(Nc):
    #Calpha=c08[i]
    #alpf=caf[i,:]
    #phi0=phi00[i] 
    #cbu=cbun[i,:]
    if i == jran[ij]:
        Ca0=1+sqrt(Nph)*randn(Nph);
        yal=rfft(Ca0)/Nph;yal[Nph//12:]=0;Ca0=irfft(yal)*Nph;
        Calpha=mean(Ca0)
        alpf=Ca0
        ij=ij+1

    if i==jr[ij1]:
        tfluc=0.
        phi0=pi*randn(1) 
        cbu=zeros(Nch)
        if int(phi0*5.) % 2:
            cbu[:Nch//2]=40
        else: 
            cbu[Nch//2::]=40
        ij1=ij1+1
    else:
        #tfluc=tfluc+1.
        #phi0=0
        cbu=zeros(Nch)#cbu*exp(-16*tfluc)#zeros(Nch)
    phi00[i]=phi0 
    cbun[i,:]=cbu
    caf[i,:]=Ca0


    D1b=tau*diag(Cmu*Calpha*alff*bet+Ch*hels);
    Da2=dot(DD1,dot(D1b,df1r))+D1b#+dot(diag(sqrt(1-xch**2)*xch),dot(D1b,df1r))
    M_1=inv(bmat([[DDa,-Ra*D1b*si_g],[-D1a*si_g-si_g*Ra2*Da2,DD]])) #+si_g*D1bu
    M1=bmat([[D_Da,Ra*D1b*sig],[D1a*sig+sig*Ra2*Da2,D_D]]) #-sig*D1bu
    MX=dot(M_1,M1)
    solut=asarray(dot(MX,append(a0,b0))).flatten()
    a0=solut[:Nch]+tau*Ra2*dot(DD_a,sour0)-Rbu*tau*dot(DD_a,soubu*a0)
    b0=solut[Nch:]+tau*Ra2*dot(DD_a,sour1)-Rbu*tau*dot(DD_a,soubu*b0) #floss*b0+
    #g10[i]=-cnm(1)*sum(ma_0[0,:]*a0);g20[i]=-cnm(2)*sum(ma_0[1,:]*a0);

    ab=dot(df1r,a0)*dot(mabp,b0)+a0*b0
    he_l=dot(M_H,hel)+tau*dot(MH,ab+abm)/Rm  
    hels=he_l-ab-abm
    alpha[i]=Calpha#1.+(Ca0[0]+Ca0[1])/sqrt(2.)
    al0[i,:]=Calpha*alff*bet+Ch*hels
    
    #
    tn=dot(uni_t,dot(unit,t_n)+Rw*tau*dot(ms1,s_n)+Ra2*tau*salf-Rbu*buot*tau)
    sn=dot(uni_s,dot(unis,s_n)+Ra2*tau*talf+Rbu*buos*tau)
    hn=dot(uni_h,dot(unih,h_n)+AB*tau/Rm) #+tau*xih*h_n)
    tn[(mk>=Nm-2)]=0.+1j*0;sn[(mk>=Nm-2)]=0.+1j*0;
    tn[NN//2:]=conjugate(tn[:NN//2])
    sn[NN//2:]=conjugate(sn[:NN//2])
    hn[NN//2:]=conjugate(hn[:NN//2])
    t_n=tn;s_n=sn;h_n=hn;heln=hn-AB
    salf,talf,bet,beta,AB,abm,bmax,brmax,sour0,sour1,buos,buot,bnaxi,soubu=bfah(a0,b0,hels,sn,tn,heln,phi0) 
    
    if i % 4:
        hl0[i//4,:]=hels
        br0t=-dot(df1r,a0)
        br0[i//4,:]=br0t
        at0[i//4,:]=a0
        bt0[i//4,:]=b0
        bm[i//4]=max(abs(b0));
        bmr[i//4]=max(abs(br0t));
        ti0[i//4,:]=tn
        si0[i//4,:]=sn
        #    g11[i]=cmm(1,1)*real(sn[((mk==1) & (nk ==1))][0])
        #    h11[i]=cmm(1,1)*imag(sn[((mk==1) & (nk ==1))][0])
        #    g21[i]=cmm(2,1)*real(sn[((mk==1) & (nk ==2))][0])
        #    h21[i]=cmm(2,1)*imag(sn[((mk==1) & (nk ==2))][0])
        #    g22[i]=cmm(2,2)*real(sn[((mk==2) & (nk ==2))][0])
        #    h22[i]=cmm(2,2)*imag(sn[((mk==2) & (nk ==2))][0])
        be[i//4]=mean(beta)-mean(abs(b0))
        abn[i//4,:]=AB
        htn[i//4,:]=heln
        bnx[i//4,:]=bnaxi
        bma[i//4]=bmax;
        brma[i//4]=brmax;
        bmo=brmod(a0,sn);
        #br1[i,:]=bmo[:,1];br2[i,:]=bmo[:,2];
        naxr[i//4,:]=dot(wch,abs(bmo[:,:])**2)/2.
        bmo=bfmod(b0,sn,tn);
        #bf1[i,:]=bmo[:,1];bf2[i,:]=bmo[:,2];
        naxf[i//4,:]=dot(wch,abs(bmo[:,:])**2)/2.
        axl[i//4,:],naxl[i//4,:]=spml(a0,sn);
        mono[i//4]=sum(sum(brfield(a0,sn),axis=1)*wch)*2*pi/Nph
    if remainder(i,1000)==0:
        print(i,mean(Ca0))
    
time2=4*tau*arange(Nc//4)
time=4*tau*arange(Nc//4)
timer=tau*arange(Nc)

time3=4*tau*arange(Nc//4)
#time2=tau*arange(Nc)
#lt
bt07=bt0;br07=br0;bx07=bnx;br107=br1; bf107=bf1; br207=br2;bf207=bf2; nax07=naxr;nxf07=naxf;nl07=naxl;al07=axl;c07=alpha;# cbu=40,caf,0.25,0,lt m3b
bt06=bt0;br06=br0;bx06=bnx;br106=br1; bf106=bf1; br206=br2;bf206=bf2; nax06=naxr;nxf06=naxf;nl06=naxl;al06=axl;c06=alpha;# cbu=40,caf,0.25,lt m3a
#
bt08s=bt0;br08s=br0;bx08s=bnx;nax08s=naxr;nxf08s=naxf;nl08s=naxl;al08s=axl;c08s=alpha;# cbu=40,caf,1,1  m2c
#npzfile=load('mfe.npz')
#at0=npzfile['arr_0']
#bt0=npzfile['arr_0']
#hl0=npzfile['arr_0']
savez('m2cfal', alpha)
savez('m2c',at08=at0,bt08=bt0,br08=br0,bx08=bnx,nax08=naxr,nxf08=naxf,nl08=naxl,al08=axl)# cbu=50,caf,0. m1a
savez('m2cs',bt08s=bt08,br08s=br08,bx08s=bx08,nax08s=nax08,nxf08s=nxf08,nl08s=nl08,al08s=al08)# cbu=50,caf,0. m1a
savez('m2csl',bt08l=bt0,br08l=br0,bx08l=bnx,nax08l=naxr,nxf08l=naxf,nl08l=naxl,al08l=axl)# cbu=50,caf,0. m1a
#npzfile=load('M2c.npz')


bt09=bt0;br09=br0;bx09=bnx;nax09=naxr;nxf09=naxf;nl09=naxl;al09=axl;c09=alpha;# cbu=40,caf,0,1  m2d
savez('m2d',at09=at0,bt09=bt0,br09=br0,bx09=bnx,nax09=naxr,nxf09=naxf,nl09=naxl,al09=axl);# cbu=40,caf,0,1  m2d

savez('m2ds',bt09s=bt0,br09s=br0,bx09s=bnx,nax09s=naxr,nxf09s=naxf,nl09s=naxl,al09s=axl);# cbu=40,caf,0,1  m2d
bt09s=bt0;br09s=br0;bx09s=bnx;nax09s=naxr;nxf09s=naxf;nl09s=naxl;al09s=axl
bt08s=bt08;br08s=br08;bx08s=bx08;nax08s=nax08;nxf08s=nxf08;nl08s=nl08;al08s=al08


bt04=bt0;br04=br0;bx04=bnx;br104=br1; bf104=bf1; br204=br2;bf204=bf2; nax04=naxr;nxf04=naxf;nl04=naxl;al04=axl;c04=alpha;# cbu=40,caf,0.5 m2b
bt03=bt0;br03=br0;bx03=bnx;br103=br1; bf103=bf1; br203=br2;bf203=bf2; nax03=naxr;nxf03=naxf;nl03=naxl;al03=axl;c03=alpha;#cbu=0,caf,0.25 m1b
bt02=bt0;br02=br0;bx02=bnx;br102=br1; bf102=bf1; br202=br2;bf202=bf2; nax02=naxr;nxf02=naxf;nl02=naxl;al02=axl;c02=alpha;#cbu=40,caf,0.25 m2a
bt01=bt0;br01=br0;bx01=bnx;br101=br1; bf101=bf1; br201=br2;bf201=bf2; nax01=naxr;nxf01=naxf;nl01=naxl;al01=axl;c01=alpha;# cbu=0,ca=1 m0
bt00=bt0;br00=br0;bx00=bnx;br100=br1; bf100=bf1; br200=br2;bf200=bf2; nax00=naxr;nxf00=naxf;nl00=naxl;al00=axl;c00=alpha;# cbu=40, ca=1  m1a


plt.figure()
lin0=plt.plot(time, sum(abs(bt02)*wch,axis=1)/2,'b',
                 time, sum(abs(br02)*wch,axis=1)/2,'r',
                 time, sum(abs(bx02)*wch,axis=1)/2,'k')
lin1=plt.plot(time, sum(abs(bt04)*wch,axis=1)/2,'b--',
                 time, sum(abs(br04)*wch,axis=1)/2,'r--',
                 time, sum(abs(bx04)*wch,axis=1)/2,'k--')
#plt.legend(lin1, ['${\\left|\\bar{B}_{\\phi}\\right|}$',
#                 '${\\left|\\bar{B}_r\\right|}$',
#                 '${\\left|\\tilde{B} \\right|}$'], loc='auto', frameon=False)
#plt.ylim(3.e-3,1.)
plt.show()

plt.figure()
lin0=plt.plot(time,sqrt(sum(naxf[:,:],axis=1)),'k')
         #time,sqrt(sum(naxf[:,:],axis=1)),'r')
         #time2,sqrt(sum(naxf[:,:],axis=1)),'b',)
         #time,sqrt(sum(nxf02[:,:],axis=1)),'k',
         #time,sqrt(sum(nxf04[:,:],axis=1)),'r')
         #timel,sqrt(sum(nxf06[:,:],axis=1)),'k',
         #timel,sqrt(sum(nxf07[:,:],axis=1)),'r')#,time,sum(nxf05[:,:],axis=1)/2.,'b')
#lin1=plt.plot(time2,sqrt(nxf08s[:,0]),'k--',time3,sqrt(naxf[:,0]),'r--')#,time,nxf05[:,0]/2.,'b--')
plt.legend(lin0, ['M2c','M2d'], loc='auto', frameon=False)
plt.ylabel('${\\langle {B}_{\\phi}\\rangle}$'+', ${\\left|\\bar{B}_{\\phi}\\right|}$')
#plt.xlim(16.,17.)
#plt.yscale('log')
plt.show()


dip02=sum((br02[:,:]-br02[:,-1::-1])**2*wch,axis=1)
qup02=sum((br02[:,:]+br02[:,-1::-1])**2*wch,axis=1)
dip04=sum((br04[:,:]-br04[:,-1::-1])**2*wch,axis=1)
qup04=sum((br04[:,:]+br04[:,-1::-1])**2*wch,axis=1)
dip08=sum((br08s[:,:]-br08s[:,-1::-1])**2*wch,axis=1)
qup08=sum((br08s[:,:]+br08s[:,-1::-1])**2*wch,axis=1)
dip09=sum((br09s[:,:]-br09s[:,-1::-1])**2*wch,axis=1)
qup09=sum((br09s[:,:]+br09s[:,-1::-1])**2*wch,axis=1)

dip06=sum((br06[:,:]-br06[:,-1::-1])**2*wch,axis=1)
qup06=sum((br06[:,:]+br06[:,-1::-1])**2*wch,axis=1)
dip07=sum((br07[:,:]-br07[:,-1::-1])**2*wch,axis=1)
qup07=sum((br07[:,:]+br07[:,-1::-1])**2*wch,axis=1)

plt.figure()
plt.plot(timel,(qup06-dip06)/(dip06+qup06),'k',
         timel,(qup07-dip07)/(dip07+qup07),'r',)#
#         time2,(qup09-dip09)/(dip09+qup09),'b',) #,time,nax0[:,1],'b',time,nax0[:,2],'r',time,nax0[:,3],'y')
plt.show()

plt.figure()
plt.plot(time2,sum(nax08s[:,1:],axis=1)/sum(nax08s[:,:],axis=1),'k',
         time2,sum(nax09s[:,1:],axis=1)/sum(nax09s[:,:],axis=1),'r',
         time2,(qup08-dip08)/(dip08+qup08),'k--',
         time2,(qup09-dip09)/(dip09+qup09),'r--',)
#         time,sum(nax02[:,1:],axis=1)/sum(nax02[:,:],axis=1),'k',
#         time,sum(nax04[:,1:],axis=1)/sum(nax04[:,:],axis=1),'r',
#         time,(qup02-dip02)/(dip02+qup02),'k--',
#         time,(qup04-dip04)/(dip04+qup04),'r--',)
         #timel,sum(nax06[:,1:],axis=1)/sum(nax06[:,:],axis=1),'k',
         #timel,sum(nax07[:,1:],axis=1)/sum(nax07[:,:],axis=1),'r',
         #timel,(qup06-dip06)/(dip06+qup07),'k--',
         #timel,(qup07-dip07)/(dip07+qup07),'r--',)
#plt.legend(lin1, ['${\\left|\\bar{B}_{\\phi}\\right|}$',
#                 '${\\left|\\bar{B}_r\\right|}$',
#                 '${\\left|\\tilde{B} \\right|}$'], loc='auto', frameon=False)
#plt.xlabel('R$^2/\\eta_T$')
plt.xlim(16.,17.)
plt.ylabel('P$_X$, P$_E$')
#plt.yscale('symlog',linthreshy=2)
plt.show()



spm0=(mean(nax00,axis=0)[:Nm-2]) #m1a
spm2=(mean(nax02,axis=0)[:Nm-2])# #M2a
spm3=(mean(nax03,axis=0)[:Nm-2])# #m1b
spm4=(mean(nax04,axis=0)[:Nm-2])# #m2b
spm8=(mean(nax08,axis=0)[:Nm-2])# #m2c
spm9=(mean(nax09,axis=0)[:Nm-2])# #m2d

spmdi=loadtxt('brmdi')/max(loadtxt('brmdi')[2:])
savetxt('m1bspm',spm3)
savetxt('m1bspl',[mean(nl03+al03,axis=0),mean(al03,axis=0)])
#spm7=loadtxt('spm7')
#spm8=loadtxt('spm8')


xa,ya=meshgrid(phi*180/pi,90-arccos(xch)*180/pi)
#nnx1=sum(nax0[:,1:],axis=1)/sum(nax0,axis=1)
brf=zeros(shape(xa))
bff=zeros(shape(xa))
hk=zeros(shape(xa))

for i in range(Nph):
    brf[:,i]=brfield(a0b,sn0)[:,i]#-dot(df1r,a0)
    bff[:,i]=bfield(bt0[-1,:],tn,sn)[:,i]#+bt0[-1,:]
    hk[:,i]=afield(heln)[:,i]+hels#[-1,:]
#brf=brd(a0,sn)    

print, sum(sum(brf,axis=1)*wch)

levs=sort(append(-max(bff.flatten())*arange(1,10)/10.,max(bff.flatten())*arange(1,10)/10.))
levr=sort(append(-max(brf.flatten())*arange(1,10)/10.,max(brf.flatten())*arange(1,10)/10.))
leh=sort(append(-max(hk.flatten())*arange(1,10)/10.,max(hk.flatten())*arange(1,10)/10.))

plt.figure()
CS2=plt.pcolor(xa,ya,brf*100,cmap=cmap.bwr,vmin=-1,vmax=1)
#plt.contour(xa,ya,bff[:,:],levs,colors='k')
plt.contour(xa,ya,hk[:,:],leh,colors='k')
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "3%", pad="3%")
CB=plt.colorbar(CS2, cax=cax,ticks=[-1,0,1])
CB.set_label('100B$_{r}$',fontsize=30)
plt.tight_layout()
plt.show()
import matplotlib.colors as colors
from matplotlib.mlab import bivariate_normal
from cartopy.util import  add_cyclic_point
bffc = add_cyclic_point(bff)
brfc = add_cyclic_point(brf)
hkc = add_cyclic_point(hk)
phin=arange(Nph+1)*2*pi/Nph

xn,yn=meshgrid(phin*180/pi,90-arccos(xch)*180/pi)

plt.figure()
ax = plt.axes(projection=ccrs.Mollweide(central_longitude=0))
#ax = plt.axes(projection=ccrs.Orthographic(central_latitude=20,central_longitude=180.0-(phi[1]-phi[0])*180/pi/2.))
ax.contour(xn,yn, bffc[:,:],levs,colors='k',lw=2, transform=ccrs.PlateCarree())
#CS=ax.pcolor(xn,yn,hkc[:,:]*2e3,cmap=cmap.bwr,vmin=-1,vmax=1,transform=ccrs.PlateCarree())
#CS=ax.pcolor(xn,yn,brfc[:,:]*200,cmap=cmap.bwr,vmin=-.5,vmax=.5,transform=ccrs.PlateCarree())
CS=ax.pcolor(xn,yn,brfc[:,:], norm=colors.SymLogNorm(linthresh=.01, linscale=.01,
                     vmin=-2.,vmax=2.),cmap=cmap.seismic, transform=ccrs.PlateCarree())
CB=plt.colorbar(CS,orientation='horizontal',fraction=0.05, pad=0.02,ticks=[-.2,0,.2],format='%.1f')#, cax=cax)
CB.set_label('B$_r$',fontsize=24)
plt.tight_layout() 
plt.show()

plt.figure()
ax = plt.axes(projection=ccrs.Mollweide(central_longitude=0))
#ax = plt.axes(projection=ccrs.Orthographic(central_latitude=20,central_longitude=180.0-(phi[1]-phi[0])*180/pi/2.))
ax.contour(xn,yn, bffc[:,:],levr,colors='k',lw=2, transform=ccrs.PlateCarree())
#CS=ax.pcolor(xn,yn,hkc[:,:]*2e3,cmap=cmap.bwr,vmin=-1,vmax=1,transform=ccrs.PlateCarree())
#CS=ax.pcolor(xn,yn,brfc[:,:]*200,cmap=cmap.bwr,vmin=-.5,vmax=.5,transform=ccrs.PlateCarree())
CS=ax.pcolor(xn,yn,hkc[:,:]*2e3, norm=colors.SymLogNorm(linthresh=.1, linscale=.1,
             vmin=-1,vmax=1),cmap=cmap.seismic, transform=ccrs.PlateCarree())
CB=plt.colorbar(CS,orientation='horizontal',fraction=0.05, pad=0.02,ticks=[-1,0,1])#, cax=cax)
CB.set_label('100 B$_r$',fontsize=24)
plt.tight_layout() 
plt.show()


Cap=0.2*randn(2);
Cmu=1#+Cap[0]*(1.+sign(Cap[1])*xch*sqrt(3.*pi/4.*(1-xch**2)))*(1-xch**2)

Ca0=1+0.25*sqrt(Nph)*randn(Nph);
yal=rfft(Ca0)/Nph;yal[Nph//8:]=0;Ca0=irfft(yal)*Nph;
#Calpha=1. #mean((Ca0+Ca_0))/(2.)#+0.25*randn(1)#(Ca0[0]+Ca0[1])/(3.) #sqrt(2.)
alpf=Ca0#(Ca0+Ca_0)/(2.);

alf=tensordot(Cmu*alff,alpf,0)
lea=sort(append(-max(alf.flatten())*arange(1,10)/10.,max(alf.flatten())*arange(1,10)/10.))
alfc = add_cyclic_point(alf)

plt.figure()
ax = plt.axes(projection=ccrs.Mollweide(central_longitude=0))
ax.contour(xn,yn, alfc[:,:],lea,colors='k',lw=2, transform=ccrs.PlateCarree())
CS=ax.pcolor(xn,yn,alfc, norm=colors.SymLogNorm(linthresh=.1, linscale=.1,
             vmin=-2,vmax=2),cmap=cmap.seismic, transform=ccrs.PlateCarree())
CB=plt.colorbar(CS,orientation='horizontal',fraction=0.05, pad=0.02,ticks=[-1,0,1])#, cax=cax)
CB.set_label('$\\alpha$',fontsize=24)
plt.tight_layout() 
plt.show()

   
sp0,sp=spml(a0,sn)
    
plt.figure()
plt.semilogy(sp,'r',sp0[:],'b')
#plt.semilogy(spm(a0,sn),'k',bspm(b0,tn,sn),'k--')
plt.show()

plt.figure()
plt.semilogy((be[:]),'r', mean(abs(bt0),axis=1),'b',  amax(abs(bt0),axis=1),'b--' )
plt.show()

brmo=brmod(a0,sn)
#bq=sum((bt0[:,:]+bt0[:,-1::-1])**2*wch,axis=1)
#bd=sum((bt0[:,:]-bt0[:,-1::-1])**2*wch,axis=1)
#peb=(bq-bd)/(bq+bd)
#plt.figure()
#plt.plot(time,peb)
#plt.show()

#ssn1=bm #*exp(-(1./bm))
#ssnr1=(bmr+bma)#*exp(-(1./(bmr+bma)))
#brn1=br0
#btn1=bt0
#bmn1=bm;bma1=bma;
#bmr1=bmr;bmar1=brma;
#hll1=hl0
#savetxt('m11',asarray([ssn11,brn11,btn11,bmn11,hl11]))
tr=time2
xt=zeros((Nc//4,Nch),float);
yt=zeros((Nc//4,Nch),float);
for i in range(Nc//4):
    yt[i,:]=(pi/2.-arccos(xch[:]))*180./pi;
    xt[i,:]=tr[i];

indx=where(((xt[:,0] >= 16) & (xt[:,0] <= 17)))[0]
xto=xt[indx,:]
yto=yt[indx,:]
bct=bt08s[indx,:] 
brt=br08s[indx,:] 
#bct1=imag(bfo3[indx,:]) 
#brt1=real(bfo3[indx,:]) 

import matplotlib.cm as cmap
#levh=-max(abs(hct).flatten())/2.+max(abs(hct).flatten())*arange(12)/11.
levb=-max(abs(bct).flatten())/2.+max(abs(bct).flatten())*arange(12)/11.
leva=-max(abs(brt).flatten())/2.+max(abs(brt).flatten())*arange(12)/11.
 
plt.figure()
CS1=plt.contour(xto[::16,:],yto[::16,:],bct[::16,:], levb,colors='k')
CS2=plt.pcolor(xto[::16,:], yto[::16,:], brt[::16,:],cmap=cmap.bwr,
               norm=colors.SymLogNorm(linthresh=.1, linscale=0.001,
             vmin=-.02,vmax=0.02)) #cmap.Greys)
#plt.xlabel('R$^2/\\eta_T$')
plt.ylabel('LATITUDE')
plt.ylim( (-90, 90) )
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "3%", pad="3%")
CB=plt.colorbar(CS2, cax=cax,ticks=[-0.01,0,0.01],format='%.2f')
CB.set_label('B$_{r}$, [B$_{eq}$]',fontsize=30)
plt.show()


#mu0=arccos(xch)
#
#for k in range(Nph):
#    for i in range(Nch):
#        xs[i,k] = sin(mu0[i])*cos(phi[k]);
#        ys[i,k] = sin(mu0[i])*sin(phi[k]);
#        zs[i,k] = cos(mu0[i]);
#
#xs,ys,zs = broadcast_arrays( xs, ys, zs)
#
levs=-1+2*arange(11)/10.
ia=0
for i in range(0,Nc,8):
    if (timer[i]>=16) & (timer[i]<=16.5):
        bff=bfield(bt0[i//4],ti0[i//4,:],si0[i//4,:])
        brf=brfield(at0[i//4],si0[i//4,:]) #brd(at0[i],si0[i,:])+brq(at0[i],si0[i,:])  #
        brfc = add_cyclic_point(brf)
        bffc = add_cyclic_point(bff)
        tim=timer[i]
        plt.figure()
        ax = plt.axes(projection=ccrs.Mollweide(central_longitude=0))
        ax.contour(xn,yn, bffc[:,:],levs,colors='k',lw=2, transform=ccrs.PlateCarree())
        CS=ax.pcolor(xn,yn,brfc[:,:], norm=colors.SymLogNorm(linthresh=.01, linscale=.01,
                     vmin=-2.,vmax=2.),cmap=cmap.seismic, transform=ccrs.PlateCarree())
        CB=plt.colorbar(CS,orientation='horizontal',fraction=0.05, pad=0.02,ticks=[-1.,0,1.],format='%.1f')#, cax=cax)
        CB.set_label('B$_r$',fontsize=24)
        plt.title(str('%.2f' %tim))
        plt.tight_layout() 
        plt.savefig(fia+'s'+str(ia)+'.png',format='png',dpi=72);
        plt.close('all')
        print(ia)
        ia=ia+1

wsot=loadtxt('wsomt')
wsof=loadtxt('wsomf')
wsobf=loadtxt('wsombf')
wsosp=mean(wsof[:,:],axis=0)[:10]
wsosb=mean(wsobf[:,:],axis=0)[1:10]

wsosp=wsosp/max(wsosp)
nxsp2=mean(naxr2[:,:],axis=0)[:15]
#nxsp2=nxsp2/max(nxsp2)
#nxspa=mean(naxr[:,:],axis=0)[:15]
#nxspa=nxspa/max(nxspa)
nxsp3=mean(naxr3[:,:],axis=0)[:15]
#nxsp3=nxsp3/max(nxsp3)
nxsp=mean(naxr[:,:],axis=0)[:15]
nxsp=nxsp/max(nxsp)

plt.figure()
plt.semilogy(nxsp,'r',nxsp2,'b',nxsp3,'k')
plt.show()

plt.figure()
plt.semilogy(mean(wsof[400:,:],axis=0)[:10],'k')
plt.show()

plt.figure()
#plt.semilogy(mean(naxf,axis=0)[:14],'r',mean(naxf9,axis=0)[:14],'b')
plt.semilogy(mean(wsof,axis=0)[:10],'k')#,mean(naxr9,axis=0)[:14],'b')
plt.semilogy(mean(wsobf,axis=0)[1:10],'r')#,mean(naxr9,axis=0)[:14],'b')
#plt.semilogy(mean(naxr10,axis=0)[:14],'y')#,mean(naxr0,axis=0)[:14],'g')
#plt.plot(time0,naxf0[:,0],'k',time0,sum(naxf0[:,:],axis=1),'k--')
plt.show()

plt.figure()
lin0=plt.semilogy(time, sum(abs(bt0)*wch,axis=1)/2,'b',
                 time, sum(abs(br0)*wch,axis=1)/2,'r',
                 time, sum(abs(bnx)*wch,axis=1)/2,'k')
#lin1=plt.semilogy(time, sum(abs(bt01)*wch,axis=1)/2,'b--',
#                 time, sum(abs(br01)*wch,axis=1)/2,'r--',
#                 time, sum(abs(bx01)*wch,axis=1)/2,'k--')
#plt.legend(lin1, ['${\\left|\\bar{B}_{\\phi}\\right|}$',
#                 '${\\left|\\bar{B}_r\\right|}$',
#                 '${\\left|\\tilde{B} \\right|}$'], loc='auto', frameon=False)
plt.ylim(3.e-3,1.)
plt.show()
