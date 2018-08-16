from scipy.linalg import *
from numpy.random import *
from func_dif import *
from scipy.special import *
#from plac import *
import matplotlib.pyplot as plt
params = {'axes.labelsize': 28,
          'font.size': 28,
          'legend.fontsize': 28,
          'xtick.labelsize': 28,
          'ytick.labelsize': 28,
          'text.usetex': True}
plt.rcParams.update(params)
import matplotlib.cm as cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
fia='/home/user/work/pap/stars/sth/ani/'

set_printoptions(precision=16)
from axmat import *
from init00ml import *



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


def spb(brr0,brr1):
    brfc=zeros(NN)+1j*zeros(NN)
    Nmax=Nn
    sp0=zeros(Nmax); sp=zeros(Nmax)
    brm0=dot(inv(M0C),brr0)
    for i in range(NN//2):
        if(mk[i]==1):
            brfc[i]=sum(MM0[:,i]*brr1[:]*wch[:])
    
    brfc[NN//2:]=conjugate(brfc[:NN//2])
    #brm1=dot(inv(M0C),brr1)
    for j in range(Nmax):
        sp0[j]=sum((abs(brfc[(nk==j)]*conj(brfc[(nk==j)]))))/((2.*j+1))
        sp[j]=(brm0[j]**2/(2.*j+1))#+sum(sqrt(abs(brfc[(nk==j)]*conj(brfc[(nk==j)]))))/sqrt((2.*j+1))#/sqrt(len(brfc[(nk==j)]))
    return sp,sp0,brm0[0],brfc[0]   

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
        but1[:,j]=cbu*vbu(ba)*exp(-100*sin((phi[j]-phi0)/2)**2)*(aa+bth[:,j])/sqrt(1-xch**2)
        but2[:,j]=dot(DD1,cbu*vbu(ba)*(ba+bf[:,j])*sqrt(1-xch**2))*exp(-100*sin((phi[j]-phi0)/2)**2)
        bus1[:,j]=cbu*vbu(ba)*exp(-100*sin((phi[j]-phi0)/2)**2)*(bf[:,j]+ba)/sqrt(1-xch**2)
        bus2[:,j]=dot(DD1,cbu*vbu(ba)*(aa+bth[:,j])*sqrt(1-xch**2))*exp(-100*sin((phi[j]-phi0)/2)**2)
        bus3[:,j]=vbu(ba)*(1.+cbu*exp(-100*sin((phi[j]-phi0)/2)**2))#*(0*bf[:,j]+ba)
        
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



Nc=70000
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
#savetxt('mfb',asarray([a0b,b0b,hel0]))
#savetxt('nxRb',real(asarray([tn0,sn0,hn0])))
#savetxt('nxIb',imag(asarray([tn0,sn0,hn0])))
tn0,sn0,hn0=loadtxt('nxRb')+1j*loadtxt('nxIb')
a0b,b0b,hel0=loadtxt('mfb')
#
a0=a0b;b0=b0b;hel=hel0;tn=tn0;sn=sn0;hn=hn0;t_n=tn0;s_n=sn0;h_n=hn0;
salf,talf,bet,beta,AB,abm,bmax,brmax,sour0,sour1,buos,buot,bnaxi,soubu=bfah(a0,b0,hels,sn,tn,heln,phi0)
si0=zeros((Nc,NN))+1j*zeros((Nc,NN))
ti0=zeros((Nc,NN))+1j*zeros((Nc,NN))
htn=zeros((Nc,NN))+1j*zeros((Nc,NN))
abn=zeros((Nc,NN))+1j*zeros((Nc,NN))
at0=zeros((Nc,Nch))
bt0=zeros((Nc,Nch))
al0=zeros((Nc,Nch))

yal=zeros(Nph)+1j*zeros(Nph)


be=zeros((Nc))
bm=zeros((Nc));bmr=zeros((Nc));
bma=zeros((Nc));
brma=zeros((Nc));
phasa=zeros((Nc));
phasa2=zeros((Nc));

br0=zeros((Nc,Nch))
hl0=zeros((Nc,Nch))
bnx=zeros((Nc,Nch))
jran=cumsum((16*(1+abs(randn(Nc))))).astype(int); 
jr=cumsum(8*ones(Nc)).astype(int) #(3*(1+abs(randn(Nc)))); 
alpha=zeros((Nc))
Cmu=ones(Nch)
Calpha=1.;alpf=Calpha*zeros(Nph)
Ca0=1+sqrt(Nph)*randn(Nph);
yal=rfft(Ca0)/Nph;yal[Nph//8:]=0;Ca0=irfft(yal)*Nph;
alpf=0*Ca0
Calpha=1#mean(Ca0)
floss=zeros(Nch)
cbu=zeros(Nch)
cbun=zeros((Nc,Nch))
phi00=zeros(Nc)
caf=zeros((Nc,Nph))
##
naxr=zeros((Nc,Nph//2+1))
naxf=zeros((Nc,Nph//2+1))
naxl=zeros((Nc,Nn))
axl=zeros((Nc,Nn))

br1=zeros((Nc,Nch))+1j*zeros((Nc,Nch))
br2=zeros((Nc,Nch))+1j*zeros((Nc,Nch))

bf1=zeros((Nc,Nch))+1j*zeros((Nc,Nch))
bf2=zeros((Nc,Nch))+1j*zeros((Nc,Nch))

#cafn=caf
#phin=phi00#zeros(Nc)
#cbunn=cbun
#if i/jran-fix(i/jran) == 0.0:
ij=0
ij1=0
xi_h=0.;tfluc=0
for i in range(Nc):
    #Calpha=1
    #alpf=caf[i,:]#Ca0#+Ca_0)/(2.);
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
            cbu[:Nch//2]=55.
        else: 
            cbu[Nch//2::]=55.
        ij1=ij1+1
    else:
        cbu=zeros(Nch)#cbu*exp(-16*tfluc)#zeros(Nch)

    caf[i,:]=Ca0
    phi00[i]=phi0 
    cbun[i,:]=cbu
#

    D1b=tau*diag(Cmu*Calpha*alff*bet+Ch*hels);
    #D1bu=Rbu*tau*diag(b0**2/(1+2*abs(b0)**3))
    #D1ba=tau*diag(Calpha*(alff*bet+Ch*hels)) #*(1-xch**2));
    Da2=dot(DD1,dot(D1b,df1r))+D1b#+dot(diag(sqrt(1-xch**2)*xch),dot(D1b,df1r))
    M_1=inv(bmat([[DDa,-Ra*D1b*si_g],[-D1a*si_g-si_g*Ra2*Da2,DD]])) #+si_g*D1bu
    M1=bmat([[D_Da,Ra*D1b*sig],[D1a*sig+sig*Ra2*Da2,D_D]]) #-sig*D1bu
    MX=dot(M_1,M1)
    solut=asarray(dot(MX,append(a0,b0))).flatten()
    a0=solut[:Nch]+tau*Ra2*dot(DD_a,sour0)-Rbu*tau*dot(DD_a,soubu*a0)
    b0=solut[Nch:]+tau*Ra2*dot(DD_a,sour1)-Rbu*tau*dot(DD_a,soubu*b0) #floss*b0+
    br0[i,:]=-dot(df1r,a0)
    at0[i,:]=a0
    bt0[i,:]=b0
    bm[i]=max(abs(b0));bmr[i]=max(abs(br0[i,:]));
    #g10[i]=-cnm(1)*sum(ma_0[0,:]*a0);g20[i]=-cnm(2)*sum(ma_0[1,:]*a0);

    ab=dot(df1r,a0)*dot(mabp,b0)+a0*b0
    he_l=dot(M_H,hel)+tau*dot(MH,ab+abm)/Rm  
    hels=he_l-ab-abm
    #hl0[i,:]=hels
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
    
    ti0[i,:]=tn
    si0[i,:]=sn
    #    g11[i]=cmm(1,1)*real(sn[((mk==1) & (nk ==1))][0])
    #    h11[i]=cmm(1,1)*imag(sn[((mk==1) & (nk ==1))][0])
    #    g21[i]=cmm(2,1)*real(sn[((mk==1) & (nk ==2))][0])
    #    h21[i]=cmm(2,1)*imag(sn[((mk==1) & (nk ==2))][0])
    #    g22[i]=cmm(2,2)*real(sn[((mk==2) & (nk ==2))][0])
    #    h22[i]=cmm(2,2)*imag(sn[((mk==2) & (nk ==2))][0])
    be[i]=mean(beta)-mean(abs(b0))
    #abn[i,:]=AB
    #htn[i,:]=heln
    bnx[i,:]=bnaxi
    bma[i]=bmax;
    brma[i]=brmax;
    #bmo=brmod(a0,sn);
    #br1[i,:]=bmo[:,1];br2[i,:]=bmo[:,2];
    naxr[i,:]=dot(wch,abs(bmo[:,:])**2)/2.
    #bmo=bfmod(b0,sn,tn);
    #bf1[i,:]=bmo[:,1];bf2[i,:]=bmo[:,2];
    naxf[i,:]=dot(wch,abs(bmo[:,:])**2)/2.
    axl[i,:],naxl[i,:]=spml(a0,sn);
    if remainder(i,1000)==0:
        print(i,mean(Ca0))
    
#time=tau*arange(20000)
time=tau*arange(Nc)
#lt
bt07=bt0;br07=br0;bx07=bnx;br107=br1; bf107=bf1; br207=br2;bf207=bf2; nax07=naxr;nxf07=naxf;nl07=naxl;al07=axl;c07=alpha;# cbu=50,caf,0. m1a
bt08=bt0;br08=br0;bx08=bnx;br108=br1; bf108=bf1; br208=br2;bf208=bf2; nax08=naxr;nxf08=naxf;nl08=naxl;al08=axl;c08=alpha;# cbu=50,caf,0.25 m2a
#bt09=bt0;br09=br0;bx09=bnx;br109=br1; bf109=bf1; br209=br2;bf209=bf2; nax09=naxr;nxf09=naxf;nl09=naxl;al09=axl;c09=alpha;# cbu=50,caf,.5 m2b
bt11=bt0;br11=br0;bx11=bnx;br111=br1; bf111=bf1; br211=br2;bf211=bf2; nax11=naxr;nxf11=naxf;nl11=naxl;al11=axl;c11=alpha;# cbu=50,caf,1.,m2c

bt12=bt0;br12=br0;bx12=bnx;br112=br1; bf112=bf1; br212=br2;bf212=bf2; nax12=naxr;nxf12=naxf;nl12=naxl;al12=axl;c12=alpha;# cbu=0,ca,1.caf=0,m2c

plt.figure()
lin0=plt.semilogy(time, sum(abs(bt07)*wch,axis=1)/2,'b',
                 time, sum(abs(br07)*wch,axis=1)/2,'r',
                 time, sum(abs(bx07)*wch,axis=1)/2,'k',
                 time, sum(abs(bx05)*wch,axis=1)/2,'g')
lin1=plt.semilogy(time, sum(abs(bt06)*wch,axis=1)/2,'b--',
                 time, sum(abs(br06)*wch,axis=1)/2,'r--',
                 time, sum(abs(bx06)*wch,axis=1)/2,'k--')
plt.legend(lin0, ['${\\langle {B}_{\\phi}\\rangle}$',
                 '${\\langle {B}_r\\rangle}$',
                 '${\\left|\\tilde{B} \\right|}$',
                 '${\\left|\\tilde{B} \\right|}$'], loc='auto', frameon=False)
plt.ylim(1.e-3,1.)
plt.show()


plt.figure()
lin0=plt.plot(time,sqrt(sum(nxf12[:,:],axis=1)),'k',
         time3,sqrt(sum(nxf11[:,:],axis=1)),'r')
         #time,sqrt(sum(nxf02[:,:],axis=1)),'k',
         #time,sqrt(sum(nxf04[:,:],axis=1)),'r')
         #timel,sqrt(sum(nxf06[:,:],axis=1)),'k',
         #timel,sqrt(sum(nxf07[:,:],axis=1)),'r')#,time,sum(nxf05[:,:],axis=1)/2.,'b')
lin1=plt.plot(time,sqrt(nxf12[:,0]),'k--',time3,sqrt(nxf11[:,0]),'r--')#,time,nxf05[:,0]/2.,'b--')
#plt.legend(lin0, ['M3a','M3b'], loc='auto', frameon=False)
plt.ylabel('${\\langle {B}_{\\phi}\\rangle}$, '+
          '${\\left|\\bar{B}_{\\phi}\\right|}$')
#plt.yscale('log')
plt.show()

dip12=sum((br12[:,:]-br12[:,-1::-1])**2*wch,axis=1)
qup12=sum((br12[:,:]+br12[:,-1::-1])**2*wch,axis=1)

plt.figure()
plt.plot(time,(qup12-dip12)/(dip12+qup12),'k')
        # timel,(qup07-dip07)/(dip07+qup07),'r',)#
#         time2,(qup09-dip09)/(dip09+qup09),'b',) #,time,nax0[:,1],'b',time,nax0[:,2],'r',time,nax0[:,3],'y')
plt.show()



plt.figure()
plt.plot(time,dot(abs(bt0),wch),'k',time, sum(naxf,axis=1),'k--')
plt.show()

plt.figure()
plt.plot(time,dot(abs(br0),wch),'k',time,dot(abs(br1),wch),'k--')
plt.show()

plt.figure()
#plt.semilogy(mean(naxf,axis=0)[:14],'r',mean(naxf9,axis=0)[:14],'b')
plt.loglog(mean(nax07,axis=0)[:Nm-2],'k')#,mean(naxr9,axis=0)[:14],'b')
#plt.loglog(mean(nax08,axis=0)[:Nm-2],'k')#,mean(naxr9,axis=0)[:14],'b')
plt.loglog(mean(nax09,axis=0)[:Nm-2],'k--')#,mean(naxr9,axis=0)[:14],'b')
plt.loglog(mean(nl07+al07,axis=0),'b')#,mean(naxr9,axis=0)[:14],'b')
plt.loglog(mean(nl09+al09,axis=0),'b--')#,mean(naxr9,axis=0)[:14],'b')
plt.loglog(mean(0*nl07+al07,axis=0),'r')#,mean(naxr9,axis=0)[:14],'b')
plt.loglog(mean(0*nl09+al09,axis=0),'r--')#,mean(naxr9,axis=0)[:14],'b')
#plt.semilogy(mean(naxr10,axis=0)[:14],'y')#,mean(naxr0,axis=0)[:14],'g')
#plt.plot(time0,naxf0[:,0],'k',time0,sum(naxf0[:,:],axis=1),'k--')
plt.show()

spm0=mean(nax07,axis=0)[:Nm-2];savetxt('m1aspm',spm0);#/max(mean(nax00,axis=0)[2:Nm-2]) #m1a
spm2=mean(nax08,axis=0)[:Nm-2];savetxt('m2aspm',spm2);#/max(mean(nax02,axis=0)[2:Nm-2]) #M2a
#spm3=mean(nax09,axis=0)[:Nm-2];savetxt('m2bspm',spm3);#/max(mean(nax03,axis=0)[2:Nm-2]) #m2b
spm4=mean(nax11,axis=0)[:Nm-2];savetxt('m2cspm',spm4);#/max(mean(nax04,axis=0)[2:Nm-2]) #m2c

#savetxt('m1aspl',[mean(nl07+al07,axis=0),mean(al07,axis=0)])
#savetxt('m2aspl',[mean(nl08+al08,axis=0),mean(al08,axis=0)])
##savetxt('m2bspl',[mean(nl09+al09,axis=0),mean(al09,axis=0)])
#savetxt('m2cspl',[mean(nl11+al11,axis=0),mean(al11,axis=0)])

m1aspm=loadtxt('m1aspm')
m1bspm=loadtxt('m1bspm')
m2aspm=loadtxt('m2aspm')
#m2bspm=loadtxt('m2bspm')
m2cspm=loadtxt('m2cspm')
m1aspl1,m1aspl2=loadtxt('m1aspl')
m1bspl1,m1bspl2=loadtxt('m1bspl')
m2aspl1,m2aspl2=loadtxt('m2aspl')
#m2bspl1,m2bspl2=loadtxt('m2bspl')
m2cspl1,m2cspl2=loadtxt('m2cspl')


m1=(arange(len(m1bspm)))
m2=(arange(len(m1aspm)))
l1=(arange(len(m1bspl1))+1)
l2=(arange(len(m1aspl1))+1)

plt.figure()
plt.plot(m2,sqrt(m1aspm),'k--', m1,sqrt(m1bspm),'b', m2, sqrt(m2aspm),'k',
         m2,sqrt(m2cspm),'r')
plt.legend(['M1a','M1b','M2a','M2c'], loc='auto', frameon=False)
plt.ylabel('[B$_{eq}$]') 
plt.xlabel('$m$') 
plt.xscale('log')
plt.yscale('log')
#plt.loglog(spm7,'b',spm8,'b--')
#plt.loglog(spmdi,'r')
plt.show()

plt.figure()
plt.plot(l2,sqrt(m1aspl1),'k--', l1, sqrt(m1bspl1),'b',
         l2,sqrt(m2aspl1),'k',
         l2,sqrt(m2cspl1),'r')
#plt.legend(['M1a','M1b','M2a','M2b','M2c'], loc='auto', frameon=False)
#plt.ylabel('$\\overline{\\langle {B}_r\\rangle}$') 
plt.xlabel('$\\ell$') 
plt.xscale('log')
plt.yscale('log')
#plt.loglog(spm7,'b',spm8,'b--')
#plt.loglog(spmdi,'r')
plt.show()

fmdi='/home/va/work/MDI/'
fkp='/home/va/work/kpvt/'

lmdi,splmdi=loadtxt(fmdi+'splmdi')
mmdi,spmmdi=loadtxt(fmdi+'spmmdi')
lkp,splkp=loadtxt(fkp+'splkp')
mkp,spmkp=loadtxt(fkp+'spmkp')



plt.figure()
plt.plot(mmdi,spmmdi,'k',m2,sqrt(m1aspm)*2.e2,'k--', m2,sqrt(m2cspm)*2.e2,'r',
         mkp,spmkp,'b')
plt.legend(['MDI','M1a','M2c','KP'], loc='auto', frameon=False)
plt.ylabel('[G]') 
plt.xlabel('$m$') 
plt.xscale('log')
plt.yscale('log')
#plt.loglog(spm7,'b',spm8,'b--')
#plt.loglog(spmdi,'r')
plt.show()

plt.figure()
plt.plot(lmdi,splmdi,'k',l2,sqrt(m1aspl1)*2.e2,'k--',l2,sqrt(m2cspl1)*2.e2,'r')
#plt.legend(['M1a','M1b','M2a','M2b','M2c'], loc='auto', frameon=False)
#plt.ylabel('$\\overline{\\langle {B}_r\\rangle}$') 
plt.xlabel('$\\ell$') 
plt.xscale('log')
plt.yscale('log')
#plt.loglog(spm7,'b',spm8,'b--')
#plt.loglog(spmdi,'r')
plt.show()

#
#plt.figure()
#plt.plot(timel,dot(abs(bt0),wch),'k',timel,dot(abs(bf1),wch),'k--')
#plt.show()
#
#
##
xa,ya=meshgrid(phi*180/pi,90-arccos(xch)*180/pi)
#nnx1=sum(nax0[:,1:],axis=1)/sum(nax0,axis=1)
brf=zeros(shape(xa))
bff=zeros(shape(xa))
hk=zeros(shape(xa))

for i in range(Nph):
    brf[:,i]=brfield(a0,sn)[:,i]#-dot(df1r,a0)
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

ia=0
for i in range(0,Nc,16):
    if (time3[i]>=4.6) & (time3[i]<=5.3):
        bff=bfield(bt0[i],ti0[i,:],si0[i,:])
        brf=brfield(at0[i],si0[i,:]) #brd(at0[i],si0[i,:])+brq(at0[i],si0[i,:])  #
        brfc = add_cyclic_point(brf)
        bffc = add_cyclic_point(bff)
        tim=time3[i]
        plt.figure()
        ax = plt.axes(projection=ccrs.Mollweide(central_longitude=0))
        #ax.contour(xn,yn, bffc[:,:],2*levs,colors='k',lw=2, transform=ccrs.PlateCarree())
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



sp00=zeros((Nc,Nn))
sp10=zeros((Nc,Nn))
brm00=zeros(Nc)
brm10=zeros(Nc)+1j*zeros(Nc)

for i in range(Nc):
    sp00[i,:],sp10[i,:],brm00[i],brm10[i]=spb(br11[i,:],br111[i,:])
    if remainder(i,500)==0:
        print(i)

plt.figure()
plt.semilogx(log10(mean(nl07+al07,axis=0)),'k')#,mean(naxr9,axis=0)[:14],'b')
plt.semilogx(log10(mean(al07,axis=0)),'b')#,mean(naxr9,axis=0)[:14],'b')
plt.semilogx(log10(mean(sp107,axis=0)),'r')
plt.show()


plt.figure()
plt.semilogx(log10(mean(nl09+al09,axis=0)),'k')#,mean(naxr9,axis=0)[:14],'b')
plt.semilogx(log10(mean(al09,axis=0)),'b')#,mean(naxr9,axis=0)[:14],'b')
plt.semilogx(log10(mean(sp109,axis=0)),'r')
plt.show()

plt.figure()
plt.semilogx(log10(mean(nl10+al10,axis=0)),'k')#,mean(naxr9,axis=0)[:14],'b')
plt.semilogx(log10(mean(al10,axis=0)),'b')#,mean(naxr9,axis=0)[:14],'b')
plt.semilogx(log10(mean(sp110,axis=0)),'r')
plt.show()


plt.figure()
plt.plot(time,brm07,'k',time,brm010,'k--')
plt.plot(time,real(brm107),'r',time,real(brm110),'b')
#plt.plot(time,imag(brm10))
plt.show()

sp07=sp00;sp107=sp10;brm07=brm00;brm107=brm10;
sp08=sp00;sp108=sp10;brm08=brm00;brm108=brm10;
sp09=sp00;sp109=sp10;brm09=brm00;brm109=brm10;
sp10=sp00;sp110=sp10;brm010=brm00;brm110=brm10;
sp11=sp00;sp111=sp10;brm011=brm00;brm111=brm10;


plt.figure()
plt.semilogx(log10(mean(sp107,axis=0)),'k')#,mean(naxr9,axis=0)[:14],'b')
plt.semilogx(log10(mean(sp110,axis=0)),'r')
plt.semilogx(log10(mean(sp111,axis=0)),'b')

plt.show()

tr=time
xt=zeros((Nc,Nch),float);
yt=zeros((Nc,Nch),float);
for i in range(Nc):
    yt[i,:]=(pi/2.-arccos(xch[:]))*180./pi;
    xt[i,:]=tr[i];

indx=where(xt[:,0] >= 0)[0]
xto=xt[indx,:]
yto=yt[indx,:]
bct=bt07[indx,:] 
brt=br07[indx,:] 
#bct1=imag(bfo3[indx,:]) 
#brt1=real(bfo3[indx,:]) 

import matplotlib.cm as cmap
#levh=-max(abs(hct).flatten())/2.+max(abs(hct).flatten())*arange(12)/11.
levb=-max(abs(bct).flatten())/2.+max(abs(bct).flatten())*arange(12)/11.
leva=-max(abs(brt).flatten())/2.+max(abs(brt).flatten())*arange(12)/11.
 
plt.figure()
CS1=plt.contour(xto[::20,:],yto[::20,:],brt[::20,:], leva,colors='k')
CS2=plt.pcolor(xto[::20,:], yto[::20,:], bct[::20,:],cmap=cmap.bwr,vmin=-.2,vmax=.2) #cmap.Greys)
plt.xlabel('R$^2/\\eta_T$')
plt.ylabel('LATITUDE')
plt.ylim( (-90, 90) )
#plt.xlim(1,5)#(time[indx[0]], time[indx[-1]])
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "3%", pad="3%")
CB=plt.colorbar(CS2, cax=cax,ticks=[-0.1,0,0.1])
CB.set_label('B$_{\\phi}$, [B$_{eq}$]',fontsize=30)
plt.show()


savez('mfe', at0,bt0,hl0)
npzfile=load('mfe.npz')
at0=npzfile['arr_0']
bt0=npzfile['arr_0']
hl0=npzfile['arr_0']
savez('mfal', alpha)
savez('nxeR', real(si0),real(ti0),real(htn))
savez('nxeI', imag(si0),imag(ti0),imag(htn))
si0,ti0,htn=load('nxeR.npz')+1j*load('nxeI.npz')


savez('M2c',bt11=bt0,br11=br0,bx11=bnx,br111=br1,bf111=bf1, br211=br2,bf211=bf2, nax11=naxr,nxf11=naxf,nl11=naxl,al11=axl,c11=alpha)# cbu=50,caf,0. m1a
npzfile=load('M2c.npz')

plt.figure()
lin0=plt.plot(time,sqrt(sum(nxf12[:,:],axis=1)),'k')
         #time,sqrt(sum(nxf04[:,:],axis=1)),'r')
         #timel,sqrt(sum(nxf06[:,:],axis=1)),'k',
         #timel,sqrt(sum(nxf07[:,:],axis=1)),'r')#,time,sum(nxf05[:,:],axis=1)/2.,'b')
lin1=plt.plot(time,sqrt(nxf12[:,0]),'b')
#plt.legend(lin0, ['M3a','M3b'], loc='auto', frameon=False)
plt.ylabel('${\\langle {B}_{\\phi}\\rangle}$, '+
          '${\\left|\\bar{B}_{\\phi}\\right|}$')
#plt.yscale('log')
plt.show()

npzfile=load('M2c.npz')
c11=npzfile['c11']
