from numpy.linalg import *
from func_dif import *
from scipy.special import *

set_printoptions(precision=16)
Nch=64

Nn=36
Nl=64
Nm=18

k=0
for i in range(Nn):
    for j in range(1,i+2): #range(-i-1,i+2): # range(1): #range(-1,2):
        if abs(j)<=Nm:
            if j!=0:        
                k=k+1


#NN=int((3+(2*Nn+1))*Nn/2)
NN0=int((Nn+3)*Nn/2)
NN=2*k #int((Nn+1m)*Nn/2)

#NN=Nn*(Nm-1)
#NN=Nn*3
#NN=Nn*2
fmt=str(NN)+'s'
Nph=48
#NN=86
#wch=real(orthogonal.legendre_roots(Nch,1/2)[1])
wch=real(orthogonal.p_roots(Nch)[1])
xch=real(orthogonal.p_roots(Nch)[0])
xch=(xch[:]-xch[-1::-1])/2.
wch=(wch[:]+wch[-1::-1])/2.

phi=arange(Nph)*2*pi/Nph
#frk=fftfreq(Nph,phi[1]-phi[0])
kfr=range(-Nn,Nn+1)
kfri=arange(-Nn,Nn+1)
ck=ones(Nph)
ck[0]=2;ck[-1]=2
theta=arccos(xch)


mk=zeros(NN,dtype=int)
mkk=zeros(NN,dtype=int)
nk=zeros(NN,dtype=int)
nk0=zeros(NN0,dtype=int)
mk0=zeros(NN0,dtype=int)
kl=zeros((NN,NN),dtype=int)
ic=range(Nch)
k=0
k0=0
for i in range(Nn):
    for j in range(1,i+2): 
        if abs(j)<=Nm:
            nk[k]=i+1   
            mk[k]=j
            mkk[k]=abs(j)
            k=k+1
nk[NN//2:]=nk[:NN//2]
mk[NN//2:]=-mk[:NN//2]
mkk[NN//2:]=mkk[:NN//2]
mka=where(mk!=0)[0]
mks=where(mk==0)[0]
mkm=where(mk<0)[0]
mkp=where(mk>0)[0]

m0=zeros((Nch,NN),dtype=float64)
m_0=zeros((Nch,NN),dtype=float64)
m1=zeros((Nch,NN),dtype=float64)
s0=zeros((Nch,NN),dtype=float64)
st1=zeros((Nch,NN),dtype=float64)
mfs=zeros((Nch,NN),dtype=float64)
mft=zeros((Nch,NN),dtype=float64)

for i in range(NN):
    m0[ic,i]=vLpnm(nk[i], mkk[i], xch[ic])
    mfs[ic,i]=mk[i]*vLpnm(nk[i], mkk[i], xch[ic])/sqrt(1-xch[ic]**2)
    m1[ic,i]=mk[i]*vLpnm(nk[i], mkk[i], xch[ic])
    s0[ic,i]=-(nk[i]+1)*nk[i]*vLpnm(nk[i], mkk[i], xch[ic])
    st1[ic,i]=(-nk[i]*vLpnm(nk[i], mkk[i], xch[ic])*xch[ic]
        +(mkk[i]+nk[i])*vLpnm(nk[i]-1, mkk[i], xch[ic])
        *cn(nk[i],mkk[i])/cn(nk[i]-1,mkk[i]))

for i in range(NN):
    mft[ic,i]=st1[ic,i]/sqrt(1-xch[ic]**2)


m00=zeros((NN,NN),dtype=float64)
ms0=zeros((NN,NN),dtype=float64)
SS0=s0
MFT=mft
MFS=mfs

for i in range(NN):
    ji=where(mk[i] == mk[range(NN)])[0]
    for j in range(NN): # ji:
        if mk[i] == mk[j]:
            ms0[i,j]=sum(m0[:,i]*s0[:,j]*wch[:])
ms_0=inv(ms0)

    
    
MRT=zeros((NN,NN),dtype=float64)
mss=zeros((NN,NN),dtype=float64)
ms1=zeros((NN,NN),dtype=float64)
m0m=zeros((NN,NN),dtype=float64)
mpp=zeros((NN,NN),dtype=float64)
m1ss=zeros((NN,NN),dtype=float64)


for i in range(NN):
    ji=where(mk[i] == mk[range(NN)])[0]
    for j in range(NN): # ji:
        if mk[i] == mk[j]:
            m00[i,j]=sum(m0[:,i]*m0[:,j]*wch[:])
            MRT[i,j]=sum(m0[:,i]*mk[j]*m0[:,j]*(0.-xch[:]**2/4.)*wch[:])
            mss[i,j]=sum(m0[:,i]*mk[j]*s0[:,j]*(0.-xch[:]**2/4.)*wch[:])
            ms1[i,j]=sum(m0[:,i]*st1[:,j]*wch[:])
            m0m[i,j]=sum(m0[:,i]*mk[j]**2*m0[:,j]*wch[:])#/(1-xch**2))
            mpp[i,j]=sum(m0[:,i]*(s0[:,j]*xch[:]**2)*wch[:])
            m1ss[i,j]=sum(m0[:,i]*mk[j]**2*s0[:,j]*wch[:])

MB1=ms0 #+4*msm #dot(ms_0,ms1)
MRS=mss
MSA=dot(ms_0,m1ss)
MM0=m0
M0M=m0m
M0c=(zeros((Nch,Nl),dtype=float64))
M0s1=(zeros((Nch,Nl),dtype=float64))
M0C=(zeros((Nch,Nl),dtype=float64))

MB_1=ms_0
modec=arange(Nl)+1
dfk2c=zeros((Nl,Nl))
dfk3c=zeros((Nl,Nl))

dfk2c[0,1]=9./5.*c(2)/c(1)
dfk2c[Nl-1,Nl-2]=-(Nl-1.)**2/(2.*Nl-1.)*c(Nl-1)/c(Nl)
dfk3c[0,1]=12./5.*c(2)/c(1)
dfk3c[Nch-1,Nch-2]=-(Nch-1.)*(Nch-2.)/(2.*Nch-1.)*c(Nch-1)/c(Nch)
for i in range(1,Nl-1):
        dfk2c[i,i+1]=(i+3.)**2/(2.*i+5)*c(i+2)/c(i+1)
        dfk2c[i,i-1]=-(i*1.)**2/(2.*i+1)*c(i)/c(i+1)
        dfk3c[i,i+1]=(i+3.)*(i+4.)/(2.*i+5)*c(i+2)/c(i+1)
        dfk3c[i,i-1]=-(i-1)*i/(2.*i+1)*c(i)/c(i+1)



for i in range(Nl):
    M0c[:,i]=vlpn(modec[i],xch[:])
    M0s1[:,i]=vlpn(modec[i],xch[:])/sqrt(1-xch[:]**2)
    M0C[:,i]=vlp(modec[i],xch[:])


M_0c=inv(M0c)

DD1=dot(M0s1,dot(dfk2c,M_0c))
DD3=dot(M0s1,dot(dfk3c,M_0c))
nfl=Nl//3
filt=append(ones(Nl-nfl),zeros(nfl))

FMc=dot(M0c,dot(diag(filt),M_0c))

