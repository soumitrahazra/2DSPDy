#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from func_dif import *
from scipy.linalg import *
Nch=72
etas=0.01
wch=real(orthogonal.p_roots(Nch)[1])
xch=real(orthogonal.p_roots(Nch)[0])
xch=(xch[:]-xch[-1::-1])/2.
wch=(wch[:]+wch[-1::-1])/2.
mut=linspace(-90,90,180)*pi/180.

Nch1=144
mu=linspace(-1,1,Nch1)

ma0=(zeros((Nch,Nch),dtype=float64))
ma00=(zeros((Nch,Nch),dtype=float64))
ma0b=(zeros((Nch,Nch),dtype=float64))
ma0s=(zeros((Nch,Nch),dtype=float64))
ma0s1=(zeros((Nch,Nch),dtype=float64))
mma0=(zeros((Nch,Nch),dtype=float64))
m0c0=(zeros((Nch,Nch),dtype=float64))
m00b=(zeros((Nch1,Nch),dtype=float64))
m00c=(zeros((Nch1,Nch),dtype=float64))
m0ca=(zeros((Nch,Nch),dtype=float64))


mode=arange(Nch)+1
mode0=arange(Nch)

dfk1=zeros((Nch,Nch))
dfk2=zeros((Nch,Nch))
dfk3=zeros((Nch,Nch))
dfk1[0,1]=6./5.*c(2)/c(1)
dfk1[Nch-1,Nch-2]=-(Nch-1.)*Nch/(2.*Nch-1.)*c(Nch-1)/c(Nch)
dfk2[0,1]=9./5.*c(2)/c(1)
dfk2[Nch-1,Nch-2]=-(Nch-1.)**2/(2.*Nch-1.)*c(Nch-1)/c(Nch)
dfk3[0,1]=12./5.*c(2)/c(1)
dfk3[Nch-1,Nch-2]=-(Nch-1.)*(Nch-2.)/(2.*Nch-1.)*c(Nch-1)/c(Nch)

for i in range(1,Nch-1):
        dfk1[i,i+1]=(i+2.)*(i+3.)/(2.*i+5)*c(i+2)/c(i+1)
        dfk1[i,i-1]=-(i+1)*i/(2.*i+1)*c(i)/c(i+1)
        dfk2[i,i+1]=(i+3.)**2/(2.*i+5)*c(i+2)/c(i+1)
        dfk2[i,i-1]=-(i*1.)**2/(2.*i+1)*c(i)/c(i+1)
        dfk3[i,i+1]=(i+3.)*(i+4.)/(2.*i+5)*c(i+2)/c(i+1)
        dfk3[i,i-1]=-(i-1)*i/(2.*i+1)*c(i)/c(i+1)


df2k=zeros((Nch,Nch))
df2k0=zeros((Nch,Nch))
for i in range(Nch):
    #if i< Nch-31:
    df2k[i,i]=-(i+1.)*(i+2.)
    #else:
    #    df2k[i,i]=-(i+1.)*(i+2.)*(i+1.)*(i+2.)*etas
    df2k0[i,i]=-i*(i+3.)
        

for i in range(Nch):
    ma0[:,i]=vlpnm(mode[i],xch[:])
    ma0s[:,i]=vlpnm(mode[i],xch[:])/(1-xch[:]**2)
    ma0s1[:,i]=vlpnm(mode[i],xch[:])/sqrt(1-xch[:]**2)
    ma0b[:,i]=vlpnc(mode[i],xch[:]) 
    ma00[:,i]=vlp00(mode[i],xch[:]) 
    m0c0[:,i]=vlpn0(mode0[i],xch[:])
    m00b[:,i]=vlpnm(mode[i],mu[:])
    m00c[:,i]=vlpn0(mode0[i],mu[:])

m_0c0=inv(m0c0)
fil0=append(ones(Nch-1),zeros(1))

m_00c=dot(diag(fil0/(mode*(mode+1))),inv(ma0b))
ma_0=inv(ma0)
mabp=-dot(ma0b,ma_0)

df1=dot(ma0,dot(dfk1,ma_0))
df1r=-dot(ma0s1,dot(dfk1,ma_0))
D1=dot(ma0,dot(dfk1,ma_0))
D1s=dot(ma0s1,dot(dfk2,ma_0))
mreca=dot(ma0,m_00c)

d2f=dot(dot(ma0,df2k),ma_0)
nfl=Nch//2
filt=append(ones(Nch-nfl),zeros(nfl))

FMb=dot(ma0,dot(diag(filt),ma_0))

filtd=zeros(Nch);filtd[0]=1
filtq=zeros(Nch);filtq[1]=1

ma_0b=inv(ma0b)
FMd=dot(ma0b,dot(diag(filtd),ma_0b))
FMq=dot(ma0b,dot(diag(filtq),ma_0b))
filt0=append(ones(10),zeros(Nch-10))
filt2=append(ones(Nch-10),zeros(10))
filt0=append(ones(10),zeros(Nch-10))

FMch=dot(ma00,dot(diag(filtq),inv(ma00)))
filt4=append(ones(10),zeros(Nch-10))#filt2#vsigS(nfi)
FMc3=dot(m0c0,dot(diag(filt4),m_0c0))


Nl=Nch
M0c=(zeros((Nch,Nl),dtype=float64))
M0s1=(zeros((Nch,Nl),dtype=float64))
M0C=(zeros((Nch,Nl),dtype=float64))

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


#savetxt('tmc'+fin,dot(m00c,dot(diag(filt2),m_0c0)))
#savetxt('tmb'+fin,dot(m00b,dot(diag(filt2),m_0)))
