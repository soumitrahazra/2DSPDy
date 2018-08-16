#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from func_dif import *
from scipy.linalg import *
Nch=64


wch=real(orthogonal.p_roots(Nch)[1])
xch=real(orthogonal.p_roots(Nch)[0])
xch=(xch[:]-xch[-1::-1])/2.
wch=(wch[:]+wch[-1::-1])/2.
mut=linspace(-90,90,180)*pi/180.

ma0=(zeros((Nch,Nch),dtype=float64))
ma00=(zeros((Nch,Nch),dtype=float64))
ma0b=(zeros((Nch,Nch),dtype=float64))
ma0s=(zeros((Nch,Nch),dtype=float64))
ma0s1=(zeros((Nch,Nch),dtype=float64))
mma0=(zeros((Nch,Nch),dtype=float64))
mode=arange(Nch)+1
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
    df2k[i,i]=-(i+1.)*(i+2.)
    df2k0[i,i]=-i*(i+3.)
        

for i in range(Nch):
    ma0[:,i]=vlpnm(mode[i],xch[:])
    ma0s[:,i]=vlpnm(mode[i],xch[:])/(1-xch[:]**2)
    ma0s1[:,i]=vlpnm(mode[i],xch[:])/sqrt(1-xch[:]**2)
    ma0b[:,i]=vlpnc(mode[i],xch[:]) 
    ma00[:,i]=vlp00(mode[i],xch[:]) 


ma_0=inv(ma0)
mabp=-dot(ma0b,ma_0)

df1=dot(ma0,dot(dfk1,ma_0))
df1r=-dot(ma0s1,dot(dfk1,ma_0))
D1=dot(ma0,dot(dfk1,ma_0))
D1s=dot(ma0s1,dot(dfk2,ma_0))

d2f=dot(dot(ma0,df2k),ma_0)
nfl=Nch//3
filt=append(ones(Nch-nfl),zeros(nfl))

FMb=dot(ma0,dot(diag(filt),ma_0))

filtd=zeros(Nch);filtd[0]=1
filtq=zeros(Nch);filtq[1]=1

ma_0b=inv(ma0b)
FMd=dot(ma0b,dot(diag(filtd),ma_0b))
FMq=dot(ma0b,dot(diag(filtq),ma_0b))
filt0=append(ones(10),zeros(Nch-10))

FMch=dot(ma00,dot(diag(filtq),inv(ma00)))
