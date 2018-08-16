from numpy import *
from scipy.special import *

def facn(n):
    return gamma(n+1)

def c(n):
    return sqrt((2.*n+1.)/(2.*n*(n+1.)))
def cnm(n):
    return sqrt((2.*n+1.)*n*(n+1.)/2.)

def cmm(n,m):
    if m==0:
        d0=1
    else:
        d0=0
    ans=-(-1)**m*sqrt((2.-d0))*n*(n+1)/sqrt((2*n+1)/2.)
    return ans

def cc(n):
    return sqrt((2.*n+1.)/2.)

def cn(n,m):
    return sqrt((2*n+1)/2.*facn(n-m)/facn(n+m))
    
def c_1(n,m):
    return sqrt((2*n+1)*facn(n+m)/facn(n-m))

def legendre_p(n,x):
    return c(n)*lpn(n,x)[0][n]

def legendre_p0(n,x):
    return cc(n)*lpn(n,x)[0][n]
vlp=vectorize(legendre_p0)

def legendre_pm(n,x):
    return c(n)*lpmn(1,n,x)[0][1][n]

vlpnc=vectorize(legendre_p)
vlpn0=vectorize(legendre_p0)
vlpn=vectorize(legendre_pm)
vlpnm=vectorize(legendre_pm)

def legendre_p00(n,x):
    return lpn(n,x)[0][n]
vlp00=vectorize(legendre_p00)

def Lpnm(n,m,x):
    mm=int(abs(m))
    if mm > n:
        ans=0
    else:
        ans=cn(n,mm)*lpmn(mm,n,x)[0][mm][n]
    return ans

def Npnm(n,m,x):
    mm=int(abs(m))
    if mm > n:
        ans=0
    else:
        ans=lpmn(mm,n,x)[0][mm][n]
    return ans

vLpnm=vectorize(Lpnm)
vNpnm=vectorize(Npnm)

def leg_d1(n,x):
    if n>=1:
        return gegenbauer(n-1,3./2.)(x);
    else:
        return 0.

             
def leg_d2(n,x):
    if n>=2 : return 3.*gegenbauer(n-2,5./2.)(x);
    
    else: return 0.;



vleg_d1=vectorize(leg_d1)
vleg_d2=vectorize(leg_d2)

def f0(b):
    if abs(b) < 0.01:
        return -1.5*b**2
    else:
        return 4.*((2.+3*b**2)/sqrt(1+b**2)**3/2.-1.)/b**2
    
def f3(b):
    if abs(b) < 0.01:
        return -1.+(56./45.+48./11.*b**2)*b**2
    else:
        return -(3*(48*b**4-25)*arctan(2.*b)/(2*b)-(304*b**4-200*b**2-75)/(4*b**2+1) )*7./(768*b**6)
    
def f1(b):
    if abs(b) < 0.01:
        return 1.-0.75*b**2
    else:
        return  2.*(1.-1./sqrt(1.+b**2))/b**2;

def fa(b):
    #return (1./(1.+b*b))
    if abs(b) < 0.01:
        return 1.-18.*b**2/7.+160.*b**4/21.
    else:
        return (16.*b**2-3-3.*(4*b**2-1)*arctan(2*b)/2/b)*5./128./b**4

def fe(b):
    return 1. #/sqrt(1.+.1*b**2)
     ## if abs(b) < 0.01:
     ##     return 1.-2.*b**2
     ## else:
     ##     return (1.+4./(1.+b**2)+4*b**2/(1+b**2)**2+(b**2-5)*arctan(b)/b)*3./8./b**2
    
vf0=vectorize(f0)  
vf1=vectorize(f1)  
vf3=vectorize(f0)  
vfa=vectorize(fa)  
vfe=vectorize(fe)  

def bu(b):
    if abs(b) <= .5:
        return 0.
    else:
        return b**2/(1+2*abs(b)**3)

vbu=vectorize(bu) 