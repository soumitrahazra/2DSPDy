from pylab import *

def plplab0(t,y,xtit,ytit):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    #plt.ylim( (min(y), max(y)) )
    
    plt.plot(t, y, 'k')
    plt.xlabel(xtit,fontsize=20)
    plt.ylabel(ytit,fontsize=20)
    plt.subplots_adjust(left=.1,bottom=0.1, top=.9,right=.9)
    plt.show()
def plpl(t,y,xtit,ytit):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    #plt.ylim( (min(y), max(y)) )
    
    plt.semilogy(t, y, 'k')
    plt.xlabel(xtit,fontsize=20)
    plt.ylabel(ytit,fontsize=20)
    plt.subplots_adjust(left=.1,bottom=0.1, top=.9,right=.9)
    plt.show()


def plplab(t,y,xtit,ytit,xmi,xma):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    #plt.ylim( (min(y), max(y)) )
    
    plt.plot(t, y, 'k')
    plt.xlabel(xtit,fontsize=20)
    plt.ylabel(ytit,fontsize=20)
    plt.subplots_adjust(left=.1,bottom=0.1, top=.9,right=.9)
    plt.xlim( (xmi, xma) )
    plt.show()

def plplab_n(n,t,y,xtit,ytit):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    #plt.ylim( (min(y), max(y)) )
    plt.xlim( (min(t), max(t)) )
    for i in range(n):
        plt.plot(t, y[-1:-1-i*2:-1,:], 'k')

    plt.xlabel(xtit,fontsize=20)
    plt.ylabel(ytit,fontsize=20)
    plt.subplots_adjust(left=.1,bottom=0.1, top=.9,right=.9)

    plt.show()


def plplab_s(t,y,xtit,ytit):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    plt.plot(t, y, 'k')
    plt.xlabel(xtit)
    plt.ylabel(ytit)
    plt.subplots_adjust(left=.1,bottom=0.1, top=.9,right=.9)

    plt.show();

def plplab3(t,y,y1,leg,xtit,ytit):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 5))
    plt.plot(t, y[0], 'k')
    plt.plot(t, y[1], 'k:')
    plt.plot(t, y[2], 'k--')
    plt.xlabel(xtit)
    plt.ylabel(ytit)
    plt.legend((leg[0], leg[1], leg[2]),
           'best', shadow=True)
    plt.plot(t, y1[0], 'r')
    plt.plot(t, y1[1], 'r:')
    plt.plot(t, y1[2], 'r--')

    plt.subplots_adjust(left=.1,bottom=0.1, top=.95,right=.95)

    plt.show();

def plplab_3d(x,y,z,tit):
    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import matplotlib.pyplot as plt

    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(x, y, z, label=tit)
    ax.legend()

    plt.show();

def plplab2_3d(x,y,z,x1,y1,z1,tit):
    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import matplotlib.pyplot as plt

    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(x, y, z, 'bo', label=tit)
    ax.plot(x1,y1,z1, 'r')
    ax.legend()
    plt.show();

def plpcon0(x,y,Z1,lev):
    import numpy as np
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,4))
    plt.contour(x,y,Z1, lev,
                colors='k', # negative contours will be dashed by default
                )
    plt.show();

def plpcon2(x,y,Z1,lev1,Z2,lev2,xti,yti,barti,T0,T1):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmap
    matplotlib.rc('text', usetex = True)
    matplotlib.rcParams['legend.fancybox'] = True
    
    plt.figure(figsize=(10,4))
    #plt.figure.subplot.wspace(0)
    #plt.figure.subplot.hspace(0)
    
    CS1=plt.contour(x,y,Z1, lev1,
                colors='k', # negative contours will be dashed by default
                )

    CS2=plt.contourf(x, y, Z2,#10, # [-1, -0.1, 0, 0.1],
                        #alpha=0.5,
                        cmap=cmap.bwr) #cmap.Greys)

    CB = plt.colorbar(CS2, shrink=0.9, extend='both')
    #CBI = plt.colorbar(im, orientation='horizontal', shrink=0.8)

    xlabel(xti,{'fontsize' : 14})
    ylabel(yti,{'fontsize' : 14})
    
    plt.ylim( (-90, 90) )
    plt.xlim((T0, T1))
    plt.subplots_adjust(left=.1,bottom=0.1, top=.9,right=.89)
    l,b,w,h = plt.gca().get_position().bounds
    ll,bb,ww,hh = CB.ax.get_position().bounds
    CB.ax.set_position([.9, b+0.025*h, ww, h])
    CB.set_label(barti,fontsize=20)


    plt.show();

def plc_1(x,y,xtm,ytm,Z1,lev1,xe):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmap
    params = {'axes.labelsize': 24,
              'text.fontsize': 24,
              'legend.fontsize': 14,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'text.usetex': True}
    plt.rcParams.update(params)

    plt.figure() #figsize=(5,10))

    #subplot(1,2,1)
    CS1=plt.contour(x,y,Z1, lev1,
                colors='k') # negative contours will be dashed by default
                
    Rad=plt.plot(x[0],y[0],'k')
    Top=plt.plot(x[:,-1],y[:,-1],'k')
    BCZ=plt.plot(xtm[:,0],ytm[:,0],'k')
    plt.ylim( (0, xe) )
    plt.xlim( (0, xe) )
    plt.xlabel('${r/R_{\\odot}}$')
    plt.ylabel('${r/R_{\\odot}}$')

    #plt.axis('off')
    #plt.axis('off')
    plt.show();

def plc2e(x,y,xtm,ytm,xxe,yye,Z1,lev1,Z2,vmi,vma,tks,barti,xei):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmap
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    params = {'axes.labelsize': 24,
              'text.fontsize': 24,
              'legend.fontsize': 20,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'text.usetex': True}
    plt.rcParams.update(params)

    
    plt.figure() #figsize=(5,10))
    CS1=plt.contour(xxe,yye,Z1, lev1,
                colors='k') # negative contours will be dashed by default
    Rad=plt.plot(x[0],y[0],'k')
    Top=plt.plot(x[:,-1],y[:,-1],'k')
    BCZ=plt.plot(xtm,ytm,'k')

    CS3=plt.pcolor(x, y, Z2,cmap=cmap.bwr,vmin=vmi,vmax=vma)
    plt.xlim(0,xei)
    plt.ylim(-xei,xei)
    plt.axis('off')

    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "4%", pad="4%")
    CB=plt.colorbar(CS3, cax=cax,ticks=tks)
    plt.tight_layout()
    CB.set_label(barti,fontsize=24)

    plt.show();

def plc2(x,y,xtm,ytm,Z1,lev1,Z2,vmi,vma,barti):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmap
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    params = {'axes.labelsize': 24,
              'text.fontsize': 24,
              'legend.fontsize': 20,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'text.usetex': True}
    plt.rcParams.update(params)
    plt.figure()
   
    CS1=plt.contour(x,y,Z1, lev1,
                colors='k', # negative contours will be dashed by default
                )
    CS3=plt.pcolor(x, y, Z2, #lev2,#10, # [-1, -0.1, 0, 0.1],
                        #alpha=0.5,
                        cmap=cmap.bwr,vmin=vmi,vmax=vma ) #cmap.Greys)
    Rad=plt.plot(x[0],y[0],'k')
    Top=plt.plot(x[:,-1],y[:,-1])
    BCZ=plt.plot(xtm[:,0],ytm[:,0])
    plt.ylim( (0, 1) )
    plt.xlim( (0, 1) )
    plt.xlabel('${r/R_{\\odot}}$')
    plt.ylabel('${r/R_{\\odot}}$')

    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "3%", pad="3%")
    CB=plt.colorbar(CS3, cax=cax)
    plt.tight_layout()
    CB.set_label(barti,fontsize=24)
    plt.show();


def plc_2(x,y,xtm,ytm,Z1,lev1,Z2,lev2):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmap
    
    plt.figure() #figsize=(5,10))

    subplot(1,2,1)
    CS1=plt.contour(x,y,Z1, lev1,
                colors='k') # negative contours will be dashed by default
                
    Rad=plt.plot(x[0],y[0],'b')
    Top=plt.plot(x[:,-1],y[:,-1],'k')
    BCZ=plt.plot(xtm[:,0],ytm[:,0],'b')
    plt.axis('off')
    subplot(1,2,2)

    CS2=plt.contourf(x, y, Z2, #lev2,#10, # [-1, -0.1, 0, 0.1],
                        #alpha=0.5,
                        cmap=cmap.bwr) #cmap.Greys)
    Rad=plt.plot(x[0],y[0],'k')
    Top=plt.plot(x[:,-1],y[:,-1],'k')
    BCZ=plt.plot(xtm[:,0],ytm[:,0],'k')

    CB = plt.colorbar(CS2, shrink=0.8, extend='both')
    plt.subplots_adjust(left=.1,bottom=0.1, top=.9,right=.89)
    l,b,w,h = plt.gca().get_position().bounds
    ll,bb,ww,hh = CB.ax.get_position().bounds
    CB.ax.set_position([.9, b+0.1*h, ww, h*0.8])
    plt.axis('off')
    plt.show();

def plc_3(x,y,xtm,ytm,Z1,lev1,Z2,lev2,Z3,vmi,vma):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmap
    
    plt.figure() #figsize=(5,10))

    subplot(1,2,1)
    CS1=plt.contour(x,y,Z1, lev1,
                colors='k') # negative contours will be dashed by default
                
    Rad=plt.plot(x[0],y[0],'k')
    Top=plt.plot(x[:,-1],y[:,-1],'k')
    BCZ=plt.plot(xtm[:],ytm[:],'k')
    plt.axis('off')
    subplot(1,2,2)

    CS2=plt.contour(x,y,Z2, lev2,
                colors='k') # negative contours will be dashed by default

    CS3=plt.pcolor(x, y, Z3,cmap=cmap.bwr,vmin=vmi,vmax=vma) 
    Rad=plt.plot(x[0],y[0],'k')
    Top=plt.plot(x[:,-1],y[:,-1],'k')
    BCZ=plt.plot(xtm[:,0],ytm[:,0],'k')

    CB = plt.colorbar(CS3, shrink=0.8, extend='both')
    plt.subplots_adjust(left=.1,bottom=0.1, top=.9,right=.89)
    l,b,w,h = plt.gca().get_position().bounds
    ll,bb,ww,hh = CB.ax.get_position().bounds
    CB.ax.set_position([.9, b+0.1*h, ww, h*0.8])
    plt.axis('off')
    plt.show();

def plc_n(n,x,y,xtm,ytm,lev1,Z2,Z3,lev2):
    import matplotlib.pyplot as plt
    fig=plt.figure()
    for i in range(n):
        fig.add_subplot((2,3,i))

        CS2=plt.contour(x,y,Z2[:,:,i], lev1,
                        colors='k') # negative contours will be dashed by default
        
        CS3=plt.contourf(x, y, Z3[:,:,i],lev2,#10, # [-1, -0.1, 0, 0.1],
                        #alpha=0.5,
                         cmap=cmap.bwr) #cmap.Greys)
        Rad=plt.plot(x[0],y[0],'k')
        Top=plt.plot(x[:,-1],y[:,-1],'k')
        BCZ=plt.plot(xtm[:,0],ytm[:,0],'k')
        ## CB = plt.colorbar(CS3, shrink=0.8, extend='both')
        ## plt.subplots_adjust(left=.1,bottom=0.1, top=.9,right=.89)
        ## l,b,w,h = plt.gca().get_position().bounds
        ## ll,bb,ww,hh = CB.ax.get_position().bounds
        ## CB.ax.set_position([.9, b+0.1*h, ww, h*0.8])
        plt.axis('off')
    plt.show();

def plc_3e(x,y,xtm,ytm,xxe,yye,Z1,lev1,Z2,lev2,Z3,vmi,vma):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmap
    
    plt.figure() #figsize=(5,10))

    subplot(1,2,1)
    CS1=plt.contour(xxe,yye,Z1, lev1,
                colors='k') # negative contours will be dashed by default
    Rad=plt.plot(x[0],y[0],'k')
    Top=plt.plot(x[:,-1],y[:,-1],'k')
    BCZ=plt.plot(xtm[:],ytm[:],'k')
    plt.axis('off')
    plt.xlim(0.,.45)
    plt.ylim(-.45,.45)
    
    subplot(1,2,2)

    CS2=plt.contour(x,y,Z2, lev2,
                colors='k') # negative contours will be dashed by default

    CS3=plt.contourf(x, y, Z3, #lev2,#10, # [-1, -0.1, 0, 0.1],
                        #alpha=0.5,
                        cmap=cmap.bwr) #cmap.Greys)
    Rad=plt.plot(x[0],y[0],'k')
    Top=plt.plot(x[:,-1],y[:,-1],'k')
    BCZ=plt.plot(xtm[:,0],ytm[:,0],'k')

    CB = plt.colorbar(CS3, shrink=0.8, extend='both')
    plt.subplots_adjust(left=.1,bottom=0.1, top=.9,right=.89)
    l,b,w,h = plt.gca().get_position().bounds
    ll,bb,ww,hh = CB.ax.get_position().bounds
    CB.ax.set_position([.9, b+0.1*h, ww, h*0.8])
    plt.axis('off')
    plt.show();

def plc_3eh(x,y,xtm,ytm,xxe,yye,Z1,lev1,Z2,lev2,Z3,vmi,vma,tks,barti):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmap
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    params = {'axes.labelsize': 24,
              'text.fontsize': 24,
              'legend.fontsize': 20,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'text.usetex': True}
    plt.rcParams.update(params)

    plt.figure() #figsize=(5,10))

    subplot(1,2,1)
    CS1=plt.contour(xxe,yye,Z1, lev1,
                colors='k') # negative contours will be dashed by default
    Rad=plt.plot(x[0],y[0],'k')
    Top=plt.plot(x[:,-1],y[:,-1],'k')
    BCZ=plt.plot(xtm,ytm,'k')
    plt.axis('off')
    plt.xlim(0.,.4)
    plt.ylim(-0.4,.4)
    
    subplot(1,2,2)

    CS2=plt.contour(x,y,Z2, lev2,
                colors='k') # negative contours will be dashed by default

    CS3=plt.pcolor(x, y, Z3, #lev2,#10, # [-1, -0.1, 0, 0.1],
                        #alpha=0.5,
                        cmap=cmap.bwr,vmin=vmi,vmax=vma ) #cmap.Greys)
    Rad=plt.plot(x[0],y[0],'k')
    Top=plt.plot(x[:,-1],y[:,-1],'k')
    BCZ=plt.plot(xtm,ytm,'k')
    plt.xlim(0.,.4)
    plt.ylim(-0.4,.4)
    plt.axis('off')

    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "3%", pad="3%")
    CB=plt.colorbar(CS3, cax=cax)
    plt.tight_layout()
    CB.set_label(barti,fontsize=24)
    plt.show();


def plc_ne(nr,nc,n,x,y,xtm,ytm,xxe,yye,Z1,lev1,Z3,lev2):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmap
    fig=plt.figure()
    for i in range(n):
        fig.add_subplot(nr,nc,i+1)
        CS1=plt.contour(xxe,yye,Z1[i,:,:], lev1,
                        colors='k') # negative contours will be dashed by default
        Rad=plt.plot(x[0],y[0],'k')
        Top=plt.plot(x[:,-1],y[:,-1],'k')
        BCZ=plt.plot(xtm,ytm,'k')
        CS3=plt.contourf(x, y, Z3[i,:,:], lev2,#10, # [-1, -0.1, 0, 0.1],
                         #alpha=0.5,
                         cmap=cmap.bwr) #cmap.Greys)
        plt.axis('off')
        plt.xlim(0.,1.15)
        plt.ylim(-1.15,1.15)
    plt.show();

def plc_ne1(nr,nc,n,x,y,xtm,ytm,Z3,lev2,ic):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmap
    fig=plt.figure()
    for i in range(n):
        fig.add_subplot(nr,nc,i+1)

        Rad=plt.plot(x[0],y[0],'k')
        Top=plt.plot(x[:,-1],y[:,-1],'k')
        BCZ=plt.plot(xtm[:,0],ytm[:,0],'k')
        CS3=plt.contourf(x, y, Z3[i,:,:], #10, # [-1, -0.1, 0, 0.1],
                         #alpha=0.5,
                         cmap=cmap.bone) #cmap.Greys)
        plt.axis('off')
        plt.xlim(0.,1.)
        plt.ylim(-1.,1.)
        if ic==1:
            CB = plt.colorbar(CS3, shrink=0.8, extend='both',ticks=[-1, 0, 1])
            plt.subplots_adjust(left=.1,bottom=0.1, top=.9,right=.89)
            l,b,w,h = plt.gca().get_position().bounds
            ll,bb,ww,hh = CB.ax.get_position().bounds
            CB.ax.set_position([.9, b+0.1*h, ww, h*0.8])
            
    plt.show();

def plpcon2bw(x,y,Z1,lev1,Z2,xti,yti):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmap
    matplotlib.rc('text', usetex = True)
    matplotlib.rcParams['legend.fancybox'] = True

    plt.figure(figsize=(10,4))
    #plt.figure.subplot.wspace(0)
    #plt.figure.subplot.hspace(0)
    
    CS1=plt.contour(x,y,Z1, lev1,
                colors='y', # negative contours will be dashed by default
                )

    CS2=plt.contourf(x, y, Z2,#10, # [-1, -0.1, 0, 0.1],
                        #alpha=0.5,
                        cmap=cmap.bone) #cmap.Greys)

    #CB = plt.colorbar(CS2, shrink=0.9, extend='both')
    #CBI = plt.colorbar(im, orientation='horizontal', shrink=0.8)

    xlabel(xti,{'fontsize' : 14})
    ylabel(yti,{'fontsize' : 14})
    
    plt.ylim( (-90, 90) )
    plt.xlim((min(x[:,0]), max(x[:,0])) )

    plt.show();

def plplab2r(t,y1,xtit,ytit,tmi,tma):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmap
    matplotlib.rc('text', usetex = True)
    matplotlib.rcParams['legend.fancybox'] = True
    params = {'axes.labelsize': 24,
              'text.fontsize': 24,
              'legend.fontsize': 16,
              'xtick.labelsize': 18,
              'ytick.labelsize': 18,
              'text.usetex': True}
    plt.rcParams.update(params)

    plt.figure()
    plt.plot(t, y1, 'k')
    plt.xlim( (tmi, tma) )
    plt.ylabel(ytit,fontsize=24)
    plt.xlabel(xtit,fontsize=24)
    plt.show();

def plplab2rT(t,y1,y2,ytit3,xtit4,ytit4,T1,T2):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmap
    matplotlib.rc('text', usetex = True)
    matplotlib.rcParams['legend.fancybox'] = True

    plt.figure()
    subplot(2,1,1)
    plt.plot(t, y1, 'k')
    plt.xlim((T1,T2) )
    #plt.xlabel(xtit,fontsize=20)
    plt.ylabel(ytit3,fontsize=12)
    #plt.subplots_adjust(left=.1,bottom=0.1, top=.9,right=.9)
    subplot(2,1,2)
    plt.plot(t, y2, 'k')
    plt.xlim( (T1,T2) )
    plt.xlabel(xtit4,fontsize=12)
    plt.ylabel(ytit4,fontsize=12)
    #plt.subplots_adjust(left=.1,bottom=0.1, top=.9,right=.9)
    plt.show();


## def plpcon2c(x,y,Z1,lev1,Z2,xti1,yti1,barti1,
##             Z3,lev3,Z4,xti2,yti2,barti2):
##     import numpy as np
##     import matplotlib.pyplot as plt
##     import matplotlib.cm as cmap
##     matplotlib.rc('text', usetex = True)
##     matplotlib.rcParams['legend.fancybox'] = True

##     plt.figure()
##     subplot(2,1,1)
    
##     CS1=plt.contour(x,y,Z1, lev1,
##                 colors='k', # negative contours will be dashed by default
##                 )

##     CS2=plt.contourf(x, y, Z2,#10, # [-1, -0.1, 0, 0.1],
##                         #alpha=0.5,
##                         cmap=cmap.bwr) #cmap.Greys)

##     CB = plt.colorbar(CS2, shrink=0.9, extend='both')
##     #CBI = plt.colorbar(im, orientation='horizontal', shrink=0.8)

##     xlabel(xti1,{'fontsize' : 12})
##     ylabel(yti1,{'fontsize' : 12})
    
##     plt.ylim( (-90, 90) )
##     plt.xlim((min(x[:,0]), max(x[:,0])) )
##     plt.subplots_adjust(left=.1,bottom=0.1, top=.9,right=.9)
##     l,b,w,h = plt.gca().get_position().bounds
##     ll,bb,ww,hh = CB.ax.get_position().bounds
##     CB.ax.set_position([.92, b+0.15*h, ww, h])
##     CB.set_label(barti1,fontsize=12)

##     subplot(2,1,2)
    
##     CS1=plt.contour(x,y,Z3, lev1,
##                 colors='k', # negative contours will be dashed by default
##                 )

##     CS2=plt.contourf(x, y, Z4,#10, # [-1, -0.1, 0, 0.1],
##                         #alpha=0.5,
##                         cmap=cmap.bwr) #cmap.Greys)

##     CB = plt.colorbar(CS2, shrink=0.9, extend='both')
##     #CBI = plt.colorbar(im, orientation='horizontal', shrink=0.8)

##     xlabel(xti2,{'fontsize' : 12})
##     ylabel(yti2,{'fontsize' : 12})
    
##     plt.ylim( (-90, 90) )
##     plt.xlim((min(x[:,0]), max(x[:,0])) )
##     plt.subplots_adjust(left=.1,bottom=0.1, top=.9,right=.9)
##     l,b,w,h = plt.gca().get_position().bounds
##     ll,bb,ww,hh = CB.ax.get_position().bounds
##     CB.ax.set_position([.92, b-0.05*h, ww, h])
##     CB.set_label(barti2,fontsize=12)
##     plt.show();

def plpcon2cT(x,y,Z1,lev1,Z2,xti1,yti1,barti1,
            Z3,lev3,Z4,xti2,yti2,barti2,T0,T1):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmap
    matplotlib.rc('text', usetex = True)
    matplotlib.rcParams['legend.fancybox'] = True

    plt.figure()
    subplot(2,1,1)
    
    CS1=plt.contour(x,y,Z1, lev1,
                colors='k', # negative contours will be dashed by default
                )

    CS2=plt.contourf(x, y, Z2,#10, # [-1, -0.1, 0, 0.1],
                        #alpha=0.5,
                        cmap=cmap.bwr) #cmap.Greys)

    CB = plt.colorbar(CS2, shrink=0.9, extend='both')
    #CBI = plt.colorbar(im, orientation='horizontal', shrink=0.8)

    xlabel(xti1,{'fontsize' : 12})
    ylabel(yti1,{'fontsize' : 12})
    
    plt.ylim( (-90, 90) )
    plt.xlim((T0,T1) )
    plt.subplots_adjust(left=.1,bottom=0.1, top=.9,right=.9)
    l,b,w,h = plt.gca().get_position().bounds
    ll,bb,ww,hh = CB.ax.get_position().bounds
    CB.ax.set_position([.92, b+0.15*h, ww, h])
    CB.set_label(barti1,fontsize=12)

    subplot(2,1,2)
    
    CS1=plt.contour(x,y,Z3, lev1,
                colors='k', # negative contours will be dashed by default
                )

    CS2=plt.contourf(x, y, Z4,#10, # [-1, -0.1, 0, 0.1],
                        #alpha=0.5,
                        cmap=cmap.bwr) #cmap.Greys)

    CB = plt.colorbar(CS2, shrink=0.9, extend='both')
    #CBI = plt.colorbar(im, orientation='horizontal', shrink=0.8)

    xlabel(xti2,{'fontsize' : 12})
    ylabel(yti2,{'fontsize' : 12})
    
    plt.ylim( (-90, 90) )
    plt.xlim((T0, T1) )
    plt.subplots_adjust(left=.1,bottom=0.1, top=.9,right=.9)
    l,b,w,h = plt.gca().get_position().bounds
    ll,bb,ww,hh = CB.ax.get_position().bounds
    CB.ax.set_position([.92, b-0.05*h, ww, h])
    CB.set_label(barti2,fontsize=12)
    plt.show();


def plpcon1(x,y,Z1,lev1,xti1,yti1,tit,tmi,tm,li,la):
    #import wxversion
    #wxversion.select('2.8')
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmap
    params = {'axes.labelsize': 24,
              'text.fontsize': 24,
              'legend.fontsize': 20,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'text.usetex': True}
    plt.rcParams.update(params)

    plt.figure()
   
    CS1=plt.contour(x,y,Z1, lev1,
                colors='k' # negative contours will be dashed by default
                )
    xlabel(xti1)
    ylabel(yti1)
    yticks( range(-60,90,30))
    title(tit)
    plt.ylim( (li, la) )
    plt.xlim((tmi, tm) )
    plt.show();

def plpcon2c(x,y,Z1,x1,y1,Z2,lev1,xti1,yti1,tit,tmi,tm,li,la):
    #import wxversion
    #wxversion.select('2.8')
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmap
    params = {'axes.labelsize': 24,
              'text.fontsize': 24,
              'legend.fontsize': 20,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'text.usetex': True}
    plt.rcParams.update(params)

    plt.figure()
   
    CS1=plt.contour(x,y,Z1, lev1,
                colors='r' # negative contours will be dashed by default
                )
    CS2=plt.contour(x1,y1,Z2, lev1,
                colors='b' # negative contours will be dashed by default
                )
    xlabel(xti1)
    ylabel(yti1)
    yticks( range(-60,90,30))
    title(tit)
    plt.ylim( (li, la) )
    plt.xlim((tmi, tm) )
    plt.show();


def plpcon(x,y,Z1,lev1,Z2,vmi,vma,xti1,yti1,tit,barti1,tmi,tma,li,la):
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmap
    params = {'axes.labelsize': 24,
              'text.fontsize': 24,
              'legend.fontsize': 20,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'text.usetex': True}
    plt.rcParams.update(params)

    plt.figure(figsize=(15, 5))
   
    CS1=plt.contour(x,y,Z1, lev1,
                colors='k' # negative contours will be dashed by default
                )
    CS2=plt.pcolor(x, y, Z2,#10, # [-1, -0.1, 0, 0.1],
                   cmap=cmap.bwr,vmin=vmi,vmax=vma) #cmap.gray)
    ## CS3=plt.pcolor(x, y, Z2,#10, # [-1, -0.1, 0, 0.1],
    ##                     alpha=1.,
    ##                     cmap=cmap.bwr,vmin=vma/2,vmax=vma) #cmap.Greys)

    CB = plt.colorbar(CS2, shrink=0.9, extend='both')

    xlabel(xti1)
    ylabel(yti1)
    yticks( range(-60,90,30))
    title(tit,fontsize=24)
    plt.ylim(li, la )
    plt.xlim(tmi, tma)
    plt.subplots_adjust(left=.1,bottom=0.1, top=.9,right=.9)
    l,b,w,h = plt.gca().get_position().bounds
    ll,bb,ww,hh = CB.ax.get_position().bounds
    CB.ax.set_position([.92, b+0.05*h, ww, h])
    CB.set_label(barti1,fontsize=24)
    plt.show();

def plpconr(x,y,Z1,lev1,Z2,vmi,vma,xti1,yti1,tit,barti1,tmi,tm,tks):
    #import wxversion
    #wxversion.select('2.8')
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmap
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    params = {'axes.labelsize': 24,
              'text.fontsize': 24,
              'legend.fontsize': 24,
              'xtick.labelsize': 24,
              'ytick.labelsize': 24,
              'text.usetex': True}
    plt.rcParams.update(params)

    plt.figure(figsize=(15, 5))
   
    CS1=plt.contour(x,y,Z1, lev1,
                    colors='k', # negative contours will be dashed by default
                    )
    CS2=plt.pcolor(x, y, Z2,#10, # [-1, -0.1, 0, 0.1],
                        alpha=1,
                        cmap=cmap.bwr,vmin=vmi,vmax=vma) #cmap.Greys)
    ## CS3=plt.pcolor(x, y, Z2,#10, # [-1, -0.1, 0, 0.1],
    ##                     alpha=1.,
    ##                     cmap=cmap.bwr,vmin=vma/2,vmax=vma) #cmap.Greys)

    xlabel(xti1)
    ylabel(yti1)
    title(tit)
    plt.ylim( (0.05, .27) )
    plt.xlim((tmi, tm) )
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "3%", pad="3%")
    CB=plt.colorbar(CS2, cax=cax, ticks=tks)
    plt.tight_layout()
    CB.set_label(barti1,fontsize=30)
    plt.show();

def plc_nev(nr,nc,n,xe,ye,Z1,lev1,x,y,xtm,ytm,Z3,vmi,vma,tim):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmap
    fig=plt.figure()
    for i in range(n):
        fig.add_subplot(nr,nc,i+1)
        CS1=plt.contour(xe,ye,Z1[i,:,:], lev1,
                        colors='k') # negative contours will be dashed by default
        #Rad=plt.plot(x[0],y[0],'k')
        Top=plt.plot(x[-1,:],y[-1,:],'k')
        Z2=Z3[:,:,i]
        BCZ=plt.plot(xtm[0,:],ytm[0,:],'k')
        CS3=plt.pcolor(x, y, Z3[i,:,:], vmin=vmi, vmax=vma,cmap=cmap.bwr)
        plt.axis('off')
        plt.xlim(0.,1.2)
        plt.ylim(0,1.2)
        plt.text(0.025,0.05,'t='+str('%.1f' %tim[i])+' Yr',fontsize=20)
    plt.show()
    
def plc_nv(nr,nc,n,x,y,xe,ye,xtm,ytm,Z1,lev1,Z3,vmi,vma):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmap
    fig=plt.figure()
    for i in range(n):
        fig.add_subplot(nr,nc,i+1)
        CS1=plt.contour(xe,ye,Z1[i,:,:], lev1,
                        colors='k') # negative contours will be dashed by default
        Rad=plt.plot(x[0],y[0],'k')
        Top=plt.plot(x[:,-1],y[:,-1],'k')
        BCZ=plt.plot(xtm[:,0],ytm[:,0],'k')
        CS3=plt.pcolor(x, y, Z3[i,:,:], cmap=cmap.bwr,vmin=vmi,vmax=vma)
        plt.axis('off')
        plt.xlim(0.,1.2)
        plt.ylim(0,1.2)
    plt.show()

def plc_nev0(nr,nc,n,x,y,xtm,ytm,Z3,vmi,vma):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmap
    #from matplotlib import colors, ticker

    fig=plt.figure()
    for i in range(n):
        fig.add_subplot(nr,nc,i+1)
        Rad=plt.plot(x[0],y[0],'k')
        Top=plt.plot(x[:,-1],y[:,-1],'k')
        BCZ=plt.plot(xtm[:,0],ytm[:,0],'k')
        CS3=plt.pcolor(x, y, Z3[i,:,:], cmap=cmap.bwr,vmin=vmi,vmax=vma)
        plt.axis('off')
        plt.xlim(0.,1.)
        plt.ylim(0,1.)
    plt.show()


def plp2ax(X,Y0,Y1,xtit,y0tit,y1tit):
    from mpl_toolkits.axes_grid1 import host_subplot
    import mpl_toolkits.axisartist as AA
    import matplotlib.pyplot as plt
    params = {'axes.labelsize': 24,
              'text.fontsize': 24,
              'legend.fontsize': 14,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'text.usetex': True}
    plt.rcParams.update(params)

    if 1:

        host = host_subplot(111, axes_class=AA.Axes)
        plt.subplots_adjust(right=0.75)

        par1 = host.twinx()

        offset = 60



        host.set_xlim(min(X), max(X))
        host.set_ylim(min(Y0), max(Y0))

        p1, = host.plot(X,Y0,'b',linewidth=2)
        p2, = par1.plot(X,Y1,'r',linewidth=2)

        par1.set_ylim(min(Y1), max(Y1))
        host.axis["left"].label.set_color('k') #p1.get_color())
        par1.axis["right"].label.set_color('k') #p2.get_color())

        host.set_xlabel(xtit,fontsize=32)
        host.set_ylabel(y0tit,fontsize=32)
        par1.set_ylabel(y1tit,fontsize=32)
        #plt.draw()
        plt.show()


##########
def plc_iau(x,y,xtm,ytm,xxe,yye,Z1,lev1,Z2,vmib,vmab,tks1,
            barti1,Z3,lev2,Z4,vmi,vma,tks2,barti2,tim,i):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmap
    params = {'axes.labelsize': 24,
              'text.fontsize': 24,
              'legend.fontsize': 20,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'text.usetex': True}
    plt.rcParams.update(params)

    plt.figure(figsize=(15,6))

    subplot(1,2,1)
    CS1=plt.contour(xxe,yye,Z1, lev1,
                colors='k') # negative contours will be dashed by default
    Rad=plt.plot(x[0],y[0],'k')
    Top=plt.plot(x[:,-1],y[:,-1],'k')
    BCZ=plt.plot(xtm[:,0],ytm[:,0],'k')
    CS2=plt.pcolor(x, y, Z2, #lev2,#10, # [-1, -0.1, 0, 0.1],
                        #alpha=0.5,
                        cmap=cmap.bwr,vmin=vmib,vmax=vmab ) #cmap.Greys)
    plt.axis('off')
    plt.xlim(0.,1.15)
    plt.ylim(0,1.15)
    CB1 = plt.colorbar(CS2, shrink=0.8, extend='both',orientation='vertical',ticks=tks1)
    plt.subplots_adjust(left=.025,bottom=0.1, top=.9,right=.89)
    l,b,w,h = plt.gca().get_position().bounds
    ll,bb,ww,hh = CB1.ax.get_position().bounds
    #CB1.ax.set_position([ll-0.1*w, b-.7*h, ww, h*0.8])
    CB1.ax.set_position([.02, b+0.05*h, ww, h])
    CB1.set_label(barti1,fontsize=24)
    plt.text(0.025,0.5,'t='+str('%.1f' %tim)+' Yr',fontsize=20)
    subplot(1,2,2)

    CS3=plt.contour(x,y,Z3, lev2,
                colors='k') # negative contours will be dashed by default

    CS4=plt.pcolor(x, y, Z4, #lev2,#10, # [-1, -0.1, 0, 0.1],
                        #alpha=0.5,
                        cmap=cmap.bwr,vmin=vmi,vmax=vma ) #cmap.Greys)
    Rad=plt.plot(x[0],y[0],'k')
    Top=plt.plot(x[:,-1],y[:,-1],'k')
    BCZ=plt.plot(xtm[:,0],ytm[:,0],'k')
    plt.xlim(0.,1.)
    plt.ylim(0,1.)

    CB2 = plt.colorbar(CS4, shrink=0.8, extend='both',orientation='vertical',ticks=tks2)
    plt.subplots_adjust(left=.1,bottom=0.1, top=.9,right=.89)
    l,b,w,h = plt.gca().get_position().bounds
    ll,bb,ww,hh = CB2.ax.get_position().bounds
    CB2.ax.set_position([.92, b+0.05*h, ww, h]) #[ll-0.1*w, b-.7*h, ww, h*0.8])
    CB2.set_label(barti2,fontsize=24)
    plt.axis('off')
    plt.savefig('p'+str(i)+'.png',format='png');

def plc_iaups(x,y,xtm,ytm,xxe,yye,Z1,lev1,Z2,vmib,vmab,tks1,
            barti1,Z3,lev2,Z4,vmi,vma,tks2,barti2,tim,i):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmap
    params = {'axes.labelsize': 24,
              'text.fontsize': 24,
              'legend.fontsize': 20,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'text.usetex': True}
    plt.rcParams.update(params)

    plt.figure(figsize=(15,6))

    subplot(1,2,1)
    CS1=plt.contour(xxe,yye,Z1, lev1,
                colors='k') # negative contours will be dashed by default
    Rad=plt.plot(x[0],y[0],'k')
    Top=plt.plot(x[:,-1],y[:,-1],'k')
    BCZ=plt.plot(xtm[:,0],ytm[:,0],'k')
    CS2=plt.pcolor(x, y, Z2, #lev2,#10, # [-1, -0.1, 0, 0.1],
                        #alpha=0.5,
                        cmap=cmap.bwr,vmin=vmib,vmax=vmab ) #cmap.Greys)
    plt.axis('off')
    plt.xlim(0.,1.15)
    plt.ylim(0,1.15)
    CB1 = plt.colorbar(CS2, shrink=0.8, extend='both',orientation='vertical',ticks=tks1)
    plt.subplots_adjust(left=.025,bottom=0.1, top=.9,right=.89)
    l,b,w,h = plt.gca().get_position().bounds
    ll,bb,ww,hh = CB1.ax.get_position().bounds
    #CB1.ax.set_position([ll-0.1*w, b-.7*h, ww, h*0.8])
    CB1.ax.set_position([.02, b+0.05*h, ww, h])
    CB1.set_label(barti1,fontsize=24)
    plt.text(0.025,0.5,'t='+str('%.1f' %tim)+' Yr',fontsize=20)
    subplot(1,2,2)

    CS3=plt.contour(x,y,Z3, lev2,
                colors='k') # negative contours will be dashed by default

    CS4=plt.pcolor(x, y, Z4, #lev2,#10, # [-1, -0.1, 0, 0.1],
                        #alpha=0.5,
                        cmap=cmap.bwr,vmin=vmi,vmax=vma ) #cmap.Greys)
    Rad=plt.plot(x[0],y[0],'k')
    Top=plt.plot(x[:,-1],y[:,-1],'k')
    BCZ=plt.plot(xtm[:,0],ytm[:,0],'k')
    plt.xlim(0.,1.)
    plt.ylim(0,1.)

    CB2 = plt.colorbar(CS4, shrink=0.8, extend='both',orientation='vertical',ticks=tks2)
    plt.subplots_adjust(left=.1,bottom=0.1, top=.9,right=.89)
    l,b,w,h = plt.gca().get_position().bounds
    ll,bb,ww,hh = CB2.ax.get_position().bounds
    CB2.ax.set_position([.92, b+0.05*h, ww, h]) #[ll-0.1*w, b-.7*h, ww, h*0.8])
    CB2.set_label(barti2,fontsize=24)
    plt.axis('off')
    plt.savefig('p'+str(i)+'.eps',format='eps');

def plc_ab(x,y,xtm,ytm,xxe,yye,Z1,lev1,Z2,vmib,vmab,tks1,barti1,tim1,n0,ns,N):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmap
    from mpl_toolkits.axes_grid1 import make_axes_locatable
        
    params = {'axes.labelsize': 24,
              'text.fontsize': 24,
              'legend.fontsize': 20,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'text.usetex': True}
    plt.rcParams.update(params)

    fig=plt.figure()
    for i in range(N):
        fig.add_subplot(N,1,i+1)
        tim=tim1[i*ns]
        CS1=plt.contour(xxe,yye,Z1[n0+i*ns,:,:], lev1,
                        colors='k') # negative contours will be dashed by default
        Rad=plt.plot(x[0],y[0],'k')
        Top=plt.plot(x[:,-1],y[:,-1],'k')
        BCZ=plt.plot(xtm[:,0],ytm[:,0],'k')
        CS2=plt.pcolor(x, y, Z2[n0+i*ns,:,:], #lev2,#10, # [-1, -0.1, 0, 0.1],
                       #alpha=0.5,
                       cmap=cmap.bwr,vmin=vmib,vmax=vmab ) #cmap.Greys)
        plt.axis('off')
        plt.xlim(0.,1.)
        plt.ylim(0,1.)
        plt.text(0.025,0.25,'t='+str('%.0f' %tim)+' Yr',fontsize=20)
        if i==N-1:
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes("bottom", "4%", pad="3%")
            CB1=plt.colorbar(CS2, cax=cax, orientation='horizontal',ticks=tks1)
            #plt.tight_layout()
            CB1.set_label(barti1,fontsize=20)
        #plt.tight_layout()
    plt.show()


def plc_ha(x,y,xtm,ytm,xxe,yye,Z1,lev1,Z2,vmib,vmab,tks1,barti1,tim1,n0,ns):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmap
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    params = {'axes.labelsize': 24,
              'text.fontsize': 24,
              'legend.fontsize': 20,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'text.usetex': True}
    plt.rcParams.update(params)

    fig=plt.figure()
    for i in range(6):
        fig.add_subplot(6,1,i+1)
        tim=tim1[i*ns]
        CS1=plt.contour(xxe,yye,Z1[n0+i*ns,:,:], lev1,
                        colors='k') # negative contours will be dashed by default
        Rad=plt.plot(x[0],y[0],'k')
        Top=plt.plot(x[:,-1],y[:,-1],'k')
        BCZ=plt.plot(xtm[:,0],ytm[:,0],'k')
        CS2=plt.pcolor(x, y, Z2[n0+i*ns,:,:], #lev2,#10, # [-1, -0.1, 0, 0.1],
                       #alpha=0.5,
                       cmap=cmap.bwr,vmin=vmib,vmax=vmab ) #cmap.Greys)
        plt.axis('off')
        plt.xlim(0.,1)
        plt.ylim(0,1)
        #plt.text(0.025,0.5,'t='+str('%.1f' %tim)+' Yr',fontsize=20)
        if i==5:
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes("bottom", "4%", pad="3%")
            CB1=plt.colorbar(CS2, cax=cax,orientation='horizontal',ticks=tks1)
            CB1.set_label(barti1,fontsize=20)
        #plt.tight_layout()
    plt.show()

def plpcon2x(x,y,Z1,lev1,x2,y2,Z2,vmi,vma,xti1,yti1,tit,barti1,tmi,tma,li,la,tks):
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri
    import matplotlib.cm as cmap
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    params = {'axes.labelsize': 24,
              'text.fontsize': 24,
              'legend.fontsize': 24,
              'xtick.labelsize': 24,
              'ytick.labelsize': 24,
              'text.usetex': True}
    plt.rcParams.update(params)

    plt.figure(figsize=(16, 4))
   
    CS1=plt.contour(x,y,Z1, lev1,
                colors='k' # negative contours will be dashed by default
                )
    CS2=plt.pcolor(x2, y2, Z2,cmap=cmap.bwr,vmin=vmi,vmax=vma) 
    xlabel(xti1,fontsize=20)
    ylabel(yti1)
    yticks( range(-60,90,30))
    title(tit,fontsize=24)
    plt.ylim(li, la )
    plt.xlim(tmi, tma)
    
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "3%", pad="3%")
    CB=plt.colorbar(CS2, cax=cax,ticks=tks)
    CB.set_label(barti1,fontsize=30)
    plt.tight_layout()
    plt.show();

def plpcon2xf(x,y,Z1,lev1,x2,y2,Z2,vmi,vma,tks1,xti1,yti1,tit,barti1,tmi,tma,li,la):
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmap
    params = {'axes.labelsize': 24,
              'text.fontsize': 24,
              'legend.fontsize': 20,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'text.usetex': True}
    plt.rcParams.update(params)

    #plt.figure(figsize=(15, 5))
   
    CS1=plt.contour(x,y,Z1, lev1,
                colors='k' # negative contours will be dashed by default
                )
    CS2=plt.contourf(x2, y2, Z2,#10, # [-1, -0.1, 0, 0.1],
                   cmap=cmap.bwr,levels=vmi+(vma-vmi)*arange(18)/17.) #cmap.gray)
    ## CS3=plt.pcolor(x, y, Z2,#10, # [-1, -0.1, 0, 0.1],
    ##                     alpha=1.,
    ##                     cmap=cmap.bwr,vmin=vma/2,vmax=vma) #cmap.Greys)

    CB = plt.colorbar(CS2, shrink=0.9, extend='both',ticks=tks1)

    xlabel(xti1)
    ylabel(yti1)
    yticks( range(-60,90,30))
    title(tit,fontsize=24)
    plt.ylim(li, la )
    plt.xlim(tmi, tma)
    plt.subplots_adjust(left=.1,bottom=0.1, top=.9,right=.9)
    l,b,w,h = plt.gca().get_position().bounds
    ll,bb,ww,hh = CB.ax.get_position().bounds
    CB.ax.set_position([.92, b+0.05*h, ww, h])
    CB.set_label(barti1,fontsize=24)
    plt.show();


def plc_abn(x,y,xtm,ytm,xxe,yye,Z1,lev1,Z2,vmib,vmab,tks1,barti1,tim1,n0,ns,n):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmap
    from mpl_toolkits.axes_grid1 import make_axes_locatable
        
    params = {'axes.labelsize': 24,
              'text.fontsize': 24,
              'legend.fontsize': 20,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'text.usetex': True}
    plt.rcParams.update(params)

    fig=plt.figure()
    for i in range(n):
        fig.add_subplot(1,n,i+1)
        tim=tim1[i*ns]
        CS1=plt.contour(xxe,yye,Z1[n0+i*ns,:,:], lev1,
                        colors='k') # negative contours will be dashed by default
        Rad=plt.plot(x[0],y[0],'k')
        Top=plt.plot(x[:,-1],y[:,-1],'k')
        BCZ=plt.plot(xtm[:,0],ytm[:,0],'k')
        CS2=plt.pcolor(x, y, Z2[n0+i*ns,:,:], #lev2,#10, # [-1, -0.1, 0, 0.1],
                       #alpha=0.5,
                       cmap=cmap.bwr,vmin=vmib,vmax=vmab ) #cmap.Greys)
        plt.axis('off')
        plt.xlim(0.,1)
        plt.ylim(0,1)
        plt.text(0.025,0.25,'t='+str('%.1f' %tim)+' Yr',fontsize=20)
        if i==n-1:
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes("right", "4%", pad="3%")
            CB1=plt.colorbar(CS2, cax=cax)
            #plt.tight_layout()
            CB1.set_label(barti1,fontsize=20)
        #plt.tight_layout()
    plt.show()

def plc_ani(x,y,xtm,ytm,xxe,yye,Z1,lev1,Z2,vmib,vmab,tks1,barti1,tim,i):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmap
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    params = {'axes.labelsize': 30,
              'text.fontsize': 30,
              'legend.fontsize': 30,
              'xtick.labelsize': 30,
              'ytick.labelsize': 30,
              'text.usetex': True}
    plt.rcParams.update(params)

    plt.figure(figsize=(15,12))

    CS1=plt.contour(xxe,yye,Z1[i,:,:], lev1,
                colors='k') # negative contours will be dashed by default
    Rad=plt.plot(x[0],y[0],'k')
    Top=plt.plot(x[:,-1],y[:,-1],'k')
    BCZ=plt.plot(xtm[:,0],ytm[:,0],'k')
    CS2=plt.pcolor(x, y, Z2[i,:,:],cmap=cmap.bwr,vmin=vmib,vmax=vmab )
    plt.xlim(0.,1.)
    plt.ylim(0,1.)
    plt.axis('off')
    plt.text(0.01,0.3,'t='+str('%.1f' %tim)+' Yr',fontsize=48)
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "4%", pad="3%")
    CB1=plt.colorbar(CS2, cax=cax)
    #plt.tight_layout()
    CB1.set_label(barti1,fontsize=30)
    plt.savefig('P'+str(i)+'.png',format='png');


def plpcon2xL(x,y,Z1,lev1,x2,y2,Z2,vmi,vma,xti1,yti1,tit,barti1,tmi,tma,li,la,tr,tet):
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri
    import matplotlib.cm as cmap
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    params = {'axes.labelsize': 24,
              'text.fontsize': 24,
              'legend.fontsize': 20,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'text.usetex': True}
    plt.rcParams.update(params)

    CS1=plt.contour(x,y,Z1, lev1,
                colors='k' # negative contours will be dashed by default
                )
    CS2=plt.pcolor(x2, y2, Z2,#10, # [-1, -0.1, 0, 0.1],
                   cmap=cmap.bwr,vmin=vmi,vmax=vma) #cmap.gray)
    lin=plt.plot(tr,tet,'g')
    plt.setp(lin[0], linewidth = 2)
    xlabel(xti1)
    ylabel(yti1)
    yticks( range(-60,90,30))
    title(tit,fontsize=24)
    plt.ylim(li, la )
    plt.xlim(tmi, tma)
    
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "3%", pad="3%")
    CB=plt.colorbar(CS2, cax=cax)
    CB.set_label(barti1) #,fontsize=24)
    plt.show();
