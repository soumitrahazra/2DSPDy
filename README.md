# 2DSPDy
Dynamo on the spherical surface. The main file is alpfluct.py (employ 990 SPH). Another, alpfluctl.py is for a higher resolution runs (1770 SPH, longer times). 
To run the code, the proper python distribution is needed. You will need numpy,scipy,
astropy and other scientific python tools. To produce the graphic output matplotlib and cartopy were employed.   
The code is provided "as it is" under GPL3.0 license it was used to produce results for the paper 
by V.V. Pipin & A.G. Kosovichev "DOES NONAXISYMMETRIC DYNAMO OPERATE IN THE SUN?". 
Please cite this paper if you use the code in your research. The observational data from KPO/SOLIS and SDO/HMI were processed with help of  nxpsp.py and nxsph.py.  
Specifically, results of the models M2c and M2d (with super-cycles events) can be reproduced by the code alp_fluctm2c.py. Note that the random realizations of the alpha-effect and the magnetic buoyancy instability differ from run to run. I keep them in my home version of the code. They are to big to keep them on github. 
