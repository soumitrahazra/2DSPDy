# 2DSPDy
Dynamo on the spherical surface. The main file is alpfluct.py (employ 990 SPH). Another, alpfluctl.py is for a higher resolution runs (1770 SPH, longer times). 
To run the code, the proper python distribution is needed. You will need numpy,scipy,
astropy and other scientific python tools. To produce the graphic output matplotlib and cartopy were employed.   
The code is provided "as it is" under GPL3.0 license it was used to produce results for the paper 
by V.V. Pipin & A.G. Kosovichev "DOES NONAXISYMMETRIC DYNAMO OPERATE IN THE SUN?". 
Please cite this paper if you use the code in your research. 
The new additions contain the output of the dynamo model with the magnetic buoyancy effect as the prime nonlinearity (mbu). The results can be processed using wavelm.py, see description of data there.

