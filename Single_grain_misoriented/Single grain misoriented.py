# -*- coding: utf-8 -*-
"""
January 2023 new equations

@author: Roy
"""


'''From grain angle 0.80V'''

import pandas as pd
#Read and import excel files
import matplotlib.pyplot as plt
#Matplotlib
import numpy as np
import math
#import scipy
from numpy import trapz
from scipy.integrate import simps
from pylab import *

'''Obtaining the data from the excel files'''
#cdte_spectra = pd.read_excel (r'D:\Matlab\cdte spectra.xlsx')
scaps = pd.read_excel (r'D:\Matlab\orientation no gb\Results nov 29 jdark no angle\scaps.xlsx')
glass = pd.read_excel (r'D:\Matlab\glass.xlsx')
#zno2 = pd.read_excel (r'D:\Matlab\Zno2.xlsx')
am4 = pd.read_excel (r'D:\Matlab\Am6.xlsx') #ojo aqui
#am4 = pd.read_excel (r'D:\Matlab\Am4.xlsx')
#am1 = pd.read_excel (r'D:\Matlab\Am1.xlsx')
cdsn = pd.read_excel (r'D:\Matlab\cdsn2.xlsx')
#cdte = pd.read_excel (r'D:\Matlab\cdte2.xlsx')
#zno = pd.read_excel (r'D:\Matlab\Zno.xlsx')
sbse=pd.read_excel (r'D:\Matlab\sbse.xlsx')
#tio2=pd.read_excel (r'D:\Matlab\tio2.xlsx')
tio2=pd.read_excel (r'D:\Matlab\tio2.xlsx')
FTO=pd.read_excel (r'D:\Matlab\FTO.xlsx')

'''Variables to fill'''
c1 = list(range(300,900))
air_n = np.zeros(600)
air_k = np.zeros(600) 
r0 = np.zeros(600)
r1 = np.zeros(600)
r2 = np.zeros(600)
r3 = np.zeros(600)
t0 = np.zeros(600)
t1 = np.zeros(600)
t2 = np.zeros(600)
t3 = np.zeros(600)
ab0 = np.zeros(600)
ab0a = np.zeros(600)
ab1 = np.zeros(600)
ab2 = np.zeros(600)
ab3 = np.zeros(600)
ab4 = np.zeros(600)
ab5 = np.zeros(600)
ab6 = np.zeros(600)
photon_flux = np.zeros(600)
photon_flux_inv = np.zeros(600)
W_photon = np.zeros(600)
photon_energy = np.zeros(600)
photon_test = np.zeros(600)
F_n1 = np.zeros(600)
F_n2 = np.zeros(600)
J_scr = np.zeros(120)
g_rate = np.zeros(120)
J_qnr = np.zeros(120)
dummy1 = np.zeros(600)
dummy3 = np.zeros(600)
dummy8 = np.zeros(600)
dummy9 = np.zeros(600)
Xp_1 = np.zeros(120)
Xp_v = np.zeros(120)
Xp_vi = np.zeros(120)
Xp_2 = np.zeros(120)
Wp = np.zeros(120)
F_top1 = np.zeros(120)
F_bot1 = np.zeros(120)
#F_top2 = np.zeros(120)
F_bot2 = np.zeros(120)
percent = np.zeros(600)

'''Table of parameters'''
#zno_width = 300*(10**-9)
#e_tio2b = 85 #at room temperature? https://www.azom.com/article.aspx?ArticleID=1179
#e_tio2 = 18.9 #at room temperature https://www.hindawi.com/journals/jnm/2014/124814/#
#e_cds = 5.4 from caraghs document
#e_cds = 10 #From https://iopscience.iop.org/article/10.1088/2053-1591/ab5fa7/pdf
#e_cdte = 13.6
#tio2 https://materialsproject.org/materials/mp-2657/
#e_tio2c = 8.10 #rutile ^
#e_sbseb= 15.76 #https://materialsproject.org/materials/mp-2160/
#cdte_gap = 1.3
#cds_gap = 2.4
#kai_cdte = 3.9
#kai_tio2= 4.0 #eV https://www.researchgate.net/publication/338496594_Photocurrent_spectra_for_above_and_below_bandgap_energies_from_photovoltaic_PbS_infrared_detectors_with_graphene_transparent_electrodes
#muh_p1 = 1.17 #hole mobility
#muh_p2 = 0.69
#muh_p3 = 2.59
#Na = 10*(10**13) # From https://iopscience.iop.org/article/10.1088/2053-1591/ab5fa7/pdf
#Nd = 10*(10**18) # From ^ same document 

#Sb2Se3 properties
sb2se3_gap = 1.17
p_width = 3000*(10**-9) #Sb2Se3 3 micrometers
kai_sbse= 4.04 #from table   
#e_p = 18 #dielectric constant
e_sbse = 18
nc_sbse = 2.2*(10**18) # 1/cm^3
nv_sbse = 1.8*(10**19) # 1/cm^3
mue_p = 16.9 #electron mobility
muh_p = 5.1 #hole mobility
Na =  1*(10**16) #From Yang paper
Tn_p = 1*(10**-9) #lifetime s
T = 300

#TiO2 properties
n_width = 50*(10**-9) #TiO2 50 nm
tio2_gap = 3.2
kai_tio2 = 4.2
e_tio2z = 10 #10  #at room temperature https://www.hindawi.com/journals/jnm/2014/124814/# #20 #at 1 MHz https://www.osti.gov/pages/servlets/purl/1351943
nc_tio2 = 2.2*(10**18) # 1/cm^3
nv_tio2 = 6.0*(10**17) # 1/cm^3 https://www.sciencedirect.com/science/article/pii/S2214785320335343
Nd_tio2 = 1*(10**17) # From  https://core.ac.uk/download/pdf/147899385.pdf

#FTO properties
fto_width = 500*(10**-9) #500 nm FTO
glass_width = 2*(10**-3) #500 nm FTO


#Tn_p = 67*(10**-9) #lifetime s
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9503613/

'''Extra parameters'''
#mu_h_cdte = 25
#mu_h_cds = 40
#mu_1 = 100 #mobility

'''Values to calculate width of SCR and Vbi'''
e = 1.6*(10**(-19))
k = 1.38064852*(10**(-23))
e_0 = 8.85418782 *(10**-14) #farads per cm
echarge = 1.602176634*(10**-19) #C
emass = 9.10938356*(10**-31) #kg
emassjc = 8.1871057769*(10**-14) #J/c^2
emassevc = 0.510998950 # Mev/c^2
h = 6.62607015*(10**-34) #J*s
h_bar = h/(2*np.pi)
hev = 4.135667696*(10**-15) #eV*s
k = 1.380649*(10**-23) #J/K
kev = 8.617333262145*(10**-5) #eV/K
c = (3*(10**8))

'''Data from Yang paper Sb2Se3'''
me_p = 0.67*emass #electron effective mass obtained from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7372245/
mh_p = 3.32*emass #hole effective mass
me_p_tio2 = 10*emass #electron effective mass obtained from https://pubs.acs.org/doi/pdf/10.1021/jp951142w?rand=ciz52uuf
mh_p_tio2 = 0.8*emass #hole effective mass



'''Orientation'''
#angu = 88 #papu
#name = '%a grain.png' %angu

S_b = 1*(10**3) #cm s^-1
S_gb = 1*(10**3) #cm/s
S_gb2 = 1*(10**7) #cm/s
angle = 44
ang = (np.pi/180)
angu = angle

'''Values to interpolate (300-900 nm), interpolated values x, interpolated values y'''
''' Interpolation if required'''
a3 = np.interp(c1, sbse.iloc[:,0], sbse.iloc[:,1])
a4 = np.interp(c1, sbse.iloc[:,0], sbse.iloc[:,2])
a1 = np.interp(c1, tio2.iloc[:,0], tio2.iloc[:,1])
a2 = np.interp(c1, tio2.iloc[:,0], tio2.iloc[:,2])
a5 = np.interp(c1, FTO.iloc[:,0], FTO.iloc[:,1])
a6 = np.interp(c1, FTO.iloc[:,0], FTO.iloc[:,2])
a7 = np.interp(c1, glass.iloc[:,0], glass.iloc[:,1])
a8 = np.interp(c1, glass.iloc[:,2], glass.iloc[:,3])


'''Irradiance to photon flux (cm^-2 s^-1)'''
for x in range(0,600):
    photon_flux[x] = ((am4.iloc[x,0]/1000)*am4.iloc[x,2])/(echarge*1.24) #https://www.pveducation.org/pvcdrom/properties-of-sunlight/photon-flux units are m-2 s-1
    W_photon[x] = (h*c)/((am4.iloc[x,0]*(10**-9)))
    photon_energy[x] = ((am4.iloc[x,2])/W_photon[x])/(10000)
    photon_test[x] = photon_flux[x]/photon_energy[x]


'''Reflection and absorption of all the different layers'''

'''Air/Glass'''
for x in range(0,600):
    air_n[x] = 1
    air_k[x] = 0
    ca = complex(air_n[x], air_k[x]) #Air data
    cb = complex(a7[x], a8[x]) #TiO2 interpolated data
    r0[x] = (abs((ca - cb)/(ca + cb)))**2
    t0[x] = (1 - r0[x])
    ab0[x] = (4*np.pi*a8[x])/((cdsn.iloc[x,0])*(10**-9))
    ab0a[x] = np.exp(-ab0[x]*glass_width)
    ca = 0
    cb = 0
#ref1 = np.multiply(am4.iloc[0:600,2],t1) #Using W/m^2 nm
ref0 = np.multiply(photon_energy[0:600],t0) # m^-2 s^-1
#ref0 = np.multiply(spec2[0:600],t0) # m^-2 s^-1
abso0 = np.multiply(ref0,ab0a)

'''Glass/FTO'''
for x in range(0,600):
    ca = complex(a7[x], a8[x]) #Air data
    cb = complex(a5[x],a6[x]) #TiO2 interpolated data
    r1[x] = (abs((ca - cb)/(ca + cb)))**2
    t1[x] = (1 - r1[x])
    ab1[x] = (4*np.pi*a6[x])/((cdsn.iloc[x,0])*(10**-9))
    ab2[x] = np.exp(-ab1[x]*fto_width)
    ca = 0
    cb = 0
#ref1 = np.multiply(am4.iloc[0:600,2],t1) #Using W/m^2 nm
ref1 = np.multiply(abso0,t1) # m^-2 s^-1
abso1 = np.multiply(ref1,ab2)

'''FTO/TiO2'''
for x in range(0,600):
    ca = complex(a5[x], a6[x])
    cb = complex(a1[x], a2[x])
    r2[x] = (abs((ca - cb)/(ca + cb)))**2
    t2[x] = (1 - r2[x])
    ab3[x] = (4*np.pi*a2[x])/((cdsn.iloc[x,0])*(10**-9))
    ab4[x] = np.exp(-ab3[x]*n_width)
    ca = 0
    cb = 0
ref2 = np.multiply(abso1,t2)
abso2 = np.multiply(ref2,ab4)


'''TiO2/Sb2Se3'''
for x in range(0,600):
    ca = complex(a1[x], a2[x])
    cb = complex(a3[x], a4[x])
    r3[x] = (abs((ca - cb)/(ca + cb)))**2
    t3[x] = (1 - r3[x])
    ab5[x] = (4*np.pi*a4[x])/((cdsn.iloc[x,0])*(10**-9))
    ab6[x] = np.exp(-ab5[x]*p_width)
    ca = 0
    cb = 0   
ref3 = np.multiply(abso2,t3) #Reflection after the interface but before absorption at Sb2Se3
#ref3 = photon_energy
#ref3 = np.multiply(photon_energy[0:600],t3) #ojo
#ref3 = ref4/0.72 #ojo
#ref3 = np.multiply(photon_energy[0:600],t3) #ojo
abso3 = np.multiply(ref3,ab6)

ab0.tofile('glass.csv',sep=',',format='%10.5f')
ab1.tofile('FTO.csv',sep=',',format='%10.5f')
ab3.tofile('TiO2.csv',sep=',',format='%10.5f')
ab5.tofile('Sb2Se3.csv',sep=',',format='%10.5f')
t1.tofile('Air-FTO trans.csv',sep=',',format='%10.5f')
ref3.tofile('ref3.csv',sep=',',format='%10.5f')
#reftest3.tofile('reftest3.csv',sep=',',format='%10.5f')
#cdsn.iloc[:,0].tofile('wavelength.csv',sep=',',format='%10.5f')
percent = np.zeros(600)
photon_flux_inv = np.zeros(600)

for i in range(0,600):
    percent[i] = ref3[i]/photon_energy[i]
    photon_flux_inv[i] = (ref3[i]*(echarge*1.24))/((am4.iloc[x,0]/1000))*100*100

photon_flux_inv.tofile('spect.csv',sep=',',format='%10.5f')

position_new = list(range(0,3000,10))
ab_new = np.zeros((300, 600))
ab_sum = np.zeros((300, 600))
intensity_new = np.zeros(300)

for i in range(0, 300):
    for j in range(0,600):
        ab_new[i,j] = np.exp(-ab5[j]*position_new[i]*(10**-9))
        ab_sum[i,j] = ab_new[i,j]*ref3[j]
    intensity_new[i] = sum(ab_sum[i,:])

plt.figure(5020)
plt.plot(position_new[0:300], intensity_new[0:300], 'r-', label='PFD at back contact', linewidth=1)
#plt.legend(loc='lower left', bbox_to_anchor=(1,0.5))
plt.xlabel('Position (nm)', fontsize=12)
plt.ylabel('Total photon flux density (cm$^{-2}$ s$^{-1}$ nm$^{-1}$)', fontsize=12) #Using m^-2 s^-1
#plt.title('Solar spectra after reflection and absorptions')
plt.savefig('spectnew', dpi=300, bbox_inches="tight")


'''Solar spectra after reflection and absorption'''
plt.figure(1)
#plt.plot(c1, photon_energy[0:600], 'k-', label='AM 1.5G', linewidth=1) #Using m^-2 s^-1
plt.plot(c1, photon_energy, 'k-', label='AM 1.5G', linewidth=1) #Using m^-2 s^-1
plt.plot(c1, abso0, 'c-', label='PFD at glass/FTO interface', linewidth=1)
plt.plot(c1, abso1, 'g-', label='PFD at FTO/TiO$_2$ interface', linewidth=1)
plt.plot(c1, abso2, 'b-', label='PFD at TiO2/Sb$_2$Se$_3$', linewidth=1)
plt.plot(c1, abso3, 'r-', label='PFD at back contact', linewidth=1)
plt.legend(loc='lower left', bbox_to_anchor=(1,0.5))
plt.xlabel('Wavelength (nm)')
plt.ylabel('Photon flux density (cm$^{-2}$ s$^{-1}$ nm$^{-1}$)') #Using m^-2 s^-1
#plt.title('Solar spectra after reflection and absorptions')
plt.savefig('spect', dpi=300, bbox_inches="tight")


plt.figure(0)
#plt.plot(c1, photon_energy[0:600], 'k-', label='AM 1.5G', linewidth=1) #Using m^-2 s^-1
plt.plot(c1, ab0/100, 'c-', label='Glass', linewidth=1)
plt.plot(c1, ab1/100, 'g-', label='FTO', linewidth=1)
plt.plot(c1, ab3/100, 'b-', label='TiO$_2$', linewidth=1)
plt.plot(c1, ab5/100, 'r-', label='Sb$_2$Se$_3$', linewidth=1)
plt.legend(loc='lower left', bbox_to_anchor=(1,0.5))
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorption (cm$^{-1}$)') #Using m^-2 s^-1
plt.yscale('log',base=10)
#plt.ylim(0, (10^6)) 
#plt.title('Solar spectra after reflection and absorptions')
plt.savefig('absort', dpi=300, bbox_inches="tight")


'''L and D'''
#Dp = (muh_p1*k*T)/echarge
#Lp = math.sqrt(Tp*Dp)
Dn_1 = (mue_p*k*T)/echarge # units are cm2
Dn = Dn_1/10000 #units are m2
Ln_1 = (math.sqrt(Tn_p*Dn_1)) #units are cm
Ln = Ln_1/100 # units are m

    
'''Ni probably wrong'''
ni_cdte = np.sqrt(nc_sbse*nv_sbse)*np.exp((-1*sb2se3_gap)/(2*kev*T)) #Is this 2 here right? Yes it is
ni = nc_sbse*nv_sbse*np.exp(-(sb2se3_gap/(kev*T))) #This ni is actually ni^2
#V_bi3 = sb2se3_gap + (kai_tio2 - kai_sbse) + ((k*T)/echarge)*math.log((Na*Nd_tio2)/(nc_tio2*nv_sbse)) #also same as carraghs document
#V_bi2 = sb2se3_gap - (kai_tio2 - kai_sbse) + ((k*T)/echarge)*math.log((Na*Nd_tio2)/(nc_tio2*nv_sbse)) #Handout2 document
V_bi2 = ((k*T)/echarge)*math.log((Na*Nd_tio2)/(ni)) #common model of the diode
V_binew = ((k*T)/echarge)*math.log((Na*Nd_tio2)/(ni)) - (kai_tio2 - kai_sbse) #common model of the diode
V_bi = ((k*T)/echarge)*math.log((Na*Nd_tio2)/(nc_tio2*nv_sbse)) + (kai_sbse - kai_tio2) + sb2se3_gap
print('Built-in voltage =', V_bi, 'V')

'''SCR width'''
num = 2*V_bi*e_sbse*e_tio2z*e_0*Nd_tio2
deno = echarge*Na*((Na*e_sbse) + (Nd_tio2*e_tio2z))
num3 = 2*V_bi*e_sbse*e_tio2z*e_0*Na
deno3 = echarge*Nd_tio2*((Na*e_sbse) + (Nd_tio2*e_tio2z))
xp = (math.sqrt(num/deno))*(1/100)*10**9
xn = (math.sqrt(num3/deno3))*(1/100)*10**9
print('At Vbi: Xp = ', xp, 'nm', 'Xn = ', xn, 'nm')

'''Charge neutrality on both sides'''
Test1 = Na*xp
Test2 = Nd_tio2*xn
Test3 = Test1/Test2
print('Charge on p/Charge on n =', Test3)

'''Orientation stuff 1'''
Grain_width=1000*(10**(-9))
Grain_height=3000*(10**(-9))


Theta = list(range(1,90))
Theta_ang = np.zeros(89)
Sint = np.zeros(89)
Cost = np.zeros(89)
Tant = np.zeros(89)
Vspace = np.zeros(89)
Vspace_corr = np.zeros(89)
Ribbon_num = np.zeros(89)
Volt = list(range(-10, 109))
V = np.arange(-0.1, 1.09, 0.01)
for i in range(0,119):
    V[i] = Volt[i]/100

#for i in range(0,120):
#    V[i] = V[i]-1

a = 4.03*(10**-10)
b = 11.54*(10**-10)
c = 12.84*(10**-10)
distance = c/2





        
for x in range(0,120):
    #Xp_1[x] = (2*(V_bi - V[x])*e_sbse*e_tio2z*e_0*Nd_tio2)/(echarge*Na*((Na*e_sbse) + (Nd_tio2*e_tio2z)))
    Xp_1[x] = (2*(V_bi - V[x])*e_sbse*e_tio2z*e_0*Nd_tio2)/(echarge*Na*((Na*e_sbse) + (Nd_tio2*e_tio2z)))
    Xp_v[x] = (math.sqrt(abs(Xp_1[x])))*(1/100)*(10**9)
    Xp_vi[x] = (math.sqrt(abs(Xp_1[x])))*(1/100)
    Wp[x] = p_width - (Xp_vi[x])

Xp_3 = np.zeros(120)
Wp_3 = np.zeros(120)
for i in range(0,120):
    Xp_3[i] = V[i]
    Wp_3[i] = Wp[i]*(10**9)

plt.figure(3)
plt.plot(Xp_3, Xp_v, label='Xp Width')
plt.xlabel('V')
plt.ylabel('Xp width (nm)') 
plt.title('Xp as a function of V')
#plt.savefig('10', dpi=300)

plt.figure(200)
plt.plot(Xp_3, Wp_3, label='Xp Width')
plt.xlabel('V')
plt.ylabel('QNR height') 
plt.title('Grain height (along X) as a function of V')

##################################################################################################################


'''Constants '''
#Dn_1 = Dn*10000 #Corrected units
#Ln_1 = Ln*100 #units
'''1 is SCR-BS, 2 is SCR-GB, 3 is GB-GB, 4 is GB-BS'''
#Alpha is ab5[x]
#C and F include absorption
#a2 depends on orientation
   

'''################################ General equations calculation #####################################'''
n_0 = ni/Na

'''Trying with the equations from the word document'''
Jdark_sin = np.zeros(120)
Jdark_sin2 = np.zeros(120)

'''Simplifications'''
cosh_bc = np.zeros(120)
sinh_bc = np.zeros(120)


for i in range(0,120):
    cosh_bc[i] = ((S_b*Ln_1)/(Dn_1*np.cos(angle*ang)))*np.cosh(Wp[i]/(Ln*np.cos(angle*ang))) + np.sinh(Wp[i]/(Ln*np.cos(angle*ang)))
    sinh_bc[i] = ((S_b*Ln_1)/(Dn_1*np.cos(angle*ang)))*np.sinh(Wp[i]/(Ln*np.cos(angle*ang))) + np.cosh(Wp[i]/(Ln*np.cos(angle*ang)))

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''
'''Jdark'''
Top21 = np.zeros(120)
Top22 = np.zeros(120)
Bot21 = np.zeros(120)
Bot22 = np.zeros(120)
Term21 = np.zeros(120)
Term22 = np.zeros(120)

for j in range(0,120):
    #Top1 = echarge*(Dn*10000)*(ni_cdte**2)*(np.exp(((echarge*V)/(k*T)) -1))
        #Jdark_sin[j,i] = ((echarge*(Dn*10000)*(ni^2)*(np.exp(((echarge*V[j])/(k*T)) -1)))/(Ln*100*Na))*(((((((S_b/Sint[i])/100)*Ln)/Dn)*np.cosh(p_width/Ln) + np.sinh(p_width/Ln)))/(((((S_b/Sint[i])/100)*Ln)/Dn)*np.sinh(p_width/Ln) + np.cosh(p_width/Ln)))
    #Top21[j] = echarge*(Dn_1)*(ni)*((np.exp(((echarge*V[j])/(k*T)))) -1)
    #Bot21[j] = (Ln_1)*Na
    #Top22[j] = (((((S_b))*Ln_1)/Dn_1*np.sin(angle*ang))*np.cosh(Wp[j]/Ln) + np.sinh(Wp[j]/Ln))
    #Bot22[j] = (((((S_b))*Ln_1)/Dn_1*np.sin(angle*ang))*np.sinh(Wp[j]/Ln) + np.cosh(Wp[j]/Ln))
    #Term21[j] = Top21[j]/Bot21[j]
    T#erm22[j] = Top22[j]/Bot22[j]
    #Jdark_sin2[j] = Term21[j]*Term22[j]
    Jdark_sin[j] = ((echarge*Dn_1*ni*(np.exp(echarge*V[j]/(k*T))-1))/(Ln_1*Na))*(cosh_bc[j]/sinh_bc[j])
    #Jdark_sin[j] = ((echarge*ni*Dn_1*(np.exp(((echarge*V[j])/(k*T))) - 1))/(Ln_1*Na))*((((S_b*Ln_1)/Dn_1*np.cos(angle*ang))*np.cosh(Wp[j]/Ln*np.cos(angle*ang)) + np.sinh(Wp[j]/Ln*np.cos(angle*ang)))/(((S_b*Ln_1)/Dn_1*np.cos(angle*ang))*np.sinh(Wp[j]/Ln*np.cos(angle*ang)) + np.cosh(Wp[j]/Ln*np.cos(angle*ang))))
    #Jdark_sin2[j] = ((echarge*(Dn_1)*(ni)*((np.exp(((echarge*V[j])/(k*T)))) -1))/((Ln_1)*Na))*((((((S_b))*Ln_1)/Dn_1)*np.cosh(Wp[j]/Ln) + np.sinh(Wp[j]/Ln))/((((((S_b))*Ln_1)/Dn_1)*np.sinh(Wp[j]/Ln) + np.cosh(Wp[j]/Ln))))
    #Top21[j] = 0
    #Top22[j] = 0
    #Bot21[j] = 0
    #Bot22[j] = 0
        

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''
'''Jscr''' #Use ref3 as Nph
dummy21 = np.zeros(600)
dummy22 = np.zeros(600)
dummy23 = np.zeros(600)
dummy24 = np.zeros(600)
dummy25 = np.zeros(600)

Jscr_sin = np.zeros(120)
Jscr_sin_testo = np.zeros(120)
for y in range(0,120):
    for x in range(0,600):
        dummy21[x] = (echarge*ref3[x]*(1 - np.exp(-ab5[x]*Xp_vi[y])))
    Jscr_sin[y] = (trapz(dummy21))
    Jscr_sin_testo[y] = np.sum(dummy21)
    #J_scr[y] = (trapz(dummy1))
    for z in range(0,600):
        dummy21[z] = 0

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''            
'''Jqnr'''
Fn1 = np.zeros(120)
Fn2 = np.zeros(120)

Jqnr_sin = np.zeros(120)
Jqnr_sin_test = np.zeros(120)
dummy31 = np.zeros(600)
dummy32 = np.zeros(600)
dummy33 = np.zeros(600)
dummy34 = np.zeros(600)
test_wave2 = np.zeros(600)



#for y in range(0,120):
#    for x in range(0,600):
#        #dummy31[x] = (echarge*ref3[x]*(np.exp(-ab5[x]*Xp_vi[y]))*(ab5[x]*Ln))/((((ab5[x]*Ln)**2) - 1))
#        #dummy32[x] = ((ab5[x]*Ln - (((((((S_b))*Ln_1)/Dn_1*np.sin(angle*ang))*np.cosh(Wp[y]/Ln) + np.sinh(Wp[y]/Ln)))/(((((S_b))*Ln_1)/Dn_1)*np.sinh(Wp[y]/Ln) + np.cosh(Wp[y]/Ln)))) - (((ab5[x]*Ln - (((((S_b))*Ln_1)/Dn_1*np.sin(angle*ang))))/(((((S_b))*Ln_1)/Dn_1*np.sin(angle*ang))*np.sinh(Wp[y]/Ln) + np.cosh(Wp[y]/Ln)))*np.exp(-ab5[x]*Wp[y])))
#        #dummy32[x] = ((Fn1[y]) - ((Fn2[y])*(np.exp(-ab5[x]*Wp[y]))))
#        #dummy33[x] = dummy31[x]*dummy32[x]
#        #print(dummy4)
#        dummy34[x] = (((echarge*(ref3[x])*(np.exp(-ab5[x]*Xp_vi[y]))*((ab5[x]*Ln))))/((((ab5[x]*Ln)**2) - 1)))*((ab5[x]*Ln*np.cos(angle*ang) - (((((((S_b))*Ln_1)/Dn_1*np.cos(angle*ang))*np.cosh(Wp[y]/Ln*np.cos(angle*ang)) + np.sinh(Wp[y]/Ln*np.cos(angle*ang))))/(((((S_b))*Ln_1)/Dn_1*np.cos(angle*ang))*np.sinh(Wp[y]/Ln*np.cos(angle*ang)) + np.cosh(Wp[y]/Ln*np.cos(angle*ang))))) - (((ab5[x]*Ln*np.cos(angle*ang) - (((((S_b))*Ln_1)/Dn_1*np.cos(angle*ang))))/(((((S_b))*Ln_1)/Dn_1*np.cos(angle*ang))*np.sinh(Wp[y]/Ln*np.cos(angle*ang)) + np.cosh(Wp[y]/Ln*np.cos(angle*ang))))*np.exp(-ab5[x]*Wp[y])))
#        dummy32[x] = (((echarge*(ref3[x])*(np.exp(-ab5[x]*Xp_vi[y]))*((ab5[x]*Ln))))/((((ab5[x]*Ln)**2) - 1)))*((ab5[x]*Ln*np.cos(angle*ang) - (cosh_bc[y]/sinh_bc[y])) - ((ab5[x]*Ln*np.cos(angle*ang) - ((S_b*Ln_1)/(Dn_1*np.cos(angle*ang))))/sinh_bc[y])*np.exp(-ab5[x]*Wp[y]))
#    Jqnr_sin[y] = trapz(dummy34)
 #   Jqnr_sin_test[y] = trapz(dummy32)
 #   for x in range(0,600):
 ##       dummy34[x] = 0
    #Jqnr_sin_test[y] = np.sum(dummy33)
#    for z in range(0,600):
#        dummy31[z] = 0
#        dummy32[z] = 0
#        dummy33[z] = 0
#        test_wave2[z] = ab5[z]*(1/100)

'''Jqnr'''
for i in range(0,120):
    for l in range(0,600):
        #dummy_bc1[l] = ((echarge*ref3[l]*Ln*ab5[l]*(np.exp(-ab5[l]*Xp_vi[i])))/(((ab5[l]*Ln)**2) - 1))
        #dummy_bc2[l] = (ab5[l]*Ln*np.cos(angu*ang) - ((((S_b*Ln_1)/(Dn_1*np.cos(angu*ang)))*np.cosh(Wp[i]/(Ln*np.cos(angu*ang))) + np.sinh(Wp[i]/(Ln*np.cos(angu*ang))))/(((S_b*Ln_1)/(Dn_1*np.cos(angu*ang)))*np.sinh(Wp[i]/(Ln*np.cos(angu*ang))) + np.cosh(Wp[i]/(Ln*np.cos(angu*ang))))))
        #dummy_bc3[l] = ((ab5[l]*Ln*np.cos(angu*ang) - ((S_b*Ln_1)/(Dn*np.cos(angu*ang))))/(((S_b*Ln_1)/(Dn_1*np.cos(angu*ang)))*np.sinh(Wp[i]/(Ln*np.cos(angu*ang))) + np.cosh(Wp[i]/(Ln*np.cos(angu*ang)))))
        #dummy_bc4[l] = dummy_bc1[l]*(dummy_bc2[l] - dummy_bc3[l]*np.exp(-ab5[l]*Wp[i]))
        dummy31[l] = (((echarge*(ref3[l])*(np.exp(-ab5[l]*Xp_vi[i]))*((ab5[l]*Ln))))/((((ab5[l]*Ln)**2) - 1)))*((ab5[l]*Ln*np.cos(angu*ang) - (cosh_bc[i]/sinh_bc[i])) - ((ab5[l]*Ln*np.cos(angu*ang) - ((S_b*Ln_1)/(Dn_1*np.cos(angu*ang))))/sinh_bc[i])*np.exp(-ab5[l]*Wp[i]))
    #Jqnr_bc[i] = np.sum(dummy_bc4)
    Jqnr_sin[i] = np.sum(dummy31)
    for m in range(0,600):
        #dummy_bc1[m] = 0
        #dummy_bc2[m] = 0
        #dummy_bc3[m] = 0
        #dummy_bc4[m] = 0
        dummy31[m] = 0


Jtotal_sin = np.zeros(120)
Jtotalma_sin = np.zeros(120)
Jtotalma_sin_ang = np.zeros(120)
Jtotalma_sin_ang2 = np.zeros(120)
Jdarkma_sin = np.zeros(120)
Jdarkma_sin2 = np.zeros(120)
Jqnrma_sin = np.zeros(120)
Jscrma_sin = np.zeros(120)

Jdarkma_sin3 = np.zeros(120)
Jqnrma_sin3 = np.zeros(120)
Jscrma_sin3 = np.zeros(120)
Jqnrma_sin3_test = np.zeros(120)

for y in range(0,120):
    #Jtotal_sin[y,i] = area1[i] - area2[i] - Jdark_sin[y,i]
    Jtotal_sin[y] = Jscr_sin[y] + Jqnr_sin[y] - Jdark_sin[y]
    Jtotalma_sin[y] = Jscr_sin[y]*1000 + Jqnr_sin[y]*1000 - (Jdark_sin[y])*1000
    Jtotalma_sin_ang[y] = Jscr_sin[y]*1000*np.cos(angle*ang) + Jqnr_sin[y]*1000*np.cos(angle*ang) - (Jdark_sin[y])*1000*np.cos(angle*ang)
    Jtotalma_sin_ang2[y] = Jscr_sin[y]*1000*np.cos(angle*ang) + Jqnr_sin[y]*1000*np.cos(angle*ang) - (Jdark_sin[y])*1000*np.cos(angle*ang)
    #Jtotalma_sin[y] = Jscr_sin[y]*1000 + Jqnr_sin[y]*1000 - Jdark_sin[y]*1000 #ojo
    Jdarkma_sin[y] = Jdark_sin[y]*1000
    #Jdarkma_sin[y] = Jdark_sin[y]*1000 #ojo
    Jdarkma_sin2[y] = Jdark_sin[y]*1000
    #Jdarkma_sin2[y] = Jdark_sin[y]*1000 #ojo
    Jqnrma_sin[y] = Jqnr_sin[y]*1000
    Jqnrma_sin3_test[y] = Jqnr_sin[y]*1000
    Jscrma_sin[y] = Jscr_sin[y]*1000
    Jqnrma_sin3[y] = Jqnr_sin[y]*1000*np.cos(angle*ang)
    Jscrma_sin3[y] = Jscr_sin[y]*1000*np.cos(angle*ang)
    Jdarkma_sin3[y] = Jdark_sin[y]*1000*np.cos(angle*ang)
    

plt.figure(5003)
plt.plot(V, Jtotalma_sin, label='$J_{Total}$', linewidth=2)
plt.plot(V, Jtotalma_sin_ang, label='Total current orientation')
#plt.plot(V, Jqnrma_sin, label='$J_{QNR}$', linewidth=2)
#plt.plot(V, -Jdarkma_sin, label='$J_{Dark}$', linewidth=2)
#plt.plot(V, Jscrma_sin, label='$J_{SCR}$', linewidth=2)
plt.plot(scaps.iloc[:,0], -scaps.iloc[:,1], label='SCAPS-1D', linewidth=2)
#plt.plot(scaps.iloc[:,0], -scaps.iloc[:,1]*0.7225, label='SCAPS-1D (adjusted)', linewidth=2)
#plt.plot(V, -Jdark, label='J dark')
#plt.plot(V, J_total, label='Jtotal')
plt.legend(loc='best')
plt.xlabel('V')
plt.ylabel('Current density (mA/cm$^2$)') #Using m^-2 s^-1
#plt.title('J-V Curve')
plt.xlim(-0.1, 0.73)
plt.ylim(-5, 35)
plt.savefig('finally36', dpi=300)

Jdarkma_sin.tofile('jdark {}.csv'.format(angle),sep=',',format='%10.5f')
Jqnrma_sin.tofile('jqnr {}.csv'.format(angle),sep=',',format='%10.5f')
Jscrma_sin.tofile('jscr {}.csv'.format(angle),sep=',',format='%10.5f')
Jdarkma_sin3.tofile('jdark {} angle.csv'.format(angle),sep=',',format='%10.5f')
Jqnrma_sin3.tofile('jqnr {} angle.csv'.format(angle),sep=',',format='%10.5f')
Jscrma_sin3.tofile('jscr {} angle.csv'.format(angle),sep=',',format='%10.5f')



'''Efficiency'''
Isc_eff = max(Jtotalma_sin_ang[:])
Vhp_min = min([n for n in Jtotalma_sin_ang[:] if n>0])
Vhp_test = np.where(Jtotalma_sin_ang[:]==Vhp_min)
Voc_eff = V[Vhp_test]
Maxpower_teo = Isc_eff*Voc_eff

Vhp_values = np.arange(Voc_eff-0.01, Voc_eff+0.01, 0.001)
Vhp_inter = np.interp(Vhp_values, V, Jtotalma_sin_ang)
Vhp_min2 = min([n for n in Vhp_inter[:] if n>0])
Vhp_test2 = np.where(Vhp_inter[:]==Vhp_min2)
Voc_eff2 = Vhp_values[Vhp_test2]
Maxpower_teo2 = Isc_eff*Voc_eff2


Maxpower_real1 = np.zeros(120)
for i in range(0,120):
    Maxpower_real1[i] = Jtotalma_sin_ang[i]*V[i]
Maxpower_real = max(Maxpower_real1[:])
Solar_power_mw = 100
Fillfactor = (Maxpower_real/Maxpower_teo2)
Eff = ((Maxpower_real)/Solar_power_mw)*100

print('Fill factor =', 100*Fillfactor, '%')
print('Efficiency =', Eff, '%')
print('Voc =', Voc_eff2*1000, 'mV')
print('Isc =', Isc_eff, 'mA')

Test_a = np.zeros(120)
for i in range(0,120):
    Test_a[i] = Jscrma_sin[i]/Jqnrma_sin[i]
    
plt.figure(520)
plt.plot(V, Test_a, label='$J_{Total}$', linewidth=2)
#plt.plot(V, Jtotalma_sin_ang, label='Total current orientation')
#plt.plot(V, Jqnrma_sin, label='$J_{QNR}$', linewidth=2)
#plt.plot(V, -Jdarkma_sin, label='$J_{Dark}$', linewidth=2)
#plt.plot(V, Jscrma_sin, label='$J_{SCR}$', linewidth=2)
#plt.step(scaps.iloc[:,0], -scaps.iloc[:,1], label='SCAPS-1D', linewidth=2)
#plt.step(scaps.iloc[:,0], -scaps.iloc[:,1]*0.7225, label='SCAPS-1D (adjusted)', linewidth=2)
#plt.plot(V, -Jdark, label='J dark')
#plt.plot(V, J_total, label='Jtotal')
plt.legend(loc='best')
plt.xlabel('Voltage (V)', fontsize=14)
plt.ylabel('Ratio between $J_{SCR}$ and $J_{QNR}$', fontsize=14) #Using m^-2 s^-1
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.title('J-V Curve')
plt.xlim(-0.1, 0.73)
#plt.ylim(-5, 40)
plt.savefig('ratio', dpi=300)

Jtotalma_sin_ang.tofile('{} degs {} Voc.csv'.format(angle, Voc_eff2),sep=',',format='%10.5f')
Jtotalma_sin.tofile('unmod {} degs {} Voc.csv'.format(angle, Voc_eff2),sep=',',format='%10.5f')
#Jtotalma_sin_ang2.tofile('{} degree dark.csv'.format(angle),sep=',',format='%10.5f')
#Jdarkma_sin.tofile('dark.csv'.format(angle),sep=',',format='%10.5f')
#Jtotalma_sin.tofile('unmod current.csv',sep=',',format='%10.5f')

plt.figure(7001)
plt.plot(V, Jtotalma_sin_ang, label='$J_{Total}$', linewidth=2)
#plt.plot(V, Jscrma_sin + Jqnrma_sin - Jdarkma_sin, label='Negative $J_{qnr}$')
plt.plot(V, Jqnrma_sin3, label='$J_{QNR}$', linewidth=2)
plt.plot(V, -Jdarkma_sin3, label='$J_{Dark}$', linewidth=2)
plt.plot(V, Jscrma_sin3, label='$J_{SCR}$', linewidth=2)
#plt.plot(V, Jqnrma_sin3_test, label='$J_{SCR test}$', linewidth=2)
#plt.plot(scaps.iloc[:,0], -scaps.iloc[:,1], label='SCAPS-1D', linewidth=2)
#plt.plot(scaps.iloc[:,0], -scaps.iloc[:,1]*0.7225, label='SCAPS-1D (adjusted)', linewidth=2)
#plt.plot(V, -Jdark, label='J dark')
#plt.plot(V, J_total, label='Jtotal')
plt.legend(loc='best', fontsize=12)
plt.xlabel('Voltage (V)', fontsize=14)
plt.ylabel('Current density (mA/cm$^2$)', fontsize=14) #Using m^-2 s^-1
#plt.title('J-V Curve')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(-0.1, 0.70)
plt.ylim(-5, 25)
plt.savefig('not ideal j-v curve', bbox_inches="tight", dpi=300)

plt.figure(7002)
plt.plot(V, Jtotalma_sin, label='$J_{Total}$', linewidth=2)
#plt.plot(V, Jscrma_sin + Jqnrma_sin - Jdarkma_sin, label='Negative $J_{qnr}$')
plt.plot(V, Jqnrma_sin, label='$J_{QNR}$', linewidth=2)
plt.plot(V, -Jdarkma_sin, label='$J_{Dark}$', linewidth=2)
plt.plot(V, Jscrma_sin, label='$J_{SCR}$', linewidth=2)
#plt.plot(V, Jqnrma_sin3_test, label='$J_{SCR test}$', linewidth=2)
#plt.plot(scaps.iloc[:,0], -scaps.iloc[:,1], label='SCAPS-1D', linewidth=2)
#plt.plot(scaps.iloc[:,0], -scaps.iloc[:,1]*0.7225, label='SCAPS-1D (adjusted)', linewidth=2)
#plt.plot(V, -Jdark, label='J dark')
#plt.plot(V, J_total, label='Jtotal')
plt.legend(loc='best', fontsize=12)
plt.xlabel('Voltage (V)', fontsize=14)
plt.ylabel('Current density (mA/cm$^2$)', fontsize=14) #Using m^-2 s^-1
#plt.title('J-V Curve')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(-0.1, 0.70)
plt.ylim(-5, 25)
plt.savefig('aaa2', bbox_inches="tight", dpi=300)

deg = [0,10,20,30,40,50,60,70,80,85]
ratio = pd.read_excel (r'D:\Matlab\ratio.xlsx')

plt.figure(7003)
plt.plot(ratio.iloc[:,0], ratio.iloc[:,1]/ratio.iloc[:,3], color='black', label='Current density', linewidth=2)
plt.plot(ratio.iloc[:,0], ratio.iloc[:,2]/ratio.iloc[:,3], color='orange', linestyle='dashed', label='Efficiency', linewidth=2)
plt.legend(loc='best', fontsize=12)
plt.xlabel('Misorientation (degress)', fontsize=14)
plt.ylabel('Ratio', fontsize=14) #Using m^-2 s^-1
#plt.title('J-V Curve')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(0.995, 1.05)
#plt.ylim(-5, 25)
plt.savefig('aaa5', bbox_inches="tight", dpi=300)

ratio = pd.read_excel (r'D:\Matlab\ratio2.xlsx')

plt.figure(7004)
plt.plot(ratio.iloc[:,0], ratio.iloc[:,1], color='black', label='Ideal 1D simulation', linewidth=2)
plt.plot(ratio.iloc[:,0], ratio.iloc[:,2], color='orange', linestyle='dashed', label='Symmetric twin GB (1$^o$)', linewidth=2)
#plt.plot(V, Jqnrma_sin3_test, label='$J_{SCR test}$', linewidth=2)
#plt.plot(scaps.iloc[:,0], -scaps.iloc[:,1], label='SCAPS-1D', linewidth=2)
#plt.plot(scaps.iloc[:,0], -scaps.iloc[:,1]*0.7225, label='SCAPS-1D (adjusted)', linewidth=2)
#plt.plot(V, -Jdark, label='J dark')
#plt.plot(V, J_total, label='Jtotal')
plt.legend(loc='best', fontsize=12)
plt.xlabel('Voltage (V)', fontsize=14)
plt.ylabel('Current density (mA/cm$^2$)', fontsize=14) #Using m^-2 s^-1
#plt.title('J-V Curve')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(-0.1, 0.70)
plt.ylim(-5, 25)
plt.savefig('aaa6', bbox_inches="tight", dpi=300)


