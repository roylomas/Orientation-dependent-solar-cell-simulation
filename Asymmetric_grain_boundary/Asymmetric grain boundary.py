# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 21:21:02 2023

Attempting to solve the asymmetric case

@author: Roy
"""

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
#scaps = pd.read_excel (r'D:\Matlab\scapsnew.xlsx')
scaps =pd.read_excel (r'D:\Matlab\scaps 2023\scaps 2023.xlsx')
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
#angle = 90
ang = (np.pi/180)
angu1 = 20 #papu
angu2 = 70
bundle = 10
name1 = '%a grain1.png' %angu1
name2 = '%a grain2.png' %angu2
S_b = 1*(10**7) #cm s^-1
S_gb = 1*(10**3) #cm/s
S_gb2 = 1*(10**7) #cm/s


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


'''Solar spectra after reflection and absorption'''
plt.figure(1)
plt.plot(c1, photon_energy[0:600], 'k-', label='AM 1.5G', linewidth=1) #Using m^-2 s^-1
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
'''Constants '''
#Dn_1 = Dn*10000 #Corrected units
#Ln_1 = Ln*100 #units

    
'''Ni probably wrong'''
ni_cdte = np.sqrt(nc_sbse*nv_sbse)*np.exp((-1*sb2se3_gap)/(2*kev*T)) #Is this 2 here right? Yes it is
ni = nc_sbse*nv_sbse*np.exp(-(sb2se3_gap/(kev*T))) #This ni is actually ni^2
#V_bi3 = sb2se3_gap + (kai_tio2 - kai_sbse) + ((k*T)/echarge)*math.log((Na*Nd_tio2)/(nc_tio2*nv_sbse)) #also same as carraghs document
#V_bi2 = sb2se3_gap - (kai_tio2 - kai_sbse) + ((k*T)/echarge)*math.log((Na*Nd_tio2)/(nc_tio2*nv_sbse)) #Handout2 document
V_bi2 = ((k*T)/echarge)*math.log((Na*Nd_tio2)/(ni)) #common model of the diode
#V_bi = ((k*T)/echarge)*math.log((Na*Nd_tio2)/(ni)) - (kai_tio2 - kai_sbse) #common model of the diode old
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
#Vspace1 = np.zeros(89)
#Vspace2 = np.zeros(89)
Vspace_corr = np.zeros(89)
#Ribbon_num1 = np.zeros(89)
#Ribbon_num2 = np.zeros(89)
#V = np.arange(0, 1.20, 0.01)
Volt = list(range(-10, 109))
V = np.arange(-0.1, 1.09, 0.01)
for i in range(0,119):
    V[i] = Volt[i]/100


a = 4.03*(10**-10) #Pnma
b = 11.54*(10**-10)
c_lat = 12.84*(10**-10)
distance = c_lat/2





Vspace1 = (c_lat/(2*np.cos(angu1*ang)))*bundle #10 ribbons bundle
Vspace2 = (c_lat/(2*np.cos(angu2*ang)))*bundle
Ribbon_num1 = int(Grain_width/Vspace1)
Ribbon_num2 = int(Grain_width/Vspace2)


Y_pos0_88_1 = np.zeros(int(Ribbon_num1))
X_pos0_88_1 = np.zeros(int(Ribbon_num1))
X_pos0_case_1 = np.zeros(int(Ribbon_num1))
Y_pos0_end_1 = np.zeros(int(Ribbon_num1))
Y_pos0_88_2 = np.zeros(int(Ribbon_num2))
X_pos0_88_2 = np.zeros(int(Ribbon_num2))
X_pos0_case_2 = np.zeros(int(Ribbon_num2))
Y_pos0_end_2 = np.zeros(int(Ribbon_num2))

'''Xp as a function of V'''

for i in range(0, (int(Ribbon_num1))):
    Y_pos0_88_1[i] =  (((i+1)*c_lat)/(2*np.cos(angu1*ang)))*10
    if angu1 == 0:
        X_pos0_88_1[i] = Grain_height
        Y_pos0_end_1[i] = Y_pos0_88_1[i]
    else:
        X_pos0_88_1[i] = Y_pos0_88_1[i]/np.tan(angu1*ang)
        if X_pos0_88_1[i] > Grain_height:
            X_pos0_88_1[i] = Grain_height
            X_pos0_case_1[i] = 1
            Y_pos0_end_1[i] = Y_pos0_88_1[i] - (Grain_height*np.tan(angu1*ang))
        else:
            Y_pos0_end_1[i] = 0
            
            
for i in range(0, (int(Ribbon_num2))):
    Y_pos0_88_2[i] =  (((i+1)*c_lat)/(2*np.cos(angu2*ang)))*10
    if angu2 == 0:
        X_pos0_88_2[i] = Grain_height
        Y_pos0_end_2[i] = Y_pos0_88_2[i]
    else:
        X_pos0_88_2[i] = Y_pos0_88_2[i]/np.tan(angu2*ang)
        if X_pos0_88_2[i] > Grain_height:
            X_pos0_88_2[i] = Grain_height
            X_pos0_case_2[i] = 1
            Y_pos0_end_2[i] = Y_pos0_88_2[i] - (Grain_height*np.tan(angu2*ang))
        else:
            Y_pos0_end_2[i] = 0
            
            
        
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
plt.plot(V, Xp_vi*10**9, label='Xp Width')
plt.xlabel('V')
plt.ylabel('Xp width (nm)') 
plt.title('Xp as a function of V')
#plt.savefig('10', dpi=300)

plt.figure(200)
plt.plot(V, Wp*10**9, label='Xp Width')
plt.xlabel('V')
plt.ylabel('QNR height') 
plt.title('Grain height (along X) as a function of V')

##################################################################################################################



'''1 is SCR-BS, 2 is SCR-GB, 3 is GB-GB, 4 is GB-BS'''
#Alpha is ab5[x]
#C and F include absorption
#a2 depends on orientation


 
'''Plot test'''
for i in range(0,(int(Ribbon_num1))):
    point1 = [(Y_pos0_88_1[i]), 0]
    point2 = [Y_pos0_end_1[i], X_pos0_88_1[i]]
    x_values1 = [-point1[0], -point2[0]]
    y_values1 = [point1[1], point2[1]]
#x_values1 = x_values1*-1
    #plt.figure(101, figsize=(10,15))
    plt.figure(101, figsize=(10,15))
    plt.plot(x_values1,y_values1, color='red', linewidth=1)
    #plt.xlim((-1*(10**-6)), (1*(10**-6)))
for i in range(0,(int(Ribbon_num2))):
    point1a = [(Y_pos0_88_2[i]), 0]
    point2a = [Y_pos0_end_2[i], X_pos0_88_2[i]]
    x_values1a = [point1a[0], point2a[0]]
    y_values1a = [point1a[1], point2a[1]]
#x_values1 = x_values1*-1
    #plt.figure(101, figsize=(10,15))
    plt.figure(101, figsize=(10,15))
    plt.plot(x_values1a,y_values1a, color='blue', linewidth=1)
    #plt.xlim((-1*(10**-6)), (1*(10**-6)))
    plt.xlim((-1*(10**-6)), (1*(10**-6)))
    plt.ylim(0, (Wp[10]))
plt.savefig('ribbons {} {}'.format(angu1, angu2), dpi=300)


Xa1 = np.zeros((120,(int(Ribbon_num1))))
Casexa1 = np.zeros((120,(int(Ribbon_num1))))
Xb1 = np.zeros((120,(int(Ribbon_num2))))
Casexb1 = np.zeros((120,(int(Ribbon_num2))))


for i in range(0,120):
    for n in range(0,(int(Ribbon_num1))):
        Xa1[i,n] = X_pos0_88_1[n]
        if Xa1[i,n] >= Wp[i]:
            Xa1[i,n] = Wp[i]
            Casexa1[i,n] = 0
        elif Xa1[i,n] < Wp[i]:
            Casexa1[i,n] = 1
            
for i in range(0,120):
    for n in range(0,(int(Ribbon_num2))):
        Xb1[i,n] = X_pos0_88_2[n]
        if Xb1[i,n] >= Wp[i]:
            Xb1[i,n] = Wp[i]
            Casexb1[i,n] = 0
        elif Xb1[i,n] < Wp[i]:
            Casexb1[i,n] = 1
            
num_ribbon1 = np.zeros(120)
num_ribbon2 = np.zeros(120)            

for i in range(0,120):
    for n in range(0,(int(Ribbon_num1))):
        if Casexa1[i,n] > 0:
            num_ribbon1[i] = n
            
for i in range(0,120):
    for n in range(0,(int(Ribbon_num2))):
        if Casexb1[i,n] > 0:
            num_ribbon2[i] = n
            
print('Red grain angle =', angu1)
print('Blue grain angle =', angu2)
print('# ribbons red =', Ribbon_num1)
print('# ribbons blue =', Ribbon_num2)

if angu1 == 0 or angu2 ==0:
    print('There is no coincidence between ribbons')
    
if X_pos0_88_1[10] > X_pos0_88_2[10]:
    ratio = X_pos0_88_1[10]/X_pos0_88_2[10]
    print('left > right, ratio is', round(ratio, 1))
    bundle1 = bundle
    bundle2 = bundle*round(ratio, 1)
else:
    ratio = X_pos0_88_2[10]/X_pos0_88_1[10]
    print('left < right, ratio is', round(ratio,1))
    bundle1 = bundle*round(ratio, 1)
    bundle2 = bundle
    
    
'''Rebundling'''
Vspace1_new = (c_lat/(2*np.cos((angu1)*ang)))*bundle1 #10 ribbons bundle
Vspace2_new = (c_lat/(2*np.cos((angu2)*ang)))*bundle2
Ribbon_num1_new = int(Ribbon_num1*bundle/bundle1)
Ribbon_num2_new = int(Ribbon_num2*bundle/bundle2)

print('#ribbons left new =', Ribbon_num1_new)
print('#ribbons right new =', Ribbon_num2_new)

 
position_new1 = np.zeros(Ribbon_num1_new)
position_new2 = np.zeros(Ribbon_num2_new)
#position_new1[0] = Xa1[10,0]*bundle/(2*bundle)
#position_new2[0] = Xb1[10,0]*bundle2/(2*bundle)

for i in range(0, Ribbon_num1_new):
    position_new1[i] = Xa1[10,0]*bundle1/(2*bundle) + (Xa1[10,0]*bundle1/(bundle))*(i)

for i in range(0, Ribbon_num2_new):
    position_new2[i] = Xb1[10,0]*bundle2/(2*bundle) + (Xb1[10,0]*bundle2/(bundle))*(i)
    
ribbona_x = list(range(0,Ribbon_num1_new))
ribbonb_x = list(range(0,Ribbon_num2_new))
    
plt.figure(1611)
plt.plot(ribbona_x, position_new1, color='k', label='Left grain')
plt.plot(ribbonb_x, position_new2, '--', color='y', label='right grain')
plt.title('Ending positions')
plt.xlabel('Ribbon bundle number', fontsize=14)
plt.ylabel('Distance from SCE', fontsize=14) #Using m^-2 s^-1
plt.legend(loc='best', fontsize=12)


    
Xa_new = np.zeros((120, Ribbon_num1_new))
Xb_new = np.zeros((120, Ribbon_num2_new))
Casexa_new = np.zeros((120, Ribbon_num1_new))
Casexb_new = np.zeros((120, Ribbon_num2_new))

for i in range(0,120):
    for n in range(0, Ribbon_num1_new):
        if position_new1[n] >= Wp[i]:
            Xa_new[i,n] = Wp[i]
            Casexa_new[i,n] = 0
        elif position_new1[n] < Wp[i]:
            Xa_new[i,n] = position_new1[n]
            Casexa_new[i,n] = 1
            
for i in range(0,120):
    for n in range(0, Ribbon_num2_new):
        if position_new2[n] >= Wp[i]:
            Xb_new[i,n] = Wp[i]
            Casexb_new[i,n] = 0
        elif position_new2[n] < Wp[i]:
            Xb_new[i,n] = position_new2[n]
            Casexb_new[i,n] = 1
            
#Casexa = 1 is SCR-GB
#Casexa = 0 is SCR-BC

'''Ribbon matching'''
if position_new1[-1] > Grain_height and position_new2[-1] > Grain_height and angu1 > 0 and angu2 > 0:
    print('All ribbons on the left are going to have a corresponding ribbon')
    
dummy_newa = np.zeros(Ribbon_num1_new)
dummy_newb = np.zeros(Ribbon_num2_new)
last_ribbon_a = 0
last_ribbon_b = 0
    
if position_new1[-1] > position_new2[-1]:
    for i in range(0, Ribbon_num1_new):
        dummy_newa[i] = abs(position_new1[i] - position_new2[-1])    
else:
    for i in range(0, Ribbon_num2_new):
        dummy_newb[i] = abs(position_new2[i] - position_new1[-1])
        
if position_new1[-1] > position_new2[-1]:
    for i in range(0, Ribbon_num1_new):
        if dummy_newa[i] == min(dummy_newa):
            last_ribbon_a = i
            last_ribbon_b = Ribbon_num2_new
else:
    for i in range(0, Ribbon_num2_new):
        if dummy_newb[i] == min(dummy_newb):
            last_ribbon_a = Ribbon_num1_new
            last_ribbon_b = i
            
if last_ribbon_b != last_ribbon_a:
    if position_new1[-1] > position_new2[-1]:
        last_ribbon_a = last_ribbon_b
    else:
        last_ribbon_b = last_ribbon_a

'''Changing Dn and Ln to include misorientation effect''' #not modified anymore

'''L and D'''
Dn_x1 = ((mue_p*k*T)/echarge) # units are cm2
Dnx1 = (Dn_1/10000) #units are m2
Ln_x1 = ((math.sqrt(Tn_p*Dn_1))) #units are cm
Lnx1 = (Ln_1/100) # units are m
Dn_x2 = ((mue_p*k*T)/echarge) # units are cm2
Dnx2 = (Dn_1/10000) #units are m2
Ln_x2 = ((math.sqrt(Tn_p*Dn_1))) #units are cm
Lnx2 = (Ln_1/100) # units are m

'''Copied code from symmetric case'''

'''################################ General equations calculation #####################################'''
n_0 = ni/Na
ang_bundle = (((bundle1/bundle)*np.sin(angu1*ang)) + ((bundle2/bundle)*np.sin(angu2*ang)))*Dn_x1

'''SCR-GB calculation'''
Jdark_bc1 = np.zeros(120)
Jqnr_bc1 = np.zeros(120)
Jdark_bc2 = np.zeros(120)
Jqnr_bc2 = np.zeros(120)
Jqnr_bctest = np.zeros(120)

inte1 = np.zeros(600)
inte2 = np.zeros(600)
inte3 = np.zeros(600)
inte4 = np.zeros(600)
inte5 = np.zeros(600)
inte6 = np.zeros(600)
inte7 = np.zeros(600)
inte8 = np.zeros(600)
inte9 = np.zeros(600)
Current_new1 = np.zeros((120,Ribbon_num1_new))
Current_new2 = np.zeros((120,Ribbon_num2_new)) 


'''Simplifications'''
cosh_bc1 = np.zeros(120)
sinh_bc1 = np.zeros(120)
cosh_bc2 = np.zeros(120)
sinh_bc2 = np.zeros(120)
cosh_gb1 = np.zeros((120,Ribbon_num1_new))
cosh_gb2 = np.zeros((120,Ribbon_num2_new))
sinh_gb1 = np.zeros((120,Ribbon_num1_new))
sinh_gb2 = np.zeros((120,Ribbon_num2_new))

cosh_gb1b = np.zeros((120,Ribbon_num1_new))
cosh_gb2b = np.zeros((120,Ribbon_num2_new))
sinh_gb1b = np.zeros((120,Ribbon_num1_new))
sinh_gb2b = np.zeros((120,Ribbon_num2_new))
sinh_k = np.zeros((120,Ribbon_num1_new))
sinh_b = np.zeros((120,Ribbon_num1_new))
cosh_k = np.zeros((120,Ribbon_num1_new))
cosh_b = np.zeros((120,Ribbon_num1_new))

cero = 0
#if bundle1 > bundle2:
#    Ar = bundle1/bundle2
#    Br = 1
#else:
#    Ar = 1
#    Br = bundle2/bundle1

if bundle1 > bundle2:
    Ar = bundle1/bundle1
    Br = bundle2/bundle1
else:
    Ar = bundle1/bundle2
    Br = bundle2/bundle2
    
#Ar = bundle1/bundle
#Br = bundle2/bundle


for i in range(0,120):
    cosh_bc1[i] = (S_b*Ln_x1)/(((Dn_x1))*np.cosh(Wp[i]/(Lnx1*np.cos(angu1*ang)))) + np.sinh(Wp[i]/(Lnx1*np.cos(angu1*ang)))
    sinh_bc1[i] = (S_b*Ln_x1)/(((Dn_x1))*np.sinh(Wp[i]/(Lnx1*np.cos(angu1*ang)))) + np.cosh(Wp[i]/(Lnx1*np.cos(angu1*ang)))
    cosh_bc2[i] = (S_b*Ln_x1)/(((Dn_x1))*np.cosh(Wp[i]/(Lnx1*np.cos(angu2*ang)))) + np.sinh(Wp[i]/(Lnx1*np.cos(angu2*ang)))
    sinh_bc2[i] = (S_b*Ln_x1)/(((Dn_x1))*np.sinh(Wp[i]/(Lnx1*np.cos(angu2*ang)))) + np.cosh(Wp[i]/(Lnx1*np.cos(angu2*ang)))
    

for i in range(0,120):
    for j in range(0,Ribbon_num1_new):
            sinh_b[i,j] = np.sinh(Xa_new[i,j]/(Lnx1*np.cos(angu1*ang)))
            cosh_b[i,j] = np.cosh(Xa_new[i,j]/(Lnx1*np.cos(angu1*ang)))

for i in range(0,120):
    for j in range(0, Ribbon_num2_new):           
            sinh_k[i,j] = np.sinh(Xb_new[i,j]/(Lnx1*np.cos(angu2*ang)))
            cosh_k[i,j] = np.cosh(Xb_new[i,j]/(Lnx1*np.cos(angu2*ang)))
    
    
for i in range(0,120):
    for j in range(0, last_ribbon_a):
        if angu1 == 0:
            cero = 0
        else:
            cosh_gb1[i,j] = sinh_k[i,j]*(Ar*np.sin(angu1*ang)*sinh_b[i,j] + Br*np.sin(angu2*ang)*sinh_k[i,j] + ((S_gb*Ln_x1)/Dn_x1)*cosh_b[i,j]) + Br*np.sin(angu2*ang)*cosh_k[i,j]*(cosh_b[i,j] - cosh_k[i,j])
            cosh_gb1b[i,j] = sinh_k[i,j]*(Ar*np.sin(angu1*ang)*sinh_b[i,j] + Br*np.sin(angu2*ang)*sinh_k[i,j] + ((S_gb2*Ln_x1)/Dn_x1)*cosh_b[i,j]) + Br*np.sin(angu2*ang)*(cosh_k[i,j]*(cosh_b[i,j] - cosh_k[i,j]))
            sinh_gb1[i,j] = Ar*np.sin(angu1*ang)*cosh_b[i,j]*sinh_k[i,j] + Br*np.sin(angu2*ang)*cosh_k[i,j]*sinh_b[i,j] + ((S_gb*Ln_x1)/Dn_x1)*sinh_b[i,j]*sinh_k[i,j]
            sinh_gb1b[i,j] = Ar*np.sin(angu1*ang)*cosh_b[i,j]*sinh_k[i,j] + Br*np.sin(angu2*ang)*cosh_k[i,j]*sinh_b[i,j] + ((S_gb2*Ln_x1)/Dn_x1)*sinh_b[i,j]*sinh_k[i,j]


for i in range(0,120):
    for j in range(0, last_ribbon_b):
        if angu2 == 0:
            cero = 0
        else:
            cosh_gb2[i,j] = sinh_b[i,j]*(Ar*np.sin(angu1*ang)*sinh_b[i,j] + Br*np.sin(angu2*ang)*sinh_k[i,j] + ((S_gb*Ln_x1)/Dn_x1)*cosh_k[i,j]) + Ar*np.sin(angu1*ang)*cosh_b[i,j]*(cosh_k[i,j] - cosh_b[i,j])
            cosh_gb2b[i,j] = sinh_b[i,j]*(Ar*np.sin(angu1*ang)*sinh_b[i,j] + Br*np.sin(angu2*ang)*sinh_k[i,j] + ((S_gb2*Ln_x1)/Dn_x1)*cosh_k[i,j]) + Ar*np.sin(angu1*ang)*cosh_b[i,j]*(cosh_k[i,j] - cosh_b[i,j])
            sinh_gb2[i,j] = Ar*np.sin(angu1*ang)*cosh_b[i,j]*sinh_k[i,j] + Br*np.sin(angu2*ang)*cosh_k[i,j]*sinh_b[i,j] + ((S_gb*Ln_x1)/Dn_x1)*sinh_b[i,j]*sinh_k[i,j]
            sinh_gb2b[i,j] = Ar*np.sin(angu1*ang)*cosh_b[i,j]*sinh_k[i,j] + Br*np.sin(angu2*ang)*cosh_k[i,j]*sinh_b[i,j] + ((S_gb2*Ln_x1)/Dn_x1)*sinh_b[i,j]*sinh_k[i,j]
            
#Uncoupled ribbons
uncoup1 = Ribbon_num1_new - last_ribbon_a
uncoup2 = Ribbon_num2_new - last_ribbon_b

if uncoup1 > 0:
    for i in range(0,120):
        for j in range(0, uncoup1):
            cosh_gb1[i,j+last_ribbon_a] = Ar*(np.cos(angu1*ang)*np.sin(angu2*ang) + np.cos(angu2*ang)*np.sin(angu1*ang))*(((((S_gb*Ln_x1)/Dn_x1)*np.cos(angu2*ang))/(Ar*(np.cos(angu1*ang)*np.sin(angu2*ang) + np.cos(angu2*ang)*np.sin(angu1*ang))*Dn_x1))*cosh_b[i,j+last_ribbon_a] + sinh_b[i,j+last_ribbon_a])
            cosh_gb1b[i,j+last_ribbon_a] = Ar*(np.cos(angu1*ang)*np.sin(angu2*ang) + np.cos(angu2*ang)*np.sin(angu1*ang))*(((((S_gb2*Ln_x1)/Dn_x1)*np.cos(angu2*ang))/(Ar*(np.cos(angu1*ang)*np.sin(angu2*ang) + np.cos(angu2*ang)*np.sin(angu1*ang))*Dn_x1))*cosh_b[i,j+last_ribbon_a] + sinh_b[i,j+last_ribbon_a])
            sinh_gb1[i,j+last_ribbon_a] = Ar*(np.cos(angu1*ang)*np.sin(angu2*ang) + np.cos(angu2*ang)*np.sin(angu1*ang))*(((((S_gb*Ln_x1)/Dn_x1)*np.cos(angu2*ang))/(Ar*(np.cos(angu1*ang)*np.sin(angu2*ang) + np.cos(angu2*ang)*np.sin(angu1*ang))*Dn_x1))*sinh_b[i,j+last_ribbon_a] + cosh_b[i,j+last_ribbon_a])
            sinh_gb1b[i,j+last_ribbon_a] = Ar*(np.cos(angu1*ang)*np.sin(angu2*ang) + np.cos(angu2*ang)*np.sin(angu1*ang))*(((((S_gb2*Ln_x1)/Dn_x1)*np.cos(angu2*ang))/(Ar*(np.cos(angu1*ang)*np.sin(angu2*ang) + np.cos(angu2*ang)*np.sin(angu1*ang))*Dn_x1))*sinh_b[i,j+last_ribbon_a] + cosh_b[i,j+last_ribbon_a])
else:
    for i in range(0,120):
        for j in range(0, uncoup2):
            cosh_gb2[i,j+last_ribbon_b] = Br*(np.cos(angu1*ang)*np.sin(angu2*ang) + np.cos(angu2*ang)*np.sin(angu1*ang))*sinh_k[i,j+last_ribbon_b] + ((S_gb*Ln_x1)/Dn_x1)*np.cos(angu1*ang)*cosh_k[i,j+last_ribbon_b]
            cosh_gb2b[i,j+last_ribbon_b] = Br*(np.cos(angu1*ang)*np.sin(angu2*ang) + np.cos(angu2*ang)*np.sin(angu1*ang))*sinh_k[i,j+last_ribbon_b] + ((S_gb2*Ln_x1)/Dn_x1)*np.cos(angu1*ang)*cosh_k[i,j+last_ribbon_b]
            sinh_gb2[i,j+last_ribbon_b] = Br*(np.cos(angu1*ang)*np.sin(angu2*ang) + np.cos(angu2*ang)*np.sin(angu1*ang))* (((((S_gb*Ln_x1)/Dn_x1)*np.cos(angu1*ang))/(Br*(np.cos(angu1*ang)*np.sin(angu2*ang) + np.cos(angu2*ang)*np.sin(angu1*ang))*Dn_x1))*sinh_k[i,j+last_ribbon_b] + cosh_k[i,j+last_ribbon_b])
            sinh_gb2b[i,j+last_ribbon_b] = Br*(np.cos(angu1*ang)*np.sin(angu2*ang) + np.cos(angu2*ang)*np.sin(angu1*ang))* (((((S_gb2*Ln_x1)/Dn_x1)*np.cos(angu1*ang))/(Br*(np.cos(angu1*ang)*np.sin(angu2*ang) + np.cos(angu2*ang)*np.sin(angu1*ang))*Dn_x1))*sinh_k[i,j+last_ribbon_b] + cosh_k[i,j+last_ribbon_b])

#cosh_gb1[i,j+last_ribbon_a] = 2*np.cos(angu2*ang)*((S_gb*Ln_x1)/(bundle1*Dn_x1*(np.sin(angu1*ang)*np.cos(angu2*ang)+np.sin(angu2*ang)*np.cos(angu1*ang))))*np.cosh(Xa_new[i,j+last_ribbon_a]/(Lnx1*np.cos(angu1*ang))) + np.sinh(Xa_new[i,j+last_ribbon_a]/(Lnx1*np.cos(angu1*ang)))


'''SCR-BC current'''
dummy_bc1 = np.zeros(600)
dummy_bc2 = np.zeros(600)
dummy_bc3 = np.zeros(600)
dummy_bc4 = np.zeros(600)
dummy_bc51 = np.zeros(600)
dummy_bc52 = np.zeros(600)



#####################################################################################
'''Jdark'''
for i in range(0,120):
    Jdark_bc1[i] = ((echarge*Dn_x1*ni*(np.exp(echarge*V[i]/(k*T))-1))/(Ln_x1*Na))*(cosh_bc1[i]/sinh_bc1[i])
    Jdark_bc2[i] = ((echarge*Dn_x1*ni*(np.exp(echarge*V[i]/(k*T))-1))/(Ln_x1*Na))*(cosh_bc2[i]/sinh_bc2[i])

'''Jqnr'''
for i in range(0,120):
    for l in range(0,600):
        dummy_bc51[l] = (((echarge*(ref3[l])*(np.exp(-ab5[l]*Xp_vi[i]))*((ab5[l]*Lnx1))))/((((ab5[l]*Lnx1)**2) - 1)))*((ab5[l]*Lnx1*np.cos(angu1*ang) - (cosh_bc1[i]/sinh_bc1[i])) - ((ab5[l]*Lnx1*np.cos(angu1*ang) - ((S_b*Ln_x1)/(Dn_x1))/sinh_bc1[i])*np.exp(-ab5[l]*Wp[i])))
        dummy_bc52[l] = (((echarge*(ref3[l])*(np.exp(-ab5[l]*Xp_vi[i]))*((ab5[l]*Lnx1))))/((((ab5[l]*Lnx1)**2) - 1)))*((ab5[l]*Lnx1*np.cos(angu2*ang) - (cosh_bc2[i]/sinh_bc2[i])) - ((ab5[l]*Lnx1*np.cos(angu2*ang) - ((S_b*Ln_x1)/(Dn_x1))/sinh_bc2[i])*np.exp(-ab5[l]*Wp[i])))
    Jqnr_bc1[i] = np.sum(dummy_bc51)
    Jqnr_bc2[i] = np.sum(dummy_bc52)
    for m in range(0,600):
        dummy_bc51[m] = 0
        dummy_bc52[m] = 0
        
        
        
        
'''Jscr'''
Jscr_num1 = np.zeros(120)
Jscr_num21 = np.zeros(120)
dummy_scr1 = np.zeros(600)
Jscr_num2 = np.zeros(120)
Jscr_num22 = np.zeros(120)
dummy_scr2 = np.zeros(600)

for i in range(0,120):
    for l in range(0,600):
        dummy_scr1[l] = (echarge*ref3[l]*(1 - np.exp(-ab5[l]*Xp_vi[i])))
        dummy_scr2[l] = (echarge*ref3[l]*(1 - np.exp(-ab5[l]*Xp_vi[i])))
    Jscr_num21[i] = (trapz(dummy_scr1))*np.cos(angu1*ang)
    Jscr_num1[i] = (sum(dummy_scr1))*np.cos(angu1*ang)
    Jscr_num22[i] = (trapz(dummy_scr2))*np.cos(angu2*ang)
    Jscr_num2[i] = (sum(dummy_scr2))*np.cos(angu2*ang)
    for z in range(0,600):
        dummy_scr1[z] = 0
        dummy_scr2[z] = 0
        
####################################################################################

'''SCR-GB current '''
Jdark_gb1 = np.zeros((120, Ribbon_num1_new))
Jqnr_gb1 = np.zeros((120, Ribbon_num1_new))
Jqnr_gb1_test = np.zeros((120, Ribbon_num1_new))
dummy_qnr1a = np.zeros(600)
dummy_qnr1b = np.zeros(600)
dummy_qnr1c = np.zeros(600)
dummy_qnr1d = np.zeros(600)
dummy_qnr2a = np.zeros(600)
dummy_qnr2b = np.zeros(600)
dummy_qnr2c = np.zeros(600)
dummy_qnr2d = np.zeros(600)
Jdark_gb2 = np.zeros((120, Ribbon_num2_new))
Jqnr_gb2 = np.zeros((120, Ribbon_num2_new))
Jqnr_gb2_test = np.zeros((120, Ribbon_num2_new))
dummy_qnr2 = np.zeros(600)

Jdark_gb1b = np.zeros((120, Ribbon_num1_new))
Jqnr_gb1b = np.zeros((120, Ribbon_num1_new))
Jqnr_gb1_testb = np.zeros((120, Ribbon_num1_new))
dummy_qnr1ab = np.zeros(600)
dummy_qnr1bb = np.zeros(600)
dummy_qnr1cb = np.zeros(600)
dummy_qnr1db = np.zeros(600)
dummy_qnr2ab = np.zeros(600)
dummy_qnr2bb = np.zeros(600)
dummy_qnr2cb = np.zeros(600)
dummy_qnr2db = np.zeros(600)
Jdark_gb2b = np.zeros((120, Ribbon_num2_new))
Jqnr_gb2b = np.zeros((120, Ribbon_num2_new))
Jqnr_gb2_testb = np.zeros((120, Ribbon_num2_new))
dummy_qnr2b = np.zeros(600)



for i in range(0,120): #fix this
    for j in range(0, last_ribbon_a):
        if Casexa_new[i,j] == 1: #this part is wrong
            Jdark_gb1[i,j] = ((echarge*Dn_x1*ni*(np.exp(echarge*V[i]/(k*T))-1))/(Ln_x1*Na))*(cosh_gb1[i,j]/sinh_gb1[i,j])
            Jdark_gb1b[i,j] = ((echarge*Dn_x1*ni*(np.exp(echarge*V[i]/(k*T))-1))/(Ln_x1*Na))*(cosh_gb1b[i,j]/sinh_gb1b[i,j])
            
for i in range(0,120): #fix this
    for j in range(0, last_ribbon_b):
        if Casexb_new[i,j] == 1: #this part is wrong
            Jdark_gb2[i,j] = ((echarge*Dn_x1*ni*(np.exp(echarge*V[i]/(k*T))-1))/(Ln_x1*Na))*(cosh_gb2[i,j]/sinh_gb2[i,j])
            Jdark_gb2b[i,j] = ((echarge*Dn_x1*ni*(np.exp(echarge*V[i]/(k*T))-1))/(Ln_x1*Na))*(cosh_gb2b[i,j]/sinh_gb2b[i,j])
            
if uncoup1 > 0:
    for i in range(0,120):
        for j in range(0, uncoup1):
            if Casexa_new[i,j+last_ribbon_a] == 1: #this part is wrong
                Jdark_gb1[i,j+last_ribbon_a] = ((echarge*Dn_x1*ni*(np.exp(echarge*V[i]/(k*T))-1))/(Ln_x1*Na))*(cosh_gb1[i,j+last_ribbon_a]/sinh_gb1[i,j+last_ribbon_a])
                Jdark_gb1b[i,j+last_ribbon_a] = ((echarge*Dn_x1*ni*(np.exp(echarge*V[i]/(k*T))-1))/(Ln_x1*Na))*(cosh_gb1b[i,j+last_ribbon_a]/sinh_gb1b[i,j+last_ribbon_a])
else:
    for i in range(0,120):
        for j in range(0, uncoup2):
            if Casexb_new[i,j+last_ribbon_b] == 1: #this part is wrong
                Jdark_gb2[i,j+last_ribbon_b] = ((echarge*Dn_x1*ni*(np.exp(echarge*V[i]/(k*T))-1))/(Ln_x1*Na))*(cosh_gb2[i,j+last_ribbon_b]/sinh_gb2[i,j+last_ribbon_b])      
                Jdark_gb2b[i,j+last_ribbon_b] = ((echarge*Dn_x1*ni*(np.exp(echarge*V[i]/(k*T))-1))/(Ln_x1*Na))*(cosh_gb2b[i,j+last_ribbon_b]/sinh_gb2b[i,j+last_ribbon_b])



for i in range(0,120): #fix this
    for j in range(0, last_ribbon_a):
        if Casexa_new[i,j] == 1:
            for l in range(0,600):
                dummy_qnr1a[l] = (((echarge*(ref3[l])*(np.exp(-ab5[l]*Xp_vi[i]))*((ab5[l]*Lnx1))))/((((ab5[l]*Lnx1)**2) - 1)))
                #dummy_qnr1ab[l] = (((echarge*(ref3[l])*(np.exp(-ab5[l]*Xp_vi[i]))*((ab5[l]*Lnx1))))/((((ab5[l]*Lnx1)**2) - 1)))
                dummy_qnr1b[l] = ((ab5[l]*Lnx1*np.cos(angu1*ang) - (cosh_gb1[i,j]/sinh_gb1[i,j])))
                dummy_qnr1bb[l] = ((ab5[l]*Lnx1*np.cos(angu1*ang) - (cosh_gb1b[i,j]/sinh_gb1b[i,j])))
                #dummy_qnr1c[l] = ((ab5[l]*Lnx1*np.cos(angu1*ang) - ((S_gb*Ln_x1)/ang_bundle))/sinh_gb1[i,j])*np.exp(-ab5[l]*Xa_new[i,j])
                dummy_qnr1c[l] = (sinh_k[i,j]*((ab5[l]*Lnx1*(Ar*np.sin(angu1*ang)*np.cos(angu1*ang) + Br*np.sin(angu2*ang)*np.cos(angu2*ang))) - ((S_gb*Ln_x1)/Dn_x1)))/sinh_gb1[i,j]
                dummy_qnr1cb[l] = (sinh_k[i,j]*((ab5[l]*Lnx1*(Ar*np.sin(angu1*ang)*np.cos(angu1*ang) + Br*np.sin(angu2*ang)*np.cos(angu2*ang))) - ((S_gb2*Ln_x1)/Dn_x1)))/sinh_gb1b[i,j]
                dummy_qnr1d[l] = dummy_qnr1a[l]*(dummy_qnr1b[l] - (dummy_qnr1c[l]*np.exp(-ab5[l]*Xa_new[i,j])))
                dummy_qnr1db[l] = dummy_qnr1a[l]*(dummy_qnr1bb[l] - (dummy_qnr1cb[l]*np.exp(-ab5[l]*Xa_new[i,j])))
            Jqnr_gb1[i,j] = np.sum(dummy_qnr1d)
            Jqnr_gb1b[i,j] = np.sum(dummy_qnr1db)
            #Jqnr_gb2[i,j] = np.sum(dummy_qnr2)
            for z in range(0,600):
                dummy_qnr1a[z] = 0
                dummy_qnr1b[z] = 0
                dummy_qnr1c[z] = 0
                dummy_qnr1d[z] = 0
                #dummy_qnr1ab[z] = 0
                dummy_qnr1bb[z] = 0
                dummy_qnr1cb[z] = 0
                dummy_qnr1db[z] = 0
                #dummy_qnr2[z] = 0
                
for i in range(0,120): #fix this
    for j in range(0, last_ribbon_b):
        if Casexb_new[i,j] == 1:
            for l in range(0,600):
                dummy_qnr2a[l] = (((echarge*(ref3[l])*(np.exp(-ab5[l]*Xp_vi[i]))*((ab5[l]*Lnx1))))/((((ab5[l]*Lnx1)**2) - 1)))
                dummy_qnr2b[l] = ((ab5[l]*Lnx1*np.cos(angu2*ang) - (cosh_gb2[i,j]/sinh_gb2[i,j])))
                #dummy_qnr2ab[l] = (((echarge*(ref3[l])*(np.exp(-ab5[l]*Xp_vi[i]))*((ab5[l]*Lnx1))))/((((ab5[l]*Lnx1)**2) - 1)))
                dummy_qnr2bb[l] = ((ab5[l]*Lnx1*np.cos(angu2*ang) - (cosh_gb2b[i,j]/sinh_gb2b[i,j])))
                #dummy_qnr1c[l] = ((ab5[l]*Lnx1*np.cos(angu1*ang) - ((S_gb*Ln_x1)/ang_bundle))/sinh_gb1[i,j])*np.exp(-ab5[l]*Xa_new[i,j])
                dummy_qnr2c[l] = (sinh_b[i,j]*((ab5[l]*Lnx1*(Ar*np.sin(angu1*ang)*np.cos(angu1*ang) + Br*np.sin(angu2*ang)*np.cos(angu2*ang))) - ((S_gb*Ln_x1)/Dn_x1)))/sinh_gb2[i,j]
                dummy_qnr2cb[l] = (sinh_b[i,j]*((ab5[l]*Lnx1*(Ar*np.sin(angu1*ang)*np.cos(angu1*ang) + Br*np.sin(angu2*ang)*np.cos(angu2*ang))) - ((S_gb2*Ln_x1)/Dn_x1)))/sinh_gb2b[i,j]
                dummy_qnr2d[l] = dummy_qnr2a[l]*(dummy_qnr2b[l] - dummy_qnr2c[l]*np.exp(-ab5[l]*Xb_new[i,j]))
                dummy_qnr2db[l] = dummy_qnr2a[l]*(dummy_qnr2bb[l] - dummy_qnr2cb[l]*np.exp(-ab5[l]*Xb_new[i,j]))
            Jqnr_gb2[i,j] = np.sum(dummy_qnr2d)
            Jqnr_gb2b[i,j] = np.sum(dummy_qnr2db)
            #Jqnr_gb2[i,j] = np.sum(dummy_qnr2)
            for z in range(0,600):
                dummy_qnr2a[z] = 0
                dummy_qnr2b[z] = 0
                dummy_qnr2c[z] = 0
                dummy_qnr2d[z] = 0
                #dummy_qnr2ab[z] = 0
                dummy_qnr2bb[z] = 0
                dummy_qnr2cb[z] = 0
                dummy_qnr2db[z] = 0
                #dummy_qnr2[z] = 0
                
#for i in range(0,120): #fix this
#    for j in range(0, last_ribbon_b):
#        if Casexb_new[i,j] == 1:
#            for l in range(0,600):
#                dummy_qnr2[l] = (((echarge*(ref3[l])*(np.exp(-ab5[l]*Xp_vi[i]))*((ab5[l]*Lnx1))))/((((ab5[l]*Lnx1)**2) - 1)))*((ab5[l]*Lnx1 - (cosh_gb2[i,j]/sinh_gb2[i,j])) - ((ab5[l]*Lnx1 - ((S_gb*Ln_x1)/ang_bundle))/sinh_gb2[i,j])*np.exp(-ab5[l]*Xb_new[i,j]))
#            Jqnr_gb2[i,j] = np.sum(dummy_qnr2)
#            for z in range(0,600):
#                dummy_qnr2[z] = 0
#                #dummy_qnr2[z] = 0

if uncoup1 > 0:
    for i in range(0,120): #fix this
        for j in range(0, uncoup1):
            if Casexa_new[i,j+last_ribbon_a] == 1:
                for l in range(0,600):
                    dummy_qnr1a[l] = (((echarge*(ref3[l])*(np.exp(-ab5[l]*Xp_vi[i]))*((ab5[l]*Lnx1))))/((((ab5[l]*Lnx1)**2) - 1)))
                    dummy_qnr1b[l] = ((ab5[l]*Lnx1*np.cos(angu1*ang) - (cosh_gb1[i,j+last_ribbon_a]/sinh_gb1[i,j+last_ribbon_a])))
                    #dummy_qnr1ab[l] = (((echarge*(ref3[l])*(np.exp(-ab5[l]*Xp_vi[i]))*((ab5[l]*Lnx1))))/((((ab5[l]*Lnx1)**2) - 1)))
                    dummy_qnr1bb[l] = ((ab5[l]*Lnx1*np.cos(angu1*ang) - (cosh_gb1b[i,j+last_ribbon_a]/sinh_gb1b[i,j+last_ribbon_a])))
                    #dummy_qnr1c[l] = ((ab5[l]*Lnx1*np.cos(angu1*ang) - ((S_gb*Ln_x1)/ang_bundle))/sinh_gb1[i,j])*np.exp(-ab5[l]*Xa_new[i,j])
                    dummy_qnr1c[l] = (ab5[l]*Lnx1*np.cos(angu1*ang) - (np.cos(angu2*ang)*S_gb*Ln_x1)/(Ar*(np.cos(angu1*ang)*np.sin(angu2*ang) + np.cos(angu2*ang)*np.sin(angu1*ang))*Dn_x1))/sinh_gb1[i,j+last_ribbon_a]
                    dummy_qnr1cb[l] = (ab5[l]*Lnx1*np.cos(angu1*ang) - (np.cos(angu2*ang)*S_gb2*Ln_x1)/(Ar*(np.cos(angu1*ang)*np.sin(angu2*ang) + np.cos(angu2*ang)*np.sin(angu1*ang))*Dn_x1))/sinh_gb1b[i,j+last_ribbon_a]
                    dummy_qnr1d[l] = dummy_qnr1a[l]*(dummy_qnr1b[l] - dummy_qnr1c[l]*np.exp(-ab5[l]*Xa_new[i,j+last_ribbon_a]))
                    dummy_qnr1db[l] = dummy_qnr1a[l]*(dummy_qnr1bb[l] - dummy_qnr1cb[l]*np.exp(-ab5[l]*Xa_new[i,j+last_ribbon_a]))
                Jqnr_gb1[i,j+last_ribbon_a] = np.sum(dummy_qnr1d)
                Jqnr_gb1_test[i,j] = np.sum(dummy_qnr1d)
                Jqnr_gb1b[i,j+last_ribbon_a] = np.sum(dummy_qnr1db)
                Jqnr_gb1_testb[i,j] = np.sum(dummy_qnr1db)
                #Jqnr_gb2[i,j] = np.sum(dummy_qnr2)
                for z in range(0,600):
                    dummy_qnr1a[z] = 0
                    dummy_qnr1b[z] = 0
                    dummy_qnr1c[z] = 0
                    dummy_qnr1d[z] = 0
                    #dummy_qnr1ab[z] = 0
                    dummy_qnr1bb[z] = 0
                    dummy_qnr1cb[z] = 0
                    dummy_qnr1db[z] = 0
                    #dummy_qnr2[z] = 0
else:
    for i in range(0,120): #fix this
        for j in range(0, uncoup2):
            if Casexb_new[i,j+last_ribbon_b] == 1:
                for l in range(0,600):
                    dummy_qnr2a[l] = (((echarge*(ref3[l])*(np.exp(-ab5[l]*Xp_vi[i]))*((ab5[l]*Lnx1))))/((((ab5[l]*Lnx1)**2) - 1)))
                    dummy_qnr2b[l] = ((ab5[l]*Lnx1*np.cos(angu2*ang) - (cosh_gb2[i,j+last_ribbon_b]/sinh_gb2[i,j+last_ribbon_b])))
                    #dummy_qnr2ab[l] = (((echarge*(ref3[l])*(np.exp(-ab5[l]*Xp_vi[i]))*((ab5[l]*Lnx1))))/((((ab5[l]*Lnx1)**2) - 1)))
                    dummy_qnr2bb[l] = ((ab5[l]*Lnx1*np.cos(angu2*ang) - (cosh_gb2b[i,j+last_ribbon_b]/sinh_gb2b[i,j+last_ribbon_b])))
                    #dummy_qnr1c[l] = ((ab5[l]*Lnx1*np.cos(angu1*ang) - ((S_gb*Ln_x1)/ang_bundle))/sinh_gb1[i,j])*np.exp(-ab5[l]*Xa_new[i,j])
                    dummy_qnr2c[l] = ab5[l]*Lnx1*np.cos(angu2*ang) - (np.cos(angu1*ang)*S_gb*Ln_x1)/(Br*(np.cos(angu1*ang)*np.sin(angu2*ang) + np.cos(angu2*ang)*np.sin(angu1*ang))*Dn_x1)
                    dummy_qnr2cb[l] = ab5[l]*Lnx1*np.cos(angu2*ang) - (np.cos(angu1*ang)*S_gb2*Ln_x1)/(Br*(np.cos(angu1*ang)*np.sin(angu2*ang) + np.cos(angu2*ang)*np.sin(angu1*ang))*Dn_x1)
                    dummy_qnr2d[l] = dummy_qnr2a[l]*(dummy_qnr2b[l] - dummy_qnr2c[l]*np.exp(-ab5[l]*Xb_new[i,j+last_ribbon_b]))
                    dummy_qnr2db[l] = dummy_qnr2a[l]*(dummy_qnr2bb[l] - dummy_qnr2cb[l]*np.exp(-ab5[l]*Xb_new[i,j+last_ribbon_b]))
                Jqnr_gb2[i,j+last_ribbon_b] = np.sum(dummy_qnr2d)
                Jqnr_gb2_test[i,j] = np.sum(dummy_qnr2d)
                Jqnr_gb2b[i,j+last_ribbon_b] = np.sum(dummy_qnr2db)
                Jqnr_gb2_testb[i,j] = np.sum(dummy_qnr2db)
                #Jqnr_gb2[i,j] = np.sum(dummy_qnr2)
                for z in range(0,600):
                    dummy_qnr2a[z] = 0
                    dummy_qnr2b[z] = 0
                    dummy_qnr2c[z] = 0
                    dummy_qnr2d[z] = 0
                    #dummy_qnr2ab[z] = 0
                    dummy_qnr2bb[z] = 0
                    dummy_qnr2cb[z] = 0
                    dummy_qnr2db[z] = 0
                    #dummy_qnr2[z] = 0
                    

'''Current of each ribbon'''
Ribbon_area1 = (c_lat*b/(2*np.cos(angu1*ang)))
Ribbon_area2 = (c_lat*b/(2*np.cos(angu2*ang)))
Curr_ribbon1 = np.zeros(120)
Curr_ribbon_dark1 = np.zeros(120)
Curr_ribbon_qnr1 = np.zeros(120)
Curr_ribbon_scr1 = np.zeros(120)
Curr_ribbon_ang1 = np.zeros(120)
Curr_ribbon2 = np.zeros(120)
Curr_ribbon_dark2 = np.zeros(120)
Curr_ribbon_qnr2 = np.zeros(120)
Curr_ribbon_scr2 = np.zeros(120)
Curr_ribbon_ang2 = np.zeros(120)
Total_current_density1 = np.zeros(120)
Total_current_density2 = np.zeros(120)
Total_current_density3 = np.zeros(120)
dummy_ribbon1 = np.zeros(Ribbon_num1_new)
dummy_ribbon_dark1 = np.zeros(Ribbon_num1_new)
dummy_ribbon_scr1 = np.zeros(Ribbon_num1_new)
dummy_ribbon_qnr1 = np.zeros(Ribbon_num1_new)
dummy_ribbon2 = np.zeros(Ribbon_num2_new)
dummy_ribbon_dark2 = np.zeros(Ribbon_num2_new)
dummy_ribbon_scr2 = np.zeros(Ribbon_num2_new)
dummy_ribbon_qnr2 = np.zeros(Ribbon_num2_new)
Jind_ribbon1 = np.zeros((120, Ribbon_num1_new))
Jind_ribbon2 = np.zeros((120, Ribbon_num2_new))
Jind_ribbon1a = np.zeros((120, Ribbon_num1_new))
Jind_ribbon2a = np.zeros((120, Ribbon_num2_new))

Curr_ribbon1b = np.zeros(120)
Curr_ribbon_dark1b = np.zeros(120)
Curr_ribbon_qnr1b = np.zeros(120)
Curr_ribbon_scr1b = np.zeros(120)
Curr_ribbon_ang1b = np.zeros(120)
Curr_ribbon2b = np.zeros(120)
Curr_ribbon_dark2b = np.zeros(120)
Curr_ribbon_qnr2b = np.zeros(120)
Curr_ribbon_scr2b = np.zeros(120)
Curr_ribbon_ang2b = np.zeros(120)
Total_current_density1b = np.zeros(120)
Total_current_density2b = np.zeros(120)
Total_current_density3b = np.zeros(120)
dummy_ribbon1b = np.zeros(Ribbon_num1_new)
dummy_ribbon_dark1b = np.zeros(Ribbon_num1_new)
dummy_ribbon_scr1b = np.zeros(Ribbon_num1_new)
dummy_ribbon_qnr1b = np.zeros(Ribbon_num1_new)
dummy_ribbon2b = np.zeros(Ribbon_num2_new)
dummy_ribbon_dark2b = np.zeros(Ribbon_num2_new)
dummy_ribbon_scr2b = np.zeros(Ribbon_num2_new)
dummy_ribbon_qnr2b = np.zeros(Ribbon_num2_new)
Jind_ribbon1b = np.zeros((120, Ribbon_num1_new))
Jind_ribbon2b = np.zeros((120, Ribbon_num2_new))
Jind_ribbon1ab = np.zeros((120, Ribbon_num1_new))
Jind_ribbon2ab = np.zeros((120, Ribbon_num2_new))

for i in range(0,120):
    for j in range(0, Ribbon_num1_new):
        if Casexa_new[i,j] == 0:
            dummy_ribbon1[j] = (-Jdark_bc1[i]*Ribbon_area1*bundle1*np.cos(angu1*ang) + Jqnr_bc1[i]*Ribbon_area1*bundle1*np.cos(angu1*ang) + Jscr_num1[i]*Ribbon_area1*bundle1)
            Jind_ribbon1[i,j] = (-Jdark_bc1[i]*Ribbon_area1*bundle1*np.cos(angu1*ang) + Jqnr_bc1[i]*Ribbon_area1*bundle1*np.cos(angu1*ang) + Jscr_num1[i]*Ribbon_area1*bundle1)
            Jind_ribbon1a[i,j] = (-Jdark_bc1[i]*Ribbon_area1*bundle1*np.cos(angu1*ang) + Jqnr_bc1[i]*Ribbon_area1*bundle1*np.cos(angu1*ang))
            dummy_ribbon_dark1[j] = (Jdark_bc1[i]*Ribbon_area1*bundle1)
            dummy_ribbon_qnr1[j] = (Jqnr_bc1[i]*Ribbon_area1*bundle1)
            dummy_ribbon_scr1[j] = (Jscr_num1[i]*Ribbon_area1*bundle1)
            ########
            dummy_ribbon1b[j] = (-Jdark_bc1[i]*Ribbon_area1*bundle1*np.cos(angu1*ang) + Jqnr_bc1[i]*Ribbon_area1*bundle1*np.cos(angu1*ang) + Jscr_num1[i]*Ribbon_area1*bundle1)
            Jind_ribbon1b[i,j] = (-Jdark_bc1[i]*Ribbon_area1*bundle1*np.cos(angu1*ang) + Jqnr_bc1[i]*Ribbon_area1*bundle1*np.cos(angu1*ang) + Jscr_num1[i]*Ribbon_area1*bundle1)
            Jind_ribbon1ab[i,j] = (-Jdark_bc1[i]*Ribbon_area1*bundle1*np.cos(angu1*ang) + Jqnr_bc1[i]*Ribbon_area1*bundle1*np.cos(angu1*ang))
            dummy_ribbon_dark1b[j] = (Jdark_bc1[i]*Ribbon_area1*bundle1)
            dummy_ribbon_qnr1b[j] = (Jqnr_bc1[i]*Ribbon_area1*bundle1)
            dummy_ribbon_scr1b[j] = (Jscr_num1[i]*Ribbon_area1*bundle1)
        else:
            dummy_ribbon1[j] = (-Jdark_gb1[i,j]*Ribbon_area1*bundle1*np.cos(angu1*ang) + Jqnr_gb1[i,j]*Ribbon_area1*bundle1*np.cos(angu1*ang) + Jscr_num1[i]*Ribbon_area1*bundle1)
            Jind_ribbon1[i,j] = (-Jdark_gb1[i,j]*Ribbon_area1*bundle1*np.cos(angu1*ang) + Jqnr_gb1[i,j]*Ribbon_area1*bundle1*np.cos(angu1*ang) + Jscr_num1[i]*Ribbon_area1*bundle1)
            Jind_ribbon1a[i,j] = (-Jdark_gb1[i,j]*Ribbon_area1*bundle1*np.cos(angu1*ang) + Jqnr_gb1[i,j]*Ribbon_area1*bundle1*np.cos(angu1*ang))
            dummy_ribbon_dark1[j] = (Jdark_gb1[i,j]*Ribbon_area1*bundle1)
            dummy_ribbon_qnr1[j] = (Jqnr_gb1[i,j]*Ribbon_area1*bundle1)
            dummy_ribbon_scr1[j] = (Jscr_num1[i]*Ribbon_area1*bundle1)
            #####
            dummy_ribbon1b[j] = (-Jdark_gb1b[i,j]*Ribbon_area1*bundle1*np.cos(angu1*ang) + Jqnr_gb1b[i,j]*Ribbon_area1*bundle1*np.cos(angu1*ang) + Jscr_num1[i]*Ribbon_area1*bundle1)
            Jind_ribbon1b[i,j] = (-Jdark_gb1b[i,j]*Ribbon_area1*bundle1*np.cos(angu1*ang) + Jqnr_gb1b[i,j]*Ribbon_area1*bundle1*np.cos(angu1*ang) + Jscr_num1[i]*Ribbon_area1*bundle1)
            Jind_ribbon1ab[i,j] = (-Jdark_gb1b[i,j]*Ribbon_area1*bundle1*np.cos(angu1*ang) + Jqnr_gb1b[i,j]*Ribbon_area1*bundle1*np.cos(angu1*ang))
            dummy_ribbon_dark1b[j] = (Jdark_gb1b[i,j]*Ribbon_area1*bundle1)
            dummy_ribbon_qnr1b[j] = (Jqnr_gb1b[i,j]*Ribbon_area1*bundle1)
            dummy_ribbon_scr1b[j] = (Jscr_num1[i]*Ribbon_area1*bundle1)
    Curr_ribbon1[i] = np.sum(dummy_ribbon1)
    Curr_ribbon_dark1[i] = np.sum(dummy_ribbon_dark1)/(b*Grain_width)
    Curr_ribbon_qnr1[i] = np.sum(dummy_ribbon_qnr1)/(b*Grain_width)
    Curr_ribbon_scr1[i] = np.sum(dummy_ribbon_scr1)/(b*Grain_width)
    Curr_ribbon_ang1[i] = Curr_ribbon1[i]
    Total_current_density1[i] = Curr_ribbon_ang1[i]/(b*Grain_width)
    ####
    Curr_ribbon1b[i] = np.sum(dummy_ribbon1b)
    Curr_ribbon_dark1b[i] = np.sum(dummy_ribbon_dark1b)/(b*Grain_width)
    Curr_ribbon_qnr1b[i] = np.sum(dummy_ribbon_qnr1b)/(b*Grain_width)
    Curr_ribbon_scr1b[i] = np.sum(dummy_ribbon_scr1b)/(b*Grain_width)
    Curr_ribbon_ang1b[i] = Curr_ribbon1b[i]
    Total_current_density1b[i] = Curr_ribbon_ang1b[i]/(b*Grain_width)
    
    
for i in range(0,120):
    for j in range(0, Ribbon_num2_new):
        if Casexb_new[i,j] == 0:
            dummy_ribbon2[j] = (-Jdark_bc2[i]*Ribbon_area2*bundle2*np.cos(angu2*ang) + Jqnr_bc2[i]*Ribbon_area2*bundle2*np.cos(angu2*ang) + Jscr_num2[i]*Ribbon_area2*bundle2)
            Jind_ribbon2[i,j] = Ar*(-Jdark_bc2[i]*Ribbon_area2*bundle2*np.cos(angu2*ang) + Jqnr_bc2[i]*Ribbon_area2*bundle2*np.cos(angu2*ang) + Jscr_num2[i]*Ribbon_area2*bundle2)
            Jind_ribbon2a[i,j] = Ar*(-Jdark_bc2[i]*Ribbon_area2*bundle2*np.cos(angu2*ang) + Jqnr_bc2[i]*Ribbon_area2*bundle2*np.cos(angu2*ang))
            dummy_ribbon_dark2[j] = (Jdark_bc2[i]*Ribbon_area1*bundle2)
            dummy_ribbon_qnr2[j] = (Jqnr_bc2[i]*Ribbon_area1*bundle2)
            dummy_ribbon_scr2[j] = (Jscr_num2[i]*Ribbon_area1*bundle2) 
            ####
            dummy_ribbon2b[j] = (-Jdark_bc2[i]*Ribbon_area2*bundle2*np.cos(angu2*ang) + Jqnr_bc2[i]*Ribbon_area2*bundle2*np.cos(angu2*ang) + Jscr_num2[i]*Ribbon_area2*bundle2)
            Jind_ribbon2b[i,j] = Ar*(-Jdark_bc2[i]*Ribbon_area2*bundle2*np.cos(angu2*ang) + Jqnr_bc2[i]*Ribbon_area2*bundle2*np.cos(angu2*ang) + Jscr_num2[i]*Ribbon_area2*bundle2)
            Jind_ribbon2ab[i,j] = Ar*(-Jdark_bc2[i]*Ribbon_area2*bundle2*np.cos(angu2*ang) + Jqnr_bc2[i]*Ribbon_area2*bundle2*np.cos(angu2*ang))
            dummy_ribbon_dark2b[j] = (Jdark_bc2[i]*Ribbon_area2*bundle2)
            dummy_ribbon_qnr2b[j] = (Jqnr_bc2[i]*Ribbon_area2*bundle2)
            dummy_ribbon_scr2b[j] = (Jscr_num2[i]*Ribbon_area2*bundle2)
        else:
            dummy_ribbon2[j] = (-Jdark_gb2[i,j]*Ribbon_area2*bundle2*np.cos(angu2*ang) + Jqnr_gb2[i,j]*Ribbon_area2*bundle2*np.cos(angu2*ang) + Jscr_num2[i]*Ribbon_area2*bundle2)
            Jind_ribbon2[i,j] = Ar*(-Jdark_gb2[i,j]*Ribbon_area2*bundle2*np.cos(angu2*ang) + Jqnr_gb2[i,j]*Ribbon_area2*bundle2*np.cos(angu2*ang) + Jscr_num2[i]*Ribbon_area2*bundle2)
            Jind_ribbon2a[i,j] = Ar*(-Jdark_gb2[i,j]*Ribbon_area2*bundle2*np.cos(angu2*ang) + Jqnr_gb2[i,j]*Ribbon_area2*bundle2*np.cos(angu2*ang))
            dummy_ribbon_dark2[j] = (Jdark_gb2[i,j]*Ribbon_area2*bundle2)
            dummy_ribbon_qnr2[j] = (Jqnr_gb2[i,j]*Ribbon_area2*bundle2)
            dummy_ribbon_scr2[j] = (Jscr_num2[i]*Ribbon_area2*bundle2)
            ####
            dummy_ribbon2b[j] = (-Jdark_gb2b[i,j]*Ribbon_area2*bundle2*np.cos(angu2*ang) + Jqnr_gb2b[i,j]*Ribbon_area2*bundle2*np.cos(angu2*ang) + Jscr_num2[i]*Ribbon_area2*bundle2)
            Jind_ribbon2b[i,j] = Ar*(-Jdark_gb2b[i,j]*Ribbon_area2*bundle2*np.cos(angu2*ang) + Jqnr_gb2b[i,j]*Ribbon_area2*bundle2*np.cos(angu2*ang) + Jscr_num2[i]*Ribbon_area2*bundle2)
            Jind_ribbon2ab[i,j] = Ar*(-Jdark_gb2b[i,j]*Ribbon_area2*bundle2*np.cos(angu2*ang) + Jqnr_gb2b[i,j]*Ribbon_area2*bundle2*np.cos(angu2*ang))
            dummy_ribbon_dark2b[j] = (Jdark_gb2b[i,j]*Ribbon_area2*bundle2)
            dummy_ribbon_qnr2b[j] = (Jqnr_gb2b[i,j]*Ribbon_area2*bundle2)
            dummy_ribbon_scr2b[j] = (Jscr_num2[i]*Ribbon_area2*bundle2)
    Curr_ribbon2[i] = np.sum(dummy_ribbon2)
    Curr_ribbon_dark2[i] = np.sum(dummy_ribbon_dark2)/(b*Grain_width)
    Curr_ribbon_qnr2[i] = np.sum(dummy_ribbon_qnr2)/(b*Grain_width)
    Curr_ribbon_scr2[i] = np.sum(dummy_ribbon_scr2)/(b*Grain_width)
    Curr_ribbon_ang2[i] = Curr_ribbon2[i]
    Total_current_density2[i] = Curr_ribbon_ang2[i]/(b*Grain_width)
    Total_current_density3[i] = (Curr_ribbon_ang2[i] + Curr_ribbon_ang1[i])/(2*b*Grain_width)
    ####
    Curr_ribbon2b[i] = np.sum(dummy_ribbon2b)
    Curr_ribbon_dark2b[i] = np.sum(dummy_ribbon_dark2b)/(b*Grain_width)
    Curr_ribbon_qnr2b[i] = np.sum(dummy_ribbon_qnr2b)/(b*Grain_width)
    Curr_ribbon_scr2b[i] = np.sum(dummy_ribbon_scr2b)/(b*Grain_width)
    Curr_ribbon_ang2b[i] = Curr_ribbon2b[i]
    Total_current_density2b[i] = Curr_ribbon_ang2b[i]/(b*Grain_width)
    Total_current_density3b[i] = (Curr_ribbon_ang2b[i] + Curr_ribbon_ang1b[i])/(2*b*Grain_width)
    
'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''
'''Graphs'''
'''I-V'''
plt.figure(5001)
plt.plot(V, Curr_ribbon_dark1*1000, label='J dark left 10^3')
plt.plot(V, Curr_ribbon_dark2*1000, label='J dark right 10^3')
plt.plot(V, Curr_ribbon_dark1b*1000, label='J dark left 10^7')
plt.plot(V, Curr_ribbon_dark2b*1000, label='J dark right10^7')
#plt.plot(V, J_total, label='Jtotal')
plt.legend(loc='lower left')
plt.xlabel('V')
plt.ylabel('Current density (mA cm^-2)') #Using m^-2 s^-1
plt.title('J dark')
#plt.xlim(0, 0.5)
#plt.ylim(-14, 0.5)
plt.savefig('new1', dpi=300)

plt.figure(5002)
plt.plot(V, Curr_ribbon_qnr1*1000, label='J qnr left 10^3')
#plt.plot(V, Curr_ribbon_qnr2*1000, label='J qnr right 10^3')
plt.plot(V, Curr_ribbon_qnr1b*1000, label='J qnr left 10^7')
#plt.plot(V, Curr_ribbon_qnr2b*1000, label='J qnr right 10^7')
#plt.plot(V, J_total, label='Jtotal')
plt.legend(loc='lower left')
plt.xlabel('V')
plt.ylabel('Current density (mA cm^-2)') #Using m^-2 s^-1
plt.title('J qnr')
#plt.xlim(0, 0.5)
#plt.ylim(-14, 0.5)
plt.savefig('new2', dpi=300)

plt.figure(5003)
plt.plot(V, Curr_ribbon_scr1*1000, label='J scr left 10^3')
plt.plot(V, Jscr_num1*1000, '--', label='Unadjusted left')
plt.plot(V, Curr_ribbon_scr2*1000, label='J scr right 10^3')
plt.plot(V, Jscr_num2*1000, '--', label='Unadjusted right')
#plt.plot(V, Curr_ribbon_scr1b*1000, label='J scr left 10^7')
#plt.plot(V, Curr_ribbon_scr2b*1000, label='J scr right 10^7')
#plt.plot(V, J_total, label='Jtotal')
plt.legend(loc='lower left')
plt.xlabel('V')
plt.ylabel('Current density (mA cm^-2)') #Using m^-2 s^-1
plt.title('J scr')
#plt.xlim(0, 0.5)
#plt.ylim(-14, 0.5)
plt.savefig('new3', dpi=300)

plt.figure(5004)
plt.plot(V, Total_current_density1*1000, label='J total left 10^3')
plt.plot(V, Total_current_density2*1000, label='J total right 10^3')
plt.plot(V, Total_current_density3*1000, label='J total both 10^3')
plt.plot(V, Total_current_density1b*1000, label='J total left 10^7')
plt.plot(V, Total_current_density2b*1000, label='J total right 10^7')
plt.plot(V, Total_current_density3b*1000, label='J total both 10^7')
#plt.plot(V, Total_current_density1b*1000, label='J total left 10^7')
#plt.plot(V, Total_current_density2b*1000, label='J total right 10^7')
#plt.plot(V, Total_current_density3b*1000, label='J total both 10^7')
#plt.plot(V, J_total, label='Jtotal')
plt.legend(loc='lower left')
plt.xlabel('V')
plt.ylabel('Current density (mA cm^-2)') #Using m^-2 s^-1
plt.title('J total')
plt.xlim(0, V_bi)
plt.ylim(-20, 50)
plt.savefig('new4', dpi=300)

Curr_ribbon_scr1.tofile('SCR current left {} {} 10^3.csv'.format(angu1, angu2),sep=',')
Curr_ribbon_scr2.tofile('SCR current right {} {} 10^3.csv'.format(angu1, angu2),sep=',')
Curr_ribbon_scr1b.tofile('SCR current left {} {} 10^7.csv'.format(angu1, angu2),sep=',')
Curr_ribbon_scr2b.tofile('SCR current right {} {} 10^7.csv'.format(angu1, angu2),sep=',')


Curr_ribbon_dark1.tofile('Dark current left {} {} 10^3.csv'.format(angu1, angu2),sep=',')
Curr_ribbon_dark2.tofile('Dark current right {} {} 10^3.csv'.format(angu1, angu2),sep=',')
Curr_ribbon_dark1b.tofile('Dark current left {} {} 10^7.csv'.format(angu1, angu2),sep=',')
Curr_ribbon_dark2b.tofile('Dark current right {} {} 10^7.csv'.format(angu1, angu2),sep=',')


Curr_ribbon_qnr1.tofile('QNR current left {} {} 10^3.csv'.format(angu1, angu2),sep=',')
Curr_ribbon_qnr1.tofile('QNR current right {} {} 10^3.csv'.format(angu1, angu2),sep=',')
Curr_ribbon_qnr1b.tofile('QNR current left {} {} 10^7.csv'.format(angu1, angu2),sep=',')
Curr_ribbon_qnr1b.tofile('QNR current right {} {} 10^7.csv'.format(angu1, angu2),sep=',')


Total_current_density1.tofile('Total current left {} {} 10^3.csv'.format(angu1, angu2),sep=',')
Total_current_density2.tofile('Total current right {} {} 10^3.csv'.format(angu1, angu2),sep=',')
Total_current_density3.tofile('Total current both {} {} 10^3.csv'.format(angu1, angu2),sep=',')
Total_current_density1b.tofile('Total current left {} {} 10^7.csv'.format(angu1, angu2),sep=',')
Total_current_density2b.tofile('Total current right {} {} 10^7.csv'.format(angu1, angu2),sep=',')
Total_current_density3b.tofile('Total current both {} {} 10^7.csv'.format(angu1, angu2),sep=',')

import winsound
duration = 10000
freq = 440
winsound.Beep(freq, duration)

Xa_save = np.zeros(Ribbon_num1_new)
Xb_save = np.zeros(Ribbon_num2_new)
Xa_save_xp = np.zeros(Ribbon_num1_new)
Xb_save_xp = np.zeros(Ribbon_num2_new)


plt.figure(6001)
plt.plot(Xa_new[10,:]*10**9, Jind_ribbon1[10,:]*1000*np.cos(angu1*ang)/(b*Grain_width), label='right ribbon 10^3')
plt.plot(Xb_new[10,:]*10**9, Jind_ribbon2[10,:]*1000*np.cos(angu1*ang)/(b*Grain_width), '--', label='left ribbon 10^3')
plt.plot(Xa_new[10,:]*10**9, Jind_ribbon1b[10,:]*1000*np.cos(angu1*ang)/(b*Grain_width), label='right ribbon 10^7')
plt.plot(Xb_new[10,:]*10**9, Jind_ribbon2b[10,:]*1000*np.cos(angu1*ang)/(b*Grain_width), '--', label='left ribbon 10^7')
plt.legend(loc='lower left')
plt.xlabel('Ending position (x)')
plt.ylabel('Current density (mA cm^-2)') #Using m^-2 s^-1
plt.title('J total individual ribbons (10 degrees)')
#plt.xlim(0, V_bi)
#plt.ylim(-20, 45)
plt.savefig('single ribbon total current 10 deg', dpi=300)

Jind_ribbon1[10,:].tofile('Individual current {}-{} deg 10^3 left rib.csv'.format(angu1, angu2),sep=',')
Jind_ribbon2[10,:].tofile('Individual current {}-{} deg 10^3 right rib.csv'.format(angu1, angu2),sep=',')
Jind_ribbon1a[10,:].tofile('Individual qnr current {}-{} deg 10^3 left rib 10^3.csv'.format(angu1, angu2),sep=',')
Jqnr_gb1[10,:].tofile('Jqnr current {}-{} deg left rib 10^3.csv'.format(angu1, angu2),sep=',')
Jdark_gb1[10,:].tofile('Jdark current {}-{} deg left rib 10^7.csv'.format(angu1, angu2),sep=',')
Jind_ribbon2a[10,:].tofile('Individual qnr current {}-{} deg 10^3 right rib 10^3.csv'.format(angu1, angu2),sep=',')
#####
Jind_ribbon1b[10,:].tofile('Individual current {}-{} deg 10^7 left rib.csv'.format(angu1, angu2),sep=',')
Jind_ribbon2b[10,:].tofile('Individual current {}-{} deg 10^7 right rib.csv'.format(angu1, angu2),sep=',')
Jind_ribbon1ab[10,:].tofile('Individual qnr current {}-{} deg 10^7 left rib 10^7.csv'.format(angu1, angu2),sep=',')
Jqnr_gb1b[10,:].tofile('Jqnr current {}-{} deg left rib 107.csv'.format(angu1, angu2),sep=',')
Jdark_gb1b[10,:].tofile('Jdark current {}-{} deg left rib 10^7.csv'.format(angu1, angu2),sep=',')
Jind_ribbon2ab[10,:].tofile('Individual qnr current {}-{} deg 10^7 right rib 10^7.csv'.format(angu1, angu2),sep=',')

for i in range(0, Ribbon_num1_new):
    Xa_save[i] = Xa_new[10,i]
    Xa_save_xp[i] = Xa_new[10,i] + Xp_v[10]*10**-9
    
for i in range(0, Ribbon_num2_new):
    Xb_save[i] = Xb_new[10,i]
    Xb_save_xp[i] = Xb_new[10,i] + Xp_v[10]*10**-9

Xa_new[10,:].tofile('Xa_new {}-{} deg.csv'.format(angu1, angu2),sep=',')
Xb_save.tofile('Xb+Xp {}-{} deg.csv'.format(angu1, angu2),sep=',')
Xb_new[10,:].tofile('Xb_new {}-{} deg.csv'.format(angu1, angu2),sep=',')
Xa_save.tofile('Xa+Xp {}-{} deg.csv'.format(angu1, angu2),sep=',')

plt.figure(6002)
plt.plot(Xa_new[10,:]*10**9, Jind_ribbon1[10,:]*1000/(b*Grain_width)*np.cos(angu2*ang), label='left grain 10^3')
plt.plot(Xa_new[10,:]*10**9, Jind_ribbon1b[10,:]*1000/(b*Grain_width)*np.cos(angu2*ang), label='left grain 10^7')
plt.plot(Xb_new[10,:]*10**9, Jind_ribbon2[10,:]*Ar*1000/(b*Grain_width)*np.cos(angu2*ang), '--', label='right grain 10^3')
plt.plot(Xb_new[10,:]*10**9, Jind_ribbon2b[10,:]*Ar*1000/(b*Grain_width)*np.cos(angu2*ang), '--', label='right grain 10^7')
plt.xlabel('Ending position (x)')
plt.legend(loc='best')
plt.ylabel('Current density (mA cm^-2)') #Using m^-2 s^-1
plt.title('J total individual ribbons (20 degrees)')
#plt.xlim(0, V_bi)
#plt.ylim(-20, 45)
plt.savefig('single ribbon total current 20 deg', dpi=300)

plt.figure(6003)
plt.plot(Xa_new[10,:]*10**9, Jind_ribbon1a[10,:]*1000*np.cos(angu1*ang)/(b*Grain_width), color='k', label='Left grain')
#plt.plot(Xb_new[10,:]*10**9, Jind_ribbon2a[10,:]*1000/(b*Grain_width)*np.cos(angu2*ang), '--', color='y', label='right grain')
plt.plot(Xa_new[10,:]*10**9, Jind_ribbon1ab[10,:]*1000*np.cos(angu1*ang)/(b*Grain_width), color='k', label='Left grain')
#plt.plot(Xb_new[10,:]*10**9, Jind_ribbon2ab[10,:]*1000/(b*Grain_width)*np.cos(angu2*ang), '--', color='y', label='right grain')
plt.legend(loc='best')
plt.xlabel('Distance from the SCE (nm)')
plt.ylabel('Current density (mA cm^-2)') #Using m^-2 s^-1
plt.title('J total')
#plt.xlim(0, V_bi)
#plt.ylim(-20, 45)
plt.savefig('single rib close to scr total current {}-{}'.format(angu1, angu2), dpi=300)


plt.figure(6004)
plt.plot(Xa_new[10,0:140]*10**9, -Jdark_gb1[0,0:140]*Ribbon_area1*bundle1*np.cos(angu1*ang)*1000/(b*Grain_width), label='$S_{GB}$ = 10$^3$ cm/s')
plt.plot(Xa_new[10,0:140]*10**9, -Jdark_gb1b[0,0:140]*Ribbon_area1*bundle1*np.cos(angu1*ang)*1000/(b*Grain_width), label='$S_{GB}$ = 10$^7$ cm/s')
plt.legend(loc='best', fontsize=12)
plt.xlabel('Distance from the SCE (nm)', fontsize=14)
plt.ylabel('Current density (mA/cm$^2$)', fontsize=14) #Using m^-2 s^-1
#plt.title('J total')
plt.yscale('log',base=10)
#plt.xlim(0, V_bi)
#plt.ylim(10**-13, 5*10**-11)
plt.savefig('save Jdark v=0', dpi=300)

plt.figure(6005)
plt.plot(Xa_new[10,0:140]*10**9, Jqnr_gb1[0,0:140]*Ribbon_area1*bundle1*np.cos(angu1*ang)*1000/(b*Grain_width), label='$S_{GB}$ = 10$^3$ cm/s left')
plt.plot(Xa_new[10,0:140]*10**9, Jqnr_gb1b[0,0:140]*Ribbon_area1*bundle1*np.cos(angu1*ang)*1000/(b*Grain_width), label='$S_{GB}$ = 10$^7$ cm/s left')
#plt.plot(Xb_new[10,0:140]*10**9, Jqnr_gb2[0,0:140]*Ribbon_area2*bundle1*np.cos(angu2*ang)*1000/(b*Grain_width), label='$S_{GB}$ = 10$^3$ cm/s right')
#plt.plot(Xb_new[10,0:140]*10**9, Jqnr_gb2b[0,0:140]*Ribbon_area2*bundle1*np.cos(angu2*ang)*1000/(b*Grain_width), label='$S_{GB}$ = 10$^7$ cm/s right')
plt.legend(loc='best', fontsize=12)
plt.xlabel('Distance from the SCE (nm)', fontsize=14)
plt.ylabel('Current density (mA/cm$^2$)', fontsize=14) #Using m^-2 s^-1
#plt.title('J total')
#plt.yscale('log',base=10)
#plt.xlim(0, V_bi)
#plt.ylim(0, 0.012)
plt.savefig('qnr left ribbons {}-{}'.format(angu1,angu2), dpi=300, bbox_inches="tight")

plt.figure(6006)
#plt.plot(Xa_new[10,0:140]*10**9, Jqnr_gb1[0,0:140]*Ribbon_area1*bundle1*np.cos(angu1*ang)*1000/(b*Grain_width), label='$S_{GB}$ = 10$^3$ cm/s left')
#plt.plot(Xa_new[10,0:140]*10**9, Jqnr_gb1b[0,0:140]*Ribbon_area1*bundle1*np.cos(angu1*ang)*1000/(b*Grain_width), label='$S_{GB}$ = 10$^7$ cm/s left')
plt.plot(Xb_new[10,0:140]*10**9, Jqnr_gb2[0,0:140]*Ribbon_area2*bundle1*np.cos(angu2*ang)*1000/(b*Grain_width), label='$S_{GB}$ = 10$^3$ cm/s right')
plt.plot(Xb_new[10,0:140]*10**9, Jqnr_gb2b[0,0:140]*Ribbon_area2*bundle1*np.cos(angu2*ang)*1000/(b*Grain_width), label='$S_{GB}$ = 10$^7$ cm/s right')
plt.legend(loc='best', fontsize=12)
plt.xlabel('Distance from the SCE (nm)', fontsize=14)
plt.ylabel('Current density (mA/cm$^2$)', fontsize=14) #Using m^-2 s^-1
#plt.title('J total')
#plt.yscale('log',base=10)
#plt.xlim(0, V_bi)
#plt.ylim(0, 0.010)
plt.savefig('qnr right ribbons {}{}'.format(angu1,angu2), dpi=300, bbox_inches="tight")

plt.figure(7002)
plt.plot(V, -Curr_ribbon_dark1*1000, label='$S_{GB}$ = 10$^3$ cm/s')
#plt.plot(V, Curr_ribbon_qnr2*1000, label='J qnr right 10^3')
plt.plot(V, -Curr_ribbon_dark1b*1000, label='$S_{GB}$ = 10$^7$ cm/s')
#plt.plot(V, Curr_ribbon_qnr2b*1000, label='J qnr right 10^7')
#plt.plot(V, J_total, label='Jtotal')
plt.legend(loc='best', fontsize=12)
plt.xlabel('Voltage (V)', fontsize=14)
plt.ylabel('Current density (mA/cm$^2$)', fontsize=14) #Using m^-2 s^-1
#plt.title('save J qnr')
#plt.yscale('log',base=10)
plt.xlim(-0.1, 0.7)
plt.ylim(-1, 1)
plt.savefig('save Jdark total', dpi=300)