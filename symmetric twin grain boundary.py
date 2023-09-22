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
import os

'''Importing the complex refractive index (n+ik) for each layer'''
current_directory = os.getcwd()
file_name1 = "glass.xlsx"
file_name2 = "Am1-5.xlsx"
file_name3 = "sb2se3.xlsx"
file_name4 = "tio2.xlsx"
file_name5 = "FTO.xlsx"
file_path1 = os.path.join(current_directory, file_name1)
file_path2 = os.path.join(current_directory, file_name2)
file_path3 = os.path.join(current_directory, file_name3)
file_path4 = os.path.join(current_directory, file_name4)
file_path5 = os.path.join(current_directory, file_name5)

glass = pd.read_excel(file_path1)
am15 = pd.read_excel(file_path2)
sbse = pd.read_excel(file_path3)
tio2 = pd.read_excel(file_path4)
FTO = pd.read_excel(file_path5)


'''initialising arrays'''
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
F_bot2 = np.zeros(120)


'''Simulation parameters'''
#Sb2Se3 parameters
sb2se3_gap = 1.17
p_width = 3000*(10**-9)
kai_sbse= 4.04  
e_sbse = 18
nc_sbse = 2.2*(10**18) # 1/cm^3
nv_sbse = 1.8*(10**19) # 1/cm^3
mue_p = 16.9 #electron mobility
muh_p = 5.1 #hole mobility
Na =  1*(10**16) 
Tn_p = 1*(10**-9) #lifetime s


#TiO2 parameters
n_width = 50*(10**-9) 
tio2_gap = 3.2
kai_tio2 = 4.2
e_tio2z = 10 
nc_tio2 = 2.2*(10**18) # 1/cm^3
nv_tio2 = 6.0*(10**17) # 1/cm^3
Nd_tio2 = 1*(10**17)

#Other parameters
fto_width = 500*(10**-9) #500 nm FTO
glass_width = 2*(10**-3) #500 nm FTO
T = 300 #Temperature K
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
me_p = 0.67*emass #electron effective mass
mh_p = 3.32*emass #hole effective mass
me_p_tio2 = 10*emass #electron effective mass
mh_p_tio2 = 0.8*emass #hole effective mass


'''Orientation (can go from 0-89)'''
angu = 45 #angle
ang = (np.pi/180) #degress to radian conversion


'''Bundling (this reduces the computational time by bundling multiple inidividual ribbons and calculating using their mean ending position)'''
bundle = 10
name = '%a grain.png' %angu #saves the data according to their orientation

'''Recombination velocities'''
S_b = 1*(10**7) #cm s^-1 back contact
S_gb = 1*(10**3) #cm/s grain boundary passivated
S_gb2 = 1*(10**7) #cm/s grain boundary unpassivated


'''Values to interpolate (300-900 nm), interpolated values x, interpolated values y'''
''' This interpolates the complex refractive index to integer values between 300-900 nm'''
a3 = np.interp(c1, sbse.iloc[:,0], sbse.iloc[:,1])
a4 = np.interp(c1, sbse.iloc[:,0], sbse.iloc[:,2])
a1 = np.interp(c1, tio2.iloc[:,0], tio2.iloc[:,1])
a2 = np.interp(c1, tio2.iloc[:,0], tio2.iloc[:,2])
a5 = np.interp(c1, FTO.iloc[:,0], FTO.iloc[:,1])
a6 = np.interp(c1, FTO.iloc[:,0], FTO.iloc[:,2])
a7 = np.interp(c1, glass.iloc[:,0], glass.iloc[:,1])
a8 = np.interp(c1, glass.iloc[:,2], glass.iloc[:,3])


'''Irradiance to photon flux (cm^-2 s^-1) conversion'''
for x in range(0,600):
    photon_flux[x] = ((am15.iloc[x,0]/1000)*am15.iloc[x,2])/(echarge*1.24) #https://www.pveducation.org/pvcdrom/properties-of-sunlight/photon-flux units are m-2 s-1
    W_photon[x] = (h*c)/((am15.iloc[x,0]*(10**-9)))
    photon_energy[x] = ((am15.iloc[x,2])/W_photon[x])/(10000)
    photon_test[x] = photon_flux[x]/photon_energy[x]



'''Reflection and absorption of all the different layers'''
#Air/Glass
for x in range(0,600):
    air_n[x] = 1
    air_k[x] = 0
    ca = complex(air_n[x], air_k[x]) #Air data
    cb = complex(a7[x], a8[x]) #TiO2 interpolated data
    r0[x] = (abs((ca - cb)/(ca + cb)))**2
    t0[x] = (1 - r0[x])
    ab0[x] = (4*np.pi*a8[x])/((am15.iloc[x,0])*(10**-9))
    ab0a[x] = np.exp(-ab0[x]*glass_width)
    ca = 0
    cb = 0
ref0 = np.multiply(photon_energy[0:600],t0) # m^-2 s^-1
abso0 = np.multiply(ref0,ab0a)

#Glass/FTO
for x in range(0,600):
    ca = complex(a7[x], a8[x]) #Air data
    cb = complex(a5[x],a6[x]) #TiO2 interpolated data
    r1[x] = (abs((ca - cb)/(ca + cb)))**2
    t1[x] = (1 - r1[x])
    ab1[x] = (4*np.pi*a6[x])/((am15.iloc[x,0])*(10**-9))
    ab2[x] = np.exp(-ab1[x]*fto_width)
    ca = 0
    cb = 0
ref1 = np.multiply(abso0,t1) # m^-2 s^-1
abso1 = np.multiply(ref1,ab2)

#FTO/TiO2
for x in range(0,600):
    ca = complex(a5[x], a6[x])
    cb = complex(a1[x], a2[x])
    r2[x] = (abs((ca - cb)/(ca + cb)))**2
    t2[x] = (1 - r2[x])
    ab3[x] = (4*np.pi*a2[x])/((am15.iloc[x,0])*(10**-9))
    ab4[x] = np.exp(-ab3[x]*n_width)
    ca = 0
    cb = 0
ref2 = np.multiply(abso1,t2)
abso2 = np.multiply(ref2,ab4)


#TiO2/Sb2Se3
for x in range(0,600):
    ca = complex(a1[x], a2[x])
    cb = complex(a3[x], a4[x])
    r3[x] = (abs((ca - cb)/(ca + cb)))**2
    t3[x] = (1 - r3[x])
    ab5[x] = (4*np.pi*a4[x])/((am15.iloc[x,0])*(10**-9))
    ab6[x] = np.exp(-ab5[x]*p_width)
    ca = 0
    cb = 0   
ref3 = np.multiply(abso2,t3) #Reflection after the interface but before absorption at Sb2Se3
abso3 = np.multiply(ref3,ab6)

'''Saving the absorption coefficients'''
ab0.tofile('glass abs coef.csv',sep=',',format='%10.5f')
ab1.tofile('FTO abs coef.csv',sep=',',format='%10.5f')
ab3.tofile('TiO2 abs coef.csv',sep=',',format='%10.5f')
ab5.tofile('Sb2Se3 abs coef.csv',sep=',',format='%10.5f')
t1.tofile('Air-FTO trans abs coef.csv',sep=',',format='%10.5f')
ref3.tofile('incident light on the sb2se3 layer.csv',sep=',',format='%10.5f')
percent = np.zeros(600)
photon_flux_inv = np.zeros(600)


for i in range(0,600):
    percent[i] = ref3[i]/photon_energy[i]
    photon_flux_inv[i] = (ref3[i]*(echarge*1.24))/((am15.iloc[x,0]/1000))*100*100

'''Saving the photon flux spectra'''
photon_flux_inv.tofile('photon flux spectra.csv',sep=',',format='%10.5f')

position_new = list(range(0,3000,10))
ab_new = np.zeros((300, 600))
ab_sum = np.zeros((300, 600))
intensity_new = np.zeros(300)

for i in range(0, 300):
    for j in range(0,600):
        ab_new[i,j] = np.exp(-ab5[j]*position_new[i]*(10**-9))
        ab_sum[i,j] = ab_new[i,j]*ref3[j]
    intensity_new[i] = sum(ab_sum[i,:])


'''Total photon flux decrease as a function of position'''
plt.figure(1)
plt.plot(position_new[0:300], intensity_new[0:300], 'r-', label='PFD at back contact', linewidth=1)
plt.xlabel('Position (nm)', fontsize=12)
plt.ylabel('Total photon flux density (cm$^{-2}$ s$^{-1}$ nm$^{-1}$)', fontsize=12) #Using m^-2 s^-1
plt.savefig('spectnew', dpi=300, bbox_inches="tight")








'''Solar spectra after reflection and absorption at each layer'''
plt.figure(2)
plt.plot(c1, photon_energy[0:600], 'k-', label='AM 1.5G', linewidth=1) #Using m^-2 s^-1
plt.plot(c1, abso0, 'c-', label='PFD at glass/FTO interface', linewidth=1)
plt.plot(c1, abso1, 'g-', label='PFD at FTO/TiO$_2$ interface', linewidth=1)
plt.plot(c1, abso2, 'b-', label='PFD at TiO2/Sb$_2$Se$_3$', linewidth=1)
plt.plot(c1, abso3, 'r-', label='PFD at back contact', linewidth=1)
plt.legend(loc='lower left', bbox_to_anchor=(1,0.5))
plt.xlabel('Wavelength (nm)')
plt.ylabel('Photon flux density (cm$^{-2}$ s$^{-1}$ nm$^{-1}$)') #Using m^-2 s^-1
plt.savefig('solar spectra after each layer', dpi=300, bbox_inches="tight")

'''Absorption coefficients'''
plt.figure(3)
plt.plot(c1, ab0/100, 'c-', label='Glass', linewidth=1)
plt.plot(c1, ab1/100, 'g-', label='FTO', linewidth=1)
plt.plot(c1, ab3/100, 'b-', label='TiO$_2$', linewidth=1)
plt.plot(c1, ab5/100, 'r-', label='Sb$_2$Se$_3$', linewidth=1)
plt.legend(loc='lower left', bbox_to_anchor=(1,0.5))
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorption (cm$^{-1}$)') #Using m^-2 s^-1
plt.yscale('log',base=10)
plt.savefig('absortion coefficients', dpi=300, bbox_inches="tight")


'''Diffusion coefficient and diffusion length'''
Dn_1 = (mue_p*k*T)/echarge # units are cm2
Dn = Dn_1/10000 #units are m2
Ln_1 = (math.sqrt(Tn_p*Dn_1)) #units are cm
Ln = Ln_1/100 # units are m

    
'''Intrinsic carrier concentration (ni) and built-in voltage'''
ni = nc_sbse*nv_sbse*np.exp(-(sb2se3_gap/(kev*T))) #This is ni^2
V_bi = kai_sbse - kai_tio2 + sb2se3_gap + ((k*T)/echarge)*math.log((Na*Nd_tio2)/(nc_tio2*nv_sbse))
n_0 = ni/Na
print('Built-in voltage =', V_bi, 'V')



'''SCR width calculation'''
num = 2*V_bi*e_sbse*e_tio2z*e_0*Nd_tio2
deno = echarge*Na*((Na*e_sbse) + (Nd_tio2*e_tio2z))
num3 = 2*V_bi*e_sbse*e_tio2z*e_0*Na
deno3 = echarge*Nd_tio2*((Na*e_sbse) + (Nd_tio2*e_tio2z))
xp = (math.sqrt(num/deno))*(1/100)*10**9
xn = (math.sqrt(num3/deno3))*(1/100)*10**9
print('At Vbi: Xp = ', xp, 'nm', 'Xn = ', xn, 'nm')

'''Charge neutrality test'''
Test1 = Na*xp
Test2 = Nd_tio2*xn
Test3 = Test1/Test2
print('Charge on p/Charge on n =', Test3)



'''Grain sizes'''
Grain_width=1000*(10**(-9))
Grain_height=3000*(10**(-9))

'''Initialising arrays'''
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

'''Lattice parameters using Pbnm'''    
a = 11.62*(10**-10)
b = 11.77*(10**-10)
c_lat = 3.962*(10**-10) #different notation because of light speed (c)
distance = b/2



'''Calculating the ending positions along the grain boundary plane'''
for i in range(0,89):
    Vspace[i] = (b/(2*np.cos((i)*ang)))*bundle #10 ribbons bundle
    Ribbon_num[i] = int(Grain_width/Vspace[i])


Y_pos0_88 = np.zeros(int(Ribbon_num[angu]))
X_pos0_88 = np.zeros(int(Ribbon_num[angu]))
X_pos0_case = np.zeros(int(Ribbon_num[angu]))
Y_pos0_end = np.zeros(int(Ribbon_num[angu]))


'''Xp as a function of V'''
for i in range(0, (int(Ribbon_num[angu]))):
    Y_pos0_88[i] =  (((i+1)*b)/(2*np.cos(angu*ang)))*10
    if angu == 0:
        X_pos0_88[i] = Grain_height
        Y_pos0_end[i] = Y_pos0_88[i]
    else:
        #X_pos0_88[i] = ((i+1)*b)/(2*np.sin(angu*ang)) 
        X_pos0_88[i] = Y_pos0_88[i]/np.tan(angu*ang)
        #X_pos0b[j,i] = Y_pos0a[i]*Cost[j]
        if X_pos0_88[i] > Grain_height:
            X_pos0_88[i] = Grain_height
            X_pos0_case[i] = 1
            Y_pos0_end[i] = Y_pos0_88[i] - (Grain_height*np.tan(angu*ang))
        else:
            Y_pos0_end[i] = 0
        
for x in range(0,120):
    Xp_1[x] = (2*(V_bi - V[x])*e_sbse*e_tio2z*e_0*Nd_tio2)/(echarge*Na*((Na*e_sbse) + (Nd_tio2*e_tio2z)))
    Xp_v[x] = (math.sqrt(abs(Xp_1[x])))*(1/100)*(10**9)
    Xp_vi[x] = (math.sqrt(abs(Xp_1[x])))*(1/100)
    Wp[x] = p_width - (Xp_vi[x])

Xp_3 = np.zeros(120)
Wp_3 = np.zeros(120)
for i in range(0,120):
    Xp_3[i] = V[i]
    Wp_3[i] = Wp[i]*(10**9)


'''Plotting Xp as a function of applied voltage'''
plt.figure(4)
plt.plot(Xp_3, Xp_v, label='Xp Width')
plt.xlabel('Voltage (V)')
plt.ylabel('Xp width (nm)') 
plt.title('Xp as a function of V')

'''Plotting Wp as a function of applied voltage'''
plt.figure(5)
plt.plot(Xp_3, Wp_3, label='Xp Width')
plt.xlabel('V')
plt.ylabel('QNR width (nm)') 
plt.title('QNR as a function of V')

##################################################################################################################


'''Ribbon orientation plot'''
for i in range(0,(int(Ribbon_num[angu]))):
    point1 = [(Y_pos0_88[i]), 0]
    point2 = [Y_pos0_end[i], X_pos0_88[i]]
    x_values1 = [point1[0], point2[0]]
    y_values1 = [point1[1], point2[1]]
    x_values1a = [-point1[0], -point2[0]]
    y_values1a = [point1[1], point2[1]]
    plt.figure(6, figsize=(10,15))
    plt.plot(x_values1,y_values1, color='red', linewidth=1)
    plt.plot(x_values1a,y_values1a, color='red', linewidth=1)
    plt.xlim((-1*(10**-6)), (1*(10**-6)))
    plt.ylim(0, (3*(10**-6)))
plt.savefig('ribbon orientation', dpi=300)


'''Initialising arrays'''
Xa = np.zeros((120,(int(Ribbon_num[angu]))))
Casexa = np.zeros((120,(int(Ribbon_num[angu]))))


'''Determines the transport case (SCR-BC or SCR-GB) for each ribbon bundle'''
for i in range(0,120):
    for n in range(0,(int(Ribbon_num[angu]))):
        Xa[i,n] = X_pos0_88[n]
        if Xa[i,n] >= Wp[i]:
            Xa[i,n] = Wp[i]
            Casexa[i,n] = 0
        elif Xa[i,n] < Wp[i]:
            Casexa[i,n] = 1
            
num_ribbon = np.zeros(120)            
for i in range(0,120):
    for n in range(0,(int(Ribbon_num[angu]))):
        if Casexa[i,n] > 0:
            num_ribbon[i] = n
        

        

'''################################ General equations calculation #####################################'''


'''Initialising arrays'''
Jdark_bc = np.zeros(120)
Jqnr_bc = np.zeros(120)
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
Current_new1 = np.zeros((120,(int(Ribbon_num[angu-1])-1))) 
cosh_bc = np.zeros(120)
sinh_bc = np.zeros(120)
cosh_gb = np.zeros((120,int(Ribbon_num[angu])))
sinh_gb = np.zeros((120,int(Ribbon_num[angu])))
cosh_gb2 = np.zeros((120,int(Ribbon_num[angu])))
sinh_gb2 = np.zeros((120,int(Ribbon_num[angu])))
cero = 0


'''Calculating the hyperbolic sin and cos functions'''
for i in range(0,120):
    cosh_bc[i] = ((S_b*Ln_1)/(Dn_1*np.cos(angu*ang)))*np.cosh(Wp[i]/(Ln*np.cos(angu*ang))) + np.sinh(Wp[i]/(Ln*np.cos(angu*ang)))
    sinh_bc[i] = ((S_b*Ln_1)/(Dn_1*np.cos(angu*ang)))*np.sinh(Wp[i]/(Ln*np.cos(angu*ang))) + np.cosh(Wp[i]/(Ln*np.cos(angu*ang)))
    
    
for i in range(0,120):
    for j in range(0, int(Ribbon_num[angu])):
        if angu == 0:
            cero = 0
        else:
            cosh_gb[i,j] = (((((S_gb))*Ln_1)/(2*Dn_1*np.sin(angu*ang)))*np.cosh(Xa[i,j]/(Ln*np.cos(angu*ang))) + np.sinh(Xa[i,j]/(Ln*np.cos(angu*ang))))
            sinh_gb[i,j] = (((((S_gb))*Ln_1)/(2*Dn_1*np.sin(angu*ang)))*np.sinh(Xa[i,j]/(Ln*np.cos(angu*ang))) + np.cosh(Xa[i,j]/(Ln*np.cos(angu*ang))))
            cosh_gb2[i,j] = (((((S_gb2))*Ln_1)/(2*Dn_1*np.sin(angu*ang)))*np.cosh(Xa[i,j]/(Ln*np.cos(angu*ang))) + np.sinh(Xa[i,j]/(Ln*np.cos(angu*ang))))
            sinh_gb2[i,j] = (((((S_gb2))*Ln_1)/(2*Dn_1*np.sin(angu*ang)))*np.sinh(Xa[i,j]/(Ln*np.cos(angu*ang))) + np.cosh(Xa[i,j]/(Ln*np.cos(angu*ang))))

        
'''SCR-BC current'''
dummy_bc1 = np.zeros(600)
dummy_bc2 = np.zeros(600)
dummy_bc3 = np.zeros(600)
dummy_bc4 = np.zeros(600)
dummy_bc5 = np.zeros(600)


'''Jdark'''
for i in range(0,120):
    Jdark_bc[i] = ((echarge*Dn_1*ni*(np.exp(echarge*V[i]/(k*T))-1))/(Ln_1*Na))*(cosh_bc[i]/sinh_bc[i])

'''Jqnr'''
for i in range(0,120):
    for l in range(0,600):
        #dummy_bc1[l] = ((echarge*ref3[l]*Ln*ab5[l]*(np.exp(-ab5[l]*Xp_vi[i])))/(((ab5[l]*Ln)**2) - 1))
        #dummy_bc2[l] = (ab5[l]*Ln*np.cos(angu*ang) - ((((S_b*Ln_1)/(Dn_1*np.cos(angu*ang)))*np.cosh(Wp[i]/(Ln*np.cos(angu*ang))) + np.sinh(Wp[i]/(Ln*np.cos(angu*ang))))/(((S_b*Ln_1)/(Dn_1*np.cos(angu*ang)))*np.sinh(Wp[i]/(Ln*np.cos(angu*ang))) + np.cosh(Wp[i]/(Ln*np.cos(angu*ang))))))
        #dummy_bc3[l] = ((ab5[l]*Ln*np.cos(angu*ang) - ((S_b*Ln_1)/(Dn*np.cos(angu*ang))))/(((S_b*Ln_1)/(Dn_1*np.cos(angu*ang)))*np.sinh(Wp[i]/(Ln*np.cos(angu*ang))) + np.cosh(Wp[i]/(Ln*np.cos(angu*ang)))))
        #dummy_bc4[l] = dummy_bc1[l]*(dummy_bc2[l] - dummy_bc3[l]*np.exp(-ab5[l]*Wp[i]))
        dummy_bc5[l] = (((echarge*(ref3[l])*(np.exp(-ab5[l]*Xp_vi[i]))*((ab5[l]*Ln))))/((((ab5[l]*Ln)**2) - 1)))*((ab5[l]*Ln*np.cos(angu*ang) - (cosh_bc[i]/sinh_bc[i])) - ((ab5[l]*Ln*np.cos(angu*ang) - ((S_b*Ln_1)/(Dn_1*np.cos(angu*ang))))/sinh_bc[i])*np.exp(-ab5[l]*Wp[i]))
    #Jqnr_bc[i] = np.sum(dummy_bc4)
    Jqnr_bc[i] = np.sum(dummy_bc5)
    for m in range(0,600):
        #dummy_bc1[m] = 0
        #dummy_bc2[m] = 0
        #dummy_bc3[m] = 0
        #dummy_bc4[m] = 0
        dummy_bc5[m] = 0
        
'''Jscr'''
Jscr_num = np.zeros(120)
Jscr_num2 = np.zeros(120)
dummy_scr = np.zeros(600)

for i in range(0,120):
    for l in range(0,600):
        dummy_scr[l] = (echarge*ref3[l]*(1 - np.exp(-ab5[l]*Xp_vi[i])))
    Jscr_num2[i] = (trapz(dummy_scr))
    Jscr_num[i] = (sum(dummy_scr))
    for z in range(0,600):
        dummy_scr[z] = 0
        
####################################################################################

'''SCR-GB current '''

'''Initialising arrays'''
Jdark_gb = np.zeros((120, int(np.max(Ribbon_num[angu]))))
Jdark_gb2 = np.zeros((120, int(np.max(Ribbon_num[angu]))))
Jqnr_gb = np.zeros((120, int(np.max(Ribbon_num[angu]))))
dummy_qnr = np.zeros(600)
Jqnr_gb2 = np.zeros((120, int(np.max(Ribbon_num[angu]))))
dummy_qnr2 = np.zeros(600)

'''Jdark'''
for i in range(0,120): #fix this
    for j in range(0, int(Ribbon_num[angu])):
        if Casexa[i,j] == 1: #this part is wrong
            Jdark_gb[i,j] = ((echarge*Dn_1*ni*(np.exp(echarge*V[i]/(k*T))-1))/(Ln_1*Na))*(cosh_gb[i,j]/sinh_gb[i,j])
            Jdark_gb2[i,j] = ((echarge*Dn_1*ni*(np.exp(echarge*V[i]/(k*T))-1))/(Ln_1*Na))*(cosh_gb2[i,j]/sinh_gb2[i,j])

'''Jqnr'''
for i in range(0,120): #fix this
    for j in range(0, int(Ribbon_num[angu])):
        if Casexa[i,j] == 1:
            for l in range(0,600):
                dummy_qnr[l] = (((echarge*(ref3[l])*(np.exp(-ab5[l]*Xp_vi[i]))*((ab5[l]*Ln))))/((((ab5[l]*Ln)**2) - 1)))*((ab5[l]*Ln*np.cos(angu*ang) - (cosh_gb[i,j]/sinh_gb[i,j])) - ((ab5[l]*Ln*np.cos(angu*ang) - ((S_gb*Ln_1)/(2*Dn_1*np.sin(angu*ang))))/sinh_gb[i,j])*np.exp(-ab5[l]*Xa[i,j]))
                dummy_qnr2[l] = (((echarge*(ref3[l])*(np.exp(-ab5[l]*Xp_vi[i]))*((ab5[l]*Ln))))/((((ab5[l]*Ln)**2) - 1)))*((ab5[l]*Ln*np.cos(angu*ang) - (cosh_gb2[i,j]/sinh_gb2[i,j])) - ((ab5[l]*Ln*np.cos(angu*ang) - ((S_gb2*Ln_1)/(2*Dn_1*np.sin(angu*ang))))/sinh_gb2[i,j])*np.exp(-ab5[l]*Xa[i,j]))
            Jqnr_gb[i,j] = np.sum(dummy_qnr)
            Jqnr_gb2[i,j] = np.sum(dummy_qnr2)
            for z in range(0,600):
                dummy_qnr[z] = 0
                dummy_qnr2[z] = 0

        
'''Current of each ribbon'''
Ribbon_area = (b*a/(2*np.cos(angu*ang)))
Curr_ribbon = np.zeros(120)
Curr_ribbon2 = np.zeros(120)
Curr_ribbon_dark = np.zeros(120)
Curr_ribbon_dark2 = np.zeros(120)
Curr_ribbon_qnr = np.zeros(120)
Curr_ribbon_qnr2 = np.zeros(120)
Curr_ribbon_scr = np.zeros(120)
Curr_ribbon_ang = np.zeros(120)
Curr_ribbon_ang2 = np.zeros(120)
Total_current_density = np.zeros(120)
Total_current_density2 = np.zeros(120)
dummy_ribbon = np.zeros(int(Ribbon_num[angu]))
dummy_ribbon2 = np.zeros(int(Ribbon_num[angu]))
dummy_ribbon_dark = np.zeros(int(Ribbon_num[angu]))
dummy_ribbon_dark2 = np.zeros(int(Ribbon_num[angu]))
dummy_ribbon_scr = np.zeros(int(Ribbon_num[angu]))
dummy_ribbon_qnr = np.zeros(int(Ribbon_num[angu]))
dummy_ribbon_qnr2 = np.zeros(int(Ribbon_num[angu]))

for i in range(0,120):
    for j in range(0, (int(Ribbon_num[angu]))):
        if Casexa[i,j] == 0:
            dummy_ribbon[j] = (-Jdark_bc[i]*Ribbon_area*bundle + Jqnr_bc[i]*Ribbon_area*bundle + Jscr_num[i]*Ribbon_area*bundle)
            dummy_ribbon2[j] = (-Jdark_bc[i]*Ribbon_area*bundle + Jqnr_bc[i]*Ribbon_area*bundle + Jscr_num[i]*Ribbon_area*bundle)
            dummy_ribbon_dark[j] = (Jdark_bc[i]*Ribbon_area*bundle)
            dummy_ribbon_qnr[j] = (Jqnr_bc[i]*Ribbon_area*bundle)
            dummy_ribbon_dark2[j] = (Jdark_bc[i]*Ribbon_area*bundle)
            dummy_ribbon_qnr2[j] = (Jqnr_bc[i]*Ribbon_area*bundle)
            dummy_ribbon_scr[j] = (Jscr_num[i]*Ribbon_area*bundle) # 
        else:
            dummy_ribbon[j] = (-Jdark_gb[i,j]*Ribbon_area*bundle + Jqnr_gb[i,j]*Ribbon_area*bundle + Jscr_num[i]*Ribbon_area*bundle)
            dummy_ribbon2[j] = (-Jdark_gb2[i,j]*Ribbon_area*bundle + Jqnr_gb2[i,j]*Ribbon_area*bundle + Jscr_num[i]*Ribbon_area*bundle)
            dummy_ribbon_dark[j] = (Jdark_gb[i,j]*Ribbon_area*bundle)
            dummy_ribbon_dark2[j] = (Jdark_gb2[i,j]*Ribbon_area*bundle)
            dummy_ribbon_qnr[j] = (Jqnr_gb[i,j]*Ribbon_area*bundle)
            dummy_ribbon_qnr2[j] = (Jqnr_gb2[i,j]*Ribbon_area*bundle)
            dummy_ribbon_scr[j] = (Jscr_num[i]*Ribbon_area*bundle)
    Curr_ribbon[i] = np.sum(dummy_ribbon)
    Curr_ribbon2[i] = np.sum(dummy_ribbon2)
    Curr_ribbon_dark[i] = np.sum(dummy_ribbon_dark)*np.cos(angu*ang)/(b*Grain_width)
    Curr_ribbon_dark2[i] = np.sum(dummy_ribbon_dark2)*np.cos(angu*ang)/(b*Grain_width)
    Curr_ribbon_qnr[i] = np.sum(dummy_ribbon_qnr)*np.cos(angu*ang)/(b*Grain_width)
    Curr_ribbon_qnr2[i] = np.sum(dummy_ribbon_qnr2)*np.cos(angu*ang)/(b*Grain_width)
    Curr_ribbon_scr[i] = np.sum(dummy_ribbon_scr)*np.cos(angu*ang)/(b*Grain_width)
    Curr_ribbon_ang[i] = Curr_ribbon[i]*np.cos(angu*ang)
    Curr_ribbon_ang2[i] = Curr_ribbon2[i]*np.cos(angu*ang)
    Total_current_density[i] = Curr_ribbon_ang[i]/(b*Grain_width)
    Total_current_density2[i] = Curr_ribbon_ang2[i]/(b*Grain_width)
    
Total_current_density.tofile('10^3 total {}.csv'.format(angu),sep=',')
Total_current_density2.tofile('10^7 total {}.csv'.format(angu),sep=',')
Curr_ribbon_dark.tofile('10^3 dark {}.csv'.format(angu),sep=',')
Curr_ribbon_dark2.tofile('10^7 dark {}.csv'.format(angu),sep=',')
Curr_ribbon_qnr.tofile('10^3 qnr {}.csv'.format(angu),sep=',')
Curr_ribbon_qnr2.tofile('10^7 qnr {}.csv'.format(angu),sep=',')
Curr_ribbon_scr.tofile('10^3 scr {}.csv'.format(angu),sep=',')
Curr_ribbon_scr.tofile('10^7 scr {}.csv'.format(angu),sep=',')


#####################################################################################################################


'''Graphs'''

'''Jdark'''
plt.figure(7)
plt.plot(V, Curr_ribbon_dark*1000, label='$J_{dark}$ $10^3$ cm/s')
plt.plot(V, Curr_ribbon_dark2*1000, label='$J_{dark}$ $10^7$ cm/s')
plt.legend(loc='best')
plt.xlabel('Voltage (V)')
plt.ylabel('Current density (mA cm$^{-2}$)') #Using m^-2 s^-1
plt.savefig('Jdark', dpi=300)

'''Jqnr'''
plt.figure(8)
plt.plot(V, Curr_ribbon_qnr*1000, label='$J_{qnr}$ $10^3$ cm/s')
plt.plot(V, Curr_ribbon_qnr2*1000, label='$J_{qnr}$ $10^7$ cm/s')
plt.legend(loc='best')
plt.xlabel('Voltage (V)')
plt.ylabel('Current density (mA cm$^{-2}$)') #Using m^-2 s^-1
plt.savefig('Jqnr', dpi=300)


'''Jscr'''
plt.figure(9)
plt.plot(V, Curr_ribbon_scr*1000, label='$J_{scr}$ $10^3$ cm/s')
plt.plot(V, Curr_ribbon_scr*1000, label='$J_{scr}$ $10^7$ cm/s')
plt.legend(loc='best')
plt.xlabel('Voltage (V)')
plt.ylabel('Current density (mA cm$^{-2}$)') #Using m^-2 s^-1
plt.savefig('Jscr', dpi=300)

'''Jtotal'''
plt.figure(10)
plt.plot(V, Total_current_density*1000, label='$J_{total}$ $10^3$ cm/s')
plt.plot(V, Total_current_density*1000, label='$J_{total}$ $10^7$ cm/s')
#plt.plot(V, J_total, label='Jtotal')
plt.legend(loc='best')
plt.xlabel('Voltage (V)')
plt.ylabel('Current density (mA cm$^{-2}$)') #Using m^-2 s^-1
plt.xlim(0, V_bi)
plt.ylim(-20, 25)
plt.savefig('Jtotal', dpi=300)

'''All current contributions'''
plt.figure(11)
plt.plot(V, Total_current_density*1000, '--', label='$J_{Total}$ $10^3$')
plt.plot(V, Total_current_density*1000, label='$J_{Total}$ $10^7$')
plt.plot(V, -Curr_ribbon_dark*1000, '--', label='$J_{Dark}$ $10^3$')
plt.plot(V, -Curr_ribbon_dark2*1000, label='$J_{Dark}$ $10^7$')
plt.plot(V, Curr_ribbon_qnr*1000, '--', label='$J_{QNR}$ $10^3$')
plt.plot(V, Curr_ribbon_qnr2*1000, label='$J_{QNR}$ $10^7$')
plt.plot(V, Curr_ribbon_scr*1000, label='$J_{SCR}$')
plt.legend(loc='lower left')
plt.xlabel('Voltage (V)')
plt.ylabel('Current density (mA cm$^{-2}$)') #Using m^-2 s^-1
plt.xlim(0, V_bi)
plt.ylim(-10, (Total_current_density[0]*1000 + 2))
plt.savefig('Jtotal contributions', dpi=300)


Jtotal_new = np.zeros(120)
Jtotal_new2 = np.zeros(120)
for i in range(0,120):
    Jtotal_new[i] = (Curr_ribbon_scr[i] - Curr_ribbon_qnr[i] - Curr_ribbon_dark[i])*1000*np.cos(angu*ang)
    Jtotal_new2[i] = (Curr_ribbon_scr[i] - Curr_ribbon_qnr2[i] - Curr_ribbon_dark2[i])*1000*np.cos(angu*ang)
Voc_num = int(V_bi*100) + 10
Curr_ribbon_qnr.tofile('{} jqnr new 107.csv'.format(angu),sep=',',format='%10.5f')


'''Solar cell parameters'''
Isc_eff = max(Jtotal_new[0:Voc_num])
Vhp_min = min([n for n in Jtotal_new[:] if n>0])
Vhp_test = np.where(Jtotal_new[:]==Vhp_min)
Voc_eff = V[Vhp_test]
Maxpower_teo = Isc_eff*Voc_eff

Maxpower_real1 = np.zeros(120)
for i in range(0,120):
    Maxpower_real1[i] = Jtotal_new[i]*V[i]
Maxpower_real = max(Maxpower_real1[:])
Solar_power_mw = 100
Fillfactor = (Maxpower_real/Maxpower_teo)
Eff = ((Maxpower_real)/Solar_power_mw)*100

print('Fill factor =', 100*Fillfactor, '%')
print('Efficiency =', Eff, '%')
print('Voc =', Voc_eff, 'mV')
print('Isc =', Isc_eff, 'mA')


'''Saving Jtotal'''
Jtotal_new.tofile('{} degree {} 10^3.csv'.format(angu, Voc_eff),sep=',',format='%10.5f')
Jtotal_new2.tofile('{} degree {} 10^7.csv'.format(angu, Voc_eff),sep=',',format='%10.5f')
