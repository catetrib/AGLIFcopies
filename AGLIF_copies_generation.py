'''
AGLIF copies generation

Requires:
- expNeuronDB_V006small_piramidali_dati_2023_06_15.json
- expNeuronDB_V006small_interneuroni_dati_2023_06_15.json
- neuronName folder for each neuron used to generate copies

The user has to:
- give the list of neuron names for which to create the copies
- spacify if these are pyramidal or interneurons. A mix of the two classes is not admitted
- specify the name of json file of experimental data (the required input file)
- specify the currents at which to make the copies
- specify coefficients for parameters 

Output:
in neuronName folder a txt file with resulting spike times for each copy is created; file name represents the set of coefficient used for the copy creation

'''

import matplotlib.pyplot as plt
import numpy as np
import time
import os

import glob

import os.path
import re




'''
INPUT
'''

# PYR
neuroni=['95810005','95810006','95810007','95810008','95810010','95810011','95810012','95810013','95810014','95810015']#tr1
#neuroni=['95810022','95810023','95810024','95810025','95810026','95810027','95810028','95810029','95810030','95810031', '95810032','95810033']
#neuroni=['95810037','95810038','95810039','95810040','95810041','95817003','95817004','95817005', '95817006','95817007','95817008']
#neuroni=['95822000','95822001','95822002','95822003','95822005','95822006','95822009', '95822010','95822011','95824000','95824004','95824006']
#neuroni=['95831000','95831001','95831002','95831003','95831004', '95912004','95912005','95912006','95912007','95914001','95914002','95914003','95914004']

# intern
#neuroni = ['96711008','99111002','99111001','99111000','97911000','97911001','97911002','97428000','97428001','98205021','98205022','98205024','98205025','97509008','97509009','97509010','97509011',
#'97717005','98D15008','98D15009','99111004','99111006','98513011','95817000','95817001','95817002']

pyramidal = True
interneuron = False


# filenames of experimental DB 
if pyramidal:
  JSONfileName = 'expNeuronDB_V006small_piramidali_dati_2023_06_15'
if interneuron:
  JSONfileName = 'expNeuronDB_V006small_interneuroni_dati_2023_06_15'

# const currents to simulate
correnti = [200,400,600,800,1000]

parName0 = 'Iadap' #Iadap_start
coeffParamPar0 = [0,0.1,0.25,0.4,0.6] #original: [0]

parName1 = 'cost_idep_ini' #Idep_start 
coeffParamPar1 = [0.75,0.9,1,1.5,1.75] #original: [1]

parName2 = 'Idep_ini_vr' #Idep0
coeffParamPar2 = [0.25,0.5,1,1.5,1.75] #original: [1]

parName3 = 'c'
coeffParamPar3 = [0.75,0.9,1,1.5,2] #original: [1]

parName4 = 'd' #eta
coeffParamPar4 = [0.5,0.75,1,1.5] #original: [1]

parName5 = 'rette' #eps to sum to original values [%valInfLineSup %CoeffSup,%costSup,%ValSupLineInf %coeffInf,%costInf]; original: [0,0,0,0,0,0]
coeffParamPar5 = [[0,0,0,0,0,0],[0,0,0,0,0,25],[0,0,0,0,25,0],[0,0,0,0,25,25],
                  [0,0,-10,0,0,0],[0,0,-10,0,0,25],[0,0,-10,0,25,0],[0,0,-10,0,25,25],
                  [0,-25,0,0,0,0],[0,-25,0,0,0,25],[0,-25,0,0,25,0],[0,-25,0,0,25,25],
                  [0,-25,-10,0,0,0],[0,-25,-10,0,0,25],[0,-25,-10,0,25,0],[0,-25,-10,0,25,25]]

parName6 = 'p'
coeffParamPar6 = [1] #original: [1]


'''
end of user input
---------------------------------------------------------------------
'''



# corrente costante
corr = [0,0,0,1]


# import neuron DB file
import json
  
# Opening JSON file
f = open(JSONfileName+'.json',)
# returns JSON object as a dictionary
pyramidalNeuronsDatabase = json.load(f)
# Closing file
f.close()



'''
FUNCTIONS
'''
# BOUNDARIES del 05/08/22

if pyramidal:
    def expTimeSogliaMin(contaSpike,myCurrent):
        if myCurrent == 200:
            expTimeAtContaSpike =  (contaSpike - 0.33333)/0.0041667
        elif myCurrent == 400:
            expTimeAtContaSpike =  (contaSpike - 2.544*10**-16)/0.0049
        elif myCurrent == 600:
            expTimeAtContaSpike =  100000
        elif myCurrent == 800:
            expTimeAtContaSpike =  ((contaSpike + 12.999)/9.6)**(1/0.08)
        elif myCurrent == 1000:
            expTimeAtContaSpike =  ((contaSpike + 195)/190)**(1/0.008)
    
        else:
            expTimeAtContaSpike = 0
       
        return(expTimeAtContaSpike)
    
    def expTimeSogliaMax(contaSpike,myCurrent):
        if myCurrent == 200:
            expTimeAtContaSpike = ((contaSpike - 0.5)/0.27 )**(1/0.6)
        elif myCurrent == 400:
            expTimeAtContaSpike =  ((contaSpike - 0.5)/0.4 )**(1/0.6)
        elif myCurrent == 600:
            expTimeAtContaSpike =  ((contaSpike - 1)/0.28 )**(1/0.7)
        elif myCurrent == 800:
            if contaSpike == 1:
                expTimeAtContaSpike = 0
            else:
                expTimeAtContaSpike =  ((contaSpike - 1.1)/0.3 )**(1/0.69)
        elif myCurrent == 1000:
            if contaSpike == 1:
                expTimeAtContaSpike = 0
            else:        
                expTimeAtContaSpike =  ((contaSpike - 1.1)/0.33 )**(1/0.7) 
        else:
            expTimeAtContaSpike = 0
        return(expTimeAtContaSpike)        


if interneuron:
    # Interneurons
    def expTimeSogliaMin(contaSpike,myCurrent):
        if myCurrent == 200:
            expTimeAtContaSpike =  (contaSpike + 0.285714)/0.0107143
        elif myCurrent == 400:
            expTimeAtContaSpike =  100000
        elif myCurrent == 600:
            expTimeAtContaSpike =  100000
        elif myCurrent == 800:
            expTimeAtContaSpike =  ((contaSpike + 17.5)/12.05)**(1/0.1)
        elif myCurrent == 1000:
            expTimeAtContaSpike =  ((contaSpike - 0.5556)/0.01111)
    
        else:
            expTimeAtContaSpike = 0
       
        return(expTimeAtContaSpike)
    
    def expTimeSogliaMax(contaSpike,myCurrent):
        if myCurrent == 200:
            expTimeAtContaSpike = ((contaSpike + 1)/1.3 )**(1/0.428)
        elif myCurrent == 400:
            expTimeAtContaSpike =  (contaSpike - 3.2)/0.14 
        elif myCurrent == 600:
            expTimeAtContaSpike =  ((contaSpike - 3)/0.2 )
        elif myCurrent == 800:
            expTimeAtContaSpike =  ((contaSpike - 4)/0.22 )
        elif myCurrent == 1000:
            expTimeAtContaSpike =  ((contaSpike - 5.0)/0.135 )
        else:
            expTimeAtContaSpike = 0
        return(expTimeAtContaSpike)        



# block lines
def tagliorette(corr,retteParParsed):
    vinc_sup = retteParParsed[0]
    coeffSup = retteParParsed[1]
    constSup = retteParParsed[2]
    vinc_inf = retteParParsed[3]
    coeffInf = retteParParsed[4]
    constInf = retteParParsed[5]
    
    dur_sign=np.inf

    if corr<vinc_inf and corr>0:
        dur_sign = coeffInf*corr + constInf
    
    if corr>vinc_sup:
        dur_sign = coeffSup*corr + constSup
    return dur_sign



    

#-------------
def V(t, delta, Psi, alpha, beta, IaA0, IdA0, t0, V0):
    return (1 / 2) * (beta + (-1) * delta) ** (-1) * (beta ** 2 + ((-1) + beta) * delta) ** (-1) * (4 * beta + (- 1) * (1 + delta) ** 2) ** (-1) * Psi * (2 * np.exp(((-1) * t + t0) * beta) * IdA0 * ((-1) + beta) * beta * (beta + (-1) * delta) * Psi + (-2) * (alpha + (-1) * beta + delta) * (beta ** 2 + ((- 1) + beta) * delta) * Psi + np.exp((1 / 2) * (t + (-1) * t0) * ((-1) + delta + (-1) * Psi)) * (IdA0 * beta * (beta + (-1) * delta) * ((-1) + (-1) * delta + beta * (3 + delta + (-1) * Psi) + Psi) + (-1) * (beta ** 2 + (-1) * delta + beta * delta) * (alpha * (1 + (-2) * beta + delta + (-1) * Psi) + (beta + (-1) * delta) * ((-1) + 2 * IaA0 * beta + (-1) * delta + Psi + V0 * ((-1) + (-1) * delta + Psi)))) + np.exp((1 / 2) * (t + (-1) * t0) * ((-1) + delta + Psi)) * ((-1) * IdA0 * beta * (beta+(-1) * delta) * ((-1) + (-1) * delta + (-1) * Psi + beta * (3 + delta + Psi)) + (beta ** 2 + (-1) * delta+beta * delta) * (alpha * (1 + (-2) * beta + delta + Psi) + (beta + (-1) * delta) * ((-1) + 2 * IaA0 * beta+(-1) * delta + (-1) * Psi + (-1) * V0 * (1 + delta + Psi)))))


def Iadap(t, delta, Psi, alpha, beta, IaA0, IdA0, t0, V0):
    return (1 / 2) * np.exp(t0 + (-1) * t0 * delta + (-1 / 2) * t * ((-1) + 2 * beta + delta + Psi)) * (beta+(-1) * delta) ** (-1) * (beta ** 2 + ((-1) + beta) * delta) ** (-1) * (4 * beta + (-1) * (1 + delta) ** 2) ** (-1) * (2 * np.exp(t0 * ((-1) + beta + delta) + (1 / 2) * t * ((-1) + delta + Psi)) * IdA0 * beta * (beta + (-1) * delta) * (4 * beta + (-1) * (1 + delta) ** 2) + (-2) * np.exp(t0 * ((-1) + delta) + (1 / 2) * t * ((-1) + 2 * beta + delta + Psi)) * alpha * (beta ** 2 + ((-1) + beta) * delta) * ((-4) * beta + (1 + delta) ** 2) + np.exp((1 / 2) * t0 * ((-1) + delta + (-1) * Psi) + t * ((-1) + beta + delta + Psi)) * ((-1) * IdA0 * beta * (beta + (-1) * delta) * ((-1) * (1 + delta) ** 2 + ((-1) + delta) * Psi + 2 * beta * (2 + Psi)) + (beta ** 2 + ((-1) + beta) * delta) * (alpha * (1+(-4) * beta + delta * (2 + delta + (-1) * Psi) + Psi) + (beta + (-1) * delta) * (4 * IaA0 * beta+(-2) * (1 + V0) * Psi + IaA0 * (1 + delta) * ((-1) + (-1) * delta + Psi))))+np.exp(t * ((-1) + beta + delta)+(1 / 2) * t0 * ((-1) + delta + Psi))*(IdA0 * beta * (beta + (-1) * delta) * ((1 + delta) ** 2 + 2 * beta * ((-2) + Psi) + ((-1) + delta) * Psi) + (beta ** 2 + ((-1) + beta) * delta) * (alpha * ((-4) * beta + (1 + delta) ** 2 + ((-1) + delta) * Psi) + (beta + (-1) * delta) * (4 * IaA0 * beta+2 * (1+V0) * Psi + (-1) * IaA0 * (1 + delta) * (1 + delta + Psi)))))


def Idep(t, beta, IdA0, t0):
    return np.exp(((-1) * t + t0) * beta) * IdA0

def exp_cum(x, a, b):
    return a * (1 - np.exp(-b * x))

def monod(x, a, b, c, alp):
    return c + (a * np.exp(b) * x) / (alp + x)



def functionAGLIF(params,corr,myCurrent,Iadap_start,filename,p):
    EL = params[0]
    vres = params[1]
    vtm = params[2]
    Cm = params[3] * p
    ith = params[4] * p
    tao_m = params[5]
    sc = params[6] * p
    bet = params[7]
    delta1 = params[8]
    cost_idep_ini = params[9]
    Idep_ini_vr = params[10]
    psi1 = ((-4)*bet+((1+delta1)**2))**0.5#params[11]
    a = params[12]
    b = params[13]
    c = params[14]
    alp = params[15]
    istim_min_spikinig_exp = params[16]
    istim_max_spikinig_exp = params[17]
    sim_lenght = params[18]
    datiRette = params[19]

    time_scale = 1 / (-sc / (Cm * EL))
    H = (90+EL)*sc*(bet-delta1)/(EL*(-200))
    

    corrcostatratti = corr[0]
    corrsin1 = corr[1]
    corrsin2 = corr[2]  
    corrcost = corr[3]

    Istim = myCurrent
    campionamento=40
    d_dt=0.005*campionamento

    
    
    if corrcostatratti:
    	campionamento=40
    	d_dt=0.005*campionamento
    	Istim0=0
    	Istim1=1000
    	Istim2=800
    	Istim3=400
    	Istim4=1000
    	cor=np.ones(int(sim_lenght/d_dt))*Istim0
    	change_cur=200
    	cor[int(change_cur/d_dt):int(sim_lenght/d_dt)+1]=np.ones(len(cor[int(change_cur/d_dt):int(sim_lenght/d_dt)+1]))*Istim1
    	change_cur=400
    	cor[int(change_cur/d_dt):int(sim_lenght/d_dt)+1]=np.ones(len(cor[int(change_cur/d_dt):int(sim_lenght/d_dt)+1]))*Istim2
    	change_cur=600
    	cor[int(change_cur/d_dt):int(sim_lenght/d_dt)+1]=np.ones(len(cor[int(change_cur/d_dt):int(sim_lenght/d_dt)+1]))*Istim3
    	change_cur=800
    	cor[int(change_cur/d_dt):int(sim_lenght/d_dt)+1]=np.ones(len(cor[int(change_cur/d_dt):int(sim_lenght/d_dt)+1]))*Istim4
    if corrsin1 or corrsin2:
    	campionamento=10
    	d_dt=0.05*campionamento
    	corr_list=[]
    	fh = open('sync_w5+1000_1.txt','r')
    	for i in enumerate(fh):
    		corr_list.append(np.float64(i[1]))
    	fh.close()
    	cor=np.array(corr_list)
    	cor=cor[0:len(cor):campionamento]
    if corrcost:
    	Istim0=0
    	cor=np.ones(int(sim_lenght/d_dt))*Istim0
    	change_cur=0
    	cor[int(change_cur/d_dt):int(sim_lenght/d_dt)+1]=np.ones(len(cor[int(change_cur/d_dt):int(sim_lenght/d_dt)+1]))*Istim
    
    
    tic = time.perf_counter()
    
    Vconvfact=-EL
    vth=vtm/Vconvfact
    vrm=vres/Vconvfact
    
    t0_val=0
    vini_neg=EL
        
    ts=np.inf
    
    dt=d_dt/time_scale
    #print('dt= ',dt)
    init_sign=0
    ref_t=2
    
    t0_val=0
    psi1=((-4)*bet+((1+delta1)**2))**(0.5)
    
    Idep_ini=0
    Iadap_ini = Iadap_start
    out=[]
    t_out=[]
    
    t_final=t0_val+dt
    v_ini=-1
    
    mul=15
    
    f=open(filename, 'w')
    i=0
    
    soglia_sign=10
    Ide=[]
    Iada=[]
    Ide2=[]
    Iada2=[]
    tetalist=[]
    
    t_spk=-3*d_dt
    afirst=0
    meancorlastis=0
    stdcorlastis=0
    sis=0
    
    spikeCount = 0
    contaMonodNegativa = 0
    toBeErased = False
    sogliaMonodNegative = 2
    
    while(t_final*time_scale<sim_lenght):
    	if cor[i]>0:
    		sis=sis+1
    		if sis>=2:
    			meancorlastprec=meancorlastis
    			stdcorlastprec=stdcorlastis
    			stdcorlastis=((sis-2)*stdcorlastis+(sis-1)*meancorlastis**2+cor[i]**2-sis*(((sis-1)*meancorlastis+cor[i])/sis)**2)/(sis-1)
    			meancorlastis=((sis-1)*meancorlastis+cor[i])/sis
    		else:
    			stdcorlastis=0
    			meancorlastis=cor[i]
    	if (t_final-init_sign)*time_scale>=tagliorette(cor[i],datiRette):
    		if corrcostatratti:
    			if cor[i]>ith:
    				if cor[i-1]<ith or i==0:
    					init_sign=t_final
    					Idep_ini = cost_idep_ini*(cor[i]-ith)
    					Iadap_ini = Iadap_start
    				if cor[i-1]>ith and cor[i-1]<cor[i]:
    					init_sign=init_sign*(1+(cor[i-1]-ith)/cor[i-1])
    					Idep_ini = cost_idep_ini*(cor[i]-ith)
    		if corrsin1 or corrcost:
    			if cor[i]>ith:
    				if cor[i-1]<ith or i==0:
    					init_sign=t_final
    					Idep_ini = cost_idep_ini*(cor[i]-ith)
    					Iadap_ini = Iadap_start
    		if corrsin2:
    			if meancorlastprec+stdcorlastisprec>700:
    				if cor[i]>ith:
    					if meancorlastis+stdcorlastis>meancorlastprec and meancorlastis<cor[i]:
    						init_sign=init_sign*(1+(meancorlastis-ith)/(meancorlastis))
    						Idep_ini = cost_idep_ini*(cor[i]-ith)
    						Iadap_ini = Iadap_start
    			else:
    				if cor[i]>ith:
    					if meancorlastis+stdcorlastis>meancorlastprec+stdcorlastprec and meancorlastis<cor[i]:
    						init_sign=init_sign*(1+(meancorlastis-ith)/(meancorlastis))
    						Idep_ini = cost_idep_ini*(cor[i]-ith)
    						Iadap_ini = Iadap_start
    				
    		
    		if cor[i-1]==0:
    			v_ini=vini_prec
    		else:
    			v_ini =(EL+(1-np.exp(-cor[i]/1000))*(vtm-EL))/Vconvfact
    		vini_prec=v_ini
    		out.append(v_ini)
    		t_out.append(t_final)
    		Iada.append(Iadap_ini)
    		Ide.append(Idep_ini)
    		#print(i)
    	else:
    		vini_prec=v_ini
    		if (cor[i] < ith and cor[i]>=0) or i==0:
    			if cor[i-1]<0:
    				Iadap_ini= 90/EL +1 
    				Idep_ini=0
    				v_ini=vini_prec
    			if ((cor[i]/ sc) / (bet - delta1) - 1)<=v_ini: 
    				Idep_ini = 0
    				Iadap_ini = (cor[i] / sc) / (bet - delta1)
    				v_ini = ((cor[i] / sc) / (bet - delta1) - 1)
    			else:
    				v_ini = V(t_final, delta1, psi1,cor[i]/sc, bet, Iadap_ini, Idep_ini, t0_val, v_ini)
    				Iadap_ini = Iadap(t0_val, delta1, psi1, cor[i] / sc, bet, Iadap_ini, Idep_ini, t0_val, v_ini)
    				Idep_ini = Idep(t_final, bet, Idep_ini, t0_val)
    				#print(i)
                    
    			if v_ini * Vconvfact < -90:
    				v_ini = -90 / Vconvfact
    				Iadap_ini = Iadap_start
    
    			out.append(v_ini)
    		else:
    			if cor[i] < cor[i-1] and cor[i]>0 and (t_spk+2*d_dt)<t_final*time_scale:
    				teta=(out[i-1]/(cor[i-1] / sc))*(1/dt-delta1)-(out[i-2]/((cor[i-1] / sc)*dt))-delta1/(cor[i-1] / sc)-1
    				if teta<0:
    					teta=0
    				Idep_ini = Iadap_ini + teta * (cor[i] / sc) / bet
    				tetalist.append(teta)
    				v_ini = V(t=t_final, delta=delta1, Psi=psi1,alpha=cor[i]/sc, beta=bet, IaA0=Iadap_ini, IdA0=Idep_ini, t0=t0_val, V0=v_ini)
    				Iadap_ini = Iadap(t=t_final, delta=delta1, Psi=psi1,alpha=cor[i]/sc, beta=bet, IaA0=Iadap_ini, IdA0=Idep_ini, t0=t0_val, V0=v_ini)
    				Idep_ini = Idep(t=t_final, beta=bet,IdA0=Idep_ini, t0=t0_val)
    			else:
    				if cor[i]>0:
    					v_ini = V(t_final, delta=delta1, Psi=psi1,alpha=cor[i]/sc, beta=bet, IaA0=Iadap_ini, IdA0=Idep_ini, t0=t0_val, V0=v_ini)
    					Iadap_ini = Iadap(t=t_final, delta=delta1, Psi=psi1,alpha=cor[i]/sc, beta=bet, IaA0=Iadap_ini, IdA0=Idep_ini, t0=t0_val, V0=v_ini)
    					Idep_ini = Idep(t=t_final, beta=bet,IdA0=Idep_ini, t0=t0_val)
    			if cor[i-1]!=cor[i] and (cor[i]<0 and cor[i]>-200):
    				Iadap_ini= (90+EL)*cor[i]/(EL*(-200))  
    				Idep_ini=0
    				v_ini=vini_prec
    			if cor[i]<0 and cor[i]>-200:	
    				v_ini = V(t=t_final, delta=delta1, Psi=psi1, alpha=H * cor[i]/sc, beta=bet, IaA0=Iadap_ini, IdA0=Idep_ini, t0=t0_val, V0=v_ini)
    				Iadap_ini = Iadap(t=t_final, delta=delta1, Psi=psi1, alpha=H * cor[i]/sc, beta=bet, IaA0=Iadap_ini, IdA0=Idep_ini, t0=t0_val, V0=v_ini)
    				Idep_ini = Idep(t=t_final, beta=bet, IdA0=Idep_ini, t0=t0_val)
    			if cor[i-1]!=cor[i] and cor[i]<=-200:
    				Iadap_ini= 90/EL +1 
    				Idep_ini=0
    				v_ini=vini_prec
    			if cor[i]<=-200:
    				v_ini = V(t=t_final, delta=delta1, Psi=psi1, alpha=H * cor[i]/sc, beta=bet, IaA0=Iadap_ini, IdA0=Idep_ini, t0=t0_val, V0=v_ini)
    				Iadap_ini = Iadap(t=t_final, delta=delta1, Psi=psi1, alpha=H * cor[i]/sc, beta=bet, IaA0=Iadap_ini, IdA0=Idep_ini, t0=t0_val, V0=v_ini)
    				Idep_ini = Idep(t=t_final, beta=bet, IdA0=Idep_ini, t0=t0_val)
    			if v_ini*Vconvfact<-90:
    				v_ini =-90/Vconvfact
    				Iadap_ini = Iadap_start
    			out.append(v_ini)
    		t_out.append(t_final)
    		Iada.append(Iadap_ini)
    		Ide.append(Idep_ini)
    		if corrcostatratti:
    			if cor[i]>ith:
    				if cor[i-1]<ith or i==0:
    					init_sign=t_final
    					Idep_ini = cost_idep_ini*(cor[i]-ith)
    					Iadap_ini = Iadap_start
    				if cor[i-1]>ith and cor[i-1]<cor[i]:
    					init_sign=init_sign*(1+(cor[i-1]-ith)/cor[i-1])
    					Idep_ini = cost_idep_ini*(cor[i]-ith)
    					Iadap_ini = Iadap_start
    		if corrsin1 or corrcost or corrsin2:
    			if cor[i]>ith:
    				if cor[i-1]<ith or i==0:
    					init_sign=t_final
    					Idep_ini = cost_idep_ini*(cor[i]-ith)
    					Iadap_ini = Iadap_start
    		if v_ini>vth:
    			t_spk=t_final*time_scale
    			spikeCount = spikeCount + 1
    			f.write(str(round(t_spk,3)) + ' \n')
    			v_ini=vrm

    			if t_spk < expTimeSogliaMax(spikeCount,cor[i]):
    				#print('t_spk: '+str(t_spk) +' - valoreFunc: '+str(expTimeSogliaMax(spikeCount,cor[i])))
    				print('erase by boundary check')
    				toBeErased = True
    				break
    
    
    			#print('***spike***')
    			#print('t ',t_final, 'expTimeSogliaMax',str(expTimeSogliaMax(spikeCount,cor[i])) )
    			#print('************')
    			if cor[i]<istim_min_spikinig_exp or cor[i]>istim_max_spikinig_exp:
    
    				c_aux=0.8*Idep_ini_vr + (cor[i]/(sc)) / bet+(delta1/bet)*(1+vrm)-a*np.exp(b*cor[i]/1000)
    				Iadap_ini =  monod((t_final - init_sign) * time_scale, a, b * (cor[i] / 1000), c_aux, alp)

    			else:
    				Iadap_ini= monod((t_final-init_sign)*time_scale,a,b*(cor[i] / 1000),c,alp)
    				if Iadap_ini < 0:
    					Iadap_ini = 0
                           
    					contaMonodNegativa = contaMonodNegativa + 1         
    					if contaMonodNegativa>sogliaMonodNegative:
    					  print('erase by negative monod - curr:'+str(Istim))
    					  toBeErased = True
    					  break
         
    			if cor[i]<ith:
    				Idep_ini=0
    				Iadap_ini = Iadap_start
    			else:
    				Idep_ini=Idep_ini_vr


    			for k in range(int(ref_t / d_dt)):
    				out.append(v_ini)
    				t_final = t_final + dt
    				t_out.append(t_final)
    				Iada.append(Iadap_ini)
    				Ide.append(Idep_ini)
    				i = i + 1
    				sis=sis+1
    				if sis>=2:
    					try:
    						stdcorlastis=((sis-2)*stdcorlastis+(sis-1)*meancorlastis**2+cor[i]**2-sis*(((sis-1)*meancorlastis+cor[i])/sis)**2)/(sis-1)
    						meancorlastis=((sis-1)*meancorlastis+cor[i])/sis
    					except IndexError:
    						aa=1
    				else:
    					stdcorlastis=0
    					meancorlastis=cor[i]
    		vini_prec=v_ini
    
    
    	i = i + 1
    	t0_val = t_final
    	t_final = t0_val+dt
        
    
    toc = time.perf_counter()
    #print(f"time: {toc - tic:0.4f} seconds")
    plt.show()
    
    f.close()

    return(toBeErased)
    
    
    
'''
---------------------------------------------------------------------
MAIN
---------------------------------------------------------------------
'''


print('-------------------------------------------------')
toBeErased = False

for nomeNeurone in neuroni:
    #print(nomeNeurone)
       
    '''
    ---------------------------------------------------------------------------------------------------------------------------
    CALL TO AGLIF FUNCTION
    ---------------------------------------------------------------------
    '''
    
    #'cost_idep_ini':
    Idep_start = pyramidalNeuronsDatabase[nomeNeurone]['parameters']['Idep_ini']
    #'Idep_ini_vr':
    Idep0 = pyramidalNeuronsDatabase[nomeNeurone]['parameters']['Idep_ini_vr']
    #'c':
    paramC = pyramidalNeuronsDatabase[nomeNeurone]['parameters']['C']
    #'d - eta':
    paramD = pyramidalNeuronsDatabase[nomeNeurone]['parameters']['alphaD']
    
    paramValues1 = np.multiply(Idep_start,coeffParamPar1)
    paramValues2 = np.multiply(Idep0,coeffParamPar2)
    paramValues3 = np.multiply(paramC,coeffParamPar3)
    paramValues4 = np.multiply(paramD,coeffParamPar4)
    
    # auxiliary variables
    sc = pyramidalNeuronsDatabase[nomeNeurone]['parameters']['sc']
    ith = pyramidalNeuronsDatabase[nomeNeurone]['parameters']['Ith']
    cost_idep_ini = pyramidalNeuronsDatabase[nomeNeurone]['parameters']['Idep_ini']
    bet = pyramidalNeuronsDatabase[nomeNeurone]['parameters']['bet']
    
    removedCount = 0
    
    for i,par1 in enumerate(paramValues1):
        for j,par2 in enumerate(paramValues2):
            for c,par3 in enumerate(paramValues3):
                for d,par4 in enumerate(paramValues4):
                    for par5 in coeffParamPar5:
                        # block lines: 
                        retteOrig = [float(x) for x in list(pyramidalNeuronsDatabase[nomeNeurone]['block_line_params'].values())]
                        rette = [0,0,0,0,0,0]
                        for it in range(6):
                            if retteOrig[it]>1000000:                                
                                rette[it] = np.inf
                            elif retteOrig[it]<-1000000:                                
                                rette[it] = -np.inf
                            else:
                                rette[it] = retteOrig[it]+retteOrig[it]*par5[it]/100
                                
                                
                            
                        for p in coeffParamPar6:
                                
                            # params
                            '''
                            EL = np.float64(valoriExp[0])
                            vres = np.float64(valoriExp[1])
                            vtm = np.float64(valoriExp[2])
                            Cm = np.float64(valoriPar[0])
                            ith = np.float64(valoriPar[1])
                            tao_m = np.float64(valoriPar[2])
                            sc = np.float64(valoriPar[3])
                            bet = np.float64(valoriPar[5])
                            delta1 = np.float64(valoriPar[6])
                            cost_idep_ini = np.float64(valoriPar[7])
                            Idep_ini_vr = np.float64(valoriPar[8])
                            psi1 = np.float64(valoriPar[9])
                            a = np.float64(valoriPar[11])
                            b = np.float64(valoriPar[12])
                            c = np.float64(valoriPar[13])
                            alp = np.float64(valoriPar[14])
                            istim_min_spikinig_exp = int(valoriExp[3].split(",")[0])
                            istim_max_spikinig_exp = int(valoriExp[3].split(",")[-1])
                            sim_lenght = 1000
                            rette
                            '''
                            params = [
                                pyramidalNeuronsDatabase[nomeNeurone]['parameters']['EL'],
                                pyramidalNeuronsDatabase[nomeNeurone]['parameters']['Vres'],
                                pyramidalNeuronsDatabase[nomeNeurone]['parameters']['VTM'],
                                pyramidalNeuronsDatabase[nomeNeurone]['parameters']['Cm'],
                                pyramidalNeuronsDatabase[nomeNeurone]['parameters']['Ith'],
                                pyramidalNeuronsDatabase[nomeNeurone]['parameters']['tao'],
                                pyramidalNeuronsDatabase[nomeNeurone]['parameters']['sc'],
                                pyramidalNeuronsDatabase[nomeNeurone]['parameters']['bet'],
                                pyramidalNeuronsDatabase[nomeNeurone]['parameters']['delta1'],
                                par1,
                                par2,
                                pyramidalNeuronsDatabase[nomeNeurone]['parameters']['psi'],
                                pyramidalNeuronsDatabase[nomeNeurone]['parameters']['A'],
                                pyramidalNeuronsDatabase[nomeNeurone]['parameters']['B'],
                                par3,
                                par4,
                                200,
                                1000,
                                400,
                                rette
                                ]
                            

                       
                            for k,iadap in enumerate(coeffParamPar0):
                                
                                for selCurr in correnti:   
                                    
                                    alfa = selCurr / sc

                                    maxIadap = alfa/bet + cost_idep_ini*(selCurr-ith)
                                
                                    iadapVariabile = np.multiply(maxIadap,coeffParamPar0) 
                                
                                    stringaRetta = str(par5).replace(', ','_').replace('[','').replace(']','')
                                    nomefile = nomeNeurone+'/'+nomeNeurone+'_t_spk_'+str(selCurr)+'pA_iadap_'+str(coeffParamPar0[k])+'_Idep_'+str(coeffParamPar1[i])+'_Idep0_'+str(coeffParamPar2[j])+'_c_'+str(coeffParamPar3[c])+'_d_'+str(coeffParamPar4[d])+'_R_'+stringaRetta+'_p_'+str(p)+'.txt'
                            
                                    valid = functionAGLIF(params,corr,selCurr,iadap,nomefile,p)
                                    
                                    if valid == True:
                                        toBeErased = 1
                                
                            # after evaluating all currents delete all file if even just one is unvalid
                                if toBeErased == 1:
                                    #print('deleting files from validation procedure')
                                    removedCount = removedCount + 1
                                    toBeErased = 0

                                    filesToErase = glob.glob(nomeNeurone+'/*pA_iadap_'+str(coeffParamPar0[k])+'_Idep_'+str(coeffParamPar1[i])+'_Idep0_'+str(coeffParamPar2[j])+'_c_'+str(coeffParamPar3[c])+'_d_'+str(coeffParamPar4[d])+'_R_'+stringaRetta+'_p_'+str(p)+'.txt')
                                    if len(filesToErase)!=0:
                                      try:
                                        for nomefileTbd in filesToErase:
                                            os.remove(nomefileTbd)
                                            #print('erase procedure')
                                      except:
                                        print("error in deleting file")
                                            
                                        
                                   
    maxCopies = len(coeffParamPar0)*len(coeffParamPar1)*len(coeffParamPar2)*len(coeffParamPar3)*len(coeffParamPar4)*len(coeffParamPar5)*len(coeffParamPar6)
    #print('-------------')
    #print('maximum number of copies expected: '+ str(maxCopies))
    #print('-------------')
    
    print(nomeNeurone + ' Exepect: '+ str(maxCopies)+' Resulting:'+str(maxCopies-removedCount))
                 
        
    

