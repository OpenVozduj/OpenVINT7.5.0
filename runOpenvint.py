#%% Libraries
import numpy as np
from src import prePostFunc as pp
from src import operGenetics as og
from src import aeroPropeller as ap
import time
import os

#%% Load surrogate model
model = ap.setup_aero_surrogate()
og.GLOBAL_MODEL = model

#%% Preprocess
t0 = time.time()
fileInputs = 'inputs.json'
Omega, flow, opt, paramBEMT, motor, mass = pp.readInputs(fileInputs)
rootDir = os.getcwd()
Metrics = pp.initMetrics()

#%% Initial population
print('Start optimization \n')
print('Genration 0 \n')
os.makedirs('tests/' + opt['testDir'], exist_ok=True)
fileGen = 'tests/' + opt['testDir'] + '/G0'
os.makedirs(fileGen, exist_ok=True)

Pg, bladeg = og.initialPopulation(fileGen, opt['N'], Omega)
TWQg = og.objectiveFunction(opt['N'], Pg, bladeg, flow, paramBEMT, mass, motor)

#%% Penalization function
fp, _ = og.compute_fitness(TWQg[:,12], TWQg[:,0], opt['Tmin'], TWQg[:,2], TWQg[:,3], opt['Qtol'])
Metrics = og.updateMetrics(Metrics, TWQg, fp)
NEF = opt['N']
print(f"Gen 0 | W_current = {Metrics['W_opt'][-1]:.2f}")
print(f"Gen 0 | T_current = {Metrics['T_opt'][-1]:.2f}")
print(f"Gen 0 | Q_u_current = {Metrics['Q_u_opt'][-1]:.2f}")
print(f"Gen 0 | Q_l_current = {Metrics['Q_l_opt'][-1]:.2f}")
print(f"Gen 0 | W_elec_u_current = {Metrics['W_e_u_opt'][-1]:.2f}")
print(f"Gen 0 | W_elec_l_current = {Metrics['W_e_l_opt'][-1]:.2f}")
print(f"Gen 0 | mass_total_current = {Metrics['m_total_opt'][-1]:.2f}")

#%% Generations
for g in range(1, opt['G']):
    print('Genration '+str(g)+'\n')
    fileGen = 'tests/' + opt['testDir'] + '/G' + str(g)
    os.makedirs(fileGen, exist_ok=True)
    Ng, NE = og.newSizePop(opt['N'], opt['G'], g)
    E = og.elitism(Pg, fp, NE)
    C = og.SBX(Pg, Ng-NE, fp, g, opt['G'])
    Q = og.mutation(C, Ng-NE, Omega, g, opt['G'])
    
    Pg = np.vstack((E, Q))
    for i in range(Ng):
        bladeg[i] = fileGen+'/blade_'+str(i)
    TWQg = og.objectiveFunction(Ng, Pg, bladeg, flow, paramBEMT, mass, motor)

    fp, _ = og.compute_fitness(TWQg[:,12], TWQg[:,0], opt['Tmin'], TWQg[:,2], TWQg[:,3], opt['Qtol'])
    Metrics = og.updateMetrics(Metrics, TWQg, fp)
    NEF += Ng
    print(f"Gen {g} | W_current = {Metrics['W_opt'][-1]:.2f}")
    print(f"Gen {g} | T_current = {Metrics['T_opt'][-1]:.2f}")
    print(f"Gen {g} | Q_u_current = {Metrics['Q_u_opt'][-1]:.2f}")
    print(f"Gen {g} | Q_l_current = {Metrics['Q_l_opt'][-1]:.2f}")
    print(f"Gen {g} | W_elec_u_current = {Metrics['W_e_u_opt'][-1]:.2f}")
    print(f"Gen {g} | W_elec_l_current = {Metrics['W_e_l_opt'][-1]:.2f}")
    print(f"Gen {g} | mass_total_current = {Metrics['m_total_opt'][-1]:.2f}")

    if NEF >= 4000:
        break
    if og.check_convergence(Metrics['W_opt'], threshold=0.0005, patience=15):
        break
        
    
#%% Results
print('Optimization completed \n')
timeCPU = time.time() - t0
print('Time CPU '+str(round(timeCPU,2))+' s')

pp.printPlots(g, Metrics, 'tests/'+opt['testDir'])
pp.saveBladeOpt(fp, bladeg, 'tests/'+opt['testDir'])
#%%
pp.blade3D('tests/'+opt['testDir'])


