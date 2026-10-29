import numpy as np
from scipy.stats import qmc
from src import geoBlade as gb
from src import aeroPropeller as ap
from src import electricFun as ef
import os
import json
import math
# import concurrent.futures

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):  
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)
    
def initialPopulation(fileGen, N, Omega):
    D = len(Omega)
    Dh = 0
    var = []
    for j in range(D):
        if Omega[j,0]!=Omega[j,1]:
            Dh += 1
            var.append(j)
    sampler = qmc.LatinHypercube(d=Dh)
    sample = sampler.random(n=N)
    LHx = qmc.scale(sample, Omega[var,0], Omega[var,1])
    Pg = np.zeros((N, D))
    idg = []
    for i in range(N):
        idg.append(fileGen+'/blade_'+str(i))
        k = 0
        for j in range(D):
            if (j in var) == True:
                Pg[i,j] = LHx[i,k]
                k += 1
            else:
                Pg[i,j] = Omega[j,0]
    Pg[:,8] = np.round(Pg[:,8])
    return Pg, idg


def objectiveFunction(N, Pg, nameBlade, flow, paramBEMT, mass, motor):
    global GLOBAL_MODEL
    model = GLOBAL_MODEL
    
    nu = flow['nu']
    rho = flow['rho']
    Vaxial = flow['V_axial']
    Vtrans = flow['V_trans']
    mass_eng = mass['m_engine'] * 2
    rho_prop = mass['rho_prop']
    Vinf = np.sqrt(Vaxial**2 + Vtrans**2)
    alphaShaft = np.degrees(np.arctan2(Vtrans, Vaxial))
    
    W_elec_u_g = np.array([])
    W_elec_l_g = np.array([])
    mass_prop_g = np.array([])
    mass_bat_g = np.array([])
    mass_total_g = np.array([])
    T_g = np.array([])
    W_g = np.array([])
    T_u_g = np.array([])
    T_l_g = np.array([])
    Q_u_g = np.array([])
    Q_l_g = np.array([])
    FM_u_g = np.array([])
    FM_l_g = np.array([])
    for i in range(N):
        os.makedirs(nameBlade[i], exist_ok=True)
        dS, volume_prop = gb.paramBlade(Pg[i], nameBlade[i], paramBEMT['rR_min'],
                                        paramBEMT['rR_max'], paramBEMT['sect_body'], 
                                        paramBEMT['sect_tip'])
        mass_prop = volume_prop * rho_prop * 2
        
        CY_batch, CX_batch = ap.reconstruct_aero_fields(model, dS[:,19])
        aeroResults = ap.TWQCoaxial_Oblique_Vectorized_Geo(Pg[i][14], Pg[i][19], Pg[i][8], Pg[i][9], 
                                                           dS[:,17], dS[:,14], dS[:,15], dS[:,16], 
                                                           CY_batch, CX_batch, Pg[i][9]*flow['Z_d'], rho, nu, 
                                                           V_inf=Vinf, alpha_shaft_deg=alphaShaft
                                                           )
        
        T, W, _, _, _, Cw, f_wake, T_u, T_l, _, _, Q_u, Q_l, FM_u, FM_l, _, _, alpha_u, alpha_l, Re_u, Re_l, _, _, _, _ = aeroResults
        
        if math.isnan(T):
            W_res, T_res, Q_u_res, Q_l_res = 100000, 100000, 100000, 100000
            W_elec_u = 100000
            W_elec_l = 100000
            mass_bat = 100000
        else:
            W_res, T_res, Q_u_res, Q_l_res = W, T, Q_u, Q_l
            elecResults_u = ef.electricPow(Pg[i][14], Q_u_res, motor)
            elecResults_l = ef.electricPow(Pg[i][19], Q_l_res, motor)
            W_elec_u = elecResults_u['battery_power_W']
            W_elec_l = elecResults_l['battery_power_W']
            mass_bat = ef.battery_mass(W_elec_u, W_elec_l, mass['autonomy'], mass['gamma_bat'])
        
        mass_total = mass_eng + mass_bat + mass_prop
        
        info_prop = {
            'nameCase': nameBlade[i], 
            'Total Thrust': float(T_res), 
            'Thrust (upper)': float(T_u),
            'Thrust (lower)': float(T_l), 
            'Total Power': float(W_res), 
            'Torque (upper)': float(Q_u_res),
            'Torque (lower)': float(Q_l_res), 
            'FM (upper)': float(FM_u), 
            'FM (lower)': float(FM_l),
            'Cw': float(Cw), 
            'f_wake': float(f_wake),
            'Electric power (upper)': float(W_elec_u),
            'Electric power (lower)': float(W_elec_l),
            'Total electric power': float(W_elec_u+W_elec_l),
            'mass motors': float(mass_eng),
            'mass propellers': float(mass_prop),
            'mass batteries': float(mass_bat),
            'Total mass': float(mass_total),
            'V_axial': flow['V_axial'], 
            'V_trans': flow['V_trans'],
            'design_parameters': {
                'c_d_r': float(Pg[i][0]), 
                'c_d_m': float(Pg[i][1]), 
                'c_d_t': float(Pg[i][2]),
                'r_cdm': float(Pg[i][3]), 
                'yt_r': float(Pg[i][4]), 
                'yt_m': float(Pg[i][5]),
                'yt_t': float(Pg[i][6]), 
                'r_yt': float(Pg[i][7]), 
                'B': int(Pg[i][8]), 
                'd': float(Pg[i][9]),
                'phi_r_up': float(Pg[i][10]), 
                'phi_m_up': float(Pg[i][11]), 
                'phi_t_up': float(Pg[i][12]),
                'r_phim_up': float(Pg[i][13]),
                'n_min_up': float(Pg[i][14]),
                'phi_r_low': float(Pg[i][15]),
                'phi_m_low': float(Pg[i][16]), 
                'phi_t_low': float(Pg[i][17]), 
                'r_phim_low': float(Pg[i][18]),
                'n_min_low': float(Pg[i][19]),
            },
            'info_sections': {
                'r_R': dS[:,13].tolist(), 
                'r': dS[:,17].tolist(), 
                'c': dS[:,14].tolist(),
                'phi_up': dS[:,15].tolist(), 
                'alpha_up': alpha_u.tolist(), 
                'phi_low': dS[:,16].tolist(),
                'alpha_low': alpha_l.tolist(), 
                'xt': dS[:,18].tolist(), 
                'yt': dS[:,19].tolist(),
                'xc': dS[:,20].tolist(), 
                'yc': dS[:,21].tolist(),
                'Re_u': Re_u.tolist(), 
                'Re_l': Re_l.tolist()
            }
        }
        
        with open(nameBlade[i] + '/info_prop.json', 'w') as outfile:
            json.dump(info_prop, outfile, indent=4, cls=NumpyEncoder)
            
        W_elec_u_g = np.append(W_elec_u_g, W_elec_u)
        W_elec_l_g = np.append(W_elec_l_g, W_elec_l)
        mass_prop_g = np.append(mass_prop_g, mass_prop)
        mass_bat_g = np.append(mass_bat_g, mass_bat)
        mass_total_g = np.append(mass_total_g, mass_total)
        T_g = np.append(T_g, T_res)
        W_g = np.append(W_g, W_res)
        T_u_g = np.append(T_u_g, T_u)
        T_l_g = np.append(T_l_g, T_l)
        Q_u_g = np.append(Q_u_g, Q_u_res)
        Q_l_g = np.append(Q_l_g, Q_l_res)
        FM_u_g = np.append(FM_u_g, FM_u)
        FM_l_g = np.append(FM_l_g, FM_l)
        
    TWQg = np.column_stack((T_g, W_g, Q_u_g, Q_l_g, T_u_g, T_l_g, FM_u_g, FM_l_g,
                            W_elec_u_g, W_elec_l_g, mass_prop_g, mass_bat_g, mass_total_g))
        
    return TWQg

def compute_fitness(mg, Tg, Tmin, Qg_u, Qg_l, Qtol):
    """
    Calcula fitness con manejo de restricciones tipo Deb (2000).
    - Factibles: fitness = Wg (valor real, sin normalizar)
    - Infactibles: fitness = peor_factible + penalización (siempre peor que cualquier factible)
    """
    # 1. Calcular violaciones de restricciones
    g1 = np.maximum(0, (Tmin - Tg) / Tmin)
    g2 = np.maximum(0, np.abs(Qg_u - Qg_l) - Qtol)
    cv = g1 + g2  # constraint violation total
    
    feasible = cv <= 1e-12
    
    fitness = np.full_like(mg, np.inf, dtype=float)
    
    if np.any(feasible):
        # --- Factibles: fitness = Wg crudo (estable entre generaciones) ---
        fitness[feasible] = mg[feasible]
        
        if np.any(~feasible):
            # --- Infactibles: SIEMPRE peores que el peor factible ---
            worst_feasible_mg = np.max(mg[feasible])
            best_feasible_mg = np.min(mg[feasible])
            spread = worst_feasible_mg - best_feasible_mg
            if spread < 1e-6:
                spread = abs(worst_feasible_mg) * 0.1  # fallback
            
            # Normalizar cv de infactibles a [0, 1]
            cv_inf = cv[~feasible]
            max_cv = np.max(cv_inf)
            cv_norm = cv_inf / max_cv if max_cv > 1e-12 else np.zeros_like(cv_inf)
            
            # Fitness infactible = peor_factible + spread + cv_norm * spread
            # Rango infactibles: [worst + spread, worst + 2*spread]
            # Siempre > cualquier factible (que está en [best, worst])
            fitness[~feasible] = worst_feasible_mg + spread + cv_norm * spread
    else:
        # --- No hay factibles: comparar solo por violación ---
        max_cv = np.max(cv)
        if max_cv > 1e-12:
            fitness = cv / max_cv
        else:
            fitness = np.zeros_like(mg)
    
    return fitness, cv

def newSizePop(N0, G, g, alpha=5, Nend=20):
    c = -np.log((Nend-alpha)/(N0+alpha))/G
    Ng = round((N0+alpha)*np.exp(-c*g)-alpha)
    if Ng < Nend:
        Ng = Nend
    if Ng%2 == 0:
        NE = 2
    else:
        NE = 3
    return Ng, NE

def tournament(N, fp, g, G, pt_min=0.6, pt_max=0.9):
    
    pt = pt_min + (pt_max - pt_min) * (g / G)
    
    xr = np.random.choice(np.arange(N), 2, replace=False)
    r = np.random.rand()
    
    if r < pt:
        if fp[xr[0]] < fp[xr[1]]:
            p = xr[0]
        else:
            p = xr[1]
    else:
        if fp[xr[0]] < fp[xr[1]]:
            p = xr[1]
        else:
            p = xr[0]
    return p

def SBX(Pg, N, fp, g, G, eta_c_min=5, eta_c_max=10):
    D = np.size(Pg, 1)
    C = np.empty([0, D])
    
    eta_c = eta_c_max - (eta_c_max - eta_c_min) * (1 - g / G)
    
    for i in range(int(N/2)):
        while 1 == 1:
            p1 = tournament(N, fp, g, G)
            p2 = tournament(N, fp, g, G)
            if p1 != p2:
                break
        P1 = Pg[p1]
        P2 = Pg[p2]
        C1 = np.zeros_like(P1)
        C2 = np.zeros_like(P2)
        for j in range(D):
            u = np.random.uniform()
            if u < 0.5:
                beta = (2*u)**(1/(eta_c+1))
            else:
                beta = 1/((2*(1-u))**(1/(eta_c+1)))
            C1[j] = 0.5*((1-beta)*P1[j] + (1+beta)*P2[j])
            C2[j] = 0.5*((1+beta)*P1[j] + (1-beta)*P2[j])
        C = np.row_stack((C, C1, C2))
    return C

def mutation(C, N, Omega, g, G, eta_m_min=5, eta_m_max=10):
    D = np.size(C, 1)
    Q = np.empty([0, D])
    
    pm = 1/D
    eta_m = eta_m_max - (eta_m_max - eta_m_min) * (1 - g / G)
    
    for i in range(N):
        p = C[i]
        m = np.copy(p) # Copiamos el padre, solo mutaremos algunos genes
        
        for j in range(D):
            # Solo mutar el gen si cae dentro de la probabilidad pm
            if np.random.rand() < pm:
                r = np.random.uniform()
                if r < 0.5:
                    delta = (2.0 * r) ** (1.0 / (eta_m + 1.0)) - 1.0
                else:
                    delta = 1.0 - (2.0 * (1.0 - r)) ** (1.0 / (eta_m + 1.0))
                    
                m[j] = p[j] + delta * (Omega[j, 1] - Omega[j, 0])
                
        # Asegurar límites
        m = np.clip(m, Omega[:, 0], Omega[:, 1])
        
        # Redondear variable discreta (índice 8) DESPUÉS del clip para asegurar que esté en rango
        m[8] = round(m[8]) 
        
        Q = np.vstack((Q, m))
    return Q

def elitism(Pg, fp, NE):
    sorted_indices = np.argsort(fp)[:NE]
    E = Pg[sorted_indices].copy()       
    return E

def updateMetrics(metrics, TWQg, fp):
    iopt = np.argmin(fp)
    metrics['T_opt'] = np.append(metrics['T_opt'], TWQg[iopt,0])
    metrics['W_opt'] = np.append(metrics['W_opt'], TWQg[iopt,1])
    metrics['Q_u_opt'] = np.append(metrics['Q_u_opt'], TWQg[iopt,2])
    metrics['Q_l_opt'] = np.append(metrics['Q_l_opt'], TWQg[iopt,3])
    metrics['T_u_opt'] = np.append(metrics['T_u_opt'], TWQg[iopt,4])
    metrics['T_l_opt'] = np.append(metrics['T_l_opt'], TWQg[iopt,5])
    metrics['FM_u_opt'] = np.append(metrics['FM_u_opt'], TWQg[iopt,6])
    metrics['FM_l_opt'] = np.append(metrics['FM_l_opt'], TWQg[iopt,7])
    metrics['W_e_u_opt'] = np.append(metrics['W_e_u_opt'], TWQg[iopt,8])
    metrics['W_e_l_opt'] = np.append(metrics['W_e_l_opt'], TWQg[iopt,9])
    metrics['m_prop_opt'] = np.append(metrics['m_prop_opt'], TWQg[iopt,10])
    metrics['m_battery_opt'] = np.append(metrics['m_battery_opt'], TWQg[iopt,11])
    metrics['m_total_opt'] = np.append(metrics['m_total_opt'], TWQg[iopt,12])
    metrics['Fitness_opt'] = np.append(metrics['Fitness_opt'], min(fp))
    return metrics

def check_convergence(W_history, threshold=0.001, patience=5):

    if len(W_history) < 2:
        return False
    
    consecutive = 0
    
    for i in range(len(W_history) - 1, 0, -1):
        curr = W_history[i]
        prev = W_history[i - 1]
        
        if prev == 0:
            variation = 0
        else:
            variation = abs(curr - prev) / abs(prev)
        
        if variation < threshold:
            consecutive += 1
        else:
            break  
    
    return consecutive >= patience