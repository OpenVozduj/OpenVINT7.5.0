import numpy as np
from sklearn.decomposition import PCA
from scipy.interpolate import CubicSpline, interpn

def setup_aero_surrogate(n_components_cy=5, n_components_cx=8):
    """
    Entrena el modelo sustituto usando PCA para AMBOS coeficientes (CY y CX).
    Nota: CX suele requerir más componentes que CY porque su campo es más 
    complejo (cubetas de arrastre, stall, etc.).
    """
    alpha_grid = np.arange(-10, 11, 1) 
    Re_grid = np.array([50000, 150000, 250000, 350000])
    yt_array = np.load('src/yt_ref.npy')
    CY_tensor = np.load('src/CY_tensor.npy')
    CX_tensor = np.load('src/CX_tensor.npy')
    
    yt_array = np.array(yt_array)
    alpha_grid = np.array(alpha_grid)
    Re_grid = np.array(Re_grid)
    
    # Verificación de seguridad para el orden de las dimensiones
    if CY_tensor.shape[1] == len(alpha_grid) and CY_tensor.shape[2] == len(Re_grid):
        CY_tensor = np.transpose(CY_tensor, (0, 2, 1))
        CX_tensor = np.transpose(CX_tensor, (0, 2, 1))
        
    N_Re = len(Re_grid)
    N_alpha = len(alpha_grid)
    
    # ==========================================
    # PCA para CY (Sustentación)
    # ==========================================
    CY_flat = CY_tensor.reshape(len(yt_array), -1)
    pca_cy = PCA(n_components=n_components_cy)
    scores_cy = pca_cy.fit_transform(CY_flat)
    splines_cy = [CubicSpline(yt_array, scores_cy[:, i]) for i in range(n_components_cy)]
    
    # ==========================================
    # PCA para CX (Arrastre) - ¡LA CORRECCIÓN CLAVE!
    # ==========================================
    CX_flat = CX_tensor.reshape(len(yt_array), -1)
    pca_cx = PCA(n_components=n_components_cx)
    scores_cx = pca_cx.fit_transform(CX_flat)
    splines_cx = [CubicSpline(yt_array, scores_cx[:, i]) for i in range(n_components_cx)]
    
    model_state = {
        'pca_cy': pca_cy,
        'splines_cy': splines_cy,
        'pca_cx': pca_cx,       # Nuevo
        'splines_cx': splines_cx, # Nuevo
        'alpha_grid': alpha_grid,
        'Re_grid': Re_grid,
        'N_alpha': N_alpha,
        'N_Re': N_Re
    }
    
    print(f"✅ Modelo sustituto configurado. Tensores validados con forma (N_perfiles, {N_Re}, {N_alpha}).")
    print(f"   - CY usa {n_components_cy} componentes PCA.")
    print(f"   - CX usa {n_components_cx} componentes PCA.")
    return model_state

def reconstruct_aero_fields(model_state, yt_batch):
    """
    Reconstruye las matrices 3D completas (N_secciones, N_Re, N_alpha) 
    para un lote de espesores dados, usando PCA para ambos coeficientes.
    """
    yt_batch = np.atleast_1d(yt_batch)
    N = len(yt_batch)
    
    # 1. Reconstruir CY
    scores_cy_new = np.column_stack([spline(yt_batch) for spline in model_state['splines_cy']])
    CY_flat_new = scores_cy_new @ model_state['pca_cy'].components_ + model_state['pca_cy'].mean_
    CY_batch = CY_flat_new.reshape(N, model_state['N_Re'], model_state['N_alpha'])
    
    # 2. Reconstruir CX 
    scores_cx_new = np.column_stack([spline(yt_batch) for spline in model_state['splines_cx']])
    CX_flat_new = scores_cx_new @ model_state['pca_cx'].components_ + model_state['pca_cx'].mean_
    CX_batch = CX_flat_new.reshape(N, model_state['N_Re'], model_state['N_alpha'])
    
    return CY_batch, CX_batch

ALPHA_GRID = np.arange(-10, 11, 1)
RE_GRID = np.array([50000, 150000, 250000, 350000])
CD_MAX_STALL = 1.2

def _prep_aero_tables(CY_batch, alpha_grid=ALPHA_GRID):
    """Precomputa por tabla: dCl/dalpha [1/deg] y alpha0 (Cl=0) por fila de Re."""
    slopes, alpha0s = [], []
    for tbl in CY_batch:
        tbl = np.asarray(tbl, dtype=float)
        slope = np.gradient(tbl, alpha_grid, axis=1)
        a0 = np.zeros(tbl.shape[0])
        for i, row in enumerate(tbl):
            cross = np.where(np.diff(np.signbit(row)))[0]
            if cross.size:
                k = int(cross[0])
                den = row[k+1] - row[k]
                if abs(den) > 1e-12:
                    a0[i] = alpha_grid[k] - row[k]*(alpha_grid[k+1]-alpha_grid[k])/den
                else:
                    a0[i] = alpha_grid[k]
            else:
                # [M-7] Fallback: ajuste lineal en la región [-5, 5] grados
                m = (alpha_grid >= -5.0) & (alpha_grid <= 5.0)
                p = np.polyfit(alpha_grid[m], row[m], 1)
                a0[i] = -p[1]/p[0] if abs(p[0]) > 1e-12 else 0.0
        slopes.append(slope)
        alpha0s.append(a0)
    return slopes, alpha0s

def read_aero_values(CY, CX, SL, A0, alpha_targets, re_targets):
    alpha_targets = np.atleast_1d(alpha_targets)
    re_targets = np.atleast_1d(re_targets)
    N = len(alpha_targets)
    a_min, a_max = ALPHA_GRID[0], ALPHA_GRID[-1]
    re_t = np.clip(re_targets, RE_GRID[0], RE_GRID[-1])
    pts = (RE_GRID, ALPHA_GRID)

    cl = np.zeros(N); cd = np.zeros(N); sl = np.zeros(N)
    a0 = np.interp(re_t, RE_GRID, A0)

    for i in range(N):
        a, re = alpha_targets[i], re_t[i]
        if a_min <= a <= a_max:
            cl[i] = interpn(pts, CY, (re, a), method='linear',
                            bounds_error=False, fill_value=None)[0]
            cd[i] = interpn(pts, CX, (re, a), method='linear',
                            bounds_error=False, fill_value=None)[0]
            sl[i] = interpn(pts, SL, (re, a), method='linear',
                            bounds_error=False, fill_value=None)[0]
        else:
            a_edge = a_max if a > a_max else a_min
            da = np.radians(a - a_edge)
            cl_e = interpn(pts, CY, (re, a_edge), method='linear',
                           bounds_error=False, fill_value=None)[0]
            cd_e = interpn(pts, CX, (re, a_edge), method='linear',
                           bounds_error=False, fill_value=None)[0]
            cl[i] = cl_e * np.cos(da)
            cd[i] = cd_e + (CD_MAX_STALL - cd_e) * np.abs(np.sin(da))
            sl[i] = interpn(pts, SL, (re, a_edge), method='linear',
                            bounds_error=False, fill_value=None)[0]
    return cl, cd, sl, a0

def getCoeffsFast(CY_batch, CX_batch, SL_batch, A0_batch, Re_flat, alpha_flat, N_r):
    """[M-6] Convención correcta: CY_batch[j] = tabla de la estación radial j.
    El plano está organizado psi-mayor, así que se itera por estación con
    rebanadas escalonadas j::N_r."""
    N_tot = len(Re_flat)
    N_psi = int(N_tot / N_r)
    cl = np.zeros(N_tot); cd = np.zeros(N_tot)
    sl = np.zeros(N_tot); a0 = np.zeros(N_tot)
    for j in range(N_r):
        idx = slice(j, N_tot, N_r)
        cl_j, cd_j, sl_j, a0_j = read_aero_values(
            CY_batch[j], CX_batch[j], SL_batch[j], A0_batch[j],
            alpha_flat[idx], Re_flat[idx])
        cl[idx] = cl_j; cd[idx] = cd_j; sl[idx] = sl_j; a0[idx] = a0_j
    return cl, cd, sl, a0

# ==============================================================================
# CORRECCIÓN 3D FÍSICA: Du-Selig (retardo de pérdida rotacional)
# + penalización de CD dependiente de la geometría local
# ==============================================================================
def apply_3d_corrections(cl_2d, cd_2d, slope, alpha0, alpha_deg, r, R, c,
                         k_ds=0.5, k_cd=0.15):
    cr = c / np.maximum(r, 0.05 * R)          # c/r local: la clave
    f_ds = np.clip(k_ds * cr, 0.0, 1.0)
    cl_p = slope * (alpha_deg - alpha0)       # sustentación adherida virtual
    cl_3d = cl_2d + f_ds * (cl_p - cl_2d)
    cd_3d = cd_2d * (1.0 + k_cd * cr)         # reemplaza al x1.15 global
    return cl_3d, cd_3d

def calculateClimbMetrics(Thrust, Power, V_axial, rho, R):
    """FM hover-equivalente = T^1.5 / (P * sqrt(2*rho*A)). Índice comparativo."""
    A = np.pi * R**2
    if Thrust > 0 and Power > 0:
        FM_hover_eq = (Thrust**1.5) / (Power * np.sqrt(2.0 * rho * A))
        P_net = max(Power - Thrust * V_axial, 0.0)
        v_h = np.sqrt(Thrust / (2.0 * rho * A))
        P_ideal_hover = Thrust * v_h
        return FM_hover_eq, P_ideal_hover, P_net
    return 0.0, 0.0, 0.0

def inducedVelocityGuess(V_axial, CT_guess, omega, R):
    lambda_c = V_axial / (omega * R)
    lambda_i = -0.5 * lambda_c + np.sqrt(0.25 * lambda_c**2 + 0.5 * CT_guess)
    return lambda_i * omega * R

def calculate_prandtl_tip_loss(nB, r, R, phi_inflow):
    sin_phi = np.maximum(np.sin(phi_inflow), 1e-4)
    f_tip = (nB / 2.0) * (R - r) / (r * sin_phi)
    F_tip = (2.0 / np.pi) * np.arccos(np.exp(-f_tip))
    return np.maximum(F_tip, 0.001)

# ==============================================================================
# 4) CARGAS DE UN ROTOR (bloque unificado superior/inferior)
# ==============================================================================
def _rotor_loads(phi_geo_g, vi_radial, skewK, omega, w_wake_g,
                 V_axial, V_trans, psi_g, r_g, c_g, dr_g, rr, nu, rho,
                 nB, R, CY_batch, CX_batch, SL_batch, A0_batch, N_r,
                 k_ds, k_cd):
    # Estela sesgada: primer armónico (Coleman): vi = vi0*(1 + K*(r/R)*sin(psi))
    f_psi = np.clip(1.0 + skewK * rr[None, :] * np.sin(psi_g), 0.15, None)
    Wa = V_axial + w_wake_g + vi_radial[None, :] * f_psi
    Wa = np.maximum(Wa, 0.05 * omega * R)

    Vt = omega * r_g + V_trans * np.sin(psi_g)
    Vres = np.hypot(Wa, Vt)
    phi_i = np.arctan2(Wa, Vt)
    alpha = phi_geo_g - np.degrees(phi_i)
    Re = Vres * c_g / nu

    cl, cd, sl, a0 = getCoeffsFast(CY_batch, CX_batch, SL_batch, A0_batch,
                                   Re.flatten(), alpha.flatten(), N_r)
    cl = cl.reshape(Vres.shape); cd = cd.reshape(Vres.shape)
    sl = sl.reshape(Vres.shape); a0 = a0.reshape(Vres.shape)
    cl, cd = apply_3d_corrections(cl, cd, sl, a0, alpha, r_g, R, c_g, k_ds, k_cd)

    F = calculate_prandtl_tip_loss(nB, r_g, R, phi_i)
    dN_psi = 0.5 * rho * Vres**2 * c_g * (cl*np.cos(phi_i) - cd*np.sin(phi_i)) \
             * F * nB * dr_g
    CT_g = cl * np.sin(phi_i) + cd * np.cos(phi_i)
    dQ_psi = 0.5 * rho * Vres**2 * c_g * CT_g * F * nB * r_g * dr_g
    return dict(alpha=alpha, Re=Re, Vres=Vres, phi_i=phi_i, cl=cl, cd=cd,
                dN_psi=dN_psi, dN_avg=np.mean(dN_psi, axis=0),
                Q=np.sum(np.mean(dQ_psi, axis=0)))


def TWQCoaxial_Oblique_Vectorized_Geo(n_min_u, n_min_l, nB, d, r, c, phi_u, phi_l,
                                      CY_batch, CX_batch, Z, rho, nu,
                                      V_inf=0.0, alpha_shaft_deg=0.0,
                                      CT_guess=0.008, N_psi=36,
                                      k_ds=0.50, k_cd=0.15,
                                      skew_gain=4.0/3.0, wake_blur=0.15,
                                      fz=None, max_iter=200, tol=5e-3,
                                      return_extras=False):
    """
    BEMT coaxial en flujo oblicuo, versión 3.0 (fusión KIMI + v2.0).

    Parámetros de calibración (con significado físico):
      k_ds     : fuerza Du-Selig (~0.3-1.0). k_ds=0 desactiva el retardo.
      k_cd     : penalización de CD local (reemplaza al antiguo x1.15).
      skew_gain: ganancia del primer armónico (4/3 = Coleman).
      wake_blur: ancho de transición del borde de estela [R].
      fz       : fracción de desarrollo de estela a altura Z; default Z/(Z+R).
                 Alternativa más física (disco actuador): fz = Z/sqrt(R^2+Z^2).
    """
    R = d / 2.0
    dr = np.gradient(r)
    dr[0] = r[1] - r[0]
    N_r = len(r)
    omega_u = 2.0 * np.pi * n_min_u / 60.0
    omega_l = 2.0 * np.pi * n_min_l / 60.0
    alpha_shaft = np.radians(alpha_shaft_deg)
    V_axial = V_inf * np.cos(alpha_shaft)
    V_trans = V_inf * np.sin(alpha_shaft)

    SL_batch, A0_batch = _prep_aero_tables(CY_batch, ALPHA_GRID)

    psi = np.linspace(0, 2*np.pi, N_psi, endpoint=False)
    psi_g = psi[:, None]
    r_g = r[None, :]
    c_g = c[None, :]
    dr_g = dr[None, :]
    rr = r / R
    X_g = r_g * np.cos(psi_g)
    Y_g = r_g * np.sin(psi_g)
    phi_u_g = np.tile(phi_u, (N_psi, 1))
    phi_l_g = np.tile(phi_l, (N_psi, 1))
    zero_wake = np.zeros((N_psi, N_r))

    # Estado: solo 2 campos radiales
    vi_u = np.full(N_r, inducedVelocityGuess(V_axial, CT_guess, omega_u, R))
    vi_l_self = np.full(N_r, inducedVelocityGuess(V_axial, CT_guess, omega_l, R))

    fz = Z / (Z + R) if fz is None else float(np.clip(fz, 0.0, 1.0))
    xi = 0.2
    converged = False
    it = 0

    for it in range(1, max_iter + 1):
        # =========================================================
        # Geometría de estela: un tubo de corriente por estación
        # =========================================================
        vi_u_c = np.maximum(vi_u, 1e-6)
        C_far = np.sqrt((V_axial + vi_u_c) / (V_axial + 2.0 * vi_u_c))
        C_z = 1.0 + (C_far - 1.0) * fz
        r_map = r * C_z
        R_wake = R * C_z[-1]
        vi_wake_up = vi_u_c * (1.0 + fz)

        v_conv = max(V_axial + 2.0 * np.mean(vi_u), 1e-3)
        delta_y = V_trans * Z / v_conv
        Cw_eff = float(C_z[-1])

        ru_of_rl = np.interp(r, r_map, r)
        vi_wake_on_lower = np.interp(ru_of_rl, r, vi_wake_up)

        # =========================================================
        # Factores de sesgo (Coleman, primer armónico)
        # =========================================================
        chi_u = np.arctan2(V_trans, V_axial + np.mean(vi_u))
        K_u = skew_gain * np.tan(0.5 * chi_u)
        chi_l = np.arctan2(V_trans, V_axial + np.mean(vi_l_self)
                           + 0.5 * np.mean(vi_wake_on_lower))
        K_l = skew_gain * np.tan(0.5 * chi_l)

        # =========================================================
        # Cargas
        # =========================================================
        res_u = _rotor_loads(phi_u_g, vi_u, K_u, omega_u, zero_wake,
                             V_axial, V_trans, psi_g, r_g, c_g, dr_g, rr, nu,
                             rho, nB, R, CY_batch, CX_batch, SL_batch, A0_batch,
                             N_r, k_ds, k_cd)

        dist = np.sqrt(X_g**2 + (Y_g - delta_y)**2)
        inside = np.clip(1.0 - (dist - R_wake) / (wake_blur * R), 0.0, 1.0)
        inside = np.where(dist <= R_wake, 1.0, inside)
        w_lower = inside * vi_wake_on_lower[None, :]

        res_l = _rotor_loads(phi_l_g, vi_l_self, K_l, omega_l, w_lower,
                             V_axial, V_trans, psi_g, r_g, c_g, dr_g, rr, nu,
                             rho, nB, R, CY_batch, CX_batch, SL_batch, A0_batch,
                             N_r, k_ds, k_cd)

        # =========================================================
        # Actualización de estados (momentum por tubo local)
        # =========================================================
        dA = 2.0 * np.pi * r * dr
        t_u = V_axial**2 / 4.0 + res_u['dN_avg'] / (2.0 * rho * dA + 1e-9)
        vi_u_req = -V_axial / 2.0 + np.sqrt(np.maximum(t_u, 0.0))

        w_avg = np.mean(inside, axis=0) * vi_wake_on_lower
        Veff = V_axial + w_avg
        t_l = Veff**2 / 4.0 + res_l['dN_avg'] / (2.0 * rho * dA + 1e-9)
        vi_l_req = -Veff / 2.0 + np.sqrt(np.maximum(t_l, 0.0))

        vi_u_new = (1 - xi) * vi_u + xi * np.clip(vi_u_req, 0.005, 0.5 * omega_u * R)
        vi_l_new = (1 - xi) * vi_l_self + xi * np.clip(vi_l_req, 0.005, 0.5 * omega_l * R)

        err = max(np.max(np.abs(vi_u_new - vi_u)) / (np.max(vi_u) + 1e-6),
                  np.max(np.abs(vi_l_new - vi_l_self)) / (np.max(vi_l_self) + 1e-6))
        vi_u, vi_l_self = vi_u_new, vi_l_new
        if err < tol:
            converged = True
            break

    # Recarga con estados convergidos
    res_u = _rotor_loads(phi_u_g, vi_u, K_u, omega_u, zero_wake,
                         V_axial, V_trans, psi_g, r_g, c_g, dr_g, rr, nu,
                         rho, nB, R, CY_batch, CX_batch, SL_batch, A0_batch,
                         N_r, k_ds, k_cd)
    dist = np.sqrt(X_g**2 + (Y_g - delta_y)**2)
    inside = np.clip(1.0 - (dist - R_wake) / (wake_blur * R), 0.0, 1.0)
    inside = np.where(dist <= R_wake, 1.0, inside)
    w_lower = inside * vi_wake_on_lower[None, :]
    res_l = _rotor_loads(phi_l_g, vi_l_self, K_l, omega_l, w_lower,
                         V_axial, V_trans, psi_g, r_g, c_g, dr_g, rr, nu,
                         rho, nB, R, CY_batch, CX_batch, SL_batch, A0_batch,
                         N_r, k_ds, k_cd)

    # =========================================================
    # Resultados
    # =========================================================
    Thrust_u = float(np.sum(res_u['dN_avg']))
    Thrust_l = float(np.sum(res_l['dN_avg']))
    Thrust_total = Thrust_u + Thrust_l
    Q_u = float(res_u['Q'])
    Q_l = float(res_l['Q'])
    Power_u = Q_u * omega_u
    Power_l = Q_l * omega_l
    Power_total = Power_u + Power_l
    P_net_total = max(Power_total - Thrust_total * V_axial, 0.0)

    n_s_u = n_min_u / 60.0
    alpha_p = Thrust_total / (rho * n_s_u**2 * d**4)
    beta_p = Power_total / (rho * n_s_u**3 * d**5)
    Torque_eq = Power_total / (2.0 * np.pi * n_s_u)

    FM_u, _, _ = calculateClimbMetrics(Thrust_u, Power_u, V_axial, rho, R)
    FM_l, _, _ = calculateClimbMetrics(Thrust_l, Power_l, V_axial, rho, R)
    A = np.pi * R**2
    V_tip = omega_u * R
    FM_global_raw = (Thrust_total**1.5) / (Power_total * np.sqrt(2.0 * rho * A))
    FM_global_coaxial = FM_global_raw / np.sqrt(2.0)

    # [M-8] Métricas honestas para flujo oblicuo (contrato v2.0)
    CT_total = Thrust_total / (rho * V_tip**2 * A)
    CP_total = Power_total / (rho * V_tip**3 * A)
    PL_total = Thrust_total / Power_total

    alpha_u_avg_out = np.mean(res_u['alpha'], axis=0)
    alpha_l_avg_out = np.mean(res_l['alpha'], axis=0)
    Re_u_avg = np.mean(res_u['Re'], axis=0)
    Re_l_avg = np.mean(res_l['Re'], axis=0)

    f_wake = float(np.mean(inside))

    outputs = (Thrust_total, Power_total, alpha_p, beta_p, Torque_eq,
               Cw_eff, f_wake, Thrust_u, Thrust_l, Power_u, Power_l,
               Q_u, Q_l, FM_u, FM_l, FM_global_coaxial, P_net_total,
               alpha_u_avg_out, alpha_l_avg_out, Re_u_avg, Re_l_avg,
               CT_total, CP_total, PL_total, FM_global_raw)

    if return_extras:
        extras = dict(converged=converged, iterations=it,
                      C_far=C_far, C_z=C_z, delta_y=delta_y,
                      chi_u_deg=np.degrees(chi_u), chi_l_deg=np.degrees(chi_l),
                      vi_u=vi_u, vi_l_self=vi_l_self,
                      w_wake_on_lower=vi_wake_on_lower,
                      CT_disk=Thrust_total / (rho * omega_u**2 * np.pi * R**4),
                      CP_disk=Power_total / (rho * omega_u**3 * np.pi * R**5),
                      P_useful=Thrust_total * V_axial)
        return outputs, extras
    return outputs