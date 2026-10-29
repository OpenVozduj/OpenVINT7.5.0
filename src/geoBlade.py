import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import simpson
from math import factorial
from src import geoAirfoil as ga

def BezierN(p, n):
    nc = 20
    t = np.linspace(0, 1, nc)
    B = 0
    for i in range(n+1):
        B += p[i]*t**i*(1-t)**(n-i)*(factorial(n)/(factorial(i)*factorial(n-i)))
    return B

def chordCurve(Pi, r, nameBlade):
    d = Pi[9]
    R = d/2    
    pR_r = np.array([r[0], 0.5*(Pi[3]*R+r[0]), Pi[3]*R])
    pc_r = np.array([Pi[0]*d, Pi[1]*d, Pi[1]*d])
    pR_t = np.array([Pi[3]*R, (r[-1]+2*Pi[3]*R)/3, (2*r[-1]+Pi[3]*R)/3, r[-1]])
    pc_t = np.array([Pi[1]*d, Pi[1]*d, (Pi[2]+5*Pi[1])*d/6, Pi[2]*d])
    BR_r = BezierN(pR_r, 2)
    Bc_r = BezierN(pc_r, 2)
    BR_t = BezierN(pR_t, 3)
    Bc_t = BezierN(pc_t, 3)
    BR = np.append(BR_r[:-1], BR_t)
    Bc = np.append(Bc_r[:-1], Bc_t)
    FC = InterpolatedUnivariateSpline(BR, Bc, k=4)
    c = FC(r)
    return c

# def phiCurve(Pi, r, R, nameBlade):
#     pR_r_up = np.array([r[0], 0.5*(Pi[13]*R+r[0]), Pi[13]*R])
#     pphi_r_up = np.array([Pi[10], Pi[11], Pi[11]])
#     pR_t_up = np.array([Pi[13]*R, (r[-1]+2*Pi[13]*R)/3, (2*r[-1]+Pi[13]*R)/3, r[-1]])
#     pphi_t_up = np.array([Pi[11], Pi[11], (Pi[12]+5*Pi[11])/6, Pi[12]])    
#     BR_r_up = BezierN(pR_r_up, 2)
#     Bphi_r_up = BezierN(pphi_r_up, 2)
#     BR_t_up = BezierN(pR_t_up, 3)
#     Bphi_t_up = BezierN(pphi_t_up, 3)
#     BR_up = np.append(BR_r_up[:-1], BR_t_up)
#     Bphi_up = np.append(Bphi_r_up[:-1], Bphi_t_up)
#     Fphi_up = InterpolatedUnivariateSpline(BR_up, Bphi_up, k=4)
#     phi_up = Fphi_up(r)
    
#     pR_r_low = np.array([r[0], 0.5*(Pi[18]*R+r[0]), Pi[18]*R])
#     pphi_r_low = np.array([Pi[15], Pi[16], Pi[16]])
#     pR_t_low = np.array([Pi[18]*R, (r[-1]+2*R*Pi[18])/3, (2*r[-1]+R*Pi[18])/3, r[-1]])
#     pphi_t_low = np.array([Pi[16], Pi[16], (Pi[17]+5*Pi[16])/6, Pi[17]])
#     BR_r_low = BezierN(pR_r_low, 2)
#     Bphi_r_low = BezierN(pphi_r_low, 2)
#     BR_t_low = BezierN(pR_t_low, 3)
#     Bphi_t_low = BezierN(pphi_t_low, 3)
#     BR_low = np.append(BR_r_low[:-1], BR_t_low)
#     Bphi_low = np.append(Bphi_r_low[:-1], Bphi_t_low)
#     Fphi_low = InterpolatedUnivariateSpline(BR_low, Bphi_low, k=4)
#     phi_low = Fphi_low(r)
    
#     return phi_up, phi_low

def phiCurve(Pi, r, R, nameBlade):
    pR_r_up = np.array([r[0], (Pi[13]*R+2*r[0])/3, (2*Pi[13]*R+r[0])/3, Pi[13]*R])    
    pphi_r_up = np.array([Pi[10], 0.8*Pi[10], 1.1*Pi[11], Pi[11]])    
    pR_t_up = np.array([Pi[13]*R, 0.5*(r[-1]+Pi[13]*R), r[-1]])
    pphi_t_up = np.array([Pi[11], Pi[11], Pi[12]]) 
    
    BR_r_up = BezierN(pR_r_up, 3)
    Bphi_r_up = BezierN(pphi_r_up, 3)
    BR_t_up = BezierN(pR_t_up, 2)
    Bphi_t_up = BezierN(pphi_t_up, 2)
    BR_up = np.append(BR_r_up[:-1], BR_t_up)
    Bphi_up = np.append(Bphi_r_up[:-1], Bphi_t_up)
    Fphi_up = InterpolatedUnivariateSpline(BR_up, Bphi_up, k=4)
    phi_up = Fphi_up(r)
    
    pR_r_low = np.array([r[0], (Pi[18]*R+2*r[0])/3, (2*Pi[18]*R+r[0])/3, Pi[18]*R])
    pphi_r_low = np.array([Pi[15], 0.8*Pi[15], 1.1*Pi[16], Pi[16]])
    pR_t_low = np.array([Pi[18]*R, (r[-1]+R*Pi[18])/2, r[-1]])
    pphi_t_low = np.array([Pi[16], Pi[16], Pi[17]])

    BR_r_low = BezierN(pR_r_low, 3)
    Bphi_r_low = BezierN(pphi_r_low, 3)
    BR_t_low = BezierN(pR_t_low, 2)
    Bphi_t_low = BezierN(pphi_t_low, 2)
    BR_low = np.append(BR_r_low[:-1], BR_t_low)
    Bphi_low = np.append(Bphi_r_low[:-1], Bphi_t_low)
    Fphi_low = InterpolatedUnivariateSpline(BR_low, Bphi_low, k=4)
    phi_low = Fphi_low(r)
    
    return phi_up, phi_low


def airfoilCurves(Pi, r, R, nameBlade):    
    pR_r = np.array([r[0], 0.5*(Pi[7]*R+r[0]), Pi[7]*R])
    pyt_r = np.array([Pi[4], Pi[5], Pi[5]])
    pR_t = np.array([Pi[7]*R, (r[-1]+R*Pi[7])*0.5, r[-1]])
    pyt_t = np.array([Pi[5], Pi[5], Pi[6]])
    BR_r = BezierN(pR_r, 2)
    Byt_r = BezierN(pyt_r, 2)
    BR_t = BezierN(pR_t, 2)
    Byt_t = BezierN(pyt_t, 2)
    BR = np.append(BR_r[:-1], BR_t)
    Byt = np.append(Byt_r[:-1], Byt_t)
    Fyt = InterpolatedUnivariateSpline(BR, Byt, k=4)
    ytMax = Fyt(r)

    return ytMax

def airfoil_area(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
        
    # Fórmula de la Lazada (Shoelace) usando operaciones vectorizadas de NumPy.
    # np.roll(y, 1) desplaza el array para emparejar (x_i, y_{i+1}) y (y_i, x_{i+1}),
    # incluyendo el salto implícito del último elemento al primero.
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area

def paramBlade(Pi, nameBlade, rR_min, rR_max, sect_body, sect_tip):
    r_R1 = np.linspace(rR_min, 0.75, sect_body)
    r_R2 = np.linspace(0.776, rR_max, sect_tip)
    r_R = np.append(r_R1, r_R2)
    R = Pi[9]/2
    r = r_R*R
    c = chordCurve(Pi, r, nameBlade)
    ytMax = airfoilCurves(Pi, r, R, nameBlade)
    phi_up, phi_low = phiCurve(Pi, r, R, nameBlade)
        
    A = np.empty([0,13])
    xt = np.array([])
    yt = np.array([])
    xc = np.array([])
    yc = np.array([])
    for s in range(len(r)):
        As = ga.creatorAirfoil(ytMax[s])
        A = np.row_stack((A, As))
        X, YU, YL = ga.cstN5(As[:-1], TE=As[-1])
        X, YT, YC = ga.cst_ST_SC(As[:-1], TE=As[-1])
        xt = np.append(xt, X[np.argmax(YT)])
        yt = np.append(yt, max(YT))
        xc = np.append(xc, X[np.argmax(YC)])
        yc = np.append(yc, max(YC))
    
    np.save(nameBlade+'/A.npy', A)
    
    dS = np.column_stack((A, r_R))
    dS = np.column_stack((dS, c))
    dS = np.column_stack((dS, phi_up))
    dS = np.column_stack((dS, phi_low))
    dS = np.column_stack((dS, r))
    dS = np.column_stack((dS, xt))
    dS = np.column_stack((dS, yt))
    dS = np.column_stack((dS, xc))
    dS = np.column_stack((dS, yc))
    
    sections_scale = []
    for s in range(len(r)):
        X, YU, YL = ga.cstN5(A[s,:12], TE=A[s,-1])
        XUI = np.delete(X, 0)
        YUI = np.delete(YU, 0)
        XUI = XUI[::-1]
        YUI = YUI[::-1]
        Xa = np.append(XUI, X, axis=0)
        Ya = np.append(YUI, YL, axis=0)
        Xscale = Xa*c[s]
        Yscale = Ya*c[s]
        sections_scale.append((Xscale, Yscale))
        
    areas = [airfoil_area(x, y) for x, y in sections_scale]
    volume_blade = simpson(areas, x=r)
    volume_prop = volume_blade * Pi[8]
    
    return dS, volume_prop

