
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt


# Angle Manupulation
# Wrapto2pi
def wrapto2pi(angle):
    return angle % (2*np.pi)


#  Semi-Major axis 
# With r_p and r_a (distance to periapsis and apoapsis)
def a_rp_ra(r_p, r_a):
    return 0.5*(r_a+r_p)


# With miu and Eps (Gravitational parameter and Specific Energy)
def a_miu_Eps(miu, Eps):
    return -miu/2/Eps


##### Eccentricity #####
# With r_p and a (distance to periapsis and Semi-major axis)
def e_rp_a(r_p, a):
    return 1-r_p/a


def e_a_p(a, p):
    return np.sqrt(1-p/a)


def e_Eps_h_miu(Eps, h, miu):
    return np.sqrt(1+2*Eps*h**2/miu**2)


##### Semi-Latus Rectum #####
# With a and e (Semi-major axis and eccentricity)
def p_a_e(a, e):
    return a*(1-e**2)


def p_miu_h(miu, h):
    return h**2/miu

##### Specific Angular Momentum #####
# With miu and p (Mass constant and Semi-Latus Rectum)
def h_miu_p(miu, p):
    return np.sqrt(miu*p)


# With r_vec and v_vec (Distance and velocity)
def h_rvec_vvec(r_vec, v_vec):
    return np.linalg.norm(np.cross(r_vec, v_vec))


##### Specific energy #####
# With miu and a (Mass constant and Semi-Major Axis)
def Eps_miu_a(miu, a):
    return -miu/2/a


def Eps_miu_r_v(miu, r, v):
    return v**2/2 -miu/r


def Eps_vinf(vinf):
    return vinf**2/2

##### Period #####
# With miu and a (Mass constant and Semi-Major Axis)
def Per_miu_a(miu, a):
    return 2*np.pi*np.sqrt(a**3/miu)


##### Speed #####
def v_a_r(miu, a, r):
    return np.sqrt(2*(miu/r - miu/2/a))


##### Change in True Anomaly #####
def delthst_r1vec_r2vec_hunit(r1_vec, r2_vec, h_unit):
    return np.sign(h_unit[-1])*np.acos( np.dot(r1_vec, r2_vec)/np.linalg.norm(r1_vec)/np.linalg.norm(r2_vec) )


##### Anomaly #####
def theta_runit_thunit(r_unit, h_unit):
    return np.arctan2(r_unit[-1],h_unit[-1])


#### Mean Motion ####
def n_miu_a(miu, a):
    return np.sqrt(miu/ a**3)

##### f and g function #####
def f_r2_p_delthst(r2, p, delthst):
    return 1 - r2/p*(1-np.cos(delthst))


def g_miu_r1_r2_p_delthst(miu, r1, r2, p, delthst):
    return r1*r2/np.sqrt(miu*p)*np.sin(delthst)


def fdot_miu_r1vec_v1vec_p_delthst(miu, r1_vec, v1_vec, p, delthst):
    r1 = np.linalg.norm(r1_vec)
    return np.dot(r1_vec, v1_vec)/p/r1*(1-np.cos(delthst)) -1/r1*np.sqrt(miu/p)*np.sin(delthst)


def gdot_r1_p_delthst(r1,p,delthst):
    return 1-r1/p*(1-np.cos(delthst))



##### Change of Frames #####
# Rotation around axis 1
def DCM_1(beta):
    return np.array([[1, 0, 0], \
                     [0, np.cos(beta), -np.sin(beta)] , \
                    [0, np.sin(beta), np.cos(beta)]])


# Rotation around axis 3
def DCM_3(beta):
    return np.array([[np.cos(beta), -np.sin(beta), 0], \
                    [np.sin(beta), np.cos(beta), 0],\
                    [0, 0, 1]])


# Satellite to Perifocal 
def sat2per(th_st):
    return DCM_3(th_st)


# ECI to Satellite frame
def sat2ECI(Omega, i, theta):
    return DCM_3(Omega) @ DCM_1(i) @ DCM_3(theta)



#### Radial to inertial ####
def rad_2_cart(r, angle):
    x = r*np.sin(angle)
    y = r*np.cos(angle)
    return x, y


# ##### Plot Functions #####
# # 3d axis
# def lims3dplot(xmax, xmin, ymax, ymin, zmax, zmin):
#     max_range = np.array([xmax-xmin, ymax-ymin, zmax-zmin]).max() 
#     mid_x = (xmax+xmin)/2
#     mid_y = (ymax+ymin)/2
#     mid_z = (zmax+zmin)/2

#     mids = np.array([mid_x, mid_y, mid_z])
#     upper_lim = np.array([-max_range, -max_range, -max_range]) + mids
#     lower_lim = np.array([max_range, max_range, max_range]) + mids
#     return lower_lim, upper_lim


# #### Plot 2d ####
# def plot_2d(a, e, thst, ax, vis='-'):
#     r = a*(1-e**2)/(1+e*np.cos(thst))
#     x = r*np.cos(thst)
#     y = r*np.sin(thst)

#     ax.plot(x, y, vis)


#### Classes ####
# Angle
class angle:
    def __init__(self, deg, rad):
        self.deg = deg
        self.rad = rad

    @classmethod
    def degree(cls, deg):
        rad = np.deg2rad(deg)
        return cls(deg, rad)


    @classmethod
    def radians(cls, rad):
        rad = wrapto2pi(rad)
        deg = np.rad2deg(rad)
        return cls(deg, rad)

    def print_table(self):
        return pd.DataFrame(self.__dict__)
    
    def __add__(self, other):
        total_rad = self.rad + other.rad
        total_deg = self.deg + other.deg
        return angle(total_deg, total_rad)
    
    def __sub__(self, other):
        total_rad = self.rad - other.rad
        total_deg = self.deg - other.deg
        return angle(total_deg, total_rad)


##### Flight Path Angle #####
class gamma(angle):

    @classmethod
    def gamma_h_r_v_thst(self, h, r, v, thst):
        gamma = np.arccos(h/r/v)
        thst = wrapto2pi(thst)
        signs = (thst > np.pi)*-1 + (thst < np.pi)
        return super().radians(gamma * signs)


##### True Anomaly #####
class thst(angle):
    @classmethod
    def thst_rvec_vvec_p_e(self, r_vec, v_vec, p, e):
        thst = np.sign(np.dot(v_vec, r_vec)) * np.arccos(1/e*(p/np.linalg.norm(r_vec) - 1))
        return super().radians(thst)

    @classmethod
    def thst_E_e(E, e):
        thst = 2*np.arctan(np.sqrt((1+e)/(1-e)) * np.tan(E/2))
        return super().radians(thst)


##### Hyperbolic Angle 1 #####
class delta(angle):
    @classmethod
    def delta_e(self, e):
        return super().radians(2*np.arcsin(1/e))
    


##### Hyperbolic Angle 2 #####
class thst_inf(angle):
    @classmethod
    def thst_inf_e(self, e):
        return super().radians(np.arccos(-1/e))


##### Eccentric Anomaly #####
class E(angle):
    # With miu and a (Mass constant and Semi-Major Axis)
    @classmethod
    def E_thst(self, thst, e):
        E = 2*np.arctan2(np.tan(thst/2), np.sqrt((1+e)/(1-e)))
        return super().radians(E)


    # With just M (mean Anomaly)
    @classmethod
    def E_M(self, M, e, acc=10^-12, **kwargs):
        E_n = M
        for n in range(100):    
            E_n1 = E_n - (E_n - e*np.sin(E_n) - M)/(1-e*np.cos(E_n))
            error = abs(E_n1 - E_n)

            if error < acc:
                break

            E_n = E_n1

        if 'n_count' in kwargs and kwargs["n_count"] == True:
            return [super().radians(E_n1), n+1]
        else:
            return super().radians(E_n1)


##### Mean Anomaly #####
class M(angle):
# With miu and a (Mass constant and Semi-Major Axis)
    @classmethod
    def M_E_e(self, E, e):
        return super().radians(E-e*np.sin(E))


##### Inclination #####
class i(angle):
    @classmethod
    def i_hunit(self, h_unit_ECI):
        return super().radians(np.arccos(h_unit_ECI[-1]))


##### Right Ascension #####
class Omega(angle):
    @classmethod
    def Omega_hunit(self, h_unit_ECI):
        return super().radians(np.arctan2(h_unit_ECI[0], -h_unit_ECI[1]))


##### Argument of Periapsis #####
class omega(angle):
    @classmethod
    def omega_thst_theta(self, thst, theta):
        return super().radians(theta - thst)
    

##### Anomaly #####
class theta(angle):
    @classmethod
    def theta_thetaunit(self, theta_unit):
        return super().radians(np.arctan2(theta_unit[-1], -theta_unit[1]))


##### Change in true Anomaly #####
class del_thst(angle):
    @classmethod
    def del_thst_r1_r2(self, r1_vec, r2_vec):
        r1 = np.linalg.norm(r1_vec)
        r2 = np.linalg.norm(r2_vec)
        h = np.cross(r1_vec, r2_vec)
        del_thst = np.sign(h[-1])*np.arccos(np.dot(r1_vec, r2_vec)/ (r1*r2))
        return super().radians(del_thst)
    
    
##### Frame #####
class frames:
    def __init__(self, per, rad, eci, mag):
        self.per = per
        self.rad = rad
        self.eci = eci
        self.mag = mag

    @classmethod
    def radial2all(cls, rad, thst, i=0, omega=0, Omega=0):
        if isinstance(thst, np.ndarray):
            per = np.zeros(rad.shape)
            eci = np.zeros(rad.shape)
            for n, th in enumerate(thst):
                per[n, ] = DCM_3(th) @ rad[n, ]
                eci[n, ] = sat2ECI(Omega=Omega, i=i, theta=omega+thst[n]) @ rad[n, ]
            mag = np.linalg.norm(per, axis=1)
            
        else:
            per = DCM_3(thst) @ rad
            eci = sat2ECI(Omega=Omega, i=i, theta=omega+thst) @ rad
            mag = np.linalg.norm(per)
                        
        return cls(per, rad, eci, mag)


    def print_table(self):
        return pd.DataFrame(self.__dict__)
    
    def __add__(self, other):
        total_per = self.per + other.per
        total_rad = self.rad + other.rad
        total_eci = self.eci + other.eci
        total_mag = np.linalg.norm(total_eci)
        return frames(total_per, total_rad, total_eci, total_mag)
    
    def __sub__(self, other):
        total_per = self.per + other.per
        total_rad = self.rad + other.rad
        total_eci = self.eci + other.eci
        total_mag = np.linalg.norm(total_eci)
        return frames(total_per, total_rad, total_eci, total_mag)
    
    def __mul__(self, other):
        if isinstance(other, frames):
            total_per = self.per * other.per
            total_rad = self.rad * other.rad
            total_eci = self.eci * other.eci
            total_mag = np.linalg.norm(total_eci)
        else:
            total_per = self.per * other
            total_rad = self.rad * other
            total_eci = self.eci * other
            total_mag = np.linalg.norm(total_eci)
            
        return frames(total_per, total_rad, total_eci, total_mag)


##### Distance #####
class distance(frames):

    @classmethod
    def keplar_r(self, p, e, thst, i=0, omega=0, Omega=0):
        r = p/(1+e*np.cos(thst))
        if isinstance(thst, np.ndarray):
            zero = np.zeros(thst.shape)
            r_vec = np.block([r, zero, zero]).reshape((len(thst), 3), order='F')
        else:
            r_vec = np.array([r, 0, 0])
            
        return super().radial2all(r_vec, thst, i, omega, Omega)



##### Velocity #####
class velocity(frames):
    
    @classmethod
    def v_gamma(self, v, gamma, thst, i=0, omega=0, Omega=0):
        v_rad = v* np.array([np.sin(gamma), np.cos(gamma), 0])
        # thst = wrapto2pi(thst)
        return super().radial2all(v_rad, thst, i, omega, Omega)
    
    @classmethod
    def v_a_miu_r(self, a, miu, r):
        v = np.sqrt(2*(miu/r - miu/2/a))
        return v
