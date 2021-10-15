import pandas as pd
import numpy as np

Solar_S = pd.DataFrame(index=["Units", "Sun", "Moon", "Earth", "Jupiter", 
                              "Saturn", "Mars", "Uranus", "Neptune", "Pluto",
                              "Titan", "Phobos", "Europa", "Oberon", "Triton",
                              "Charon"],
                       columns=["a", "miu", "r", "Per", "e", "i"])
Solar_S.loc["Units"] = ["km", "km^3/s^2", "km", "s", "-", "deg"]
Solar_S.loc["Sun"] = [np.NAN, 132712440017.990, 695990.00, np.NaN, np.NAN,
                      np.NAN]
Solar_S.loc["Moon"] = [384400.000, 4902.8011, 1738.0, 2360592, 0.0549, 5.145]
Solar_S.loc["Earth"] = [149597870.7, 398600.4415, 6378.136, 31558148.63,
                        0.01671022, 0.00005]
Solar_S.loc["Jupiter"] = [778340816.693, 126712767.8578, 71492, 37434456.8,
                          0.04838624, 1.304397]          
Solar_S.loc["Saturn"] = [1426666414.180, 37940626.0611, 60268, 929277960.1,
                         0.05386179, 2.485992]
Solar_S.loc["Mars"] = [227943822.428, 42828.3143, 3397, 59356149.69, 0.0933941, 
                       1.849691]
Solar_S.loc["Uranus"] = [2870658170.656, 5794549.0070719, 60268, 929277960.1,
                         0.05386179, 2.485992]
Solar_S.loc["Neptune"] = [498396417.0095, 6836534.0639, 25269, 5203546650, 
                          0.008590480, 1.770043]
Solar_S.loc["Pluto"] = [5906440596.5288, 981.6009, 1162, 7829117488, 0.2488273,
                        17.140012]
Solar_S.loc["Titan"] = [1221865, 8935.395624, 16177.5, np.NAN , 0.0288, np.NAN]
Solar_S.loc["Phobos"] = [9376, 0.00070, 11.1, np.NAN, 0.0151, np.NAN]
Solar_S.loc["Europa"] = [671100, 3187.47774, 1560.8, np.NAN, 0.0094, np.NAN]
Solar_S.loc["Oberon"] = [583500, 191.48348, 761.4, np.NAN, 0.0014,np.NAN]
Solar_S.loc["Triton"] = [354759, 1420.79916, 1353.4, np.NAN, 0, np.NAN]
Solar_S.loc["Charon"] = [17536, 102.70860, 603.6, np.NAN, 0.0022, np.NAN]

G = 6.67408 * 10**-20


def charac_quant(body1, body2):
    m_star = body1['miu']/G + body2['miu']/G
    l_star = body2['a']
    t_star = np.sqrt( l_star**3/ G / m_star )
    return m_star, l_star, t_star

body1_list = ["Sun", "Earth", "Sun", "Saturn", "Sun", "Mars", 
              "Jupiter", "Uranus", "Neptune", "Pluto"]
body2_list = ["Earth", "Moon", "Jupiter", "Titan", "Mars", "Phobos",
              "Europa", "Oberon", "Triton", "Charon"]
system_characteristics = np.zeros((len(body2_list),8))
index = np.array([])

systems = zip(body1_list,body2_list)

for i, (body1, body2) in enumerate(systems):
    m_star, l_star, t_star = charac_quant(Solar_S.loc[body1], Solar_S.loc[body2])
    m1 = Solar_S.loc[body1,'miu']/G
    m2 = Solar_S.loc[body2,'miu']/G
    miu = m2/m_star
    d1 = miu
    d2 = (1-miu)
    system_characteristics[i,:] = [m_star, l_star, t_star, m1, m2, miu, d1, d2]
    index = np.append(index, [(body1+'-'+body2)])

    

Solar_3 = pd.DataFrame(system_characteristics, columns=['m*','l*','t*', 'm1', 'm2', 'miu', 'd1', 'd2'],index=index)


