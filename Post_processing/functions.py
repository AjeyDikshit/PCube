import numpy as np


#################################################################################
# f02
def derivative(t, x):
    h = t[1] - t[0]
    y = np.zeros(len(t))
    i = 1
    while i <= len(x) - 1:
        y[i] = (x[i] - x[i - 1]) / h
        i = i + 1
    return y


def integration(t, x):
    h = t[1] - t[0]
    y = np.zeros(len(t))
    i = 1
    while i <= len(x) - 1:
        y[i] = y[i - 1] + h * (x[i - 1])
        i = i + 1
    return y


# Example
# t = np.arange(0, 0.1, 1e-5)
# x = np.sin(2*np.pi*50*t)
###################################################################################
# f03
def myhighpass(t, u, tc):
    x = [u[0]]
    y = np.zeros(len(t))
    h = t[1] - t[0]
    for i in range(len(t) - 1):
        y1 = (x[i] + (h / tc * u[i + 1])) / (1 + h / tc)
        x.append(y1)
    for k in range(len(t)):
        y[k] = -x[k] + u[k]
    return y


def mylowpass(t, u, tc):
    x = [u[0]]
    y = np.zeros(len(t))
    h = t[1] - t[0]
    for i in range(len(t) - 1):
        y1 = (x[i] + (h / tc * u[i + 1])) / (1 + h / tc)
        x.append(y1)
    for k in range(len(t)):
        y[k] = x[k]
    return y


# Example for both
# t = np.arange(0, 2, 1e-5)
# x = []
# for i in range(len(t)):
#     if t[i] < 0.5:
#         x1 = 10*np.sin(2*np.pi*50*t[i])
#         x.append(x1)
#     elif 0.5 <= t[i] < 1:
#         x1 = 10*np.sin(2*np.pi*50*t[i])+2*np.sin(2*np.pi*500*t[i])
#         x.append(x1)
#     else:
#         x.append(8)
###################################################################################
# f05

def mw_dft(data, t, omega):
    X_data = 0
    win_len = len(t)
    for k in range(len(t)):
        X_data = X_data + data[k] * np.exp((-omega * t[k] * 1j))
    X_data = np.sqrt(2) / win_len * X_data
    return X_data


# Take frequency as input as well
def window_phasor(x, t, sr, cycles):
    x = list(x)
    t = list(t)
    va = x[0::sr]
    t = t[0::sr]
    tnew = t
    h = tnew[1]-tnew[0]

    va_mw = np.zeros(len(va), dtype='complex_')
    dom_freq = 50
    period = round(cycles / (dom_freq * h))

    for i in range(period, len(t)):
        va_mw[i] = mw_dft(va[i - period:i], t[i - period:i], dom_freq * 2 * np.pi)
    return [abs(va_mw), tnew]

#def window_phasor(x, t, sr, cycles, out_format):
#    x = list(x)
#    t = list(t)
#    va = x[0::sr]
#    t = t[0::sr]
#    tnew = t
#    h = tnew[1]-tnew[0]

#    va_mw = np.zeros(len(va), dtype='complex_')
#    dom_freq = 50
#    period = round(cycles / (dom_freq * h))

#    for i in range(period, len(t)):
#        va_mw[i] = mw_dft(va[i - period:i], t[i - period:i], dom_freq * 2 * np.pi)
#    if out_format == -1:
#        return [np.angle(va_mw, deg=True), tnew]
#    else:
#        return [abs(va_mw), tnew]

def freq4mdft(va, t, sr, cycles):
    va = list(va)
    t = list(t)
    va = va[::sr]
    t = t[::sr]
    tnew = t
    h = t[1] - t[0]
    tnew = np.array(tnew, dtype='complex_')

    va_mw = np.zeros(len(va), dtype='complex_')
    dom_freq = 50
    period = round(cycles / (dom_freq * h))

    # Note index starts from zero and not one!!
    for i in range(period, len(t)):
        va_mw[i] = mw_dft(va[i - period:i], t[i - period:i], dom_freq * 2 * np.pi)
    Vmw = va_mw

    va_r = np.real(va_mw)
    va_im = np.imag(va_mw)
    freq_a = np.zeros(len(Vmw), dtype='complex_')
    np.seterr(invalid='ignore')
    va_r_dot_l1 = [0]
    va_im_dot_l1 = [0]
    for i in range(0, len(va_r)-1):
        freq_a[i] = np.real(50 + ((va_r[i] * va_im_dot_l1[i]) - (va_im[i] * va_r_dot_l1[i])) / ((
                    va_r[i] ** 2 + va_im[i] ** 2)*(2 * np.pi)))
        va_r_dot = (va_r[i+1] - va_r[i]) / h
        va_im_dot = (va_im[i+1] - va_im[i]) / h
        va_r_dot_l1.append(va_r_dot)
        va_im_dot_l1.append(va_im_dot)

    # Last element is becoming 0 for some reason
    freq_a[-1] = freq_a[-2]

    fa = np.zeros(len(freq_a), dtype='complex_')
    dom_freq = 50
    period = round(cycles / (dom_freq * h))

    for i in range(period, len(t)):
        fa[i] = (mw_dft(freq_a[i - period:i], t[i - period:i], 0)) / np.sqrt(2)

    return [fa, tnew, freq_a]

def freq4mdftPhasor(va, t, cycles):
    sr = 1
    va = va[::sr]
    t = t[::sr]

    h = t[1] - t[0]

    va_r = np.real(va)
    va_im = np.imag(va)
    freq_a = np.zeros(len(va_r), dtype='complex_')
    np.seterr(invalid='ignore')
    va_r_dot_l1 = [0]
    va_im_dot_l1 = [0]
    for i in range(1, len(va_r)):
        freq_a[i - 1] = np.real(50 + ((va_r[i - 1] * va_im_dot_l1[i - 1]) - (va_im[i - 1] * va_r_dot_l1[i - 1])) / ((va_r[i - 1] ** 2 +va_im[i - 1] ** 2) * (2 * np.pi)))
        va_r_dot = (va_r[i] - va_r[i - 1]) / h
        va_im_dot = (va_im[i] - va_im[i - 1]) / h
        va_r_dot_l1.append(va_r_dot)
        va_im_dot_l1.append(va_im_dot)
    freq_a[-1] = freq_a[-2]

    fa = np.zeros(len(freq_a), dtype='complex_')
    dom_freq = 50
    period = round(cycles / (dom_freq * h))

    for i in range(period, len(t)):
        fa[i] = (mw_dft(freq_a[i - period:i], t[i - period:i], 0)) / np.sqrt(2)

    return [fa, t, freq_a]


# EXAMPLE f05
# t = np.arange(0, 0.5, 1e-5)
# x = np.zeros(len(t))
#
# for i in range(len(t)):
#     if t[i] < 0.25:
#         x[i] = 10*np.sin(2*np.pi*50*t[i])
#     else:
#         x[i] = 20*np.sin(2*np.pi*52*t[i])

# t = np.arange(0, 0.5, 1e-5)
# va = 10*np.sin(2*np.pi*50*t)
###################################################################################
#  f08

def instant_power(va, vb, vc, ia, ib, ic):
    p = np.zeros(len(va))
    q = np.zeros(len(va))
    for i in range(len(va)):
        p[i] = va[i] * ia[i] + vb[i] * ib[i] + vc[i] * ic[i]
        q[i] = (1 / np.sqrt(3)) * ((va[i] - vb[i]) * ic[i] + (vb[i] - vc[i]) * ia[i] + (vc[i] - va[i]) * ib[i])
    return [p, q]


def line_current(ia, ib, ic):
    I = np.zeros(len(ia))
    for i in range(len(ia)):
        I[i] = (np.sqrt(ia[i] ** 2 + ib[i] ** 2 + ic[i] ** 2)) / np.sqrt(3)
    return I


def line_voltage(va, vb, vc):
    V = np.zeros(len(va))
    for i in range(len(va)):
        V[i] = np.sqrt((va[i]) ** 2 + (vb[i]) ** 2 + (vc[i]) ** 2)
    return V


# EXAMPLE f08
# t = np.arange(0, 40e-3, 1e-4)
# phi = 30
#
# va = 10*np.sin(2*np.pi*50*t)
# vb = 10*np.sin(2*np.pi*50*t-2*np.pi/3)
# vc = 10*np.sin(2*np.pi*50*t+2*np.pi/3)
#
# ia = 2*np.sin(2*np.pi*50*t-phi*np.pi/180)
# ib = 2.01*np.sin(2*np.pi*50*t-2*np.pi/3-phi*np.pi/180)
# ic = 2*np.sin(2*np.pi*50*t+2*np.pi/3-phi*np.pi/180)
#
# V = line_voltage(va, vb, vc)
# I = line_current(ia, ib, ic)
# [p, q] = instant_power(va, vb, vc, ia, ib, ic)
###################################################################################

def clarkestranform(va, vb, vc, t):
    mat = [[1, 0, np.sqrt(0.5)],
           [-0.5, -np.sqrt(3) / 2, np.sqrt(0.5)],
           [-0.5, np.sqrt(3) / 2, np.sqrt(0.5)]]
    mat = np.array(mat)
    B = np.linalg.inv(np.sqrt(2 / 3) * mat)
    fabg = np.zeros([3, len(t)])
    for i in range(len(t)):
        for k in range(np.shape(fabg)[0]):
            fabg[k][i] = np.dot(B, [[va[i]], [vb[i]], [vc[i]]])[k]
    return [fabg[0], fabg[1], fabg[2]]


def kronstransform(t, va, vb, vc, w, gamma):
    fdqo = np.zeros([3, len(t)])
    for i in range(len(t)):
        a = (w * t[i]) + gamma
        mat = [[np.cos(a), np.cos(a - (2 * np.pi / 3)), np.cos(a - (4 * np.pi / 3))],
               [np.sin(a), np.sin(a - (2 * np.pi / 3)), np.sin(a - (4 * np.pi / 3))],
               [np.sqrt(0.5), np.sqrt(0.5), np.sqrt(0.5)]]
        mat = np.array(mat)
        B = np.sqrt(2 / 3) * mat
        for k in range(np.shape(fdqo)[0]):
            fdqo[k][i] = np.dot(B, [[va[i]], [vb[i]], [vc[i]]])[k]
    return [fdqo[0], fdqo[1], fdqo[2]]


def sequencetransform(va, vb, vc, t):
    a = np.exp(2 * np.pi * 1j / 3)
    mat = [[1, 1, 1],
           [a ** 2, a, 1],
           [a, a ** 2, 1]]
    mat = np.array(mat)
    B = np.linalg.inv(mat)
    fpno = np.zeros([3, len(t)], dtype='complex_')
    for i in range(len(t)):
        for k in range(np.shape(fpno)[0]):
            fpno[k][i] = np.dot(B, [[va[i]], [vb[i]], [vc[i]]])[k]
    return [fpno[0], fpno[1], fpno[2]]

# Example f07
# t = np.arange(0, 100e-3, 1e-4)
# delta = 0*np.pi/180
#
# va = 10*np.sin(2*np.pi*50*t+delta)
# vb = 10*np.sin(2*np.pi*50*t-2*np.pi/3+delta)
# vc = 10*np.sin(2*np.pi*50*t+2*np.pi/3+delta)
# w = 1*2*np.pi*50
# gamma = 90
###########################################################################################

def trendfilter(t, x, lamda1):
    n = len(t)

    I = np.eye(n)
    D = np.zeros((n-1, n))

    pp = 0
    for kk in range(n-2):
        D[kk, 0+pp:3+pp] = [1, -2, 1]
        pp += 1

    y = np.linalg.inv(I+2*lamda1*np.transpose(D)@D)@x[0:n]

    return y

############################################################################################
