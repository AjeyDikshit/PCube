import numpy as np


def user_func(t, x, tc):
    h = t[1] - t[0]
    y = np.zeros(len(t))
    i = 0
    while i <= len(x) - 1:
        y[i] = (x[i] - x[i - 1]) / h
        i = i + 1
    return [y, t]

#def user_func(t, u, tc):
#    x = [u[0]]
#    y = np.zeros(len(t))
#    h = t[1] - t[0]
#    for i in range(len(t) - 1):
#        y1 = (x[i] + (h / tc[0] * u[i + 1])) / (1 + h / tc[0])
#        x.append(y1)
#    for k in range(len(t)):
#        y[k] = x[k]
#    return [y, t]

#def mw_dft(data, t, omega):
#    X_data = 0
#    win_len = len(t)
#    for k in range(len(t)):
#        X_data = X_data + data[k] * np.exp((-omega * t[k] * 1j))
#    X_data = np.sqrt(2) / win_len * X_data
#    return X_data

#def window_phasor(x, t, sr, cycles):
#    va = x[0::sr]
#    t = t[0::sr]
#    h = t[1] - t[0]
#    tnew = t

#    va_mw = np.zeros(len(va), dtype='complex_')
#    dom_freq = 50
#    period = round(cycles / (dom_freq * h))

#    for i in range(period, len(t)):
#        va_mw[i] = mw_dft(va[i - period:i], t[i - period:i], dom_freq * 2 * np.pi)
#    return [va_mw, tnew]
