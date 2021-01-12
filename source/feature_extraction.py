# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import pdb
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from ssqueezepy.synsq_cwt import synsq_cwt_fwd
from ssqueezepy.synsq_stft import synsq_stft_fwd
from ssqueezepy.stft_transforms import stft_fwd
from ssqueezepy.wavelet_transforms import cwt_fwd

"""
Created on Tue Apr  7 17:11:51 2020

@author: sid.talha
"""


def feature1(data, wn, tp, step):
    nsub = np.unique(data.subject)
    nexp = np.unique(data.experience)
    nac = np.unique(data.action)
    # pdb.set_trace()
    F = np.array([])
    experience = np.array([])
    subject = np.array([])
    lab = np.array([])
    z = np.array([])
    fs = 50
    for k1 in nsub:
        print(k1)
        for k2 in nexp:
            for k3 in nac:
                A = data[(data.subject == k1) & (data.experience == k2) & (data.action == k3)]

                n = A.shape[0]
                dec1 = 50
                ov = int(2)

                z = np.append(z, len(A))

                if not A.empty and n >= dec1:

                    # pdb.set_trace()
                    Sa = np.array(A.iloc[:, 0:6])
                    siz = wn
                    ov = step
                    dec = wn
                    #####
                    San = np.zeros((siz, 6))
                    for r in range(6):
                        t = np.linspace(0, len(Sa[:, r]) - 1, num=len(Sa[:, r]), endpoint=True)

                        f1 = interp1d(t, Sa[:, r], kind='cubic')

                        tnew = np.linspace(0, len(Sa[:, r]) - 1, num=siz, endpoint=True)
                        San[:, r] = f1(tnew)

                    Sa = San

                    #####
                    n = Sa.shape[0]
                    nw = range(0, int((n - dec) / ov))
                    nw = range(1)




                    var = 0.005
                    aug = 5              # increase for data augmentation
                    Sau = np.zeros((wn * aug, 6))
                    for m1 in range(1, aug + 1):
                        for m2 in range(6):
                            Sau[0+(m1-1)*wn:wn*m1, m2] = Sa[:, m2]+var*(m1 -aug/2)


                    nw = range(0, aug)
                    ov = wn
                    for j1 in nw:

                        S2 = np.array([])

                        for k in range(6):
                            S1 = np.array(Sau[ov * j1:ov * j1 + dec, k])

                            if tp == 'fft':

                             #%%
                                f, t, S1 = signal.spectrogram(S1, fs, nperseg=100, noverlap=95)
                                S1 = S1.reshape((S1.shape[0], S1.shape[1], 1, 1))

                            elif tp == 'synchro':
                                OPTS = {'type': 'bump', 'mu': 1}
                                S1, *_ = synsq_stft_fwd(tnew, S1, opts=OPTS)
                                S1 = np.abs(S1)

                             # %%
                                S1 = S1.reshape((S1.shape[0], S1.shape[1], 1, 1))

                            elif tp == 'statistics':
                                 S1 = np.array([S1.std(), S1.mean(), S1.min(), S1.max()])
                                 S1 = S1.reshape((S1.shape[0], 1, 1))

                            else:
                                S1 = S1.reshape((S1.shape[0], 1, 1))


                            if len(S2) != 0:
                                # Shape for temporal data
                                # Shape for STFT
                                if tp == 'synchro' or tp == 'fft':
                                    S2 = np.concatenate((S2, S1), axis=2)
                                else:
                                    S2 = np.concatenate((S2, S1), axis=0)


                            else:

                                S2 = S1
                        # pdb.set_trace()
                        if len(F) != 0:
                            if tp == 'synchro' or tp == 'fft':
                                F = np.concatenate((F, S2), axis=3)
                            else:
                                F = np.concatenate((F, S2), axis=2)


                        else:
                            if tp == 'synchro' or tp == 'fft':
                                F = S2
                                F = F.reshape((S2.shape[0], S2.shape[1], S2.shape[2], 1))
                            else:
                                F = np.concatenate((F, S2), axis=None)
                                F = F.reshape((S2.shape[0], S2.shape[1], 1))
                        # pdb.set_trace()

                        lab = np.concatenate((lab, k3), axis=None)
                        experience = np.concatenate((experience, k2), axis=None)
                        subject = np.concatenate((subject, k1), axis=None)

            # pdb.set_trace()
    lab = lab.reshape((len(lab), 1))
    experience = experience.reshape((len(lab), 1))
    subject = subject.reshape((len(lab), 1))

    Data_1 = {"feature": F, "labels": lab, "subject": subject, "experience": experience}

    return Data_1, z
