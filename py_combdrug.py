#!/usr/local/bin/python3
"""
py_combdrug - python library for calculating the effect of combination drugs

author: Sungcheol Kim @ IBM
email: kimsung@us.ibm.com
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import scipy.stats

# for increasing speed of algorithm
from numba import njit


class POPULATION(object):
    """ object of patient/cell population for drug effects
    example:
        import py_combdrug
        p = py_combdrug.POPULATION()
        p.get_scurve_hill('A', 5, 2)
        p.get_scurve_hill('B', 10, 8)
    """

    def __init__(self, tmax=30, tnum=100, snum=100, debug=False):
        """
        initialize with scores and classes
        input: data - panda DataFrame with "Score" and "Class" columns
        """

        self._tmax = tmax
        self._tnum = tnum
        self._snum = snum
        self._debug = debug

        self._info = {}
        self._pdata = pd.DataFrame()
        self._tdata = pd.DataFrame()
        self._trange = np.linspace(0, self._tmax, num=self._tnum)

    def get_scurve_file(self, drugname, filename):
        """ survival curve data from file """

        a = pd.read_csv(filename, header=None, names=['T','D'])
        t = sorted(np.append(a['T'].values, [np.max(a['T'].values), 0]))
        sur = sorted(np.append(a['D'].values, [1.0,  0]), reverse=True)
        if self._debug:
            print('t: min, mean, max - {}, {}, {}'.format(t.min(), t.mean(), t.max()))
            print('d: min, mean, max - {}, {}, {}'.format(d.min(), d.mean(), d.max()))

        self._pdata[drugname+'_Time'] = np.append(t, [np.nan] * (self._tnum - len(t)))
        self._pdata[drugname] = np.append(sur, [np.nan] * (self._tnum - len(sur)))
        self._pdata[drugname] = self._pdata[drugname]/self._pdata[drugname].max()
        self._info.update({drugname: [a['T'].mean(), 1.0, a['T'].max()]})

    def get_scurve_hill(self, drugname, t50, hillcoeff):
        """ survival curve data from hill function """

        self._pdata[drugname+'_Time'] = np.append(self._trange, self._tmax)
        self._pdata[drugname] = np.append(hill(self._trange, t50, hillcoeff), 0)
        self._info.update({drugname: [t50, hillcoeff, self._tmax]})

    def gen_stime(self, drugname, show=False):
        """ generate individual survival time from survival curve """

        sur = self._pdata[drugname].values/self._pdata[drugname].max()
        t = self._pdata[drugname+'_Time'].values

        if self._debug:
            print('sur: {}\nt: {}'.format(sur, t))

        self._tdata['t_'+drugname] = patient_sur(self._snum, t, sur, show=show)

    def gen_drugpair(self, drugnames, rho, **kwargs):
        """ generate correlated pairing between two survival times """

        t_drugnames = [ 't_'+x for x in drugnames ]
        A = self._tdata[t_drugnames[0]].sort_values()
        B = self._tdata[t_drugnames[1]].sort_values()

        stime = time.time()

        # computer shuffle with correlation
        y, it = MC_suffle(len(A), rho, **kwargs)

        print('compute time: {} secs, # of iteration: {}'.format(time.time() - stime, it))
        rho, p = scipy.stats.spearmanr(np.arange(len(A)), y)
        print('... rho: {:.4f}, p: {:.4f}'.format(rho, p))

        self._tdata[t_drugnames[0]] = A.values
        self._tdata[t_drugnames[1]] = B.values[y]

    def gen_combdrug(self, drugnames, new_drug=None):
        """ calculate survival time using independent drug action """

        if new_drug is None:
            new_drug = '+'.join(drugnames)
        t_drugnames = [ 't_'+x for x in drugnames ]
        smax = self._tdata[t_drugnames].max(axis=1)
        self._tdata['t_'+new_drug] = smax
        self._info.update({new_drug: [smax.mean(), 1.0, smax.max()]})

    def gen_comb_formula(self, drugnames, rho, new_drug=None):
        """ calculate combination survival curve using formular """

        if (rho < -1) or (rho > 1):
            print('... out of range')
            return

        ssize = len(self._pdata)

        if new_drug is None:
            new_drug = '+'.join(drugnames)

        if len(drugnames) > 1:
            A = self._pdata[drugnames[0]].values
            A_t = self._pdata[drugnames[0]+'_Time'].values
            B = self._pdata[drugnames[1]].values
            B_t = self._pdata[drugnames[1]+'_Time'].values

            if A_t.max() != B_t.max():
                tmax = np.nanmax(B_t) if np.nanmax(A_t) > np.nanmax(B_t) else np.nanmax(A_t)
                t = np.linspace(0, tmax, ssize)
                A = np.interp(t, A_t, A)
                B = np.interp(t, B_t, B)
            else:
                t = np.append(self._trange, self._tmax)

        else:
            print('... put two drug names')
            return

        if rho >= 0:
            tmp = A + B - A*B - rho*np.minimum(A, B) * (1 - np.maximum(A, B))
            self._pdata[new_drug+'_Time'] = t
            self._pdata[new_drug] = tmp
            self._info.update({new_drug: [tmp.mean(), 1.0, tmp.max()]})
        else:
            tmp1 = (A + B - A*B)*(1 + rho) - rho
            tmp2 = A + B - A*B*(1 + rho)

            tstar = np.argmin(np.abs(tmp1 - tmp2))
            self._pdata[new_drug+'_Time'] = t
            self._pdata[new_drug] = tmp2
            self._pdata[new_drug][:tstar] = tmp1[:tstar]

    def plot_scurve(self, drugnames, ax=None, filename=None, labels=None, save=True):
        """ plot survival curve of monotherapy """

        if ax is None: ax = plt.gcf().gca()
        if labels is None: labels = drugnames

        for i, drugname in enumerate(drugnames):
            if self._pdata[drugname].isna().sum() > 10:
                ax.plot(self._pdata[drugname+'_Time'], self._pdata[drugname], '.-', label=labels[i], drawstyle='steps-mid')
            else:
                ax.plot(self._pdata[drugname+'_Time'], self._pdata[drugname], label=labels[i])
        ax.set_xlabel('Time')
        ax.set_ylabel('Survival')
        ax.legend()

        if filename is None: filename = 'survival_curve_'+'_'.join(labels)+'.pdf'
        if save: plt.savefig(filename, dpi=150)

    def plot_scurve_stime(self, drugnames, ax=None, filename=None, labels=None, save=True):
        """ plot survival curves from patient/cell survival times """

        if ax is None: ax = plt.gcf().gca()
        if labels is None: labels = drugnames

        for i, drugname in enumerate(drugnames):
            s = []
            t = np.linspace(0, self._info[drugname][2], num=self._tnum)
            if 't_'+drugname not in self._tdata.columns:
                self.gen_stime(drugname)

            N = len(self._tdata['t_'+drugname])
            for t_n in t:
                s.append((self._tdata['t_'+drugname] >= t_n).sum()/N)
            if N > 50:
                ax.plot(t, np.array(s), '.-', label=labels[i])
            else:
                ax.plot(t, np.array(s), '.-', drawstyle='steps-mid', label=labels[i])

        ax.set_xlabel('Time')
        ax.set_ylabel('Survival')
        ax.set_ylim(-0.05, 1.05)
        ax.legend()

        if filename is None: filename = 'survival_time_'+'_'.join(labels)+'.pdf'
        if save: plt.savefig(filename, dpi=150)

    def plot_stime(self, drugnames, filename=None, save=True):
        """ scatterplot of individual survival time """

        #if ax is None:
        #    ax = plt.gcf().gca()

        A = 't_'+drugnames[0]
        B = 't_'+drugnames[1]
        if A not in self._tdata.columns:
            self.gen_stime(drugnames[0])
        if B not in self._tdata.columns:
            self.gen_stime(drugnames[1])

        rho, p = scipy.stats.spearmanr(self._tdata[A], self._tdata[B])
        xmax = np.max(self._tdata[[A, B]].max().values)

        g = sns.JointGrid(x=A, y=B, data=self._tdata, xlim=(-5, xmax+5), ylim=(-5, xmax+5))
        g = g.plot_joint(sns.regplot)
        ax = plt.gcf().gca()
        ax.plot([0, xmax], [0, xmax], 'k-.', alpha=0.8)
        g = g.plot_marginals(sns.distplot, bins=int(xmax/3))
        g.annotate(scipy.stats.spearmanr)

        if filename is None: filename = 'corr_plot'+'_'.join(drugnames)+'_{:.4f}.pdf'.format(rho)
        if save: g.savefig(filename, dpi=150)

    def plot_rainbow(self, drugnames, filename=None, num=11, save=True):
        """ rainbow plot (rho from -1 to 1) """

        fig = plt.figure(figsize=(8,6))
        ax = fig.gca()

        rhos = np.linspace(-1, 1, num=num)
        for rho in rhos:
            name = 'rho={:.2f}'.format(rho)
            self.gen_comb_formula(drugnames, rho, new_drug=name)
            ax.plot(self._pdata[name+'_Time'], self._pdata[name], label=name, alpha=0.3)

        self.plot_scurve(drugnames, ax=ax, save=False)

        if filename is None: filename = 'rainbow_'+'_'.join(drugnames)+'.pdf'
        if save: plt.savefig(filename, dpi=150)


def hill(t, t50, h):
    """ hill function with half life """

    return 1./(1. + np.power(t/t50, h))


def patient_sur(N, t, survival, show=False):
    """ generate N patient survival time from survival function """

    y_n = np.random.uniform(0, 1.0, N)
    x_n = np.interp(1 - y_n, 1 - survival, t)

    if show:
        plt.plot(x_n, y_n, '.')
        plt.plot(t, survival)
        plt.xlabel('Time')
        plt.ylabel('Survival')

    return x_n.flatten()


@njit(fastmath=True, cache=True)
def MC_suffle(N, rho, conf=0.001, rate=200.0, annealing=1.0, max_iter=50000000):
    """ create jittered rank list """

    it = 0
    N1 = int(N/4); N2 = int(N/2); N3 = int(N*3/4)

    # prepare initial values
    x = np.arange(N)
    if rho >= 1.0:
        return x, it
    elif rho > 0.95:
        y = x.copy()
    elif rho > 0.9:
        y = x.copy()
        y[:N1] = np.random.permutation(y[:N1])
        y[N1:N2] = np.random.permutation(y[N1:N2])
        y[N2:N3] = np.random.permutation(y[N2:N3])
        y[N3:] = np.random.permutation(y[N3:])
    elif rho > 0.7:
        y = x.copy()
        y[:N2] = np.random.permutation(y[:N2])
        y[N2:] = np.random.permutation(y[N2:])
    elif rho <= -1.0:
        return x[::-1], it
    elif rho < -0.95:
        y = x.copy()[::-1]
    elif rho < -0.9:
        y = x.copy()[::-1]
        y[:N1] = np.random.permutation(y[:N1])
        y[N1:N2] = np.random.permutation(y[N1:N2])
        y[N2:N3] = np.random.permutation(y[N2:N3])
        y[N3:] = np.random.permutation(y[N3:])
    elif rho < -0.7:
        y = x.copy()[::-1]
        y[:N2] = np.random.permutation(y[:N2])
        y[N2:] = np.random.permutation(y[N2:])
    else:
        y = np.random.permutation(x)

    #rho_s0, p = scipy.stats.spearmanr(x, y)
    #rho_s0 = np.corrcoef(x, y)[0, 1]
    rho_s0 = rank_coeff(x, y)
    N2_2 = N*(N*N - 1.)/12.

    while(it < max_iter):
        # select two items and exchange order
        idx = np.random.randint(0, N, 2)
        #y_orig = y.copy()
        #y[idx] = y_orig[idx[::-1]]

        #rho_s, p = scipy.stats.spearmanr(x, y)
        #rho_s = np.corrcoef(x, y)[0, 1]
        #rho_s = rank_coeff(x, y)
        rho_s = rho_s0 - (idx[0] - idx[1])*(y[idx[0]] - y[idx[1]])/N2_2
        dist = np.abs(rho - rho_s)
        dist0 = np.abs(rho - rho_s0)
        it = it + 1

        # check random exchange result
        if dist < dist0:
            rho_s0 = rho_s
            y[idx] = y[idx[::-1]]
            continue
        elif dist >= annealing*conf + dist0:
            # selection
            p = np.exp(-rate*dist)
            if np.random.random(1)[0] > p:
                #y[idx] = y_orig[idx]
                #rho_s = rho_s0
                continue
            else:
                rho_s0 = rho_s
                y[idx] = y[idx[::-1]]
                continue
        elif dist < conf:
            break

    return y, it


@njit(fastmath=True, cache=True)
def rank_coeff(x, y):
    """ calculate rank correlation coefficient """

    N = len(x)
    return 12.*(np.sum((x+1)*(y+1))/N - (N+1.)*(N+1.)/4.)/(N*N - 1.)

