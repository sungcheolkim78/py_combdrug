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
        a.sort_values('T', inplace=True)
        self._pdata[drugname+'_Time'] = np.append(a['T'].values, [np.nan] * (self._tnum - len(a['T'])))
        self._pdata[drugname] = np.append(a['D'].values, [np.nan] * (self._tnum - len(a['D'])))
        self._info.update({drugname: [a['T'].mean(), 1.0, a['T'].max()]})

    def get_scurve_hill(self, drugname, t50, hillcoeff):
        """ survival curve data from hill function """

        self._pdata[drugname+'_Time'] = self._trange
        self._pdata[drugname] = hill(self._trange, t50, hillcoeff)
        self._info.update({drugname: [t50, hillcoeff, self._tmax]})

    def gen_stime(self, drugname, show=False):
        """ generate individual survival time from survival curve """

        sur = self._pdata[drugname].values/self._pdata[drugname].max()
        t = self._pdata[drugname+'_Time'].values

        # confirm last point hit 0 for inverse function
        sur = np.append(sur[~np.isnan(sur)], 0)
        t = np.append(t[~np.isnan(t)], t.max())

        if self._debug:
            print('sur: {}\nt: {}'.format(sur, t))

        self._tdata['t_'+drugname] = patient_sur(self._snum, t, sur, show=show)

    def gen_drugpair(self, drugnames, rho, **kwargs):
        """ generate correlated pairing between two survival times """

        t_drugnames = [ 't_'+x for x in drugnames ]
        A = self._tdata[t_drugnames[0]].sort_values()
        B = self._tdata[t_drugnames[1]].sort_values()

        y = MC_suffle(len(A), rho, debug=self._debug, **kwargs)

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

    def plot_scurve(self, drugnames, ax=None, filename=None, labels=None):
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
        plt.savefig(filename, dpi=150)

    def plot_scurve_stime(self, drugnames, num=30, ax=None, filename=None, labels=None):
        """ plot survival curves from patient/cell survival times """

        if ax is None: ax = plt.gcf().gca()
        if labels is None: labels = drugnames

        for i, drugname in enumerate(drugnames):
            s = []
            t = np.linspace(0, self._info[drugname][2], num=num)
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
        plt.savefig(filename, dpi=150)

    def plot_stime(self, drugnames, filename=None):
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
        g.savefig(filename, dpi=150)


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


def MC_suffle(N, rho, conf=0.01, rate=50.0, max_iter=20000, debug=False):
    """ create jittered rank list """

    stime = time.time()
    it = 0

    # prepare initial values
    x = np.arange(N)
    if rho >= 1.0:
        return x
    elif rho > 0.7:
        y = x.copy()
        y[:int(N/2)] = np.random.permutation(y[:int(N/2)])
        y[int(N/2):] = np.random.permutation(y[int(N/2):])
    elif rho <= -1.0:
        return x[::-1]
    elif rho < -0.7:
        y = x.copy()[::-1]
        y[:int(N/2)] = np.random.permutation(y[:int(N/2)])
        y[int(N/2):] = np.random.permutation(y[int(N/2):])
    else:
        y = np.random.permutation(x)

    rho_s0, p = scipy.stats.spearmanr(x, y)

    while(it < max_iter):
        # select two items and exchange order
        idx = np.random.randint(0, N, 2)
        y_orig = y.copy()
        y[idx] = y_orig[idx[::-1]]

        rho_s, p = scipy.stats.spearmanr(x, y)
        dist = rho - rho_s
        dist0 = rho - rho_s0
        it = it + 1

        # check random exchange result
        if np.abs(dist) < np.abs(dist0):
            if debug:
                print('... rho:{:.4f}, p:{:.4f}, idx: {}, dist: {:.4f}'.format(rho_s, p, idx, dist))
            rho_s0 = rho_s
        elif np.abs(dist) >= conf + np.abs(dist0):
            # selection
            p = np.exp(-rate*np.abs(dist))
            if np.random.random(1)[0] > p:
                y[idx] = y_orig[idx]
            else:
                if debug:
                    print('... rho:{:.4f}, p:{:.4f}, idx: {}, dist: {:.4f}, p: {:.4f}'.format(rho_s, p, idx, dist, p))
                rho_s0 = rho_s
        elif np.abs(dist) < conf:
            break

    print('compute time: {} secs'.format(time.time() - stime))
    if it == max_iter:
        print('... not converge! rho: {:.4f}, p: {:.4f}'.format(rho_s, p))
    else:
        print('... rho: {:.4f}, p: {:.4f}'.format(rho_s, p))

    return y

def jitter(orderedlist, j=0):
    """ shuffle ordered list with j amount """

    temp = orderedlist.copy()
    newlist = []

    for i in range(len(orderedlist)):
        if j > len(temp)-1:
            idx = len(temp)
        else:
            idx = j
        element = np.random.choice(temp[:idx], 1) if idx > 0 else temp[0]
        newlist.append(element)

        # remove one element
        mask = temp == element
        if sum(mask) > 1:
            print('... multiple {} - {}'.format(element, sum(mask)))
        else:
            temp = temp[~mask]

    return np.array(newlist).flatten()


def jitter2(orderedlist, j=0, sampleN=-1):
    """ shuffle ordered list with j amount """

    temp = orderedlist.copy()
    if sampleN == -1:
        sampleN = len(orderedlist)
    newlist = []

    if j == 0:
        return temp

    for i in range(sampleN):
        idx1 = i-j if (i-j > 0) else 0
        idx2 = len(temp) if (i+j > len(temp)) else i+j+1

        element = np.random.choice(temp[idx1:idx2], 1)
        newlist.append(element)

    return np.array(newlist).flatten()


def plot_sur_t(pA, pB, pBj, pAB, j=0):
    #p0 = np.corrcoef(pA, y=pB)[0,1]
    p0 = scipy.stats.spearmanr(pA, pB)[0]

    #p = np.corrcoef(pA, y=pBj)[0,1]
    p = scipy.stats.spearmanr(pA, pBj)[0]

    fig = plt.figure(figsize=(14,5))
    ax = plt.subplot(121)

    plot_survival(pA, t, label='A')
    plot_survival(pB, t, label='B')
    plot_survival(pAB, t, label='A+B')
    plt.legend()

    ax = plt.subplot(122)
    plt.plot(pA, pB, '.', label='j=0 p={0:.2f}'.format(p0))
    plt.plot(pA, pBj, '.', label='j={0:d} p={1:.2f}'.format(j, p))
    plt.plot(pA, pA, label='y=x')
    plt.xlabel('t_A')
    plt.ylabel('t_B')
    plt.legend()
