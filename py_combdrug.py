#!/usr/local/bin/python3
"""
py_auc - python library for calculating the area under the curve (ROC, PR) of binary classifiers

author: Sungcheol Kim @ IBM
email: kimsung@us.ibm.com
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score


class AUC(object):
    """ object for area under the curve (AUC) calculation
    example:
        import py_auc
        a = py_auc.AUC(sg.get_asDataFrame())
        a.cal_auc_rank()
        a.plot_ROC()
    """

    def __init__(self, data=None, debug=False):
        """
        initialize with scores and classes
        input: data - panda DataFrame with "Score" and "Class" columns
        """

        self._scores = []
        self._classes = []
        self._data = data

        self._debug = debug

        self._n0 = 0       # class 0
        self._n1 = 0       # class 1
        self._n = 0

        self._spm = None

        if (data is not None) and ("Score" in data.columns) and ("Class" in data.columns):
            self.get_classes_scores(data["Class"], data["Score"])

    def get_classes_scores(self, c, s):
        """ add scores and classes """

        if len(c) != len(s):
            print('... dimension is not matches: score - %d    class - %d'.format(len(s), len(c)))
            return

        self._scores = s
        self._classes = c

        self._n = len(self._scores)
        self._n1 = np.count_nonzero(self._classes)
        self._n0 = self._n - self._n1

        self._prepare()

    def get_scores(self, s0, s1):
        """ add class 0 score and class 1 score separately """

        self._n0 = len(s0)
        self._n1 = len(s1)
        self._n = self._n0 + self._n1

        self._scores = np.zeros(self._n)
        self._scores[:self._n0] = s0
        self._scores[self._n0:] = s1
        self._classes = np.ones(self._n)
        self._classes[:self._n0] = 0

        self._prepare()

    def n(self):
        return self._n

    def n0(self):
        return self._n0

    def n1(self):
        return self._n1

    def _prepare(self):
        """ calculate rank """

        self._data = pd.DataFrame()

        self._data['Score'] = self._scores
        self._data['Class'] = self._classes

        self._spm = self._data.sort_values(by='Score', ascending=False)
        self._spm['Rank'] = range(self._n+1)[1:]
        self._spm['FPR'] = (self._spm['Class'] == 0).cumsum().values/self._n0         # FPR
        self._spm['TPR'] = (self._spm['Class'] == 1).cumsum().values/self._n1         # TPR or recall
        self._spm['Prec'] = (self._spm['Class'] == 1).cumsum().values/(np.arange(self._n) + 1)    # precision

    def cal_auc_rank(self, measure_time=False):
        """ calculate area under ROC using rank algorithm """

        if measure_time: start_time = time.time()

        self._spm = self._data.sort_values(by='Score', ascending=False)
        self._spm['Rank'] = np.arange(self._n+1)[1:]/self._n
        mask = self._spm['Class'] == 0

        auc = 0.5 + (self._spm[mask].Rank.mean() - self._spm[~mask].Rank.mean())

        if measure_time:
            return (auc, (time.time() - start_time))
        else:
            return (auc)

    def cal_auc_bac(self, measure_time=False):
        """ calculate area under ROC using rank algorithm """

        if measure_time: start_time = time.time()

        self._spm = self._data.sort_values(by='Score', ascending=False)
        self._spm['FPR'] = (self._spm['Class'] == 0).cumsum().values/self._n0         # FPR
        self._spm['TPR'] = (self._spm['Class'] == 1).cumsum().values/self._n1         # TPR or recall

        auc = np.sum(self._spm['TPR'] + 1 - self._spm['FPR'])/self._n - 0.5

        if measure_time:
            return (auc, (time.time() - start_time))
        else:
            return (auc)

    def cal_auc_trapz(self, measure_time=False):
        """ calculate area under ROC using trapz function """

        if measure_time: start_time = time.time()

        self._spm = self._data.sort_values(by='Score', ascending=False)
        x = (self._spm['Class'] == 0).cumsum().values/self._n0         # FPR
        y = (self._spm['Class'] == 1).cumsum().values/self._n1         # TPR or recall

        auc = np.trapz(y, x=x)

        if measure_time:
            return (auc, (time.time() - start_time))
        else:
            return (auc)

    def cal_auc_sklearn(self, measure_time=False):
        """ calculate area under ROC using scikit-learning """

        if measure_time: start_time = time.time()

        auc = roc_auc_score(self._classes, self._scores)

        if measure_time:
            return (auc, (time.time() - start_time))
        else:
            return (auc)

    def cal_auprc_rank(self, measure_time=False):
        """ calculate area under precision-recall curve using rank algorithm """

        if measure_time: start_time = time.time()

        rho = self._n1/self._n

        self._spm = self._data.sort_values(by='Score', ascending=False)
        p = (self._spm['Class'] == 1).cumsum().values/(np.arange(self._n) + 1)    # precision

        auprc = 0.5*rho*(1.0 + np.sum(p*p)/(self._n*rho*rho))

        if measure_time:
            return (auprc, (time.time() - start_time))
        else:
            return(auprc)

    def cal_auprc_trapz(self, measure_time=False):
        """ calculate area under precision-recall curve using trapz algorithm """

        if measure_time: start_time = time.time()

        self._spm = self._data.sort_values(by='Score', ascending=False)
        y = (self._spm['Class'] == 1).cumsum().values/self._n1         # TPR or recall
        p = (self._spm['Class'] == 1).cumsum().values/(np.arange(self._n) + 1)    # precision

        auprc = np.trapz(p, x=y)

        if measure_time:
            return (auprc, (time.time() - start_time))
        else:
            return(auprc)

    def cal_auprc_sklearn(self, measure_time=False):
        """ calculate area under PRC using scikit-learning """

        if measure_time: start_time = time.time()

        auprc = average_precision_score(self._classes, self._scores)

        if measure_time:
            computetime = time.time() - start_time
            return (auprc, computetime)
        else:
            return auprc

    def plot_rank(self, sampling=10, filename=''):
        """ plot rank vs class """

        self._prepare()

        cmatrix = self._spm['Class'].values.reshape(sampling, -1)
        prob = cmatrix.mean(axis=1)

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,4))

        ax0 = axs[0]
        ax0.plot(prob, '.')
        ax0.set_xlabel('Sampled Rank r\'')
        ax0.set_ylabel('P(1|r\')')

        ax1 = axs[1]
        ax1.plot(self._spm['Rank'], self._spm['TPR'], label='TPR')
        ax1.plot(self._spm['Rank'], self._spm['FPR'], '--', label='FPR')
        ax1.set_xlabel('Rank r')
        ax1.set_ylabel('TPR(r), FPR(r)')
        ax1.legend()

        ax2 = axs[2]
        ax2.plot(self._spm['Rank'], self._spm['Prec'], label='prec')
        bac = (self._spm['TPR'].values + 1.0 - self._spm['FPR'].values)/2.0
        ax2.plot(self._spm['Rank'], bac, '--', label='bac')
        ax2.set_xlabel('Rank r')
        ax2.set_ylabel('Precision(r), bac(r)')
        ax2.legend()

        if filename == '': filename = 'rank_plot.pdf'
        plt.savefig(filename, dpi=150)

    def plot_ROC(self, bins=50, filename=''):
        """ calculate ROC curve (receiver operating characteristic curve) """

        self._prepare()

        auc = np.trapz(self._spm['TPR'], x=self._spm['FPR'])
        print('AUC (area under the ROC curve): {0:8.3f}'.format(auc))

        auc = np.trapz(self._spm['Prec'], x=self._spm['TPR'])
        print('AUPRC (area under the PRC curve): {0:8.3f}'.format(auc))

        # ROC plot
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,4))

        ax0 = axs[0]
        ax0.plot(self._spm['FPR'], self._spm['TPR'], label='ROC')
        ax0.set_xlabel('FPR')
        ax0.set_ylabel('TPR')
        #ax0.set_xlim(0, 1.1)
        #ax0.set_ylim(0, 1.1)

        # PR plot
        ax1 = axs[1]
        ax1.plot(self._spm['TPR'], self._spm['Prec'], label='PRC')
        ax1.set_xlabel('Recall (TPR)')
        ax1.set_ylabel('Precision')
        #ax1.set_xlim(0, 1.1)
        #ax1.set_ylim(0, 1.1)
        #ax1.set_title('Precision-Recall Curve')

        # score distribution
        ax2 = axs[2]
        sns.distplot(self._spm.loc[self._spm['Class']==0, 'Score'], bins=bins, kde=False, rug=True, label='Class 0')
        sns.distplot(self._spm.loc[self._spm['Class']==1, 'Score'], bins=bins, kde=False, rug=True, label='Class 1')
        ax2.legend(loc='upper right')
        ax2.set_xlabel('Scores')
        ax2.set_ylabel('#')

        if filename == '':
            filename = 'auc_summary.pdf'
        plt.savefig(filename, dpi=150)
        plt.show()


class Score_generator(object):
    """ two class score generator
    example:
        import py_auc
        sg = py_auc.Score_generator()
        sg.set0('gaussian', 0, 1, 1000)
        sg.set1('gaussian', 3, 1, 1000)
    """

    def __init__(self):
        """ """

        self._kind0 = ''
        self._kind1 = ''
        self._mu0 = 0
        self._mu1 = 0
        self._std0 = 0
        self._std1 = 0
        self._n0 = 100
        self._n1 = 100
        self._n = self._n0 + self._n1
        self._rho = self._n1/self._n

        self._s0 = []
        self._s1 = []
        self._prob = []
        self._sampleN = 0
        self._sampling = []

    def _generate(self, kind, mu, std, n):
        """ set parameters of class """

        if kind.lower() not in ['uniform', 'gaussian', 'triangle']:
            kind = 'uniform'

        if kind.lower() == 'uniform':
            temp = np.random.uniform(low=mu-std, high=mu+std, size=n)
        elif kind.lower() == 'gaussian':
            temp = np.random.normal(loc=mu, scale=std, size=n)
        elif kind.lower() == 'triangle':
            temp = np.random.triangular(mu-std, mu, mu+std, size=n)

        return temp, kind, mu, std, n

    def set0(self, kind, mu, std, n):
        """
        kind : ['uniform', 'gaussian', 'triangle']
        mu : mean
        std : standard deviation
        n : number of samples
        """

        self._s0, self._kind0, self._mu0, self._std0, self._n0 = self._generate(kind, mu, std, n)

    def set1(self, kind, mu, std, n):
        """
        kind : ['uniform', 'gaussian', 'triangle']
        mu : mean
        std : standard deviation
        n : number of samples
        """
        self._s1, self._kind1, self._mu1, self._std1, self._n1 = self._generate(kind, mu, std, n)

        self._n = self._n0 + self._n1
        self._rho = float(self._n1/self._n)

    def get(self):
        """ get scores """

        return [self._s0, self._s1]

    def get_asDataFrame(self):
        """ get scores as DataFrame """

        scores = np.zeros(self._n)
        scores[:self._n0] = self._s0
        scores[self._n0:] = self._s1
        classes = np.ones(self._n)
        classes[:self._n0] = 0

        return pd.DataFrame({'Score': scores, 'Class': classes})

    def get_randomSample(self, n):
        """ get sample of scores """

        temp = self.get_asDataFrame()
        return temp.sample(n)

    def get_classProbability(self, sampleSize=100, sampleN=100, measure_time=False):
        """ calculate probability of class 1 at given rank r """

        if measure_time: start_time = time.time()

        temp = self.get_asDataFrame()
        temp0 = temp[temp['Class'] == 0]
        temp1 = temp[temp['Class'] == 1]
        n1 = int(sampleSize*self._rho)
        n0 = sampleSize - n1

        res = pd.DataFrame()
        res['Rank'] = range(sampleSize+1)[1:]

        for i in range(sampleN):
            a = pd.concat([temp0.sample(n0), temp1.sample(n1)]).sort_values(by='Score', ascending=False)
            res['Class_{}'.format(i)] = a['Class'].values

        self._prob = res.values[:, 1:sampleN+1].mean(axis=1)
        res['P(1|r)'] = self._prob
        res['P(0|r)'] = 1 - self._prob

        res['TPR'] = np.cumsum(res['P(1|r)'])/n1
        res['FPR'] = np.cumsum(res['P(0|r)'])/n0
        res['Prec'] = np.cumsum(res['P(1|r)'])/res['Rank']
        res['bac'] = 0.5*(res['TPR'] + 1.0 - res['FPR'])
        self._sampling = res

        self._sampleN = sampleN
        self._sampleSize = sampleSize
        self._sampleN0 = n0
        self._sampleN1 = n1
        self._auc = np.sum(res['P(0|r)']*res['Rank']/n0 - res['P(1|r)']*res['Rank']/n1)/sampleSize + 0.5
        self._aucbac = 2*np.sum(res['bac'])/sampleSize - 0.5
        self._auprc = 0.5*self._rho + 0.5*np.sum(res['Prec']*res['Prec'])/n1

        if measure_time:
            return (res, (time.time()-start_time))
        else:
            return res

    def plot(self, filename='', show=True):
        """ plot histogram """

        plt.close('all')
        fig = plt.figure(figsize=(6,5))

        sns.distplot(self._s0, bins=50, kde=False, rug=True, label='Class 0 (#={})'.format(self._n0))
        sns.distplot(self._s1, bins=50, kde=False, rug=True, label='Class 1 (#={})'.format(self._n1))
        plt.annotate("mu={}\ns={}".format(self._mu0, self._std0), xy=(self._mu0, 0), xytext=(0.25, 0.25),
                textcoords='axes fraction', horizontalalignment='left',
                arrowprops=dict(facecolor='black', shrink=0.05))
        plt.annotate("mu={}\ns={}".format(self._mu1, self._std0), xy=(self._mu1, 0), xytext=(0.75, 0.25),
                textcoords='axes fraction', horizontalalignment='right',
                arrowprops=dict(facecolor='black', shrink=0.05))
        plt.xlabel('Score')
        plt.ylabel('#')
        plt.legend()

        if filename == '':
            filename = 'score_hist.pdf'

        plt.savefig(filename, dpi=150)
        if show: plt.show()

    def plot_prob(self, filename='', ss=100, sn=100, fig=None, show=True, sample=None):
        """ plot class probability """

        if sample is not None:
            a = sample
        else:
            a = self.get_classProbability(sampleSize=ss, sampleN=sn)

        if fig is None:
            plt.close('all')
            fig = plt.figure(figsize=(14, 6))

        ax1 = fig.add_subplot(121)
        r = range(len(self._prob))
        ax1.plot(r, self._prob, '.', label='Size={}, #={}'.format(len(self._prob), self._sampleN), alpha=0.5)
        ax1.set_xlabel('Rank')
        ax1.set_ylabel('P(1|r)')
        ax1.set_ylim((-0.05, 1.05))
        ax1.legend()

        ax2 = fig.add_subplot(122)
        ax2.plot(a['FPR'], a['TPR'], '.', label='Size={}, #={}, auc={:.4f}'.format(len(self._prob), self._sampleN, self._auc), alpha=0.5)
        ax2.set_xlabel('FPR')
        ax2.set_ylabel('TPR')
        ax2.legend()

        if filename == '':
            filename = 'classProbability.pdf'
        plt.savefig(filename, dpi=150)

        if show:
            plt.show()
            return
        else:
            return fig

