import numpy as np
import dA as AE
import corClust as CC


class KitNET:
    def __init__(self, n, max_autoencoder_size=10, FM_grace_period=None, AD_grace_period=10000, learning_rate=0.1,
                 hidden_ratio=0.75, feature_map=None):

        self.AD_grace_period = AD_grace_period
        if FM_grace_period is None:
            self.FM_grace_period = AD_grace_period
        else:
            self.FM_grace_period = FM_grace_period
        if max_autoencoder_size <= 0:
            self.m = 1
        else:
            self.m = max_autoencoder_size
        self.lr = learning_rate
        self.hr = hidden_ratio
        self.n = n

        self.n_trained = 0
        self.n_executed = 0
        self.v = feature_map
        if self.v is None:
            print("Feature-Mapper: train-mode, Anomaly-Detector: off-mode")
        else:
            self.__createAD__()
            print("Feature-Mapper: execute-mode, Anomaly-Detector: train-mode")
        self.FM = CC.corClust(self.n)
        self.ensembleLayer = []
        self.outputLayer = None

    def process(self, x):
        if self.n_trained > self.FM_grace_period + self.AD_grace_period:
            return self.execute(x)
        else:
            self.train(x)
            return 0.0


    def train(self, x):
        if self.n_trained <= self.FM_grace_period and self.v is None:
            self.FM.update(x)
            if self.n_trained == self.FM_grace_period:
                self.v = self.FM.cluster(self.m)
                self.__createAD__()
                print("The Feature-Mapper found a mapping: " + str(self.n) + " features to " + str(
                    len(self.v)) + " autoencoders.")
                print("Feature-Mapper: execute-mode, Anomaly-Detector: train-mode")
        else:  # 训练
            S_l1 = np.zeros(len(self.ensembleLayer))
            for a in range(len(self.ensembleLayer)):
                xi = x[self.v[a]]
                S_l1[a] = self.ensembleLayer[a].train(xi)
            self.outputLayer.train(S_l1)
            if self.n_trained == self.AD_grace_period + self.FM_grace_period:
                print("Feature-Mapper: execute-mode, Anomaly-Detector: exeute-mode")
        self.n_trained += 1

    def execute(self, x):
        if self.v is None:
            raise RuntimeError(
                'KitNET Cannot execute x, because a feature mapping has not yet been learned or provided. '
                'Try running process(x) instead.')
        else:
            self.n_executed += 1
            S_l1 = np.zeros(len(self.ensembleLayer))
            for a in range(len(self.ensembleLayer)):
                xi = x[self.v[a]]
                S_l1[a] = self.ensembleLayer[a].execute(xi)
            return self.outputLayer.execute(S_l1)

    def __createAD__(self):
        for map in self.v:
            params = AE.dA_params(n_visible=len(map), n_hidden=0, lr=self.lr, corruption_level=0, gracePeriod=0,
                                  hiddenRatio=self.hr)
            self.ensembleLayer.append(AE.dA(params))

        params = AE.dA_params(len(self.v), n_hidden=0, lr=self.lr, corruption_level=0, gracePeriod=0,
                              hiddenRatio=self.hr)
        self.outputLayer = AE.dA(params)
