import numpy as np


class OutlierDetection:
    def __init__(self, thrs = 0.2, coef = 1):
        self.thrs = thrs
        self.coef = coef
        self.hash_table = {}
        
    def fit(self, X):
        self.X = np.array(X)
        self.length = len(self.X)
        self.limit = int(self.length*self.thrs)
        self.multipliers = []
        self.predictions = []
        
        for i in range(len(self.X)):
            self.hash_table[i] = 0
            
        for i in range(self.X.shape[1]):
            self.multipliers.append(self.coef*(self.X[:, i].std()/self.X[:, i].mean()))
        
        return self

    def fit_predict(self, X):
        self.fit(X)
        #         print(f"multipliers = {self.multipliers}")
        for i in range(self.length):
            total = 0   
            
            if self.hash_table[i] >= self.limit:
                self.predictions.append(1)
                continue
                
            for j in range(self.length):
                temp = 1
                for k in range(self.X.shape[1]):
                    if (self.X[i][k]*(1-self.multipliers[k]) <= self.X[j][k] <= (self.X[i][k]*(1+self.multipliers[k]))):
                        temp *= 1
                    else:
                        temp *= 0
                self.hash_table[j] += temp
                total += temp        
            if total >= self.limit:
                self.predictions.append(1)
            else:
                self.predictions.append(0)
                
        return self.predictions
    
    def predict(self, X):
        to_predict = np.array(X)
        
        if to_predict.shape[1] != self.X.shape[1]:
            raise Exception("Dimension of the data is not same with the fitted data!")
            
        predictions = []
        for i in range(len(to_predict)):
            total = 0                  
            for j in range(self.length):
                temp = 1
                for k in range(self.X.shape[1]):
                    if (to_predict[i][k]*(1-self.multipliers[k]) <= self.X[j][k] <= (to_predict[i][k]*(1+self.multipliers[k]))):
                        temp *= 1
                    else:
                        temp *= 0
#                 self.hash_table[j] += temp
                total += temp        
            if total >= self.limit:
                predictions.append(1)
            else:
                predictions.append(0)
                
        return predictions
    
    def __repr__(self):
        return f"OutlierDetection(threshold = {self.thrs}, coefficient = {self.coef})"