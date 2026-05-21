import pandas as pd
import numpy as np
data=pd.read_csv("Weather.csv")
X=data.drop(['Play'],axis=1)
Y=data['Play']
print("X=",X)
print("Y=",Y)
features=list(X)
print(features)
prior={}
likelihoods={}
pred_priors={}
for f in features:
    likelihoods[f] = {}
    pred_priors[f] = {}
    for f_val in np.unique(X[f]):
        pred_priors[f].update({f_val:0})
        for outcome in np.unique(Y):
            likelihoods[f].update({f_val+''+outcome:0})
            prior.update({outcome:0})
print("likelihoods=",likelihoods)
####################################################
