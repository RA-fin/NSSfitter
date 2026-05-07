

import numpy as np
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt

import pricing

#select bond universe to fit

bondData = pd.DataFrame(columns=[
    "Frequency",
    "issue date",
    "maturity",
    "reference date",
    "notional",
    "coupon",
    "market value",
    "yield to maturity"
])

#function to minimize
def priceError(NSSparams):
    myPrices=[]
    for i,thisBond in bondData.iterrows():
        t_vector=pricing.bondPrice.dates_vector(thisBond["reference date"],thisBond["maturity"],thisBond["Frequency"])
        spot_rate=1/100*pricing.NSScurve.nss_yield_curve(t_vector[:,2],*NSSparams)
        thisPrice=pricing.bondPrice.market_value_calc(thisBond["notional"],thisBond['coupon'] / 100,spot_rate,t_vector,thisBond["Frequency"],0)
        myPrices.append(thisPrice)
        
    prices=bondData.loc[:,["market value"]]
    prices["myPrices"]=myPrices
    pricingError=(sum(np.log(prices["market value"]/prices["myPrices"])**2)/bondData.shape[0])**0.5
    
    return pricingError

#create starting values from YTM
NSSparameters=pricing.NSScurve.nssFitter(tenors=np.array((bondData["maturity"]-bondData["issue date"])/365.25),yields=bondData["yield to maturity"])

plt.plot((bondData["maturity"]-bondData["issue date"])/365.25,bondData["yield to maturity"],'o')
tenors=np.sort((bondData["maturity"]-bondData["issue date"])/365.25)
plt.plot(tenors,pricing.NSScurve.nss_yield_curve(tenors,*NSSparameters.values()))
plt.show

#minimize the function
res = minimize(priceError, list(NSSparameters.values()), method='L-BFGS-B')
res = minimize(priceError, list(NSSparameters.values()), method='BFGS', options={'gtol': 1e-3})


plt.plot(tenors,pricing.NSScurve.nss_yield_curve(tenors,*res.x))
