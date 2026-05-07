import scipy as sp
import numpy as np
from dateutil.relativedelta import relativedelta


###################################################################################################
class NSScurve:
    
    def nss_yield_curve(tenors, beta0, beta1, beta2, beta3, lambda1, lambda2):
        """
        Compute the Nelson-Siegel-Svensson (NSS) yield curve.
        
        Parameters
        ----------
        tenors : array-like
            List or numpy array of tenors (in years).
        beta0, beta1, beta2, beta3 : float
            NSS curve parameters.
        lambda1, lambda2 : float
            Decay parameters.
        
        Returns
        -------
        numpy.ndarray
            NSS yields for each tenor.
        """
        tenors = np.asarray(tenors, dtype=float)
        
        # Avoid division by zero for very small tenors
        t_over_l1 = tenors / lambda1
        t_over_l2 = tenors / lambda2
    
        def factor(t_over_l):
            return (1 - np.exp(-t_over_l)) / t_over_l
    
        f1 = factor(t_over_l1)
        f2 = f1 - np.exp(-t_over_l1)
        f3 = factor(t_over_l2) - np.exp(-t_over_l2)
    
        return beta0 + beta1 * f1 + beta2 * f2 + beta3 * f3

    
    @classmethod 
    def nssFitter(self,tenors, yields):
        """
        Fits the Nelson-Siegel-Svensson (NSS) yield curve.
        
        Parameters
        ----------
        tenors : array-like
            List or numpy array of tenors (in years).
        yields : array-like
            List or numpy array of yields (in %).
        
        Returns
        -------
        beta0, beta1, beta2, beta3 : float
            NSS curve parameters.
        lambda1, lambda2 : float
            Decay parameters.
        All packaged as a dictionary
        """
        param, param_cov=sp.optimize.curve_fit(self.nss_yield_curve,tenors, yields)
        results=dict(zip(["beta0", "beta1", "beta2", "beta3", "lambda1", "lambda2"],param))
        
        return results


###################################################################################################
class bondPrice:
    
    def dates_vector(today, maturity, frequency):
    # Ensure that the inputs are of the correct type
    # if not isinstance(today, datetime.date):
    #     raise ValueError(f"Expected 'today' to be a date, got {type(today)}")
    # if not isinstance(maturity, datetime.date):
    #     raise ValueError(f"Expected 'maturity' to be a date, got {type(maturity)}")
    # if not isinstance(frequency, int):
    #     raise ValueError(f"Expected 'frequency' to be an int, got {type(frequency)}")
    
        dates = []
        years = []
        delta = []
        n = 0
        
        while maturity - relativedelta(months=frequency * n) > today:
            
            maturity_n = maturity - relativedelta(months=frequency * n)
            dates.append(maturity_n)
            years.append(maturity_n.year - today.year + 1)
            maturity_n1 = maturity - relativedelta(months=frequency * n)
            n += 1
            delta.append((max(today, maturity_n1) - today).days / 360)
    
        if min(years) == 2:
            years = [y - 1 for y in years]
    
        output = np.column_stack((list(reversed(dates)), list(reversed(years)), list(reversed(delta))))
    
        return output

    # YTM calculation (target, [inputs])
    def ytm_calc(x, T_Vector, Nominal, Coupon, Frequency, MarketV):
        T = T_Vector.shape[0]
        cashflow = np.zeros(T)
        deflator = np.zeros(T)
        for t in range(T):
            time = T_Vector[t, 2]
            deflator[t] = (1 + x) ** -time
            if t < T - 1:
                cashflow[t] = Nominal * Coupon * Frequency / 12 * deflator[t]
            else:
                cashflow[t] = Nominal * (1 + Coupon * Frequency / 12) * deflator[t]
    
        return np.sum(cashflow) - MarketV
    
    # Spread calculation (target, [inputs])
    def spread_calc(x, nominal, coupon, spot_rate, t_vector, frequency, market_value):
        T = t_vector.shape[0]
        cashflow = np.zeros(T)
        deflator = np.zeros(T)
        for t in range(T):
            rate = spot_rate[min(t_vector[t, 1], len(spot_rate) - 1)]
            time = t_vector[t, 2]
            deflator[t] = (1 + rate + x) ** -time
            if t < T - 1:
                cashflow[t] = nominal * coupon * frequency / 12 * deflator[t]
            else:
                cashflow[t] = nominal * (1 + coupon * frequency / 12) * deflator[t]
    
        return np.sum(cashflow) - market_value
    
    # Macaulay duration calculation
    def macaulay_duration(nominal, coupon, spot_rate, t_vector, frequency, spread):
        T = t_vector.shape[0]
        cashflow = np.zeros(T)
        deflator = np.zeros(T)
        weight_dur = np.zeros(T)
        for t in range(T):
            rate = spot_rate[min(t_vector[t, 1], len(spot_rate) - 1)]
            time = t_vector[t, 2]
            deflator[t] = (1 + rate + spread) ** -time
            if t < T - 1:
                cashflow[t] = nominal * coupon * frequency / 12 * deflator[t]
            else:
                cashflow[t] = nominal * (1 + coupon * frequency / 12) * deflator[t]
            weight_dur[t] = cashflow[t] * time
        return round(np.sum(weight_dur) / np.sum(cashflow), 6)
    
    # Market Values calculation
    def market_value_calc(nominal, coupon, spot_rate, t_vector, frequency, spread):
        T = t_vector.shape[0]
        cashflow = np.zeros(T)
        deflator = np.zeros(T)
        for t in range(T):
            rate = spot_rate[min(t_vector[t, 1], len(spot_rate) - 1)]
            time = t_vector[t, 2]
            deflator[t] = (1 + rate + spread) ** -time
            if t < T - 1:
                cashflow[t] = nominal * coupon * frequency / 12 * deflator[t]
            else:
                cashflow[t] = nominal * (1 + coupon * frequency / 12) * deflator[t]
    
        return round(np.sum(cashflow), 2)
    
    # Effective Duration calculation
    def effective_duration(nominal, coupon, spot_rate, t_vector, frequency, spread, market_value, shift):
        value_up = market_value_calc(nominal, coupon, spot_rate + shift, t_vector, frequency, spread)
        value_down = market_value_calc(nominal, coupon, spot_rate - shift, t_vector, frequency, spread)
        return round((value_down - value_up) / (market_value * shift * 2), 6)
    
    # Convexity calculation 
    def convexity(nominal, coupon, spot_rate, t_vector, frequency, spread, market_value, shift):
        value_up = market_value_calc(nominal, coupon, spot_rate + shift, t_vector, frequency, spread)
        value_down = market_value_calc(nominal, coupon, spot_rate - shift, t_vector, frequency, spread)
        return round((value_down + value_up - 2 * market_value) / (market_value * shift ** 2), 6)