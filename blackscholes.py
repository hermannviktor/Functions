class BlackScholes(object):

    ''' Creating the BlackScholes class'''

    def __init__(self, spot, strike, interest, vol, time):
        '''
        Creating the inputs that are needed for all operations

        interest: is NOT in percentage

        vol: is NOT in percentage

        time: is in years (one year is 365 days)
        '''
        import numpy as np
        self.spot = spot
        self.strike = strike
        self.interest = interest
        self.vol = vol
        self.time = time

        assert spot >= 0, 'Spot price needs to be at least 0'
        assert strike >= 0, 'Strike price needs to be at least 0'
        assert vol >= 0, 'Volatility needs to be positive'

        d1 = (np.log(self.spot / self.strike) + (self.interest + (self.vol**2 / 2))*self.time) / (self.vol * np.sqrt(self.time))
        d2 = d1 - self.vol * np.sqrt(self.time)

        self.d1 = d1
        self.d2 = d2

    def call(self):
        '''C = S * N(d1) - K * e^(-rt) * N(d2)'''
        import numpy as np
        from scipy.stats import norm

        call = self.spot * norm.cdf(self.d1) - self.strike * np.exp(- self.interest * self.time) * norm.cdf(self.d2)
        call = float(call)

        return call

    def put(self):
        '''P = K * e^(-rt) * N(-d2) - S * N(-d1)'''
        import numpy as np
        from scipy.stats import norm

        put = self.strike * np.exp(- self.interest * self.time) * norm.cdf(-self.d2) - self.spot * norm.cdf(-self.d1)
        put = float(put)

        return put

    def delta_call(self):
        '''delta_call = N(d1)'''
        from scipy.stats import norm

        delta_call = norm.cdf(self.d1)
        delta_csll = float(delta_call)

        return delta_call

    def delta_put(self):
        '''delta_put = N(d1) - 1'''
        from scipy.stats import norm

        delta_put = norm.cdf(self.d1) - 1
        delta_put = float(delta_put)

        return delta_put

    def gamma(self):
        '''gamma = N'(d1) / S * sigma * sqrt(T)'''
        import numpy as np
        from scipy.stats import norm

        gamma = norm.pdf(self.d1) / (self.spot * self.vol * np.sqrt(self.time))
        gamma = float(gamma)

        assert gamma >= 0, 'Gamma needs to be above 1'

        return gamma

    def vega(self):
        '''vega = (S * N'(d1) * sqrt(T) )'''
        import numpy as np
        from scipy.stats import norm

        vega = self.spot * norm.pdf(self.d1) * np.sqrt(self.time)
        vega = float(vega)

        return vega

    def theta_call(self):
        '''theta_call = - ((S * N'(d1) * sigma) / (2 * sqrt(T))) - r * K * e^(-rT) N(d2)'''
        import numpy as np
        from scipy.stats import norm

        theta_call = -((self.spot * norm.pdf(self.d1) * self.vol) / (2 * np.sqrt(self.time))) - (self.interest * self.strike * np.exp(- self.interest * self.time) * norm.cdf(self.d2))
        theta_call = float(theta_call / (self.time * 365))

        return theta_call

    def theta_put(self):
        '''theta_put = - ((S * N'(d1) * sigma) / (2 * sqrt(T))) + r * K * e^(-rT) N(-d2)'''
        import numpy as np
        from scipy.stats import norm

        theta_put = -((self.spot * norm.pdf(self.d1) * self.vol) / (2 * np.sqrt(self.time))) + self.interest * self.strike * np.exp(- self.interest * self.time) * norm.cdf(-self.d2)
        theta_put = float(theta_put / (self.time * 365))

        return theta_put

    def rho_call(self):
        '''rho_call = T * K * e^(-rT) * N(d2)'''
        import numpy as np
        from scipy.stats import norm

        rho_call = self.time * self.strike * np.exp(- self.interest * self.time) * norm.cdf(self.d2)
        rho_call = float(rho_call / 100)

        return rho_call

    def rho_put(self):
        '''rho_put = - T * K * e^(-rT) * N(-d2)'''
        import numpy as np
        from scipy.stats import norm

        rho_put =  - self.time * self.strike * np.exp(- self.interest * self.time) * norm.cdf(-self.d2)
        rho_put = float(rho_put / 100)

        return rho_put

    def call_montecarlo(self, simulations = 100000):
        '''Compute call option price with Monte Carlo simulation'''
        from random import gauss
        import numpy as np

        def generate_asset_price(S, v, r, T):
            return S * np.exp((r - 0.5 * v**2) * T + v * np.sqrt(T) * gauss(0,1.0))

        def call_payoff(S_T,K):
            return max(0.0,S_T-K)

        payoffs = []
        discount_factor = np.exp(-r * T)

        for i in range(simulations):
            S_T = generate_asset_price(self.spot, self.vol, self.interest, self.time)
            payoffs.append(call_payoff(S_T, self.strike))

        call_montecarlo = discount_factor * (sum(payoffs) / float(simulations))
        call_montecarlo = float(call_montecarlo)

        return call_montecarlo

    def put_montecarlo(self, simulations = 100000):
        '''Compute call option price with Monte Carlo simulation'''
        from random import gauss
        import numpy as np

        def generate_asset_price(S, v, r, T):
            return S * np.exp((r - 0.5 * v**2) * T + v * np.sqrt(T) * gauss(0,1.0))

        def put_payoff(S_T,K):
            return max(0.0,K - S_T)

        payoffs = []
        discount_factor = np.exp(-r * T)

        for i in range(simulations):
            S_T = generate_asset_price(self.spot, self.vol, self.interest, self.time)
            payoffs.append(put_payoff(S_T, self.strike))

        put_montecarlo = discount_factor * (sum(payoffs) / float(simulations))
        put_montecarlo = float(put_montecarlo)

        return put_montecarlo