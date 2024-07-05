import numpy as np
from numpy.polynomial import laguerre as lag
import matplotlib.pyplot as plt

T=2 #Time to maturity in years
steps=252*T #Time steps; 252 trading days in a year
S0=100.0 #Initial underlying price
r=0.05 #Risk-free rate
M=10000 #number of underlying paths to simulate
sig=0.4 #Volatility of underlying
q=0.1 #Dividend yield
face=100 #Face value of the bond
N=1 #Conversion ratio - the number of shares the holder recieves upon conversion
P=face/N #Conversion price


def GBM(T, steps, S0, r, M, sig, q):
    dt=T/steps
    WSinc=np.random.normal(0, 1, (steps, M))
    St=np.full(shape=(steps,M), fill_value=S0)
    for i in range(1,steps):
        St[i]=St[i-1]+(r-q)*St[i-1]*dt+sig*St[i-1]*np.sqrt(dt)*WSinc[i-1,:]
    return np.transpose(St) #Transpose makes it so that each row is one path, rather than each column

def conversion_value(ratio, stock_price):
    return ratio*stock_price

def cash_flow_mat(ratio, stock_price, face):
    return np.maximum(conversion_value(ratio, stock_price), face) #At maturity, the holder can either have the face value of the bond 
                                                                  #returned to them, or they can convert into stock if that's more profitable.

def LSMC(T, steps, S0, r, M, sig, q, N, face, deg):
    paths=GBM(T, steps, S0, r, M, sig, q)
    time=np.linspace(0, T, steps)
    ST=paths[:,-1]
    CB=np.array(cash_flow_mat(N,ST,face)) #Initialise the bond value to be the cash flow at maturity for each path
    P=face/N #Conversion Price
    
    for t in range(steps-1, 0, -1): #Looping backwards; start at the last timestep and move back by 1 until the first time step
    
        dt=time[t]-time[t-1] #Time steps are uniform, so this is the same for all t
        df=np.exp(-r*dt) #Defining discount factor
        
        
        curr_price=paths[:,t] #Current price of each path at time t
        itm_idx = curr_price > P #Array of True/False, where True signifies path ITM the current timestep. Conversion price is used to determine if the convertible is ITM
        
        itm_curr_price = curr_price[itm_idx] #Array of current prices of the paths that are ITM at the current timestep. 1 Price at the current
                                             #timestep for each ITM path.
        
        
        
        fit_laguerre=lag.lagfit(itm_curr_price, df*CB[itm_idx], deg) #Fitting a laguerre series to the pair (x=current price, y=discounted cashflows)
        
        contvalues=lag.lagval(curr_price, fit_laguerre) #Compute the laguerre series at each price. This is an array of 
                                                        #continuation values for each path at the current time step.
                                                        
        immediate_conv_value=conversion_value(N, curr_price) #Array of immediate conversion values for each path.
        
        conv_idx = itm_idx & (immediate_conv_value>contvalues)
        
       
        CB=df*CB #Set the value of the bond at time t to be the value of the bond at time t+1, discounted back to time t 
        CB[conv_idx]=immediate_conv_value[conv_idx] #For all the paths where conversion is optimal, set the value of the bond to the conversion value
        
            
    return CB


Mvis=50 #Use a smaller number of paths and steps for plotting, just for visualisation purposes
stepsvis=75
paths=GBM(T=T, steps=stepsvis, S0=S0, r=r, M=Mvis, sig=sig, q=q)
time=np.linspace(0, T, stepsvis)

plt.figure(figsize=(16, 10))
plt.plot(time, np.transpose(paths), label="Stock Price")
plt.xlabel("Time (Years)", size=20)
plt.ylabel(r"Price $S(t)$, USD", size=20)
plt.title(f"Underlying Paths | Geometric Brownian Motion, {Mvis} Paths", size=20)
plt.grid(True)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)    
plt.tight_layout()
plt.show()

#color paths teal when they become ITM, orange for ATM or OTM. Plot a single path
#to show the idea
plt.figure(figsize=(16, 10))
colours = ['teal' if (S>P) else 'orangered' for S in paths[0]]
plt.plot(time, paths[0], color="gray")
plt.scatter(time, paths[0], c=colours, s=60, zorder=3, alpha=0.75)
plt.xlabel("Time (Years)", size=20)
plt.ylabel(r"Price $S(t)$, USD", size=20)
plt.title("One Underlying Path | Teal = ITM, Orange = ATM/OTM", size=20)
plt.grid(True)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)    
plt.tight_layout()    
plt.show() 

#Repeat for every path
plt.figure(figsize=(16, 10))
for path in paths:
    colours = ['teal' if (S>P) else 'orangered' for S in path]
    plt.plot(time, path, color="gray")
    plt.scatter(time, path, c=colours, s=60, zorder=3, alpha=0.5)
plt.xlabel("Time (Years)", size=20)
plt.ylabel(r"Price $S(t)$, USD", size=20)
plt.title("Underlying Paths | Teal = ITM, Orange = ATM/OTM", size=20)
plt.grid(True)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)    
plt.tight_layout()    
plt.show()

#Show the laguerre series fitting for a single pair of time steps:
Stn=paths[:,-1] #Final price of the equity at maturity for every path, at t_n=T
Stnm1=paths[:,-2] #Price of the equity for every path, at t_{n-1}.
deg=6 #Highest degree in laguerre series
dt=time[-1]-time[-2]
df=np.exp(-r*dt)
disc_cashflows=df*cash_flow_mat(N, Stn, face) #Cashflow at t_n=T, discounted to t_{n-1}. 1 for each path
fit_laguerre=lag.lagfit(Stnm1, disc_cashflows, deg)#This just returns the coefficients of the fitted laguerre series up to degree deg
s_linspace=np.linspace(min(Stnm1), max(Stnm1), len(Stnm1)) #Generate values on the x axis to plot the laguerre series
contvalues=lag.lagval(s_linspace, fit_laguerre) #Compute the laguerre series at each x value, with coefficients
#given by fit_laguerre. This gives us an approximation to the conditional expectation of the future 
#value of the bond, discounted to today. Namely, contvalues is the value of continuation at t_{n-1} for each path, which is
#the expectation value of the cashflow at the next timestep t_{n} given the underlying price at t_{n-1}. 


plt.figure(figsize=(16, 10))
plt.scatter(Stnm1, disc_cashflows, label="Discounted Cashflows")
plt.plot(s_linspace, contvalues, label=f"Fitted Laguerre Series, degree={deg}", color="orange")
plt.title("Discounted Cashflows vs Underlying Price", size=20)
plt.xlabel("Underlying $S(t_{n-1})$, USD", size=20)
plt.ylabel("Cashflows at $t_{n-1}$, USD", size=20)
plt.legend(loc="upper left")
plt.grid(True)
plt.show()



price=np.average(LSMC(T, steps, S0, r, M, sig, q, N, face, deg=deg))
print(f"Convertible Bond Price = ${price}")






