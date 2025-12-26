import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm

def get_nifty_data(ticker="^NSEI"):
    print(f"Fetching data for {ticker}...")

    stock_data = yf.download(ticker, period="1y", interval="1d")

    close_prices = stock_data['Close']

    if isinstance(close_prices, pd.DataFrame):
        close_prices = close_prices.iloc[:, 0]  
        
    log_returns = np.log(close_prices / close_prices.shift(1))
    
    volatility = log_returns.std() * np.sqrt(252)
    current_price = float(close_prices.iloc[-1])
    
    return current_price, volatility


def monte_carlo_option_pricer(S, K, T, r, sigma, n_simulations=50000, n_steps=252):
    """
    S: Spot Price
    K: Strike Price
    T: Time to Expiry
    r: Risk-free rate 
    sigma: Volatility 
    """
    dt = T / n_steps
    Z = np.random.standard_normal((n_steps, n_simulations))
    
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    
    price_paths = np.zeros((n_steps + 1, n_simulations))
    price_paths[0] = S
    
    growth_factors = np.exp(drift + diffusion)
    price_paths[1:] = S * np.cumprod(growth_factors, axis=0)
    
    ST = price_paths[-1]
    payoffs = np.maximum(ST - K, 0)
    
    option_price = np.exp(-r * T) * np.mean(payoffs)
    
    return option_price, price_paths


def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = (S * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
    return call_price

if __name__ == "__main__":

    try:
        S, sigma = get_nifty_data("^NSEI")
    except:
        S, sigma = 24000, 0.12 
    
    print(f"Current NIFTY Price: ₹{S:.2f}")
    print(f"Annualized Volatility: {sigma*100:.2f}%")
    

    K = float(input(f"Enter Strike Price (Close to {int(S)}): ")) 
    T_days = int(input("Days to Expiry (e.g., 30): "))
    T = T_days / 365.0
    r = 0.068  
    

    print("\nRunning 10,000 simulations...")
    mc_price, paths = monte_carlo_option_pricer(S, K, T, r, sigma, n_simulations=50000)
    

    bs_price = black_scholes_call(S, K, T, r, sigma)
    
    print("-" * 40)
    print(f"Monte Carlo Price:   ₹{mc_price:.2f}")
    print(f"Black-Scholes Price: ₹{bs_price:.2f}")
    print(f"Difference:          ₹{abs(mc_price - bs_price):.2f}")
    print("-" * 40)
    
    plt.figure(figsize=(10, 6))
    plt.plot(paths[:, :100]) 
    plt.axhline(K, color='black', linestyle='--', label=f'Strike Price ({K})')
    plt.title(f'Monte Carlo Simulation: NIFTY 50 Option Pricing\n({T_days} Days to Expiry)')
    plt.xlabel('Time Steps')
    plt.ylabel('NIFTY Index Value')
    plt.legend()
    plt.show()
    

    plt.figure(figsize=(10, 6))
    ST = paths[-1]
    plt.hist(ST, bins=50, alpha=0.6, color='blue', edgecolor='black')
    plt.axvline(K, color='red', linestyle='dashed', linewidth=2, label=f'Strike: {K}')
    plt.axvline(S, color='green', linestyle='dashed', linewidth=2, label=f'Start: {int(S)}')
    plt.title('Distribution of Final NIFTY Prices')
    plt.legend()
    plt.show()