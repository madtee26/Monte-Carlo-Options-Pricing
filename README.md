# Monte-Carlo-Options-Pricing
A high-performance, vectorized Monte Carlo simulation developed in Python to price European Call options on the NIFTY 50 index. This project benchmarks numerical simulation results against the analytical Black-Scholes-Merton model, incorporating real-time market data calibration.

1. Theoretical Framework
The Physics: Geometric Brownian Motion (GBM)
Stock prices are modeled as a stochastic process known as Geometric Brownian Motion. We assume that the log-returns of the stock follow a random walk with a consistent trend. The movement is governed by the following Stochastic Differential Equation (SDE):$$dS_t = \mu S_t dt + \sigma S_t dW_t$$Where:$S_t$: The asset price at time $t$.$\mu$ (Drift): The deterministic trend of the asset. In a risk-neutral world, $\mu = r$ (the risk-free rate).$\sigma$ (Volatility): The statistical measure of price dispersion.$dW_t$: A Wiener Process (Brownian Motion) representing market randomness.

The Math: Itô’s Lemma & The Volatility Drag
To solve this SDE for use in a discrete computer simulation, we apply Itô's Lemma to the function $f(S) = \ln(S)$. This transformation is necessary because log-returns are time-additive, making them ideal for simulation.The resulting discrete-time equation used in this model is:$$S_{t+\Delta t} = S_t \exp \left( (r - \frac{1}{2}\sigma^2)\Delta t + \sigma \sqrt{\Delta t} Z \right)$$The term $-\frac{1}{2}\sigma^2$ is known as the volatility drag. It is a mathematical correction required because the expected value of a log-normal distribution is higher than the median; without this adjustment, the Monte Carlo simulation would consistently overestimate the option price.

2. Features & Implementation
Vectorized Engine: The simulation leverages NumPy to generate a $(Steps \times Simulations)$ matrix of random shocks. By utilizing np.cumprod, the model calculates 50,000 parallel price paths simultaneously, bypassing slow Python loops.
Live Market Calibration: Rather than using static volatility, the script utilizes the yfinance API to fetch 1 year of historical NIFTY 50 data and calculates the Annualized Historical Volatility dynamically.Risk-Neutral Valuation: The final option price is calculated by taking the average payoff across all simulated universes and discounting it back to the present value using continuous compounding: $PV = E[\text{Payoff}] \cdot e^{-rt}$.

3.Benchmarking: Monte Carlo vs. Black-ScholesTo validate the simulation, the results are compared against the Black-Scholes-Merton analytical formula:$$C = S_0 N(d_1) - K e^{-rt} N(d_2)$$Where $N(d)$ is the Cumulative Distribution Function (CDF) of the standard normal distribution. In testing, with $N=50,000$ simulations, the numerical approximation typically converges to within 0.5% of the theoretical value.
