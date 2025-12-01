import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import multivariate_normal, wishart, invwishart
import warnings
import yfinance as yf
warnings.filterwarnings('ignore')

def fetch_log_returns(tickers, start_date, end_date, price_field="Adj Close"):
    """
    Download historical prices via yfinance and convert to daily log returns.
    """
    tickers = [ticker.upper() for ticker in tickers]
    price_data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        progress=False
    )
    
    if price_data.empty:
        raise ValueError("No price data returned from yfinance. Check tickers or date range.")
    
    if isinstance(price_data.columns, pd.MultiIndex):
        if price_field in price_data.columns.get_level_values(0):
            prices = price_data[price_field]
        else:
            prices = price_data.xs("Close", level=0, axis=1)
    else:
        prices = price_data
    
    prices = prices.ffill().dropna(how="all")
    log_returns = np.log(prices / prices.shift(1)).dropna()
    log_returns = log_returns.dropna(axis=1, how="all")
    log_returns.columns = [str(col).upper() for col in log_returns.columns]
    
    missing_cols = [ticker for ticker in tickers if ticker not in log_returns.columns]
    if missing_cols:
        raise ValueError(f"Missing return columns for tickers: {missing_cols}")
    
    log_returns = log_returns.loc[:, tickers]
    log_returns.index.name = "Date"
    return log_returns


class MichaudOptimization:
    """
    Michaud Resampled Efficiency Optimization Model (Bayesian/Statistical Approach)
    
    This model uses proper Bayesian sampling to account for estimation error in both
    the mean vector and covariance matrix. This is the theoretically correct approach
    for Michaud's Resampled Efficiency.
    
    The key insight: We don't know the true mean and covariance - we only have estimates.
    This model properly accounts for that uncertainty.
    """
    
    def __init__(self, returns_data, num_simulations=500, num_portfolios=100,
                 risk_free_rate=0.0, random_state=None):
        """
        Initialize the Michaud Optimization Model
        
        Parameters:
        -----------
        returns_data : pandas DataFrame
            Historical returns data with stocks as columns and dates as rows
        num_simulations : int
            Number of Monte Carlo simulations to run (default: 500)
        num_portfolios : int
            Number of portfolios along the efficient frontier (default: 100)
        risk_free_rate : float
            Risk-free rate expressed in the same units as returns (default: 0.0)
        random_state : int or numpy.random.Generator
            Seed or generator for reproducibility (default: None)
        """
        self.returns_data = returns_data
        self.num_simulations = num_simulations
        self.num_portfolios = num_portfolios
        self.num_assets = returns_data.shape[1]
        self.num_observations = len(returns_data)
        self.asset_names = returns_data.columns.tolist()
        # Convert annual rf → daily rf for daily returns
        self.rf_daily = (1 + risk_free_rate)**(1/252) - 1
        self.risk_free_rate = risk_free_rate  # Keep original for backward compatibility if needed
        self.rng = np.random.default_rng(random_state)
        
        # Calculate sample statistics from historical data
        self.sample_mean_returns = returns_data.mean().values
        self.sample_cov_matrix = returns_data.cov().values
        
        # These are our point estimates, but we acknowledge they have uncertainty
        print(f"Number of observations: {self.num_observations}")
        print(f"Number of assets: {self.num_assets}")
        
        # Storage for results
        self.resampled_weights = None
        self.resampled_returns = None
        self.resampled_risks = None
        self.target_returns = None
        self.base_frontier_weights = None
        self.base_frontier_returns = None
        self.base_frontier_risks = None

    def _normalize_weights(self, weights):
        """
        Project numerical optimizer output back onto the feasible simplex.
        """
        weights = np.clip(weights, 0, None)
        total = np.sum(weights)
        if total <= 0:
            return None
        return weights / total
        
    def generate_bayesian_monte_carlo_sample(self):
        """
        Generate a Monte Carlo sample using proper Bayesian approach
        
        This method accounts for estimation error in BOTH the mean vector and 
        covariance matrix, which is the theoretically correct Michaud approach.
        
        Statistical theory:
        - The true covariance matrix Σ follows an Inverse-Wishart distribution
        - Given a simulated Σ, the true mean μ follows a multivariate normal
        
        Returns:
        --------
        simulated_mean : numpy array
            Simulated mean returns vector
        simulated_cov : numpy array
            Simulated covariance matrix
        """
        T = self.num_observations  # Number of time periods
        n = self.num_assets  # Number of assets
        
        # Step 1: Sample the covariance matrix from Inverse-Wishart distribution
        # The sample covariance S follows: (T-1)*S ~ Wishart(T-1, Σ)
        # Therefore: Σ ~ Inverse-Wishart(T-1, (T-1)*S)
        
        degrees_of_freedom = T - 1
        scale_matrix = (T - 1) * self.sample_cov_matrix
        
        # Sample from Inverse-Wishart distribution
        # Note: scipy's invwishart uses df and scale parameterization
        try:
            simulated_cov = invwishart.rvs(
                df=degrees_of_freedom,
                scale=scale_matrix,
                random_state=self.rng
            )
        except Exception:
            # Fall back to sampling a precision matrix and invert so distribution stays consistent
            precision_sample = wishart.rvs(
                df=degrees_of_freedom,
                scale=np.linalg.pinv(scale_matrix),
                random_state=self.rng
            )
            simulated_cov = np.linalg.inv(precision_sample)
        
        # Ensure the covariance matrix is symmetric and positive definite
        simulated_cov = (simulated_cov + simulated_cov.T) / 2
        
        # Add small regularization to ensure positive definiteness if needed
        min_eigenval = np.min(np.linalg.eigvals(simulated_cov))
        if min_eigenval < 1e-8:
            simulated_cov += np.eye(n) * (1e-8 - min_eigenval)
        
        # Step 2: Sample the mean vector conditional on the covariance matrix
        # The sampling distribution of the mean is: μ̂ ~ N(μ, Σ/T)
        # Therefore, we sample: μ ~ N(μ̂, Σ_simulated/T)
        
        mean_covariance = simulated_cov / T
        simulated_mean = multivariate_normal.rvs(
            mean=self.sample_mean_returns,
            cov=mean_covariance,
            random_state=self.rng
        )
        
        return simulated_mean, simulated_cov
    
    def portfolio_performance(self, weights, mean_returns, cov_matrix):
        """
        Calculate portfolio return and risk
        
        Parameters:
        -----------
        weights : numpy array
            Portfolio weights
        mean_returns : numpy array
            Expected returns
        cov_matrix : numpy array
            Covariance matrix
            
        Returns:
        --------
        portfolio_return : float
            Expected portfolio return
        portfolio_risk : float
            Portfolio standard deviation (risk)
        """
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return portfolio_return, portfolio_risk
    
    def optimize_portfolio_for_target_return(self, target_return, mean_returns, cov_matrix,
                                             initial_weights=None):
        """
        Optimize portfolio to minimize risk for a given target return
        
        Parameters:
        -----------
        target_return : float
            Target expected return
        mean_returns : numpy array
            Expected returns
        cov_matrix : numpy array
            Covariance matrix
            
        Returns:
        --------
        optimal_weights : numpy array
            Optimal portfolio weights, or None if optimization fails
        """
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        if initial_weights is None:
            initial_weights = np.array([1.0 / self.num_assets] * self.num_assets)
        else:
            initial_weights = self._normalize_weights(initial_weights)
            if initial_weights is None:
                initial_weights = np.array([1.0 / self.num_assets] * self.num_assets)
        
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        sum_constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        exact_return_constraint = {'type': 'eq', 'fun': lambda w: np.dot(w, mean_returns) - target_return}
        
        result = minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=[sum_constraint, exact_return_constraint],
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if result.success:
            weights = self._normalize_weights(result.x)
            if weights is not None:
                achieved_return = np.dot(weights, mean_returns)
                if np.isclose(achieved_return, target_return, rtol=1e-4, atol=1e-6):
                    return weights
        
        # Fallback: allow small overshoot on the target return if equality solver struggled
        relaxed_constraint = {'type': 'ineq', 'fun': lambda w: np.dot(w, mean_returns) - target_return}
        result = minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=[sum_constraint, relaxed_constraint],
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if result.success:
            weights = self._normalize_weights(result.x)
            if weights is not None:
                achieved_return = np.dot(weights, mean_returns)
                if achieved_return >= target_return - 1e-6:
                    return weights
        
        return None
    
    def optimize_minimum_variance_portfolio(self, mean_returns, cov_matrix):
        """
        Find the global minimum variance portfolio (no return constraint)
        
        Parameters:
        -----------
        mean_returns : numpy array
            Expected returns (not used in optimization, but needed for return calculation)
        cov_matrix : numpy array
            Covariance matrix
            
        Returns:
        --------
        optimal_weights : numpy array
            Optimal portfolio weights
        """
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
        ]
        
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial_weights = np.array([1.0 / self.num_assets] * self.num_assets)
        
        result = minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            weights = self._normalize_weights(result.x)
            if weights is not None:
                return weights
        
        return np.array([1.0 / self.num_assets] * self.num_assets)
    
    def compute_efficient_frontier_for_sample(self, mean_returns, cov_matrix, target_returns):
        """
        Compute efficient frontier for a single Monte Carlo sample
        
        Parameters:
        -----------
        mean_returns : numpy array
            Expected returns for this sample
        cov_matrix : numpy array
            Covariance matrix for this sample
        target_returns : numpy array
            Array of target returns to optimize for
            
        Returns:
        --------
        frontier_weights : list
            List of optimal weights for each target return
        frontier_returns : list
            Achieved returns for each portfolio
        frontier_risks : list
            Risk (std dev) for each portfolio
        """
        frontier_weights = []
        frontier_returns = []
        frontier_risks = []
        
        previous_weights = None
        
        for target in target_returns:
            weights = self.optimize_portfolio_for_target_return(
                target,
                mean_returns,
                cov_matrix,
                initial_weights=previous_weights
            )
            
            if weights is None:
                raise ValueError(
                    f"Efficient frontier solver failed for target return={target:.8f}"
                )
            
            ret, risk = self.portfolio_performance(weights, mean_returns, cov_matrix)
            frontier_weights.append(weights)
            frontier_returns.append(ret)
            frontier_risks.append(risk)
            previous_weights = weights
        
        return frontier_weights, frontier_returns, frontier_risks
    
    def run_resampled_optimization(self):
        """
        Run the full Michaud Resampled Efficiency optimization
        
        This is the main method that executes the Bayesian Monte Carlo simulation
        and creates the resampled efficient frontier.
        
        The key steps:
        1. Generate Bayesian samples of (μ, Σ) that reflect estimation uncertainty
        2. For each sample, compute an efficient frontier
        3. Average the portfolio weights across all frontiers (Michaud's key insight)
        4. Evaluate the averaged portfolios using the original sample estimates
        """
        print(f"\nRunning Michaud Resampled Efficiency (Bayesian Approach)")
        print(f"Simulations: {self.num_simulations}")
        print(f"Portfolios per frontier: {self.num_portfolios}")
        print("-" * 60)
        
        # Determine a feasible range of target returns
        min_var_weights = self.optimize_minimum_variance_portfolio(
            self.sample_mean_returns,
            self.sample_cov_matrix
        )
        min_return = np.dot(min_var_weights, self.sample_mean_returns)
        
        max_asset_return = np.max(self.sample_mean_returns)
        max_return_weights = self.optimize_portfolio_for_target_return(
            max_asset_return,
            self.sample_mean_returns,
            self.sample_cov_matrix
        )
        if max_return_weights is None:
            max_return = max_asset_return
        else:
            max_return = np.dot(max_return_weights, self.sample_mean_returns)
        
        if np.isclose(max_return, min_return):
            max_return = min_return + 1e-4
        
        # Create array of target returns within the feasible band
        raw_target_returns = np.linspace(min_return, max_return, self.num_portfolios)
        
        # Compute the sample efficient frontier to anchor Michaud's resampling grid
        base_weights, base_returns, base_risks = self.compute_efficient_frontier_for_sample(
            self.sample_mean_returns,
            self.sample_cov_matrix,
            raw_target_returns
        )
        self.base_frontier_weights = np.array(base_weights)
        self.base_frontier_returns = np.array(base_returns)
        self.base_frontier_risks = np.array(base_risks)
        self.target_returns = self.base_frontier_returns.copy()
        
        # Initialize storage for all simulation results
        all_weights = []
        all_returns = []
        all_risks = []
        
        # Run Bayesian Monte Carlo simulations
        successful_simulations = 0
        
        for sim in range(self.num_simulations):
            if (sim + 1) % 50 == 0:
                print(f"  Completed {sim + 1}/{self.num_simulations} simulations")
            
            try:
                # Generate a Bayesian Monte Carlo sample
                # This accounts for estimation error in both mean and covariance
                sample_mean, sample_cov = self.generate_bayesian_monte_carlo_sample()
                
                # Compute efficient frontier for this sample
                weights, returns, risks = self.compute_efficient_frontier_for_sample(
                    sample_mean, sample_cov, self.target_returns
                )
                
                # Store results
                all_weights.append(weights)
                all_returns.append(returns)
                all_risks.append(risks)
                successful_simulations += 1
                
            except Exception as e:
                # If a simulation fails, skip it and continue
                # This can happen with numerical issues in rare cases
                if sim < 10:  # Only print first few errors
                    print(f"    Warning: Simulation {sim+1} failed: {str(e)[:50]}")
                continue
        
        print(f"\nSuccessful simulations: {successful_simulations}/{self.num_simulations}")
        
        if successful_simulations == 0:
            raise ValueError("All simulations failed. Check your input data.")
        
        # Convert lists to arrays for easier manipulation
        all_weights_array = np.array(all_weights)  # Shape: (num_sims, num_portfolios, num_assets)
        
        # MICHAUD'S KEY STEP: Average the weights across all simulations
        # This creates portfolios that are robust to estimation error
        self.resampled_weights = np.mean(all_weights_array, axis=0)
        
        # Normalize weights to ensure they sum to exactly 1
        for i in range(self.num_portfolios):
            weight_sum = np.sum(self.resampled_weights[i, :])
            if weight_sum > 0:
                self.resampled_weights[i, :] = self.resampled_weights[i, :] / weight_sum
        
        # Calculate resampled frontier performance using SAMPLE statistics
        # Important: we use the averaged weights with our original sample mean/cov estimates
        self.resampled_returns = np.zeros(self.num_portfolios)
        self.resampled_risks = np.zeros(self.num_portfolios)
        
        for i in range(self.num_portfolios):
            ret, risk = self.portfolio_performance(
                self.resampled_weights[i, :],
                self.sample_mean_returns,
                self.sample_cov_matrix
            )
            self.resampled_returns[i] = ret
            self.resampled_risks[i] = risk
        
        print("\n" + "="*60)
        print("Resampled optimization complete!")
        print("="*60)
    
    def compute_traditional_efficient_frontier(self):
        """
        Compute traditional mean-variance efficient frontier for comparison
        
        This uses only the sample estimates without accounting for estimation error.
        
        Returns:
        --------
        traditional_returns : numpy array
            Returns along traditional efficient frontier
        traditional_risks : numpy array
            Risks along traditional efficient frontier
        traditional_weights : numpy array
            Weights along traditional efficient frontier
        """
        if self.target_returns is None:
            raise ValueError("Must run run_resampled_optimization() before calling this method.")
        
        traditional_weights = []
        traditional_returns = []
        traditional_risks = []
        
        for target in self.target_returns:
            weights = self.optimize_portfolio_for_target_return(
                target, self.sample_mean_returns, self.sample_cov_matrix
            )
            
            if weights is not None:
                ret, risk = self.portfolio_performance(
                    weights, self.sample_mean_returns, self.sample_cov_matrix
                )
                traditional_weights.append(weights)
                traditional_returns.append(ret)
                traditional_risks.append(risk)
        
        return (np.array(traditional_returns), 
                np.array(traditional_risks), 
                np.array(traditional_weights))
    
    def plot_efficient_frontiers(self, show_traditional=True, figsize=(14, 9)):
        """
        Plot the resampled efficient frontier and optionally the traditional one
        
        Parameters:
        -----------
        show_traditional : bool
            Whether to also plot the traditional efficient frontier for comparison
        figsize : tuple
            Figure size (width, height)
        """
        if self.resampled_returns is None:
            raise ValueError("Must run run_resampled_optimization() first!")
        
        plt.figure(figsize=figsize)
        
        # Plot resampled efficient frontier
        plt.plot(self.resampled_risks * 100, self.resampled_returns * 100, 
                 'b-', linewidth=3, label='Resampled Efficient Frontier (Michaud)', zorder=3)
        plt.scatter(self.resampled_risks * 100, self.resampled_returns * 100,
                   c='blue', s=30, alpha=0.6, zorder=3)
        
        # Plot traditional efficient frontier if requested
        if show_traditional:
            trad_ret, trad_risk, _ = self.compute_traditional_efficient_frontier()
            plt.plot(trad_risk * 100, trad_ret * 100, 
                    'r--', linewidth=2.5, label='Traditional Efficient Frontier (MVO)', 
                    alpha=0.7, zorder=2)
            plt.scatter(trad_risk * 100, trad_ret * 100,
                       c='red', s=25, alpha=0.4, zorder=2)
        
        # Plot individual assets
        asset_returns = self.sample_mean_returns * 100
        asset_risks = np.sqrt(np.diag(self.sample_cov_matrix)) * 100
        plt.scatter(asset_risks, asset_returns, 
                   c='green', s=150, alpha=0.7, marker='D', 
                   label='Individual Assets', zorder=4, edgecolors='darkgreen', linewidths=2)
        
        # Add asset labels
        for i, name in enumerate(self.asset_names):
            plt.annotate(name, (asset_risks[i], asset_returns[i]),
                        xytext=(8, 8), textcoords='offset points', 
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
        
        plt.xlabel('Risk (Standard Deviation) %', fontsize=13, fontweight='bold')
        plt.ylabel('Expected Return %', fontsize=13, fontweight='bold')
        plt.title('Michaud Resampled Efficient Frontier\n(Bayesian Approach with Estimation Error)', 
                 fontsize=15, fontweight='bold', pad=20)
        plt.legend(fontsize=11, loc='best', framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.show()
    
    def get_portfolio_weights(self, target_risk=None, target_return=None):
        """
        Get portfolio weights for a specific risk or return target
        
        Parameters:
        -----------
        target_risk : float
            Target risk level (standard deviation)
        target_return : float
            Target return level
            
        Returns:
        --------
        weights_df : pandas DataFrame
            DataFrame with asset names and their weights
        """
        if self.resampled_weights is None:
            raise ValueError("Must run run_resampled_optimization() first!")
        
        if target_risk is not None:
            # Find portfolio closest to target risk
            idx = np.argmin(np.abs(self.resampled_risks - target_risk))
        elif target_return is not None:
            # Find portfolio closest to target return
            idx = np.argmin(np.abs(self.resampled_returns - target_return))
        else:
            raise ValueError("Must specify either target_risk or target_return")
        
        weights = self.resampled_weights[idx, :]
        portfolio_return = self.resampled_returns[idx]
        portfolio_risk = self.resampled_risks[idx]
        
        weights_df = pd.DataFrame({
            'Asset': self.asset_names,
            'Weight': weights,
            'Weight %': weights * 100
        })
        
        # Sort by weight descending for display
        weights_df = weights_df.sort_values('Weight', ascending=False)
        
        print(f"\n{'='*60}")
        print(f"Portfolio Characteristics:")
        print(f"{'='*60}")
        # 1) Convert mean log return → simple return
        # For log-normal returns: E[R] = exp(E[log(R)] + 0.5*Var[log(R)])
        # portfolio_return = E[log(R)], portfolio_risk² = Var[log(R)]
        simple_return = np.exp(portfolio_return + 0.5 * portfolio_risk**2) - 1
        
        # 2) Use daily rf
        excess_return = simple_return - self.rf_daily
        
        # 3) Compute daily Sharpe
        daily_sharpe = excess_return / portfolio_risk if portfolio_risk > 0 else np.nan
        
        # 4) Annualize Sharpe for reporting
        annual_sharpe = daily_sharpe * np.sqrt(252)
        annual_return = simple_return * 252
        annual_risk = portfolio_risk * np.sqrt(252)
        
        print(f"Expected Return (annualized): {annual_return*100:.4f}%")
        print(f"Risk (annualized std dev): {annual_risk*100:.4f}%")
        print(f"Sharpe Ratio (annualized): {annual_sharpe:.4f}")
        print(f"\nAsset Allocation (all assets):")
        print(weights_df.to_string(index=False))
        print(f"{'='*60}\n")
        
        return weights_df
    
    def get_summary_statistics(self):
        """
        Get summary statistics of the resampled efficient frontier
        
        Returns:
        --------
        summary : pandas DataFrame
            Summary statistics including min/max risk and return
        """
        if self.resampled_returns is None:
            raise ValueError("Must run run_resampled_optimization() first!")
        
        # Find minimum variance portfolio
        min_risk_idx = np.argmin(self.resampled_risks)
        
        # Find maximum return portfolio
        max_return_idx = np.argmax(self.resampled_returns)
        
        # Find maximum Sharpe ratio portfolio using specified risk-free rate
        # Convert log → simple return
        # For log-normal returns: E[R] = exp(E[log(R)] + 0.5*Var[log(R)])
        simple_returns = np.exp(self.resampled_returns + 0.5 * self.resampled_risks**2) - 1
        
        # Compute daily Sharpe
        daily_sharpes = (simple_returns - self.rf_daily) / self.resampled_risks
        daily_sharpes = np.where(self.resampled_risks > 0, daily_sharpes, np.nan)
        
        # Annualize Sharpe
        sharpe_ratios = daily_sharpes * np.sqrt(252)
        
        # Annualize return/risk
        annual_returns = simple_returns * 252
        annual_risks = self.resampled_risks * np.sqrt(252)
        
        max_sharpe_idx = np.argmax(sharpe_ratios)
        
        summary_data = {
            'Metric': [
                'Minimum Risk Portfolio - Return',
                'Minimum Risk Portfolio - Risk',
                'Minimum Risk Portfolio - Sharpe',
                '',
                'Maximum Return Portfolio - Return',
                'Maximum Return Portfolio - Risk',
                'Maximum Return Portfolio - Sharpe',
                '',
                'Maximum Sharpe Portfolio - Return',
                'Maximum Sharpe Portfolio - Risk',
                'Maximum Sharpe Portfolio - Sharpe',
                '',
                'Number of Portfolios',
                'Number of Simulations',
                'Number of Assets'
            ],
            'Value': [
                f"{annual_returns[min_risk_idx]*100:.4f}%",
                f"{annual_risks[min_risk_idx]*100:.4f}%",
                f"{sharpe_ratios[min_risk_idx]:.4f}",
                '',
                f"{annual_returns[max_return_idx]*100:.4f}%",
                f"{annual_risks[max_return_idx]*100:.4f}%",
                f"{sharpe_ratios[max_return_idx]:.4f}",
                '',
                f"{annual_returns[max_sharpe_idx]*100:.4f}%",
                f"{annual_risks[max_sharpe_idx]*100:.4f}%",
                f"{sharpe_ratios[max_sharpe_idx]:.4f}",
                '',
                self.num_portfolios,
                self.num_simulations,
                self.num_assets
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        print("\n" + "="*60)
        print("MICHAUD OPTIMIZATION SUMMARY STATISTICS")
        print("="*60)
        print(summary_df.to_string(index=False))
        print("="*60 + "\n")
        
        return summary_df
    
    def compare_portfolio_stability(self, target_return=None):
        """
        Compare the stability of Michaud vs Traditional portfolio weights
        
        This demonstrates why Michaud's approach is superior: the resampled
        weights are more stable and diversified.
        
        Parameters:
        -----------
        target_return : float
            Target return to compare (if None, uses median return)
        """
        if self.resampled_weights is None:
            raise ValueError("Must run run_resampled_optimization() first!")
        
        if target_return is None:
            target_return = np.median(self.resampled_returns)
        
        # Get Michaud weights
        michaud_idx = np.argmin(np.abs(self.resampled_returns - target_return))
        michaud_weights = self.resampled_weights[michaud_idx, :]
        
        # Get Traditional weights
        trad_weights_opt = self.optimize_portfolio_for_target_return(
            target_return, self.sample_mean_returns, self.sample_cov_matrix
        )
        if trad_weights_opt is None:
            raise ValueError(
                "Target return is infeasible for the traditional frontier under current constraints."
            )
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Asset': self.asset_names,
            'Michaud Weight %': michaud_weights * 100,
            'Traditional Weight %': trad_weights_opt * 100,
            'Difference %': (michaud_weights - trad_weights_opt) * 100
        })
        
        print(f"\n{'='*70}")
        print(f"PORTFOLIO WEIGHT COMPARISON (Target Return: {target_return*100:.4f}%)")
        print(f"{'='*70}")
        print(comparison_df.to_string(index=False))
        
        # Calculate concentration metrics
        michaud_hhi = np.sum(michaud_weights**2)
        trad_hhi = np.sum(trad_weights_opt**2)
        
        print(f"\n{'='*70}")
        print(f"Concentration Metrics (Herfindahl Index):")
        print(f"  Michaud (lower is more diversified):    {michaud_hhi:.4f}")
        print(f"  Traditional (higher is concentrated):    {trad_hhi:.4f}")
        print(f"{'='*70}\n")
        
        return comparison_df


# Example usage
if __name__ == "__main__":
    """
    Example script using live market data for Google, Apple, Amazon, eBay, NVIDIA, and AMD.
    Falls back to simulated data if downloading fails (e.g., offline environment).
    """
    preferred_tickers = ["CENX","BZ","CART","CALM","ARLP","ERIE","FAST","GLPI","ERIC","CHKP"] 
    start_date = "2018-01-01"
    end_date = "2025-11-26"
    
    try:
        returns_df = fetch_log_returns(preferred_tickers, start_date, end_date)
        print("="*60)
        print("YFINANCE DAILY LOG RETURNS (auto-adjusted)")
        print("="*60)
        print(returns_df.head(10))
        print(f"\nData shape: {returns_df.shape}")
        print(f"Date range: {returns_df.index[0]} to {returns_df.index[-1]}")
        data_source = "yfinance"
    except Exception as exc:
        print("="*60)
        print("WARNING: Failed to download live prices. Falling back to simulated data.")
        print(f"Reason: {exc}")
        print("="*60)
        np.random.seed(42)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        num_days = len(dates)
        mean_returns = np.array([0.0005, 0.0007, 0.0004, 0.0006, 0.0003, 0.00055])
        correlation = np.array([
            [1.0, 0.6, 0.3, 0.4, 0.2, 0.5],
            [0.6, 1.0, 0.5, 0.3, 0.4, 0.45],
            [0.3, 0.5, 1.0, 0.6, 0.3, 0.35],
            [0.4, 0.3, 0.6, 1.0, 0.5, 0.25],
            [0.2, 0.4, 0.3, 0.5, 1.0, 0.4],
            [0.5, 0.45, 0.35, 0.25, 0.4, 1.0]
        ])
        volatilities = np.array([0.02, 0.025, 0.018, 0.022, 0.03, 0.028])
        cov_matrix = np.outer(volatilities, volatilities) * correlation
        returns = np.random.multivariate_normal(mean_returns, cov_matrix, num_days)
        returns_df = pd.DataFrame(returns, index=dates, columns=preferred_tickers)
        data_source = "simulated"
    
    print("\n" + "="*60)
    print(f"INITIALIZING MICHAUD OPTIMIZATION MODEL ({data_source.upper()} DATA)")
    print("="*60)
    
    michaud = MichaudOptimization(
        returns_data=returns_df,
        num_simulations=500,    # Number of Bayesian Monte Carlo simulations
        num_portfolios=100,     # Number of portfolios along the frontier
        risk_free_rate=0.00005, # Example risk-free rate (~1.25% annualized)
        random_state=42
    )
    
    michaud.run_resampled_optimization()
    michaud.get_summary_statistics()
    michaud.plot_efficient_frontiers(show_traditional=True)
    
    # Find maximum Sharpe portfolio automatically
    simple_returns = np.exp(michaud.resampled_returns + 0.5 * michaud.resampled_risks**2) - 1
    daily_sharpes = (simple_returns - michaud.rf_daily) / michaud.resampled_risks
    daily_sharpes = np.where(michaud.resampled_risks > 0, daily_sharpes, np.nan)
    sharpe_ratios = daily_sharpes * np.sqrt(252)
    max_sharpe_idx = np.argmax(sharpe_ratios)
    max_sharpe_return = michaud.resampled_returns[max_sharpe_idx]
    
    print("\n" + "="*60)
    print("OPTIMIZED PORTFOLIO: Maximum Sharpe Ratio Portfolio")
    print("="*60)
    michaud.get_portfolio_weights(target_return=max_sharpe_return)
    
    print("\n" + "="*60)
    print("MINIMUM RISK PORTFOLIO")
    print("="*60)
    min_risk = np.min(michaud.resampled_risks)
    michaud.get_portfolio_weights(target_risk=min_risk)
    
    # Use median return for comparison if available
    median_return = np.median(michaud.resampled_returns)
    if not np.isnan(median_return):
        michaud.compare_portfolio_stability(target_return=median_return)
