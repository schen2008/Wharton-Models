import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import multivariate_normal, wishart, invwishart
import warnings
import yfinance as yf
from datetime import datetime
warnings.filterwarnings('ignore')


def fetch_log_returns(tickers, start_date, end_date, price_field="Adj Close"):
    """
    Download historical prices via yfinance and convert to daily log returns.
    Returns a DataFrame of daily log returns (index = dates, columns = tickers uppercase).
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

    Main features implemented/fixed:
      - Proper Bayesian resampling of (mu, Sigma) using Inverse-Wishart and conditional normal
      - Compute resampled efficient frontier and average weights (Michaud)
      - Traditional MVO frontier for comparison
      - Plotting with Capital Market Line (CML) and moderate risk band
      - Methods to get portfolio weights and summary
      - Method to find best-Sharpe portfolio inside an annual risk band (9%-12% by default)
      - Export helpers (CSV)
    """

    def __init__(self, returns_data, num_simulations=1000, num_portfolios=100,
                 risk_free_rate=0.0, random_state=None):
        self.returns_data = returns_data.copy()
        self.num_simulations = int(num_simulations)
        self.num_portfolios = int(num_portfolios)
        self.num_assets = returns_data.shape[1]
        self.num_observations = len(returns_data)
        self.asset_names = returns_data.columns.tolist()

        self.risk_free_rate = float(risk_free_rate)
        self.rf_daily = (1 + self.risk_free_rate) ** (1 / 252) - 1.0

        self.rng = np.random.default_rng(random_state)

        self.sample_mean_returns = returns_data.mean().values  
        self.sample_cov_matrix = returns_data.cov().values     


        self.base_frontier_weights = None
        self.base_frontier_returns = None
        self.base_frontier_risks = None

        self.resampled_weights = None
        self.resampled_returns = None
        self.resampled_risks = None

        print(f"Number of observations: {self.num_observations}")
        print(f"Number of assets: {self.num_assets}")

    def _normalize_weights(self, weights):
        weights = np.clip(weights, 0.0, None)
        total = np.sum(weights)
        if total <= 0:
            return None
        return weights / total

    def generate_bayesian_monte_carlo_sample(self):
        """
        Sample (mu, Sigma) using inverse-Wishart for Sigma and multivariate normal for mu | Sigma.
        Returns (simulated_mean, simulated_cov)
        """
        T = self.num_observations
        n = self.num_assets

        df = max(T - 1, n + 1) 
        scale = (T - 1) * self.sample_cov_matrix

        try:
            simulated_cov = invwishart.rvs(df=df, scale=scale, random_state=self.rng)
        except Exception:
            precision_sample = wishart.rvs(df=df, scale=np.linalg.pinv(scale), random_state=self.rng)
            simulated_cov = np.linalg.inv(precision_sample)

        simulated_cov = (simulated_cov + simulated_cov.T) / 2.0

        eigvals = np.linalg.eigvalsh(simulated_cov)
        min_eig = np.min(eigvals)
        if min_eig <= 1e-10:
            simulated_cov += np.eye(n) * (1e-8 - min_eig)

        mean_cov = simulated_cov / T
        simulated_mean = multivariate_normal.rvs(mean=self.sample_mean_returns, cov=mean_cov, random_state=self.rng)

        return simulated_mean, simulated_cov

    def portfolio_performance(self, weights, mean_returns, cov_matrix):
        """
        weights: 1D array
        mean_returns: 1D daily mean returns
        cov_matrix: daily covariance matrix
        returns: (portfolio_return_daily_logmean, portfolio_risk_daily_std)
        """
        pr = float(np.dot(weights, mean_returns))
        prisk = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
        return pr, prisk

    def optimize_portfolio_for_target_return(self, target_return, mean_returns, cov_matrix, initial_weights=None):
        """
        Minimize portfolio variance given target return (equality). Boundaries: no shorting, weights in [0,1].
        Returns normalized weights or None if solver fails.
        """
        n = self.num_assets

        def portfolio_variance(w):
            return float(np.dot(w.T, np.dot(cov_matrix, w)))

        if initial_weights is None:
            x0 = np.ones(n) / n
        else:
            x0 = self._normalize_weights(initial_weights)
            if x0 is None:
                x0 = np.ones(n) / n

        bounds = tuple((0.0, 1.0) for _ in range(n))
        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'eq', 'fun': lambda w: float(np.dot(w, mean_returns) - target_return)}
        ]

        try:
            res = minimize(portfolio_variance, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 2000, 'ftol': 1e-9})
            if res.success:
                w = self._normalize_weights(res.x)
                if w is not None:
                    achieved = float(np.dot(w, mean_returns))
                    if np.isclose(achieved, target_return, rtol=1e-4, atol=1e-6):
                        return w
        except Exception:
            pass

        cons_relaxed = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'ineq', 'fun': lambda w: np.dot(w, mean_returns) - target_return}
        ]
        try:
            res = minimize(portfolio_variance, x0, method='SLSQP', bounds=bounds, constraints=cons_relaxed, options={'maxiter': 2000, 'ftol': 1e-9})
            if res.success:
                w = self._normalize_weights(res.x)
                if w is not None:
                    if float(np.dot(w, mean_returns)) >= target_return - 1e-8:
                        return w
        except Exception:
            pass

        return None

    def optimize_minimum_variance_portfolio(self, mean_returns, cov_matrix):
        """
        Minimum variance portfolio (subject to long-only, sum to 1).
        """
        n = self.num_assets

        def portfolio_variance(w):
            return float(np.dot(w.T, np.dot(cov_matrix, w)))

        x0 = np.ones(n) / n
        bounds = tuple((0.0, 1.0) for _ in range(n))
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

        res = minimize(portfolio_variance, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 2000})
        if res.success:
            w = self._normalize_weights(res.x)
            if w is not None:
                return w
        return np.ones(n) / n


    def compute_efficient_frontier_for_sample(self, mean_returns, cov_matrix, target_returns):
        frontier_weights = []
        frontier_returns = []
        frontier_risks = []

        prev_w = None
        for t in target_returns:
            w = self.optimize_portfolio_for_target_return(t, mean_returns, cov_matrix, initial_weights=prev_w)
            if w is None:
                raise ValueError(f"Efficient frontier solver failed for target return {t:.8e}")
            r, risk = self.portfolio_performance(w, mean_returns, cov_matrix)
            frontier_weights.append(w)
            frontier_returns.append(r)
            frontier_risks.append(risk)
            prev_w = w

        return frontier_weights, frontier_returns, frontier_risks

    def run_resampled_optimization(self, verbose=True):
        """
        Execute the Michaud resampled efficiency algorithm.
        After running, resampled_weights, resampled_returns, resampled_risks are populated.
        """
        if verbose:
            print("Running Michaud Resampled Efficiency (Bayesian Approach)")
            print(f"Simulations: {self.num_simulations}, Portfolios per frontier: {self.num_portfolios}")
            print("-" * 60)

        min_var_w = self.optimize_minimum_variance_portfolio(self.sample_mean_returns, self.sample_cov_matrix)
        min_return = float(np.dot(min_var_w, self.sample_mean_returns))
        max_asset_return = float(np.max(self.sample_mean_returns))
        max_return_w = self.optimize_portfolio_for_target_return(max_asset_return, self.sample_mean_returns, self.sample_cov_matrix)
        if max_return_w is None:
            max_return = max_asset_return
        else:
            max_return = float(np.dot(max_return_w, self.sample_mean_returns))

        if np.isclose(max_return, min_return):
            max_return = min_return + 1e-6

        target_returns = np.linspace(min_return, max_return, self.num_portfolios)

        base_w, base_r, base_risk = self.compute_efficient_frontier_for_sample(self.sample_mean_returns, self.sample_cov_matrix, target_returns)
        self.base_frontier_weights = np.array(base_w)
        self.base_frontier_returns = np.array(base_r)
        self.base_frontier_risks = np.array(base_risk)
        self.target_returns = self.base_frontier_returns.copy()

        all_weights = []
        successful = 0

        for sim in range(self.num_simulations):
            if verbose and (sim + 1) % 50 == 0:
                print(f"Completed {sim + 1}/{self.num_simulations} simulations")

            try:
                sample_mean, sample_cov = self.generate_bayesian_monte_carlo_sample()
                w_sim, r_sim, risk_sim = self.compute_efficient_frontier_for_sample(sample_mean, sample_cov, self.target_returns)
                all_weights.append(w_sim)
                successful += 1
            except Exception as e:
                if verbose and sim < 10:
                    print(f"Warning: simulation {sim+1} failed: {str(e)[:200]}")
                continue

        if verbose:
            print(f"Successful simulations: {successful}/{self.num_simulations}")

        if successful == 0:
            raise RuntimeError("All simulations failed. Check input data or reduce num_simulations.")

        all_weights_array = np.array(all_weights, dtype=float)

        self.resampled_weights = np.mean(all_weights_array, axis=0)

        for i in range(self.resampled_weights.shape[0]):
            s = np.sum(self.resampled_weights[i])
            if s > 0:
                self.resampled_weights[i] /= s

        self.resampled_returns = np.zeros(self.num_portfolios, dtype=float)
        self.resampled_risks = np.zeros(self.num_portfolios, dtype=float)
        for i in range(self.num_portfolios):
            r, risk = self.portfolio_performance(self.resampled_weights[i], self.sample_mean_returns, self.sample_cov_matrix)
            self.resampled_returns[i] = r
            self.resampled_risks[i] = risk

        if verbose:
            print("Resampled optimization complete!")
            print("=" * 60)

    def compute_traditional_efficient_frontier(self):
        if getattr(self, "target_returns", None) is None:
            raise ValueError("Must run run_resampled_optimization() before calling this method.")
        trad_w = []
        trad_r = []
        trad_risk = []
        for t in self.target_returns:
            w = self.optimize_portfolio_for_target_return(t, self.sample_mean_returns, self.sample_cov_matrix)
            if w is not None:
                r, risk = self.portfolio_performance(w, self.sample_mean_returns, self.sample_cov_matrix)
                trad_w.append(w)
                trad_r.append(r)
                trad_risk.append(risk)
        return np.array(trad_r), np.array(trad_risk), np.array(trad_w)
    
    def compute_levered_tangent_portfolio(self, target_annual_risk):
        """
        Compute levered portfolio along the CML to reach a target annual risk (decimal, e.g. 0.15 = 15%).
        Returns a dict with weights (including 'CASH' for risk-free), annual_return, annual_risk, sharpe.
        """
        if self.resampled_returns is None:
            raise ValueError("Must run run_resampled_optimization() first!")

        simple_returns = np.exp(self.resampled_returns + 0.5 * (self.resampled_risks ** 2)) - 1.0
        annual_returns = simple_returns * 252.0
        annual_risks = self.resampled_risks * np.sqrt(252.0)
        annual_rf = self.risk_free_rate
        with np.errstate(divide='ignore', invalid='ignore'):
            sharpe = np.where(annual_risks > 0, (annual_returns - annual_rf) / annual_risks, -np.inf)
        tangent_idx = int(np.nanargmax(sharpe))

        tangent_w = self.resampled_weights[tangent_idx]
        tangent_annual_risk = annual_risks[tangent_idx]
        tangent_annual_return = annual_returns[tangent_idx]

        if tangent_annual_risk <= 0:
            raise ValueError("Tangent portfolio has non-positive risk; cannot lever.")
        scale = float(target_annual_risk) / float(tangent_annual_risk)

        levered_weights = tangent_w * scale
        weight_rf = 1.0 - np.sum(levered_weights)

        levered_annual_return = weight_rf * annual_rf + scale * tangent_annual_return
        levered_annual_risk = abs(scale) * tangent_annual_risk
        levered_sharpe = (levered_annual_return - annual_rf) / levered_annual_risk if levered_annual_risk > 0 else np.nan

        weights_df = pd.DataFrame({
            'Asset': self.asset_names + ['CASH_RISK_FREE'],
            'Weight': np.concatenate([levered_weights, np.array([weight_rf])]),
            'Weight %': np.concatenate([levered_weights, np.array([weight_rf])]) * 100.0
        }).sort_values('Weight', ascending=False).reset_index(drop=True)

        return {
            'weights_df': weights_df,
            'annual_return': levered_annual_return,
            'annual_risk': levered_annual_risk,
            'sharpe': levered_sharpe,
            'scale': scale,
            'tangent_idx': tangent_idx
        }

    def plot_efficient_frontiers(self, show_traditional=True, show_cml=True, moderate_risk_band=(0.09, 0.12), figsize=(14, 9)):
        if self.resampled_returns is None:
            raise ValueError("Must run run_resampled_optimization() first!")

        plt.figure(figsize=figsize)

        annual_res_risk = self.resampled_risks * np.sqrt(252)  
        simple_returns = np.exp(self.resampled_returns + 0.5 * self.resampled_risks ** 2) - 1
        annual_res_return = simple_returns * 252

        plt.plot(annual_res_risk * 100, annual_res_return * 100, 'b-', linewidth=3, label='Resampled Efficient Frontier (Michaud)', zorder=3)
        plt.scatter(annual_res_risk * 100, annual_res_return * 100, c='blue', s=25, alpha=0.6, zorder=3)
        
        if show_traditional:
            trad_r, trad_risk, _ = self.compute_traditional_efficient_frontier()
            if len(trad_r) > 0:
                trad_annual_risk = trad_risk * np.sqrt(252)
                trad_simple_ret = np.exp(trad_r + 0.5 * trad_risk ** 2) - 1
                trad_annual_ret = trad_simple_ret * 252
                plt.plot(trad_annual_risk * 100, trad_annual_ret * 100, 'r--', linewidth=2.2, label='Traditional MVO Frontier', alpha=0.8, zorder=2)
                plt.scatter(trad_annual_risk * 100, trad_annual_ret * 100, c='red', s=20, alpha=0.4, zorder=2)

        asset_annual_ret = self.sample_mean_returns * 252
        asset_annual_risk = np.sqrt(np.diag(self.sample_cov_matrix)) * np.sqrt(252)
        plt.scatter(asset_annual_risk * 100, asset_annual_ret * 100, c='green', s=120, alpha=0.8, marker='D', label='Assets', zorder=4)
        for i, name in enumerate(self.asset_names):
            plt.annotate(name, (asset_annual_risk[i] * 100, asset_annual_ret[i] * 100), xytext=(6, 6), textcoords='offset points',
                         fontsize=9, bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.25))

        min_band, max_band = moderate_risk_band
        plt.axvspan(min_band * 100, max_band * 100, color='orange', alpha=0.12, label=f"Moderate risk band ({min_band*100:.0f}%–{max_band*100:.0f}%)")

        cml_point = None
        if show_cml:
            annual_rf = self.risk_free_rate
            with np.errstate(divide='ignore', invalid='ignore'):
                sharpe_array = (annual_res_return - annual_rf) / annual_res_risk
            valid = np.isfinite(sharpe_array)
            if np.any(valid):
                tangent_idx = int(np.nanargmax(np.where(valid, sharpe_array, -np.inf)))
                risk_t = annual_res_risk[tangent_idx]
                ret_t = annual_res_return[tangent_idx]
                slope = (ret_t - annual_rf) / risk_t if risk_t > 0 else 0.0
                max_x = max(np.nanmax(annual_res_risk) * 1.1, max_band)
                xvals = np.linspace(0.0, max_x, 200)
                cml_vals = annual_rf + slope * xvals
                plt.plot(xvals * 100, cml_vals * 100, linestyle=':', linewidth=2.2, color='black', label='CML (resampled tangent)')
                plt.scatter(risk_t * 100, ret_t * 100, marker='X', s=110, c='black', zorder=6, label='Tangent (max Sharpe)')
                cml_point = {'tangent_idx': tangent_idx, 'risk': risk_t, 'return': ret_t, 'slope': slope}

        best_mod = self.get_best_sharpe_within_risk_band(min_annual_risk=min_band, max_annual_risk=max_band)
        if best_mod is not None:

            plt.scatter(best_mod['annual_risk'] * 100, best_mod['annual_return'] * 100, marker='*', s=250, c='magenta', zorder=7, label=f"Best Sharpe in {min_band*100:.0f}%–{max_band*100:.0f}% band")
            plt.annotate('Best (9%–12%)', (best_mod['annual_risk'] * 100, best_mod['annual_return'] * 100), xytext=(8, -18), textcoords='offset points', fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        else:
            if cml_point is not None:
                mid_band = 0.5 * (min_band + max_band)
                levered = self.compute_levered_tangent_portfolio(mid_band)
                plt.scatter(levered['annual_risk'] * 100, levered['annual_return'] * 100, marker='*', s=250, c='magenta', zorder=7, label=f"CML point in {min_band*100:.0f}%–{max_band*100:.0f}% band (levered)")
                plt.annotate('CML Best (levered)', (levered['annual_risk'] * 100, levered['annual_return'] * 100), xytext=(8, -18), textcoords='offset points', fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

                print("No feasible long-only portfolio inside the requested risk band.Using the Capital Market Line (levered tangent portfolio) to produce a portfolio inside the band:")
                print("Levered CML portfolio (includes CASH_RISK_FREE which may be negative when leveraging):")
                print(levered['weights_df'].to_string(index=False))
                print(f"Annual Return: {levered['annual_return']*100:.4f}%")
                print(f"Annual Risk:   {levered['annual_risk']*100:.4f}%")
                print(f"Sharpe:        {levered['sharpe']:.6f}")

        plt.xlabel('Risk (Annualized Std Dev) %', fontsize=13, fontweight='bold')
        plt.ylabel('Expected Return (Annualized) %', fontsize=13, fontweight='bold')
        plt.title('Resampled Efficient Frontier (Michaud) — with CML and Moderate-risk Band', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, loc='best', framealpha=0.9)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def get_portfolio_weights(self, target_risk=None, target_return=None, annualized=False):
        """
        Returns (weights_df, summary) for the portfolio nearest to target_risk OR target_return.
        - target_risk expects same units as self.resampled_risks (daily std) unless annualized=True.
        - target_return expects same units as self.resampled_returns (daily log-mean) unless annualized=True.
        """
        if self.resampled_weights is None:
            raise ValueError("Must run run_resampled_optimization() first!")

        if target_risk is not None and annualized:
            target_risk_daily = float(target_risk) / np.sqrt(252)
            idx = int(np.argmin(np.abs(self.resampled_risks - target_risk_daily)))
        elif target_risk is not None:
            idx = int(np.argmin(np.abs(self.resampled_risks - float(target_risk))))
        elif target_return is not None and annualized:
            daily_log = np.log(1.0 + float(target_return)) / 252.0
            idx = int(np.argmin(np.abs(self.resampled_returns - daily_log)))
        elif target_return is not None:
            idx = int(np.argmin(np.abs(self.resampled_returns - float(target_return))))
        else:
            raise ValueError("Specify target_risk or target_return")

        weights = self.resampled_weights[idx, :]
        port_ret = self.resampled_returns[idx]
        port_risk = self.resampled_risks[idx]

        weights_df = pd.DataFrame({
            'Asset': self.asset_names,
            'Weight': weights,
            'Weight %': weights * 100.0
        }).sort_values('Weight', ascending=False).reset_index(drop=True)

        simple_ret = np.exp(port_ret + 0.5 * (port_risk ** 2)) - 1.0  
        annual_return = simple_ret * 252.0
        annual_risk = port_risk * np.sqrt(252.0)
        annual_excess = annual_return - self.risk_free_rate
        annual_sharpe = annual_excess / annual_risk if annual_risk > 0 else np.nan

        summary = {
            'Selected Index': idx,
            'Annual Return (simple)': annual_return,
            'Annual Risk (std)': annual_risk,
            'Annual Sharpe': annual_sharpe
        }

        print("" + "=" * 60)
        print("PORTFOLIO CHARACTERISTICS")
        print("=" * 60)
        print(f"Annual Return (simple): {annual_return*100:.4f}%")
        print(f"Annual Risk (std dev):   {annual_risk*100:.4f}%")
        print(f"Annual Sharpe:           {annual_sharpe:.4f}")
        print("Asset Allocation:")
        print(weights_df.to_string(index=False))
        print("=" * 60 + "")

        return weights_df, summary

    def get_summary_statistics(self):
        if self.resampled_returns is None:
            raise ValueError("Must run run_resampled_optimization() first!")

        min_idx = int(np.argmin(self.resampled_risks))
        max_idx = int(np.argmax(self.resampled_returns))
        simple_returns = np.exp(self.resampled_returns + 0.5 * (self.resampled_risks ** 2)) - 1.0
        annual_returns = simple_returns * 252.0
        annual_risks = self.resampled_risks * np.sqrt(252.0)

        daily_sharpes = (simple_returns - self.rf_daily) / np.where(self.resampled_risks > 0, self.resampled_risks, np.nan)
        sharpe_ratios = daily_sharpes * np.sqrt(252.0)
        max_sharpe_idx = int(np.nanargmax(sharpe_ratios))

        data = {
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
                f"{annual_returns[min_idx]*100:.4f}%",
                f"{annual_risks[min_idx]*100:.4f}%",
                f"{sharpe_ratios[min_idx]:.4f}",
                '',
                f"{annual_returns[max_idx]*100:.4f}%",
                f"{annual_risks[max_idx]*100:.4f}%",
                f"{sharpe_ratios[max_idx]:.4f}",
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

        df = pd.DataFrame(data)
        print("" + "=" * 60)
        print("MICHAUD OPTIMIZATION SUMMARY STATISTICS")
        print("=" * 60)
        print(df.to_string(index=False))
        print("=" * 60 + "")
        return df

    def get_best_sharpe_within_risk_band(self, min_annual_risk=0.09, max_annual_risk=0.12):
        if self.resampled_returns is None:
            raise ValueError("Must run run_resampled_optimization() first!")

        simple_returns = np.exp(self.resampled_returns + 0.5 * (self.resampled_risks ** 2)) - 1.0
        annual_returns = simple_returns * 252.0
        annual_risks = self.resampled_risks * np.sqrt(252.0)
        annual_rf = self.risk_free_rate

        with np.errstate(divide='ignore', invalid='ignore'):
            sharpe = np.where(annual_risks > 0, (annual_returns - annual_rf) / annual_risks, -np.inf)

        idxs = np.where((annual_risks >= min_annual_risk) & (annual_risks <= max_annual_risk))[0]
        if idxs.size == 0:
            print(f"No portfolios found in annual risk band {min_annual_risk*100:.1f}%–{max_annual_risk*100:.1f}%")
            return None

        rel = np.nanargmax(sharpe[idxs])
        best_idx = int(idxs[rel])

        best_weights = self.resampled_weights[best_idx]
        best_annual_return = annual_returns[best_idx]
        best_annual_risk = annual_risks[best_idx]
        best_sharpe = sharpe[best_idx]

        weights_df = pd.DataFrame({
            'Asset': self.asset_names,
            'Weight': best_weights,
            'Weight %': best_weights * 100.0
        }).sort_values('Weight', ascending=False).reset_index(drop=True)

        print("" + "=" * 60)
        print(f"BEST SHARPE PORTFOLIO WITHIN {min_annual_risk*100:.1f}%–{max_annual_risk*100:.1f}% ANNUAL RISK")
        print("=" * 60)
        print(f"Annual Return: {best_annual_return*100:.4f}%")
        print(f"Annual Risk:   {best_annual_risk*100:.4f}%")
        print(f"Sharpe:        {best_sharpe:.4f}")
        print("Allocation:")
        print(weights_df.to_string(index=False))
        print("=" * 60 + "")

        return {
            'index': best_idx,
            'weights_df': weights_df,
            'annual_return': best_annual_return,
            'annual_risk': best_annual_risk,
            'sharpe': best_sharpe
        }

    def export_resampled_frontier(self, path="resampled_frontier.csv"):
        if self.resampled_weights is None:
            raise ValueError("Must run run_resampled_optimization() first!")
        df = pd.DataFrame(self.resampled_weights, columns=self.asset_names)
        df['daily_log_return'] = self.resampled_returns
        df['daily_risk'] = self.resampled_risks
        simple_returns = np.exp(self.resampled_returns + 0.5 * (self.resampled_risks ** 2)) - 1.0
        df['annual_return_simple'] = simple_returns * 252.0
        df['annual_risk_std'] = self.resampled_risks * np.sqrt(252.0)
        df.to_csv(path, index=False)
        return path

    def export_portfolio(self, index, path="portfolio_weights.csv"):
        if self.resampled_weights is None:
            raise ValueError("Must run run_resampled_optimization() first!")
        if index < 0 or index >= self.resampled_weights.shape[0]:
            raise IndexError("index out of range")
        df = pd.DataFrame({
            'Asset': self.asset_names,
            'Weight': self.resampled_weights[index],
            'Weight %': self.resampled_weights[index] * 100.0
        }).sort_values('Weight', ascending=False).reset_index(drop=True)
        df.to_csv(path, index=False)
        return path


if __name__ == "__main__":
    preferred_tickers = ["ARGX", "COCO", "GTX", "BZ", "INVA", "ERIC", "GBDC", "KRYS", "CALM", "CART", "ARLP", "GLPI", "IEF", "TIP", "TLT"]
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    start_date = (pd.Timestamp.today() - pd.DateOffset(years=5)).strftime("%Y-%m-%d")

    try:
        returns_df = fetch_log_returns(preferred_tickers, start_date, end_date)
        print("=" * 60)
        print("YFINANCE DAILY LOG RETURNS (auto-adjusted)")
        print("=" * 60)
        print(returns_df.head(6))
        print(f"Data shape: {returns_df.shape}")
        print(f"Date range: {returns_df.index[0]} to {returns_df.index[-1]}")
        data_source = "yfinance"
    except Exception as exc:
        print("=" * 60)
        print("WARNING: Failed to download live prices. Falling back to simulated data.")
        print(f"Reason: {exc}")
        print("=" * 60)
        np.random.seed(42)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        num_days = len(dates)
        mu = np.full(15, 0.00045)
        vol = np.linspace(0.015, 0.035, 15)
        corr = 0.3 + 0.4 * np.eye(15)  
        cov = np.outer(vol, vol) * 0.2 + np.diag(vol**2) * 0.8
        returns = np.random.multivariate_normal(mu, cov, size=num_days)
        returns_df = pd.DataFrame(returns, index=dates, columns=preferred_tickers)
        data_source = "simulated"

    print("" + "=" * 60)
    print(f"INITIALIZING MICHAUD OPTIMIZATION MODEL ({data_source.upper()} DATA)")
    print("=" * 60)

    m = MichaudOptimization(
        returns_data=returns_df,
        num_simulations=500,   
        num_portfolios=120,
        risk_free_rate=0.0125,
        random_state=42
    )

    m.run_resampled_optimization(verbose=True)
    m.get_summary_statistics()

    m.plot_efficient_frontiers(show_traditional=True, show_cml=True, moderate_risk_band=(0.09, 0.12))

    simple_returns = np.exp(m.resampled_returns + 0.5 * (m.resampled_risks**2)) - 1.0
    annual_returns = simple_returns * 252.0
    annual_risks = m.resampled_risks * np.sqrt(252.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        sharpe_arr = np.where(annual_risks > 0, (annual_returns - m.risk_free_rate) / annual_risks, -np.inf)
    max_sharpe_idx = int(np.nanargmax(sharpe_arr))

    print("" + "=" * 60)
    print("OPTIMIZED PORTFOLIO: Maximum Sharpe Ratio Portfolio (unconstrained)")
    print("=" * 60)
    m.get_portfolio_weights(target_risk=None, target_return=m.resampled_returns[max_sharpe_idx])

    min_risk_idx = int(np.argmin(m.resampled_risks))
    print("" + "=" * 60)
    print("MINIMUM RISK PORTFOLIO")
    print("=" * 60)
    m.get_portfolio_weights(target_risk=m.resampled_risks[min_risk_idx])

    best_mod = m.get_best_sharpe_within_risk_band(min_annual_risk=0.09, max_annual_risk=0.12)
    if best_mod is None:
        print("No feasible moderate-risk portfolio found inside the band.")
        levered = m.compute_levered_tangent_portfolio(0.5*(0.09+0.12))
        path = "best_moderate_portfolio_levered.csv"
        levered['weights_df'].to_csv(path, index=False)
        print(f"Computed levered CML portfolio and exported to {path}")
    else:
        path = m.export_portfolio(best_mod['index'], path="best_moderate_portfolio.csv")
        print(f"Exported best moderate-risk portfolio to {path}")
