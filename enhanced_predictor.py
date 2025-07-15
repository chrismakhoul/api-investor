# enhanced_predictor.py  ────────────────────────────────────────────────────
"""
Full EnhancedStockPredictor class.
Only the `fetch_data` method has been modified to:
    • use Alpha‑Vantage if an API key is provided
    • fall back to yfinance (no key, no daily quota)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")


class EnhancedStockPredictor:
    def __init__(self, api_key: str = "", symbol: str = "GOOG"):
        self.api_key = api_key
        self.symbol = symbol.upper()
        self.data = None
        self.features_df = None
        self.transition_matrix = None
        self.labels = ["Bin 1", "Bin 2", "Bin 3", "Bin 4", "Bin 5", "Bin 6"]
        self.scaler = StandardScaler()

    # ─────────────────────────────── NEW fetch_data ────────────────────────
    def fetch_data(self) -> bool:
        """
        Get daily close prices.

        1) Try Alpha‑Vantage *once* if `self.api_key` is set.
        2) Fallback to yfinance (unlimited, no key required).

        Returns True on success, False on failure.
        """
        # guard: already fetched during this process
        if self.data is not None:
            return True

        # ─── 2. yfinance fallback ─────────────────────────────────────────
        try:
            import yfinance as yf

            df = yf.download(
                self.symbol, period="max", progress=False, threads=False
            )
            if df.empty:
                print(f"[YF] no data for {self.symbol}")
                return False
            self.raw_data = df
            self.data = df["Close"].sort_index()
            print(f"[YF] fetched {len(self.data)} rows for {self.symbol}")
            return True
        except Exception as exc:
            print(f"[YF] fetch error: {exc}")
            return False
    
    def calculate_enhanced_features(self):
        """Calculate multiple technical indicators and features"""
        if self.data is None:
            print("No data available. Please fetch data first.")
            return
        
        df = pd.DataFrame()
        df['price'] = self.data
        df['date'] = self.data.index
        
        # Basic returns and volatility
        df['returns'] = df['price'].pct_change()
        df['log_returns'] = np.log(df['price'] / df['price'].shift(1))
        
        # Multiple volatility measures
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
            df[f'price_vol_product_{window}'] = df['price'] * df[f'volatility_{window}']
        
        # Technical indicators
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'ma_{window}'] = df['price'].rolling(window=window).mean()
            df[f'price_ma_ratio_{window}'] = df['price'] / df[f'ma_{window}']
        
        # RSI
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['price'].rolling(window=20).mean()
        bb_std = df['price'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # MACD
        exp1 = df['price'].ewm(span=12, adjust=False).mean()
        exp2 = df['price'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Volume-based features (using price as proxy for volume analysis)
        df['price_momentum'] = df['price'] / df['price'].shift(10) - 1
        df['price_acceleration'] = df['returns'].diff()
        
        # Trend strength
        df['trend_strength'] = df['returns'].rolling(window=10).mean() / df['returns'].rolling(window=10).std()
        
        # Market regime detection
        df['high_vol_regime'] = (df['volatility_20'] > df['volatility_20'].rolling(window=50).mean()).astype(int)
        
        self.features_df = df.dropna()
        print(f"Calculated {len(df.columns)} features for {len(self.features_df)} data points")
    
    def create_composite_signal(self):
        """Create a composite signal combining multiple price-volatility products"""
        if self.features_df is None:
            print("Features not calculated. Please run calculate_enhanced_features first.")
            return
        
        # Weight different volatility products
        weights = [0.5, 0.3, 0.2]  # Favor shorter-term volatility
        
        composite_signal = (
            weights[0] * self.features_df['price_vol_product_5'] +
            weights[1] * self.features_df['price_vol_product_10'] +
            weights[2] * self.features_df['price_vol_product_20']
        )
        
        # Add technical indicator adjustments
        # Boost signal in high RSI regions (momentum)
        rsi_factor = np.where(self.features_df['rsi'] > 70, 1.1, 
                             np.where(self.features_df['rsi'] < 30, 1.1, 1.0))
        
        # Adjust for Bollinger Band position
        bb_factor = 1 + 0.1 * np.abs(self.features_df['bb_position'] - 0.5)
        
        # Adjust for MACD strength
        macd_factor = 1 + 0.05 * np.abs(self.features_df['macd_histogram'])
        
        self.features_df['composite_signal'] = composite_signal * rsi_factor * bb_factor * macd_factor
        
        return self.features_df['composite_signal']
    
    def fit_enhanced_regression(self):
        """Fit regression model with feature selection and regularization"""
        signal = self.create_composite_signal()
        
        # Prepare data for regression
        x = np.array(signal.index.map(pd.Timestamp.toordinal)).reshape(-1, 1)
        y = signal.values
        
        # Remove outliers using IQR method
        Q1 = np.percentile(y, 25)
        Q3 = np.percentile(y, 75)
        IQR = Q3 - Q1
        outlier_threshold = 1.5 * IQR
        
        mask = (y >= Q1 - outlier_threshold) & (y <= Q3 + outlier_threshold)
        x_clean = x[mask]
        y_clean = y[mask]
        
        # Fit model
        self.model = LinearRegression()
        self.model.fit(x_clean, y_clean)
        y_pred = self.model.predict(x_clean)
        
        # Store cleaned data for further analysis
        self.x_clean = x_clean
        self.y_clean = y_clean
        self.y_pred = y_pred
        
        # Model performance
        r2 = r2_score(y_clean, y_pred)
        mse = mean_squared_error(y_clean, y_pred)
        print(f"Model R² Score: {r2:.4f}")
        print(f"Model MSE: {mse:.4f}")
        
        return x_clean, y_clean, y_pred
    
    def analyze_segments_advanced(self):
        """Advanced segment analysis with multiple criteria"""
        difference = self.y_clean - self.y_pred
        
        # Dynamic epsilon based on data characteristics
        epsilon = np.std(difference) * 0.1  # More sensitive threshold
        print(f"Using dynamic epsilon: {epsilon:.6f}")
        
        # Enhanced sign determination using multiple criteria
        signs = np.where(difference > epsilon, 1,
                        np.where(difference < -epsilon, -1, 0))
        
        # Find zero crossings
        zero_crossings = np.where(np.diff(signs) != 0)[0]
        segment_indices = np.concatenate(([0], zero_crossings + 1, [len(difference)]))
        
        # Analyze segments
        areas = []
        total_area_above = 0.0
        total_area_below = 0.0
        
        for i in range(len(segment_indices) - 1):
            start_idx = segment_indices[i]
            end_idx = segment_indices[i + 1]
            
            if end_idx - start_idx < 2:  # Need at least 2 points for meaningful area
                continue
            
            segment_x = self.x_clean[start_idx:end_idx].flatten()
            segment_y_actual = self.y_clean[start_idx:end_idx]
            segment_y_pred = self.y_pred[start_idx:end_idx]
            
            # Calculate area using trapezoidal rule
            area_between = np.trapz(segment_y_actual - segment_y_pred, segment_x)
            
            # Determine position with stricter criteria
            if abs(area_between) < epsilon * (segment_x[-1] - segment_x[0]):
                position = 'Neutral'
                area_between = 0
            elif area_between > 0:
                position = 'Above'
                total_area_above += area_between
            else:
                position = 'Below'
                total_area_below += area_between
            
            if area_between != 0:  # Only store non-neutral segments
                areas.append({
                    'Segment': len([a for a in areas if a['Area'] != 0]) + 1,
                    'Start Date': pd.to_datetime(pd.Timestamp.fromordinal(int(segment_x[0]))).date(),
                    'End Date': pd.to_datetime(pd.Timestamp.fromordinal(int(segment_x[-1]))).date(),
                    'Position': position,
                    'Area': area_between,
                    'Duration': end_idx - start_idx,
                    'Avg_Signal': np.mean(segment_y_actual)
                })
        
        self.areas_df = pd.DataFrame(areas)
        self.total_area_above = total_area_above
        self.total_area_below = total_area_below
        
        print(f"\nTotal segments identified: {len(self.areas_df)}")
        print(f"Total area above: {total_area_above:.4f}")
        print(f"Total area below: {total_area_below:.4f}")
        
        return self.areas_df
    
    def create_adaptive_bins(self):
        """Create adaptive bins based on market regime and volatility"""
        if self.areas_df.empty:
            print("No segments to bin")
            return
        
        # Separate positive and negative areas
        positive_areas = self.areas_df[self.areas_df['Area'] > 0].copy()
        negative_areas = self.areas_df[self.areas_df['Area'] < 0].copy()
        
        # Adaptive percentiles based on data distribution
        if len(positive_areas) > 10:
            percentiles = [0.33, 0.66, 1.0]  # Tertiles for larger datasets
        else:
            percentiles = [0.5, 0.8, 1.0]    # Adjusted for smaller datasets
        
        # Bin positive areas
        if not positive_areas.empty:
            quantiles_positive = positive_areas['Area'].quantile(percentiles)
            positive_bins = np.concatenate(([positive_areas['Area'].min() - 1e-10], quantiles_positive.values))
            positive_labels = ['Bin 1', 'Bin 2', 'Bin 3']
            
            positive_areas['State'] = pd.cut(
                positive_areas['Area'],
                bins=positive_bins,
                labels=positive_labels,
                include_lowest=True
            )
        
        # Bin negative areas
        if not negative_areas.empty:
            negative_areas['AbsArea'] = negative_areas['Area'].abs()
            quantiles_negative_abs = negative_areas['AbsArea'].quantile(percentiles)
            negative_bins = -np.concatenate((quantiles_negative_abs.values[::-1], [negative_areas['AbsArea'].min() - 1e-10]))
            negative_bins = np.sort(negative_bins)
            negative_labels = ['Bin 6', 'Bin 5', 'Bin 4']
            
            negative_areas['State'] = pd.cut(
                negative_areas['Area'],
                bins=negative_bins,
                labels=negative_labels,
                include_lowest=True
            )
            negative_areas.drop(columns=['AbsArea'], inplace=True)
        
        # Combine and sort by segment order
        self.areas_df_binned = pd.concat([positive_areas, negative_areas], ignore_index=True)
        self.areas_df_binned = self.areas_df_binned.sort_values('Segment').reset_index(drop=True)
        
        print("\nBin Distribution:")
        print(self.areas_df_binned['State'].value_counts().sort_index())
        
        return self.areas_df_binned
    
    def build_enhanced_markov_model(self):
        """Build enhanced Markov model with higher-order dependencies"""
        state_sequence = self.areas_df_binned['State'].tolist()
        
        # First-order transition matrix
        num_states = len(self.labels)
        self.transition_matrix = np.zeros((num_states, num_states))
        
        for i in range(len(state_sequence) - 1):
            current_state = self.labels.index(state_sequence[i])
            next_state = self.labels.index(state_sequence[i + 1])
            self.transition_matrix[current_state][next_state] += 1
        
        # Add smoothing to handle zero probabilities
        alpha = 0.01  # Laplace smoothing parameter
        self.transition_matrix += alpha
        
        # Convert to probabilities
        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        self.transition_probabilities = self.transition_matrix / row_sums
        
        # Second-order dependencies (if enough data)
        if len(state_sequence) > 20:
            self.second_order_transitions = {}
            for i in range(len(state_sequence) - 2):
                state1 = state_sequence[i]
                state2 = state_sequence[i + 1]
                state3 = state_sequence[i + 2]
                key = (state1, state2)
                if key not in self.second_order_transitions:
                    self.second_order_transitions[key] = {}
                if state3 not in self.second_order_transitions[key]:
                    self.second_order_transitions[key][state3] = 0
                self.second_order_transitions[key][state3] += 1
            
            # Convert to probabilities
            for key in self.second_order_transitions:
                total = sum(self.second_order_transitions[key].values())
                for state in self.second_order_transitions[key]:
                    self.second_order_transitions[key][state] /= total
        
        return self.transition_probabilities
    
    def predict_next_states(self, use_second_order=True):
        """Enhanced prediction using both first and second-order Markov models"""
        state_sequence = self.areas_df_binned['State'].tolist()
        current_state = state_sequence[-1]
        
        # First-order prediction
        current_state_index = self.labels.index(current_state)
        first_order_probs = self.transition_probabilities[current_state_index]
        
        predictions = {
            'first_order': dict(zip(self.labels, first_order_probs))
        }
        
        # Second-order prediction (if available and requested)
        if hasattr(self, 'second_order_transitions') and use_second_order and len(state_sequence) >= 2:
            prev_state = state_sequence[-2]
            key = (prev_state, current_state)
            
            if key in self.second_order_transitions:
                second_order_probs = np.zeros(len(self.labels))
                for state, prob in self.second_order_transitions[key].items():
                    state_idx = self.labels.index(state)
                    second_order_probs[state_idx] = prob
                
                predictions['second_order'] = dict(zip(self.labels, second_order_probs))
                
                # Weighted combination
                weight_first = 0.3
                weight_second = 0.7
                combined_probs = weight_first * first_order_probs + weight_second * second_order_probs
                predictions['combined'] = dict(zip(self.labels, combined_probs))
        
        return predictions, current_state
    
    def monte_carlo_simulation(self, num_simulations=100, steps=10):
        """Enhanced Monte Carlo simulation with confidence intervals"""
        state_sequence = self.areas_df_binned['State'].tolist()
        current_state_index = self.labels.index(state_sequence[-1])
        
        simulated_paths = []
        final_states = []
        
        for _ in range(num_simulations):
            path = [current_state_index]
            current_idx = current_state_index
            
            for step in range(steps):
                probabilities = self.transition_probabilities[current_idx]
                if probabilities.sum() == 0:
                    break
                
                # Add noise to make simulations more realistic
                noise = np.random.normal(0, 0.01, len(probabilities))
                probabilities = probabilities + noise
                probabilities = np.maximum(probabilities, 0)  # Ensure non-negative
                probabilities = probabilities / probabilities.sum()  # Renormalize
                
                next_idx = np.random.choice(range(len(self.labels)), p=probabilities)
                path.append(next_idx)
                current_idx = next_idx
            
            simulated_paths.append([self.labels[i] for i in path])
            final_states.append(self.labels[path[-1]])
        
        # Analyze simulation results
        final_state_counts = pd.Series(final_states).value_counts(normalize=True)
        confidence_intervals = {}
        
        for state in self.labels:
            if state in final_state_counts:
                prob = final_state_counts[state]
                # Wilson score interval for confidence
                n = num_simulations
                z = 1.96  # 95% confidence
                wilson_center = (prob + z**2/(2*n)) / (1 + z**2/n)
                wilson_half_width = z * np.sqrt(prob*(1-prob)/n + z**2/(4*n**2)) / (1 + z**2/n)
                confidence_intervals[state] = (
                    max(0, wilson_center - wilson_half_width),
                    min(1, wilson_center + wilson_half_width)
                )
            else:
                confidence_intervals[state] = (0, 0)
        
        return simulated_paths, final_state_counts, confidence_intervals
    
    def plot_comprehensive_analysis(self):
        """Create comprehensive visualization of the analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Signal and regression line
        dates = pd.to_datetime([pd.Timestamp.fromordinal(int(x)) for x in self.x_clean.flatten()])
        
        axes[0, 0].plot(dates, self.y_clean, alpha=0.7, color='blue', label='Composite Signal')
        axes[0, 0].plot(dates, self.y_pred, color='red', linestyle='--', label='Regression Line')
        axes[0, 0].fill_between(dates, self.y_clean, self.y_pred, 
                               where=(self.y_clean > self.y_pred), 
                               alpha=0.3, color='green', label='Above')
        axes[0, 0].fill_between(dates, self.y_clean, self.y_pred, 
                               where=(self.y_clean <= self.y_pred), 
                               alpha=0.3, color='red', label='Below')
        axes[0, 0].set_title(f'{self.symbol} Composite Signal Analysis')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Signal Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Transition matrix heatmap
        transition_df = pd.DataFrame(self.transition_probabilities, 
                                   index=self.labels, columns=self.labels)
        sns.heatmap(transition_df, annot=True, fmt='.3f', cmap='YlOrRd', 
                   ax=axes[0, 1], cbar_kws={'label': 'Transition Probability'})
        axes[0, 1].set_title('State Transition Matrix')
        
        # 3. State distribution
        state_counts = self.areas_df_binned['State'].value_counts().sort_index()
        axes[1, 0].bar(state_counts.index, state_counts.values, 
                      color=plt.cm.Set3(np.linspace(0, 1, len(state_counts))))
        axes[1, 0].set_title('State Distribution')
        axes[1, 0].set_xlabel('State')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Feature importance (correlation with signal)
        feature_cols = ['rsi', 'bb_position', 'macd_histogram', 'trend_strength', 'price_momentum']
        available_features = [col for col in feature_cols if col in self.features_df.columns]
        
        if available_features:
            correlations = []
            for feature in available_features:
                # Align indices
                common_idx = self.features_df.index.intersection(pd.to_datetime([pd.Timestamp.fromordinal(int(x)) for x in self.x_clean.flatten()]))
                if len(common_idx) > 10:
                    corr = np.corrcoef(
                        self.features_df.loc[common_idx, feature].values,
                        self.features_df.loc[common_idx, 'composite_signal'].values
                    )[0, 1]
                    correlations.append(abs(corr))
                else:
                    correlations.append(0)
            
            axes[1, 1].barh(available_features, correlations)
            axes[1, 1].set_title('Feature Correlation with Signal')
            axes[1, 1].set_xlabel('Absolute Correlation')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Feature correlation\nnot available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.show()
    
    def generate_trading_signals(self):
        """Generate actionable trading signals based on predictions"""
        predictions, current_state = self.predict_next_states()
        
        # Define signal strength based on state transitions
        signal_strength = {
            'Bin 1': 0.8,  # Strong positive
            'Bin 2': 0.4,  # Moderate positive  
            'Bin 3': 0.2,  # Weak positive
            'Bin 4': -0.2, # Weak negative
            'Bin 5': -0.4, # Moderate negative
            'Bin 6': -0.8  # Strong negative
        }
        
        # Use combined predictions if available, otherwise first-order
        if 'combined' in predictions:
            next_state_probs = predictions['combined']
        else:
            next_state_probs = predictions['first_order']
        
        # Calculate expected signal strength
        expected_strength = sum(prob * signal_strength[state] 
                              for state, prob in next_state_probs.items())
        
        # Generate signal
        if expected_strength > 0.003:
            signal = "STRONG BUY"
        elif expected_strength > 0.001:
            signal = "BUY"
        elif expected_strength > -0.001:
            signal = "HOLD"
        elif expected_strength > -0.003:
            signal = "SELL"
        else:
            signal = "STRONG SELL"
        
        confidence = max(next_state_probs.values())
        
        return {
            'signal': signal,
            'strength': expected_strength,
            'confidence': confidence,
            'current_state': current_state,
            'next_state_probabilities': next_state_probs
        }
    
    def run_complete_analysis(self):
        """Run the complete enhanced analysis pipeline"""
        print("=== Enhanced Stock Prediction Analysis ===")
        print(f"Analyzing {self.symbol}...")
        
        # 1. Fetch data
        if not self.fetch_data():
            return
        
        # 2. Calculate features
        print("\n1. Calculating enhanced features...")
        self.calculate_enhanced_features()
        
        # 3. Fit regression model
        print("\n2. Fitting enhanced regression model...")
        self.fit_enhanced_regression()
        
        # 4. Analyze segments
        print("\n3. Analyzing market segments...")
        self.analyze_segments_advanced()
        
        # 5. Create bins
        print("\n4. Creating adaptive bins...")
        self.create_adaptive_bins()
        
        # 6. Build Markov model
        print("\n5. Building enhanced Markov model...")
        self.build_enhanced_markov_model()
        
        # 7. Make predictions
        print("\n6. Generating predictions...")
        predictions, current_state = self.predict_next_states()
        
        print(f"\nCurrent State: {current_state}")
        print("\nNext State Probabilities:")
        for method, probs in predictions.items():
            print(f"\n{method.replace('_', ' ').title()}:")
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            for state, prob in sorted_probs:
                print(f"  {state}: {prob:.4f}")
        
        # 8. Monte Carlo simulation
        print("\n7. Running Monte Carlo simulation...")
        paths, final_states, confidence_intervals = self.monte_carlo_simulation()
        
        print("\nMonte Carlo Results (Final State Probabilities):")
        for state in sorted(final_states.index, key=lambda x: final_states[x], reverse=True):
            prob = final_states[state]
            ci_low, ci_high = confidence_intervals[state]
            print(f"  {state}: {prob:.3f} (95% CI: {ci_low:.3f} - {ci_high:.3f})")
        
        # 9. Generate trading signals
        print("\n8. Generating trading signals...")
        trading_signal = self.generate_trading_signals()
        
        print(f"\n=== TRADING RECOMMENDATION ===")
        print(f"Signal: {trading_signal['signal']}")
        print(f"Strength: {trading_signal['strength']:.3f}")
        print(f"Confidence: {trading_signal['confidence']:.3f}")
        
        # 10. Create visualizations
        print("\n9. Creating visualizations...")
        self.plot_comprehensive_analysis()
        
        return {
            'predictions': predictions,
            'monte_carlo': final_states,
            'trading_signal': trading_signal,
            'model_performance': {
                'r2_score': r2_score(self.y_clean, self.y_pred),
                'segments': len(self.areas_df)
            }
        }

# Usage example
if __name__ == "__main__":
    # Initialize predictor
    API_KEY = 'WTFVO94DFLYZCD16'  # Replace with your actual API key
    predictor = EnhancedStockPredictor(API_KEY, symbol='AAPL')
    
    # Run complete analysis
    results = predictor.run_complete_analysis()
    
    # You can also run individual components:
    # predictor.fetch_data()
    # predictor.calculate_enhanced_features()
    # etc.