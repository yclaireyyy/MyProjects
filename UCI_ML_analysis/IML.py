import pandas as pd  # data analysis
import numpy as np  # calculation
import matplotlib.pyplot as plt  # draw graphs
import seaborn as sns  # visualize result
from sklearn.model_selection import StratifiedKFold, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc)
from ucimlrepo import fetch_ucirepo
import warnings
import time  # Add missing imports
import gc    # For memory management

warnings.filterwarnings('ignore')

# Set random state for reproducibility
RANDOM_STATE = 42


class OnlineShoppersMLProject:
    def __init__(self):
        self.X = None
        self.y = None
        self.X_processed = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.results = {}

    def load_data(self):
        """Load the UCI Online Shoppers Purchasing Intention Dataset"""
        try:
            # Fetch dataset from UCI repository
            dataset = fetch_ucirepo(id=468)

            # Extract features and target variables
            self.X = dataset.data.features
            self.y = dataset.data.targets

            # Target variable handling
            if hasattr(self.y, 'squeeze'):
                self.y = self.y.squeeze()

            # Ensure y is 1D array
            if len(self.y.shape) > 1:
                self.y = self.y.iloc[:, 0] if hasattr(self.y, 'iloc') else self.y[:, 0]

            # Ensure target variable is properly encoded (True/False -> 1/0)
            if self.y.dtype == 'object' or self.y.dtype == 'bool':
                # Type conversion
                unique_vals = self.y.unique()
                if len(unique_vals) == 2:
                    # Boolean or binary classification
                    if self.y.dtype == 'bool':
                        self.y = self.y.astype(int)
                    else:
                        # Handle string type boolean values
                        label_encoder = LabelEncoder()
                        self.y = label_encoder.fit_transform(self.y)
                else:
                    # Multi-class classification
                    label_encoder = LabelEncoder()
                    self.y = label_encoder.fit_transform(self.y)

            print(f"Dataset loaded successfully!")
            print(f"Features shape: {self.X.shape}")
            print(f"Target shape: {self.y.shape}")
            print(f"Target distribution: {np.bincount(self.y)}")

            return self.X, self.y

        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def explore_data(self):
        """Enhanced exploratory data analysis"""
        print("\nExploratory Data Analysis...")
        start_time = time.time()

        try:
            # variable initialization
            selected_features = []

            print("  Identifying numeric features...")
            numeric_features = self.X.select_dtypes(include=[np.number]).columns.tolist()

            # Set random seed
            np.random.seed(RANDOM_STATE)
            sample_size = min(5000, len(self.X))

            if len(self.X) > sample_size:
                sample_idx = np.random.choice(len(self.X), sample_size, replace=False)
                X_sample = self.X.iloc[sample_idx]
                # Index handling
                if isinstance(self.y, np.ndarray):
                    y_sample = self.y[sample_idx]
                else:
                    y_sample = self.y.iloc[sample_idx]
            else:
                X_sample = self.X
                y_sample = self.y

            plt.figure(figsize=(15, 5))

            # Target variable distribution
            plt.subplot(1, 3, 1)
            target_counts = pd.Series(y_sample).value_counts()
            target_counts.plot(kind='bar', color=['skyblue', 'lightcoral'])
            plt.title('Target Distribution')
            plt.xlabel('Revenue (Purchase Intention)')
            plt.ylabel('Count')
            plt.xticks(rotation=0)

            # Feature correlation
            plt.subplot(1, 3, 2)

            if len(numeric_features) > 0:
                max_features_for_corr = 20

                if len(numeric_features) > max_features_for_corr:
                    feature_vars = X_sample[numeric_features].var().sort_values(ascending=False)
                    selected_features = feature_vars.head(max_features_for_corr).index.tolist()
                else:
                    selected_features = numeric_features

                corr_matrix = X_sample[selected_features].corr(method='pearson')

                sns.heatmap(
                    corr_matrix,
                    annot=False,
                    cmap='RdBu_r',
                    center=0,
                    square=True,
                    cbar_kws={"shrink": .8}
                )
                plt.title(f'Feature Correlations (Top {len(selected_features)})')
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
            else:
                plt.text(0.5, 0.5, 'No numeric features found', ha='center', va='center')
                plt.title('Feature Correlations - No Data')

            # Key feature distribution
            plt.subplot(1, 3, 3)

            priority_features = ['PageValues', 'ExitRates', 'BounceRates', 'ProductRelated_Duration']
            selected_feature = None

            for feature in priority_features:
                if feature in self.X.columns:
                    selected_feature = feature
                    break

            if selected_feature is None and len(numeric_features) > 0:
                selected_feature = numeric_features[0]

            if selected_feature:
                data_to_plot = X_sample[selected_feature].dropna()
                n_bins = min(30, max(10, int(np.sqrt(len(data_to_plot)))))

                plt.hist(data_to_plot, bins=n_bins, alpha=0.7, color='steelblue', edgecolor='black')
                plt.title(f'{selected_feature} Distribution')
                plt.xlabel(selected_feature)
                plt.ylabel('Frequency')

                mean_val = data_to_plot.mean()
                median_val = data_to_plot.median()
                plt.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
                plt.axvline(median_val, color='green', linestyle='--', alpha=0.7, label=f'Median: {median_val:.2f}')
                plt.legend()
            else:
                plt.text(0.5, 0.5, 'No numeric features available', ha='center', va='center')
                plt.title('Feature Distribution - No Data')

            plt.tight_layout()

            elapsed_time = time.time() - start_time
            print(f"  EDA completed in {elapsed_time:.2f} seconds")

            plt.show()

            # Direct variable usage
            print(f"\nDataset Overview:")
            print(f"  - Sample count: {len(self.X):,}")
            print(f"  - Feature count: {len(self.X.columns)}")
            print(f"  - Numeric features: {len(numeric_features)}")
            print(f"  - Target distribution: {dict(target_counts)}")

            print("Data exploration completed!")

        except Exception as e:
            print(f"Error in data exploration: {e}")
            import traceback
            traceback.print_exc()

    def feature_engineering(self):
        """
        Feature Construction: Expand from original features to 50+ features
        """
        print("Starting feature engineering...")
        try:
            df = self.X.copy()

            # Categorical feature handling
            expected_categorical = ['Month', 'VisitorType', 'Weekend']
            categorical_features = []

            for feature in expected_categorical:
                if feature in df.columns:
                    categorical_features.append(feature)
                    if df[feature].dtype == 'object' or df[feature].dtype == 'bool':
                        encoder = LabelEncoder()
                        # Handle missing values
                        df[feature] = df[feature].fillna('Unknown')
                        df[f'{feature}_Encoded'] = encoder.fit_transform(df[feature].astype(str))

            # Check if required numeric columns exist
            required_numeric = ['Administrative', 'Informational', 'ProductRelated',
                                'Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration']

            missing_cols = [col for col in required_numeric if col not in df.columns]
            if missing_cols:
                print(f"Warning: Missing expected columns: {missing_cols}")
                # Create default values for missing columns
                for col in missing_cols:
                    df[col] = 0

            # Group 1: Page behavior aggregation features
            df['Total_Pages'] = df['Administrative'] + df['Informational'] + df['ProductRelated']
            df['Total_Duration'] = df['Administrative_Duration'] + df['Informational_Duration'] + df[
                'ProductRelated_Duration']

            # Avoid division by zero
            df['Avg_Duration_Per_Page'] = np.where(df['Total_Pages'] > 0,
                                                   df['Total_Duration'] / df['Total_Pages'], 0)

            # Page type ratios - use safe division
            df['Admin_Page_Ratio'] = np.where(df['Total_Pages'] > 0,
                                              df['Administrative'] / df['Total_Pages'], 0)
            df['Info_Page_Ratio'] = np.where(df['Total_Pages'] > 0,
                                             df['Informational'] / df['Total_Pages'], 0)
            df['Product_Page_Ratio'] = np.where(df['Total_Pages'] > 0,
                                                df['ProductRelated'] / df['Total_Pages'], 0)

            # Duration ratios - use safe division
            df['Admin_Duration_Ratio'] = np.where(df['Total_Duration'] > 0,
                                                  df['Administrative_Duration'] / df['Total_Duration'], 0)
            df['Info_Duration_Ratio'] = np.where(df['Total_Duration'] > 0,
                                                 df['Informational_Duration'] / df['Total_Duration'], 0)
            df['Product_Duration_Ratio'] = np.where(df['Total_Duration'] > 0,
                                                    df['ProductRelated_Duration'] / df['Total_Duration'], 0)

            # Efficiency metrics - check if PageValues exists
            if 'PageValues' in df.columns:
                df['Page_Efficiency'] = np.where(df['Total_Pages'] > 0,
                                                 df['PageValues'] / df['Total_Pages'], 0)
                df['Duration_Efficiency'] = np.where(df['Total_Duration'] > 0,
                                                     df['PageValues'] / df['Total_Duration'], 0)
            else:
                df['Page_Efficiency'] = 0
                df['Duration_Efficiency'] = 0

            # Exit and Bounce rates - check if columns exist
            if 'ExitRates' in df.columns and 'BounceRates' in df.columns:
                df['Bounce_Exit_Diff'] = df['ExitRates'] - df['BounceRates']
            else:
                df['Bounce_Exit_Diff'] = 0

            # Group 2: Seasonal features
            if 'Month' in df.columns:
                # Flexible month mapping
                def map_month_to_season(month):
                    month_str = str(month).lower()
                    if any(m in month_str for m in ['nov', 'dec', '11', '12']):
                        return 1  # Holiday season
                    elif any(m in month_str for m in ['jan', 'feb', 'mar', '1', '2', '3']):
                        return 2  # Winter
                    elif any(m in month_str for m in ['apr', 'may', 'jun', '4', '5', '6']):
                        return 3  # Spring
                    else:
                        return 4  # Summer/Fall

                df['Season'] = df['Month'].apply(map_month_to_season)

                # More flexible holiday season detection
                holiday_months = df['Month'].astype(str).str.lower()
                df['Is_Holiday_Season'] = (holiday_months.str.contains('nov|dec|11|12', na=False)).astype(int)

            # Special day features
            if 'SpecialDay' in df.columns:
                # Binning operation
                try:
                    # Ensure data is in reasonable range
                    special_day_clean = pd.to_numeric(df['SpecialDay'], errors='coerce').fillna(0)
                    df['Special_Day_Category'] = pd.cut(
                        special_day_clean,
                        bins=[-0.1, 0.2, 0.6, 1.1],
                        labels=[0, 1, 2],
                        include_lowest=True
                    )
                    df['Special_Day_Category'] = df['Special_Day_Category'].fillna(0).astype(int)
                except Exception as e:
                    print(f"Warning: Error in Special Day categorization: {e}")
                    df['Special_Day_Category'] = 0

            # Weekend features
            if 'Weekend' in df.columns:
                # Boolean conversion
                if df['Weekend'].dtype == 'bool':
                    df['Weekend_Numeric'] = df['Weekend'].astype(int)
                else:
                    # Handle string or other types
                    weekend_str = df['Weekend'].astype(str).str.lower()
                    df['Weekend_Numeric'] = (weekend_str.isin(['true', '1', 'yes'])).astype(int)

                if 'SpecialDay' in df.columns:
                    df['Weekend_Special_Combo'] = df['Weekend_Numeric'] * pd.to_numeric(df['SpecialDay'],
                                                                                        errors='coerce').fillna(0)
                if 'Is_Holiday_Season' in df.columns:
                    df['Weekend_Holiday_Combo'] = df['Weekend_Numeric'] * df['Is_Holiday_Season']

            # Group 3: Technical environment features
            if 'VisitorType' in df.columns:
                visitor_str = df['VisitorType'].astype(str).str.lower()
                df['Returning_Visitor_Flag'] = (visitor_str.str.contains('return', na=False)).astype(int)

            # Technical diversity - safer handling
            if 'OperatingSystems' in df.columns and 'Browser' in df.columns:
                try:
                    df['Tech_Diversity'] = (df['OperatingSystems'].astype(str) + '_' +
                                            df['Browser'].astype(str))
                    tech_encoder = LabelEncoder()
                    df['Tech_Diversity_Encoded'] = tech_encoder.fit_transform(df['Tech_Diversity'])

                    # Technical popularity
                    os_counts = df['OperatingSystems'].value_counts()
                    browser_counts = df['Browser'].value_counts()
                    df['OS_Popularity'] = df['OperatingSystems'].map(os_counts).fillna(0)
                    df['Browser_Popularity'] = df['Browser'].map(browser_counts).fillna(0)
                except Exception as e:
                    print(f"Warning: Error in technical diversity calculation: {e}")
                    df['Tech_Diversity_Encoded'] = 0
                    df['OS_Popularity'] = 0
                    df['Browser_Popularity'] = 0

            # Regional traffic combination
            if 'Region' in df.columns and 'TrafficType' in df.columns:
                try:
                    df['Region_Traffic_Combo'] = (pd.to_numeric(df['Region'], errors='coerce').fillna(0) * 100 +
                                                  pd.to_numeric(df['TrafficType'], errors='coerce').fillna(0))
                except Exception as e:
                    print(f"Warning: Error in region traffic combination: {e}")
                    df['Region_Traffic_Combo'] = 0

            # Group 4: Advanced composite features
            df['User_Engagement'] = (df['Total_Pages'] * 0.3 +
                                     df['Total_Duration'] * 0.0001 +
                                     (df['PageValues'] if 'PageValues' in df.columns else 0) * 0.7)

            df['Purchase_Propensity'] = ((df['PageValues'] if 'PageValues' in df.columns else 0) * 2 -
                                         (df['BounceRates'] if 'BounceRates' in df.columns else 0) * 10 -
                                         (df['ExitRates'] if 'ExitRates' in df.columns else 0) * 5)

            # Composite indicator calculation
            bounce_rates = df['BounceRates'] if 'BounceRates' in df.columns else 0
            exit_rates = df['ExitRates'] if 'ExitRates' in df.columns else 0

            df['Site_Quality'] = (1 - bounce_rates) * (1 - exit_rates)
            df['User_Experience'] = df['Site_Quality'] * df['Page_Efficiency']
            df['Depth_Value_Ratio'] = np.where(df['Total_Pages'] > 0,
                                               (df['PageValues'] if 'PageValues' in df.columns else 0) / df[
                                                   'Total_Pages'], 0)
            df['Time_Value_Efficiency'] = np.where(df['Total_Duration'] > 0,
                                                   (df['PageValues'] if 'PageValues' in df.columns else 0) / df[
                                                       'Total_Duration'], 0)

            # User behavior features
            if 'Returning_Visitor_Flag' in df.columns:
                df['New_User_Behavior'] = (1 - df['Returning_Visitor_Flag']) * df['User_Engagement']
                df['Returning_User_Behavior'] = df['Returning_Visitor_Flag'] * df['User_Engagement']

            if 'Weekend_Numeric' in df.columns:
                df['Weekday_Engagement'] = (1 - df['Weekend_Numeric']) * df['User_Engagement']
                df['Weekend_Engagement'] = df['Weekend_Numeric'] * df['User_Engagement']

            df['Overall_User_Value'] = df['Purchase_Propensity'] * 0.4 + df['User_Experience'] * 0.6

            # Group 5: Interaction features
            df['Interaction_1'] = df['Total_Pages'] * (df['PageValues'] if 'PageValues' in df.columns else 0)
            df['Interaction_2'] = df['Total_Duration'] * df['Site_Quality']
            if 'SpecialDay' in df.columns:
                df['Interaction_3'] = pd.to_numeric(df['SpecialDay'], errors='coerce').fillna(0) * df['User_Engagement']

            # Polynomial features - Non-negative values before squaring
            if 'PageValues' in df.columns:
                page_values_safe = np.maximum(df['PageValues'], 0)  # Ensure non-negative
                df['PageValues_Squared'] = page_values_safe ** 2
            else:
                df['PageValues_Squared'] = 0

            total_duration_safe = np.maximum(df['Total_Duration'], 0)  # Ensure non-negative
            df['Total_Duration_Squared'] = total_duration_safe ** 2

            # Clean up and ensure all features are numeric
            # Remove original categorical columns to avoid confusion
            columns_to_remove = ['Month', 'VisitorType', 'Weekend', 'Tech_Diversity']
            for col in columns_to_remove:
                if col in df.columns:
                    df = df.drop(columns=[col])

            # Infinite value handling
            df = df.replace([np.inf, -np.inf], np.nan)

            # Get numeric columns only
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

            # Keep only numeric columns
            df = df[numeric_columns]

            # Fill missing values with median for numeric columns
            df = df.fillna(df.median())

            # Final check for any remaining non-numeric columns
            for col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Fill any NaN values created by conversion
            df = df.fillna(0)

            # Store processed features
            self.X_processed = df
            self.feature_names = list(df.columns)

            print(f"Feature engineering completed!")
            print(f"Original features: {self.X.shape[1]} → Final features: {len(self.feature_names)}")

            return df

        except Exception as e:
            print(f"Error in feature engineering: {e}")
            # If feature engineering fails, at least keep original numeric features
            numeric_features = self.X.select_dtypes(include=[np.number])
            self.X_processed = numeric_features.fillna(numeric_features.median())
            self.feature_names = list(self.X_processed.columns)
            print(f"Fallback: Using {len(self.feature_names)} original numeric features")
            return self.X_processed

    def preprocess_data(self):
        """Data preprocessing and standardization"""
        print("\nData preprocessing...")
        try:
            # Ensure we have processed data
            if self.X_processed is None:
                raise ValueError("Must run feature_engineering() first")

            # Variable name consistency
            numeric_columns = self.X_processed.select_dtypes(include=[np.number]).columns

            for column in numeric_columns: 
                Q1 = self.X_processed[column].quantile(0.25)
                Q3 = self.X_processed[column].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:  # Only clip if IQR is positive
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    self.X_processed[column] = self.X_processed[column].clip(lower_bound, upper_bound)

            # Standardization
            standardized_data = self.scaler.fit_transform(self.X_processed)

            self.X_processed = pd.DataFrame(
                standardized_data,
                columns=self.feature_names,
                index=self.X_processed.index
            )

            # Check data quality after standardization
            if np.any(np.isnan(self.X_processed.values)):
                print("Warning: NaN values detected after standardization, filling with zeros")
                self.X_processed = self.X_processed.fillna(0)

            if np.any(np.isinf(self.X_processed.values)):
                print("Warning: Infinite values detected after standardization, clipping")
                self.X_processed = self.X_processed.replace([np.inf, -np.inf], [1e6, -1e6])

            print("Preprocessing completed!")
            return self.X_processed

        except Exception as e:
            print(f"Error in preprocessing: {e}")
            raise

    def setup_algorithms(self):
        """Setup the three algorithms with hyperparameter grids"""
        # Algorithm selection covering bias-variance spectrum
        algorithms = {
            'Naive_Bayes': {
                'model': GaussianNB(),
                'params': {
                    'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7]  # Extended parameter range
                },
                'bias_variance': 'High Bias / Low Variance'
            },
            'Random_Forest': {
                'model': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],  # Add None option
                    'min_samples_split': [2, 5, 10]  # Add more parameters
                },
                'bias_variance': 'Medium Bias / Medium Variance'
            },
            'SVM': {
                'model': SVC(random_state=RANDOM_STATE, probability=True),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'gamma': ['scale', 0.001, 0.01, 0.1],  # Add 'scale' option
                    'kernel': ['rbf', 'linear']  # Add kernel option
                },
                'bias_variance': 'Low Bias / High Variance'
            }
        }
        return algorithms

    def cross_validation_experiment(self):
        """5-fold cross-validation with nested hyperparameter tuning"""
        print("\nRunning cross-validation experiment...")
        try:
            # Setup cross-validation
            outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

            algorithms = self.setup_algorithms()

            for algo_name, algo_config in algorithms.items():
                print(f"\nTraining {algo_name}...")
                # Lists to store results
                fold_scores = []
                fold_predictions = []
                fold_probabilities = []
                fold_true_labels = []
                best_params_per_fold = []

                for fold, (train_idx, test_idx) in enumerate(outer_cv.split(self.X_processed, self.y)):
                    # Ensure index alignment
                    try:
                        # Split data - use iloc to ensure correct indexing
                        X_train = self.X_processed.iloc[train_idx].copy()
                        X_test = self.X_processed.iloc[test_idx].copy()
                        y_train = self.y[train_idx].copy() if hasattr(self.y, '__getitem__') else self.y.iloc[
                            train_idx].copy()
                        y_test = self.y[test_idx].copy() if hasattr(self.y, '__getitem__') else self.y.iloc[
                            test_idx].copy()

                        # Ensure correct data types
                        if hasattr(y_train, 'values'):
                            y_train = y_train.values.ravel()
                        if hasattr(y_test, 'values'):
                            y_test = y_test.values.ravel()

                        # Nested cross-validation for hyperparameter tuning
                        grid_search = GridSearchCV(
                            algo_config['model'],
                            algo_config['params'],
                            cv=inner_cv,
                            scoring='accuracy',
                            n_jobs=-1,
                            error_score='raise'  # Explicitly handle errors
                        )

                        grid_search.fit(X_train, y_train)
                        best_params_per_fold.append(grid_search.best_params_)

                        # Train best model and predict
                        best_model = grid_search.best_estimator_
                        y_pred = best_model.predict(X_test)

                        # Probability prediction
                        try:
                            if hasattr(best_model, 'predict_proba'):
                                y_prob = best_model.predict_proba(X_test)
                                if y_prob.shape[1] > 1:
                                    y_prob = y_prob[:, 1]  # Take positive class probability
                                else:
                                    y_prob = y_prob[:, 0]
                            else:
                                y_prob = None
                        except Exception:
                            y_prob = None

                        # Calculate metrics - add error handling
                        try:
                            scores = {
                                'accuracy': accuracy_score(y_test, y_pred),
                                'precision': precision_score(y_test, y_pred, zero_division=0),
                                'recall': recall_score(y_test, y_pred, zero_division=0),
                                'f1': f1_score(y_test, y_pred, zero_division=0)
                            }
                        except Exception:
                            scores = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}

                        fold_scores.append(scores)
                        fold_predictions.extend(y_pred.tolist())
                        fold_true_labels.extend(y_test.tolist())
                        if y_prob is not None:
                            fold_probabilities.extend(y_prob.tolist())

                    except Exception as fold_e:
                        print(f"    Error in fold {fold + 1}: {fold_e}")
                        # Add default values to maintain list length consistency
                        default_scores = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
                        fold_scores.append(default_scores)
                        best_params_per_fold.append({})

                # Store results
                self.results[algo_name] = {
                    'fold_scores': fold_scores,
                    'predictions': fold_predictions,
                    'true_labels': fold_true_labels,
                    'probabilities': fold_probabilities,
                    'best_params': best_params_per_fold,
                    'bias_variance': algo_config['bias_variance']
                }

                # Print summary
                if fold_scores:  # Ensure results exist before calculating statistics
                    avg_scores = {}
                    for metric in ['accuracy', 'precision', 'recall', 'f1']:
                        scores = [fold[metric] for fold in fold_scores if metric in fold]
                        if scores:
                            avg_scores[metric] = np.mean(scores)
                            std_scores = np.std(scores)
                            print(f"    {metric.capitalize()}: {avg_scores[metric]:.4f} (±{std_scores:.4f})")

            print("Cross-validation experiment completed!")

        except Exception as e:
            print(f"Error in cross-validation: {e}")
            raise

    def generate_experimental_results(self):
        """Generate comprehensive experimental results"""
        print("\nGenerating results...")
        try:
            # Check if there are valid results
            if not self.results:
                print("No results to display")
                return

            # 1. Performance comparison table
            self.plot_performance_comparison()

            # 2. Confusion matrices
            self.plot_confusion_matrices()

            # 3. ROC curves
            self.plot_roc_curves()

            # 4. Learning curves
            print("\n4. Learning Curves (This may take longer time)")
            self.plot_learning_curves()

            # 5. Feature importance (Random Forest)
            self.plot_feature_importance()
            print("All results generated successfully!")

        except Exception as e:
            print(f"Error generating results: {e}")

    def plot_performance_comparison(self):
        """Create performance comparison table"""
        print("\n1. Performance Comparison Table")
        try:
            # Create performance comparison table
            metrics_data = []
            for algo_name, results in self.results.items():
                if 'fold_scores' in results and results['fold_scores']:
                    fold_scores = results['fold_scores']
                    row = {'Algorithm': algo_name}

                    # calculate mean and variance for each metric
                    for metric in ['accuracy', 'precision', 'recall', 'f1']:
                        scores = [fold[metric] for fold in fold_scores if metric in fold and not np.isnan(fold[metric])]
                        if scores:
                            row[f'{metric.capitalize()}'] = f"{np.mean(scores):.4f} ± {np.std(scores):.4f}"
                        else:
                            row[f'{metric.capitalize()}'] = "N/A"

                    row['Bias-Variance'] = results['bias_variance']
                    metrics_data.append(row)

            if not metrics_data:
                print("No valid performance data to display")
                return

            df_metrics = pd.DataFrame(metrics_data)
            print(df_metrics)

            # Visualization
            plt.figure(figsize=(12, 6))
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            x = np.arange(len(metrics))
            width = 0.25

            valid_results = {name: results for name, results in self.results.items()
                             if 'fold_scores' in results and results['fold_scores']}

            for i, (algo_name, results) in enumerate(valid_results.items()):
                fold_scores = results['fold_scores']
                means = []
                stds = []

                for metric in metrics:
                    scores = [fold[metric] for fold in fold_scores if metric in fold and not np.isnan(fold[metric])]
                    if scores:
                        means.append(np.mean(scores))
                        stds.append(np.std(scores))
                    else:
                        means.append(0)
                        stds.append(0)

                plt.bar(x + i * width, means, width, label=algo_name, yerr=stds, capsize=5)

            plt.xlabel('Metrics')
            plt.ylabel('Score')
            plt.title('Algorithm Performance Comparison')
            plt.xticks(x + width, [m.capitalize() for m in metrics])
            plt.legend()
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error in performance comparison: {e}")

    def plot_confusion_matrices(self):
        """Plot confusion matrices for all algorithms"""
        print("\n2. Confusion Matrices")
        try:
            valid_results = {name: results for name, results in self.results.items()
                             if 'predictions' in results and 'true_labels' in results
                             and results['predictions'] and results['true_labels']}

            if not valid_results:
                print("No valid prediction data for confusion matrices")
                return

            fig, axes = plt.subplots(1, len(valid_results), figsize=(5 * len(valid_results), 4))
            if len(valid_results) == 1:
                axes = [axes]

            for i, (algo_name, results) in enumerate(valid_results.items()):
                try:
                    cm = confusion_matrix(results['true_labels'], results['predictions'])
                    sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
                    axes[i].set_title(f'{algo_name}')
                    axes[i].set_xlabel('Predicted')
                    axes[i].set_ylabel('Actual')
                except Exception as cm_e:
                    print(f"Error creating confusion matrix for {algo_name}: {cm_e}")
                    axes[i].text(0.5, 0.5, f'Error: {str(cm_e)}', ha='center', va='center')

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error in confusion matrices: {e}")

    def plot_roc_curves(self):
        """Plot ROC curves for all algorithms"""
        print("\n3. ROC Curves")
        try:
            plt.figure(figsize=(10, 8))

            valid_curves = False
            for algo_name, results in self.results.items():
                if ('probabilities' in results and results['probabilities'] and
                        'true_labels' in results and results['true_labels']):
                    try:
                        # Convert boolean labels to integers
                        true_labels = [int(label) for label in results['true_labels']]
                        probabilities = results['probabilities']

                        if len(true_labels) == len(probabilities):
                            fpr, tpr, _ = roc_curve(true_labels, probabilities)
                            roc_auc = auc(fpr, tpr)
                            plt.plot(fpr, tpr, label=f'{algo_name} (AUC = {roc_auc:.4f})', linewidth=2)
                            valid_curves = True
                        else:
                            print(f"Warning: Mismatched lengths for {algo_name}")
                    except Exception as roc_e:
                        print(f"Error creating ROC curve for {algo_name}: {roc_e}")

            if valid_curves:
                plt.plot([0, 1], [0, 1], 'k--', label='Random')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curves Comparison')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.show()
            else:
                print("No valid probability data for ROC curves")

        except Exception as e:
            print(f"Error in ROC curves: {e}")

    def plot_learning_curves(self):
        """Plot learning curves for all algorithms"""
        try:
            algorithms = self.setup_algorithms()
            valid_algos = {name: config for name, config in algorithms.items()
                           if name in self.results and 'best_params' in self.results[name]
                           and self.results[name]['best_params']}

            if not valid_algos:
                print("No valid algorithms for learning curves")
                return

            fig, axes = plt.subplots(1, len(valid_algos), figsize=(6 * len(valid_algos), 5))
            if len(valid_algos) == 1:
                axes = [axes]

            for i, (algo_name, algo_config) in enumerate(valid_algos.items()):
                try:
                    # Use best parameters from first fold
                    best_params = self.results[algo_name]['best_params'][0] if self.results[algo_name][
                        'best_params'] else {}
                    model = algo_config['model'].set_params(**best_params)

                    train_sizes, train_scores, val_scores = learning_curve(
                        model, self.X_processed, self.y, cv=3,  # Reduce CV folds for speed
                        train_sizes=np.linspace(0.1, 1.0, 8),  # Reduce training size points
                        scoring='accuracy', random_state=RANDOM_STATE, n_jobs=-1
                    )

                    train_mean = np.mean(train_scores, axis=1)
                    train_std = np.std(train_scores, axis=1)
                    val_mean = np.mean(val_scores, axis=1)
                    val_std = np.std(val_scores, axis=1)

                    axes[i].plot(train_sizes, train_mean, 'o-', label='Training Score', color='blue')
                    axes[i].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2,
                                         color='blue')

                    axes[i].plot(train_sizes, val_mean, 'o-', label='Validation Score', color='red')
                    axes[i].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='red')

                    axes[i].set_title(f'{algo_name} Learning Curve')
                    axes[i].set_xlabel('Training Set Size')
                    axes[i].set_ylabel('Accuracy Score')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)

                except Exception as lc_e:
                    print(f"  Error generating learning curve for {algo_name}: {lc_e}")
                    axes[i].text(0.5, 0.5, f'Error: {str(lc_e)}', ha='center', va='center', transform=axes[i].transAxes)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error in learning curves: {e}")

    def plot_feature_importance(self):
        """Plot feature importance for Random Forest"""
        print("\n5. Feature Importance (Random Forest)")
        try:
            if 'Random_Forest' in self.results and 'best_params' in self.results['Random_Forest']:
                best_params = self.results['Random_Forest']['best_params'][0] if self.results['Random_Forest'][
                    'best_params'] else {}
                rf_model = RandomForestClassifier(random_state=RANDOM_STATE, **best_params)
                rf_model.fit(self.X_processed, self.y)

                # Get feature importance
                importance = rf_model.feature_importances_
                feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)

                # Plot top 15 features
                plt.figure(figsize=(12, 8))
                top_features = feature_importance.head(15)
                plt.barh(range(len(top_features)), top_features['importance'])
                plt.yticks(range(len(top_features)), top_features['feature'])
                plt.xlabel('Feature Importance')
                plt.title('Top 15 Most Important Features (Random Forest)')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.show()

                # Display numerical table
                print("\nTop 10 Feature Importances:")
                print(feature_importance.head(10).to_string(index=False))
            else:
                print("Random Forest results not available for feature importance")

        except Exception as e:
            print(f"Error in feature importance: {e}")

    def run_complete_experiment(self):
        """Run the complete machine learning experiment"""
        print("=== UCI Online Shoppers Purchasing Intention Dataset Analysis ===")

        try:
            # Step 1: Load and explore data
            self.load_data()
            self.explore_data()

            # Step 2: Feature engineering (original → 50+ features)
            self.feature_engineering()

            # Step 3: Data preprocessing
            self.preprocess_data()

            # Step 4: Cross-validation with hyperparameter tuning
            self.cross_validation_experiment()

            # Step 5: Generate experimental results
            self.generate_experimental_results()

            # Summary for report
            print("\n" + "=" * 50)
            print("EXPERIMENT SUMMARY:")
            print("=" * 50)
            print(f"Dataset: UCI Online Shoppers Purchasing Intention ({self.X.shape[0]} samples)")
            print(f"Feature Engineering: {self.X.shape[1]} → {len(self.feature_names)} features")
            print("Algorithms: Naive Bayes (High Bias), Random Forest (Medium), SVM (Low Bias)")
            print("Cross-validation: 5-fold stratified CV with nested hyperparameter tuning")
            print("Evaluation Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC")
            print("Additional Analysis: Learning curves, Confusion matrices, Feature importance")
            print("=" * 50)

        except Exception as e:
            print(f"\nCritical Error in experiment: {e}")
            print("Experiment terminated due to unexpected error.")
            raise


# Run the complete experiment
if __name__ == "__main__":
    project = OnlineShoppersMLProject()
    project.run_complete_experiment()
