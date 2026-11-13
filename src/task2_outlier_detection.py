"""
Task 2: Outlier Detection and Treatment with Advanced Visualization

This module implements various outlier detection and treatment techniques,
including statistical methods, machine learning-based detection, and 
comprehensive visualization of outliers and their impact.

Author: MLDS Course
Date: December 2025
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from scipy import stats
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings('ignore')


class OutlierAnalyzer:
    """
    A comprehensive outlier analyzer that detects and treats outliers using
    multiple methods and provides detailed visualizations.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the analyzer with a dataset.
        
        Parameters:
            data (pd.DataFrame): Input DataFrame to analyze
        """
        self.data = data.copy()
        self.original_data = data.copy()
        self.outlier_masks = {}
        self.treated_datasets = {}
        
    def detect_outliers_zscore(self, threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers using Z-score method.
        
        Parameters:
            threshold (float): Z-score threshold for outlier detection
            
        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        mask = pd.DataFrame(False, index=self.data.index, columns=numeric_cols, dtype=bool)

        if len(self.data) > 1:
            for col in numeric_cols:
                std = self.data[col].std()
                if std and not np.isnan(std) and std > 0:
                    z_scores = np.abs((self.data[col] - self.data[col].mean()) / std)
                    mask[col] = (z_scores > threshold).fillna(False)

        self.outlier_masks['zscore'] = mask
        return mask
    
    def detect_outliers_iqr(self, multiplier: float = 1.5) -> pd.DataFrame:
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Parameters:
            multiplier (float): IQR multiplier for boundary calculation
            
        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        mask = pd.DataFrame(False, index=self.data.index, columns=numeric_cols, dtype=bool)

        for col in numeric_cols:
            q1 = self.data[col].quantile(0.25)
            q3 = self.data[col].quantile(0.75)
            iqr = q3 - q1
            if not np.isnan(iqr) and iqr > 0:
                lower = q1 - multiplier * iqr
                upper = q3 + multiplier * iqr
                mask[col] = ((self.data[col] < lower) | (self.data[col] > upper)).fillna(False)

        self.outlier_masks['iqr'] = mask
        return mask
    
    def detect_outliers_isolation_forest(self, contamination: float = 0.1, 
                                         random_state: int = 42) -> pd.DataFrame:
        """
        Detect outliers using Isolation Forest algorithm.
        
        Parameters:
            contamination (float): Expected proportion of outliers in the dataset
            random_state (int): Random seed for reproducibility
            
        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        mask = pd.DataFrame(False, index=self.data.index, columns=numeric_cols, dtype=bool)

        if len(numeric_cols) > 0 and len(self.data) > 1:
            X = self.data[numeric_cols].fillna(self.data[numeric_cols].median())
            iso = IsolationForest(contamination=contamination, random_state=random_state)
            preds = iso.fit_predict(X)
            row_outliers = preds == -1
            for col in numeric_cols:
                mask[col] = row_outliers

        self.outlier_masks['isolation_forest'] = mask
        return mask
    
    def detect_outliers_lof(self, n_neighbors: int = 20, 
                            contamination: float = 0.1) -> pd.DataFrame:
        """
        Detect outliers using Local Outlier Factor (LOF) algorithm.
        
        Parameters:
            n_neighbors (int): Number of neighbors to consider
            contamination (float): Expected proportion of outliers
            
        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        mask = pd.DataFrame(False, index=self.data.index, columns=numeric_cols, dtype=bool)

        if len(numeric_cols) > 0 and len(self.data) > 1:
            X = self.data[numeric_cols].fillna(self.data[numeric_cols].median())
            effective_neighbors = min(n_neighbors, max(1, len(self.data) - 1))
            lof = LocalOutlierFactor(n_neighbors=effective_neighbors, contamination=contamination)
            preds = lof.fit_predict(X)
            row_outliers = preds == -1
            for col in numeric_cols:
                mask[col] = row_outliers

        self.outlier_masks['lof'] = mask
        return mask
    
    def get_outlier_summary(self) -> pd.DataFrame:
        """
        Get summary statistics of outliers detected by different methods.
        
        Returns:
            DataFrame with outlier counts for each method
        """
        summary_data = {}
        for method_name, mask in self.outlier_masks.items():
            summary_data[method_name] = mask.sum()
        return pd.DataFrame(summary_data)
    
    def remove_outliers(self, method: str = 'iqr') -> pd.DataFrame:
        """
        Remove outliers from the dataset.
        
        Parameters:
            method (str): Detection method to use
            
        Returns:
            pd.DataFrame: Dataset with outliers removed
        """
        if method not in self.outlier_masks:
            if method == 'zscore':
                self.detect_outliers_zscore()
            elif method == 'iqr':
                self.detect_outliers_iqr()
            elif method == 'isolation_forest':
                self.detect_outliers_isolation_forest()
            elif method == 'lof':
                self.detect_outliers_lof()
            else:
                raise ValueError(f"Unknown outlier detection method: '{method}'")

        mask = self.outlier_masks[method]
        rows_with_outliers = mask.any(axis=1)
        cleaned_df = self.data.loc[~rows_with_outliers].copy()
        self.treated_datasets[f'removed_{method}'] = cleaned_df
        return cleaned_df
    
    def cap_outliers(self, method: str = 'iqr', multiplier: float = 1.5) -> pd.DataFrame:
        """
        Cap outliers using winsorization.
        
        Parameters:
            method (str): Detection method to use
            multiplier (float): IQR multiplier for bounds
            
        Returns:
            pd.DataFrame: Dataset with capped outliers
        """
        capped_df = self.data.copy()
        numeric_cols = capped_df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            q1 = capped_df[col].quantile(0.25)
            q3 = capped_df[col].quantile(0.75)
            iqr = q3 - q1
            if not np.isnan(iqr) and iqr > 0:
                lower = q1 - multiplier * iqr
                upper = q3 + multiplier * iqr
                capped_df[col] = capped_df[col].clip(lower=lower, upper=upper)

        self.treated_datasets['capped'] = capped_df
        return capped_df
    
    def transform_outliers_log(self) -> pd.DataFrame:
        """
        Transform data using log transformation.
        
        Returns:
            pd.DataFrame: Dataset with log-transformed values
        """
        log_df = self.data.copy()
        numeric_cols = log_df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            col_min = log_df[col].min()
            if col_min <= 0:
                shift = abs(col_min) + 1.0
                log_df[col] = np.log1p(log_df[col] + shift)
            else:
                log_df[col] = np.log1p(log_df[col])

        self.treated_datasets['log_transform'] = log_df
        return log_df
    
    def transform_outliers_boxcox(self) -> pd.DataFrame:
        """
        Transform data using Box-Cox/Yeo-Johnson transformation.
        
        Returns:
            pd.DataFrame: Dataset with power-transformed values
        """
        boxcox_df = self.data.copy()
        numeric_cols = boxcox_df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 0 and len(boxcox_df) > 1:
            pt = PowerTransformer(method='yeo-johnson', standardize=False)
            boxcox_df[numeric_cols] = pt.fit_transform(boxcox_df[numeric_cols])

        self.treated_datasets['boxcox_transform'] = boxcox_df
        return boxcox_df
    
    def evaluate_model_performance(self, target_col: str, 
                                   dataset_name: str = 'original') -> Dict[str, float]:
        """
        Evaluate model performance with and without outliers.
        
        Args:
            target_col: Name of target column
            dataset_name: Dataset to evaluate ('original' or treated dataset name)
            
        Returns:
            Dictionary with performance metrics
        """
        if dataset_name == 'original':
            df = self.original_data.copy()
        elif dataset_name in self.treated_datasets:
            df = self.treated_datasets[dataset_name].copy()
        else:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        X = numeric_df.drop(columns=[target_col])
        y = numeric_df[target_col]

        cv_folds = min(5, len(numeric_df))
        model = LinearRegression()
        scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')

        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))

        return {
            'r2_score': mean_score,
            'r2_std': std_score,
            'mean_cv_score': mean_score,
        }
    
    def visualize_outliers_boxplot(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Create box plots showing outliers.
        
        Parameters:
            figsize (Tuple[int, int]): Figure size for plots
            
        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns[:6]
        if len(numeric_cols) == 0:
            print("No numeric columns to visualize")
            return None

        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.atleast_1d(axes).flatten()

        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            ax.boxplot(self.data[col].dropna(), patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7))
            ax.set_title(f'Box Plot: {col}', fontweight='bold')
            ax.set_ylabel(col)
            ax.grid(axis='y', alpha=0.3)

        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Outlier Detection - Box Plots', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def visualize_outliers_violin(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Create violin plots showing distribution with outliers.
        
        Args:
            figsize: Figure size for plots
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns[:6]
        if len(numeric_cols) == 0:
            print("No numeric columns to visualize")
            return None

        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.atleast_1d(axes).flatten()

        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            col_data = self.data[col].dropna()
            ax.violinplot(col_data, showmeans=True, showmedians=True)
            ax.set_title(f'Violin Plot: {col}', fontweight='bold')
            ax.set_ylabel(col)
            ax.grid(axis='y', alpha=0.3)

        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Outlier Distribution - Violin Plots', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def visualize_outliers_scatter(self, x_col: str, y_col: str, 
                                   figsize: Tuple[int, int] = (15, 5)):
        """
        Create scatter plots showing outliers detected by different methods.
        
        Args:
            x_col: Column for x-axis
            y_col: Column for y-axis
            figsize: Figure size for plots
        """
        if x_col not in self.data.columns or y_col not in self.data.columns:
            print(f"Columns '{x_col}' or '{y_col}' not found")
            return None

        methods = list(self.outlier_masks.keys())
        if not methods:
            self.detect_outliers_zscore()
            self.detect_outliers_iqr()
            methods = list(self.outlier_masks.keys())

        fig, axes = plt.subplots(1, len(methods), figsize=figsize)
        axes = np.atleast_1d(axes).flatten()

        for idx, method in enumerate(methods):
            ax = axes[idx]
            mask = self.outlier_masks[method].any(axis=1)
            ax.scatter(self.data.loc[~mask, x_col], self.data.loc[~mask, y_col],
                       c='steelblue', alpha=0.6, label='Normal')
            ax.scatter(self.data.loc[mask, x_col], self.data.loc[mask, y_col],
                       c='crimson', alpha=0.8, label='Outlier')
            ax.set_title(f'{method.upper()} Outliers', fontweight='bold')
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.legend()
            ax.grid(alpha=0.3)

        plt.tight_layout()
        return fig
    
    def visualize_3d_outliers(self, x_col: str, y_col: str, z_col: str, 
                             method: str = 'isolation_forest',
                             figsize: Tuple[int, int] = (12, 8)):
        """
        Create three-dimensional scatter plot showing multivariate outliers.
        
        Args:
            x_col, y_col, z_col: Columns for three-dimensional axes
            method: Detection method to use
            figsize: Figure size for plot
        """
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        if any(c not in self.data.columns for c in (x_col, y_col, z_col)):
            print("One or more columns not found")
            return None

        if method not in self.outlier_masks:
            print(f"Detection method '{method}' not found in outlier_masks")
            return None

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        mask = self.outlier_masks[method].any(axis=1)
        ax.scatter(self.data.loc[~mask, x_col], self.data.loc[~mask, y_col], self.data.loc[~mask, z_col],
                   c='steelblue', alpha=0.6, label='Normal')
        ax.scatter(self.data.loc[mask, x_col], self.data.loc[mask, y_col], self.data.loc[mask, z_col],
                   c='crimson', alpha=0.85, label='Outlier')

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
        ax.set_title(f'3D Outliers ({method.upper()})', fontweight='bold')
        ax.legend()
        plt.tight_layout()
        return fig
    
    def visualize_treatment_comparison(self, column: str, 
                                      figsize: Tuple[int, int] = (15, 10)):
        """
        Compare distributions before and after different treatments.
        
        Args:
            column: Column to visualize
            figsize: Figure size for plots
        """
        if column not in self.data.columns:
            print(f"Column '{column}' not found")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        # Original data
        ax = axes[0]
        self.original_data[column].hist(bins=30, ax=ax, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(self.original_data[column].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax.set_title('Original Data', fontweight='bold')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Treated datasets
        treatment_names = list(self.treated_datasets.keys())[:5]
        for idx, name in enumerate(treatment_names, start=1):
            ax = axes[idx]
            treated_data = self.treated_datasets[name]
            if column in treated_data.columns:
                treated_data[column].hist(bins=30, ax=ax, color='lightcoral', 
                                        edgecolor='black', alpha=0.7)
                ax.axvline(treated_data[column].mean(), color='red', 
                         linestyle='--', linewidth=2, label='Mean')
                ax.set_title(f'{name.replace("_", " ").title()}', fontweight='bold')
                ax.set_xlabel(column)
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(treatment_names) + 1, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Treatment Comparison: {column}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def visualize_qq_plots(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Create Q-Q plots to assess normality before and after treatment.
        
        Args:
            figsize: Figure size for plots
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns[:6]
        
        if len(numeric_cols) == 0:
            print("No numeric columns to visualize")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            
            stats.probplot(self.data[col].dropna(), dist="norm", plot=ax)
            ax.set_title(f'Q-Q Plot: {col}', fontweight='bold')
            ax.grid(alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Q-Q Plots for Normality Assessment', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig


def create_sample_dataset_with_outliers() -> pd.DataFrame:
    """
    Create a sample dataset with outliers for demonstration.
    
    Returns:
        pd.DataFrame: Dataset with outliers
    """
    np.random.seed(42)
    n_samples = 300
    
    # Create base data
    data = {
        'feature1': np.random.normal(100, 15, n_samples),
        'feature2': np.random.exponential(50, n_samples),
        'feature3': np.random.normal(500, 100, n_samples),
        'feature4': np.random.uniform(0, 100, n_samples),
        'target': np.random.normal(200, 30, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add synthetic target relationship
    df['target'] = 0.5 * df['feature1'] + 0.3 * df['feature2'] + np.random.normal(0, 10, n_samples)
    
    # Inject outliers
    # Extreme high values in feature1
    outlier_indices = np.random.choice(df.index, size=15, replace=False)
    df.loc[outlier_indices, 'feature1'] = np.random.uniform(200, 300, len(outlier_indices))
    
    # Extreme low values in feature2
    outlier_indices = np.random.choice(df.index, size=10, replace=False)
    df.loc[outlier_indices, 'feature2'] = np.random.uniform(200, 400, len(outlier_indices))
    
    # Random extreme values in feature3
    outlier_indices = np.random.choice(df.index, size=20, replace=False)
    # Split outliers into low and high
    low_outliers = outlier_indices[:len(outlier_indices)//2]
    high_outliers = outlier_indices[len(outlier_indices)//2:]
    df.loc[low_outliers, 'feature3'] = np.random.uniform(0, 100, len(low_outliers))
    df.loc[high_outliers, 'feature3'] = np.random.uniform(1000, 1500, len(high_outliers))
    
    return df


def main():
    """
    Main function demonstrating the complete Task 2 workflow.
    
    Returns:
        OutlierAnalyzer: Configured analyzer object
    """
    print("=" * 80)
    print("Task 2: Outlier Detection and Treatment with Advanced Visualization")
    print("=" * 80)
    print()
    
    # Create sample dataset
    print("Creating sample dataset with outliers...")
    df = create_sample_dataset_with_outliers()
    print(f"Dataset created: {df.shape[0]} rows, {df.shape[1]} columns")
    print()
    
    # Initialize analyzer
    analyzer = OutlierAnalyzer(df)
    
    # Detect outliers using different methods
    print("1. OUTLIER DETECTION")
    print("-" * 80)
    
    print("  a) Z-Score Detection...")
    zscore_outliers = analyzer.detect_outliers_zscore(threshold=3.0)
    print(f"     Outliers detected: {zscore_outliers.sum().sum()}")
    
    print("  b) IQR Detection...")
    iqr_outliers = analyzer.detect_outliers_iqr(multiplier=1.5)
    print(f"     Outliers detected: {iqr_outliers.sum().sum()}")
    
    print("  c) Isolation Forest Detection...")
    iso_outliers = analyzer.detect_outliers_isolation_forest(contamination=0.1)
    print(f"     Outliers detected: {iso_outliers.sum().sum()}")
    
    print("  d) Local Outlier Factor Detection...")
    lof_outliers = analyzer.detect_outliers_lof(n_neighbors=20, contamination=0.1)
    print(f"     Outliers detected: {lof_outliers.sum().sum()}")
    print()
    
    # Get outlier summary
    print("2. OUTLIER SUMMARY")
    print("-" * 80)
    summary = analyzer.get_outlier_summary()
    print(summary)
    print()
    
    # Apply different treatment strategies
    print("3. OUTLIER TREATMENT")
    print("-" * 80)
    
    print("  a) Removing outliers (IQR method)...")
    removed_data = analyzer.remove_outliers(method='iqr')
    print(f"     Rows after removal: {len(removed_data)} (removed {len(df) - len(removed_data)})")
    
    print("  b) Capping outliers (Winsorization)...")
    capped_data = analyzer.cap_outliers(multiplier=1.5)
    print(f"     Data capped at IQR bounds")
    
    print("  c) Log transformation...")
    log_data = analyzer.transform_outliers_log()
    print(f"     Log transformation applied")
    
    print("  d) Box-Cox transformation...")
    boxcox_data = analyzer.transform_outliers_boxcox()
    print(f"     Box-Cox/Yeo-Johnson transformation applied")
    print()
    
    # Evaluate model performance
    print("4. MODEL PERFORMANCE COMPARISON")
    print("-" * 80)
    
    datasets_to_evaluate = ['original', 'removed_iqr', 'capped', 'log_transform']
    for dataset_name in datasets_to_evaluate:
        try:
            performance = analyzer.evaluate_model_performance('target', dataset_name)
            print(f"  {dataset_name.replace('_', ' ').title()}:")
            print(f"    R² Score: {performance['r2_score']:.4f} (±{performance['r2_std']:.4f})")
        except Exception as e:
            print(f"  {dataset_name}: Could not evaluate - {str(e)}")
    print()
    
    # Generate visualizations
    print("5. GENERATING VISUALIZATIONS")
    print("-" * 80)
    
    print("  Creating box plots...")
    analyzer.visualize_outliers_boxplot()
    plt.savefig('task2_boxplots.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_boxplots.png")
    
    print("  Creating violin plots...")
    analyzer.visualize_outliers_violin()
    plt.savefig('task2_violin_plots.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_violin_plots.png")
    
    print("  Creating scatter plots with outlier detection...")
    analyzer.visualize_outliers_scatter('feature1', 'feature2')
    plt.savefig('task2_scatter_outliers.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_scatter_outliers.png")
    
    print("  Creating 3D outlier visualization...")
    analyzer.visualize_3d_outliers('feature1', 'feature2', 'feature3', method='isolation_forest')
    plt.savefig('task2_3d_outliers.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_3d_outliers.png")
    
    print("  Creating treatment comparison plots...")
    analyzer.visualize_treatment_comparison('feature1')
    plt.savefig('task2_treatment_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_treatment_comparison.png")
    
    print("  Creating Q-Q plots...")
    analyzer.visualize_qq_plots()
    plt.savefig('task2_qq_plots.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_qq_plots.png")
    print()
    
    print("=" * 80)
    print("Task 2 completed successfully!")
    print("=" * 80)
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()
    plt.show()