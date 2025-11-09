# ==========================================================
# data_analysis.py — Advanced Data Analysis Engine
# ==========================================================
"""
Comprehensive data analysis engine with visualization capabilities.
Supports CSV and Excel files with automatic insights generation.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd

# Configure matplotlib for non-interactive backend
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization defaults
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class DataAnalysisEngine:
    """
    Comprehensive data analysis with visualization and insights
    """

    def __init__(self, output_dir: str = "./analysis_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_df: Optional[pd.DataFrame] = None
        self.current_filename: str = ""

    def analyze_file(self, filepath: str) -> Dict[str, Any]:
        """
        Main entry point for file analysis
        """
        try:
            # Load data
            df = self._load_data(filepath)
            if df is None:
                return {"error": "Could not load data"}

            self.current_df = df
            self.current_filename = Path(filepath).name

            # Perform comprehensive analysis
            result = {
                "success": True,
                "filename": self.current_filename,
                "basic_info": self._get_basic_info(df),
                "statistics": self._get_statistics(df),
                "data_quality": self._check_data_quality(df),
                "insights": self._generate_insights(df),
                "visualizations": self._create_visualizations(df)
            }

            return result

        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

    def _load_data(self, filepath: str) -> Optional[pd.DataFrame]:
        """Load CSV or Excel file"""
        try:
            ext = Path(filepath).suffix.lower()

            if ext == '.csv':
                # Try different encodings
                for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                    try:
                        df = pd.read_csv(filepath, encoding=encoding)
                        return df
                    except UnicodeDecodeError:
                        continue

            elif ext in ['.xlsx', '.xls']:
                df = pd.read_excel(filepath)
                return df

            return None

        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def _get_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset information"""
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB"
        }

    def _get_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive statistics"""
        stats = {}

        # Numerical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            stats["numerical"] = {}
            for col in numeric_cols:
                stats["numerical"][col] = {
                    "mean": float(df[col].mean()),
                    "median": float(df[col].median()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "q25": float(df[col].quantile(0.25)),
                    "q75": float(df[col].quantile(0.75))
                }

        # Categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            stats["categorical"] = {}
            for col in categorical_cols[:5]:  # Limit to first 5
                value_counts = df[col].value_counts().head(10)
                stats["categorical"][col] = {
                    "unique_values": int(df[col].nunique()),
                    "top_values": {str(k): int(v) for k, v in value_counts.items()}
                }

        return stats

    def _check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data quality issues"""
        quality = {
            "missing_values": {},
            "duplicates": int(df.duplicated().sum()),
            "outliers": {}
        }

        # Missing values
        missing = df.isnull().sum()
        quality["missing_values"] = {
            col: int(count)
            for col, count in missing.items()
            if count > 0
        }

        # Outliers (for numerical columns using IQR method)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            if len(outliers) > 0:
                quality["outliers"][col] = len(outliers)

        return quality

    def _generate_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate automatic insights"""
        insights = []

        # Dataset size insight
        insights.append(f"Dataset contains {len(df):,} rows and {len(df.columns)} columns")

        # Missing data insight
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_pct > 5:
            insights.append(f"⚠️ Dataset has {missing_pct:.1f}% missing values")
        elif missing_pct == 0:
            insights.append("✅ No missing values detected")

        # Numerical insights
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            insights.append(f"Found {len(numeric_cols)} numerical columns")

            # Check for high correlation
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                high_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > 0.8:
                            high_corr.append(
                                (corr_matrix.columns[i],
                                 corr_matrix.columns[j],
                                 corr_matrix.iloc[i, j])
                            )
                if high_corr:
                    for col1, col2, corr in high_corr[:3]:
                        insights.append(
                            f"🔗 Strong correlation ({corr:.2f}) between '{col1}' and '{col2}'"
                        )

        # Categorical insights
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            insights.append(f"Found {len(categorical_cols)} categorical columns")

            for col in categorical_cols[:3]:
                unique_count = df[col].nunique()
                if unique_count < 10:
                    insights.append(
                        f"'{col}' has {unique_count} unique categories"
                    )

        # Duplicates insight
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            insights.append(f"⚠️ Found {dup_count} duplicate rows")

        return insights

    def _create_visualizations(self, df: pd.DataFrame) -> List[str]:
        """Create and save visualizations"""
        viz_paths = []

        try:
            # 1. Numerical distributions
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                viz_path = self._plot_distributions(df, numeric_cols[:4])
                if viz_path:
                    viz_paths.append(viz_path)

            # 2. Correlation heatmap
            if len(numeric_cols) > 1:
                viz_path = self._plot_correlation(df, numeric_cols)
                if viz_path:
                    viz_paths.append(viz_path)

            # 3. Categorical value counts
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_cols:
                viz_path = self._plot_categorical(df, categorical_cols[:3])
                if viz_path:
                    viz_paths.append(viz_path)

            # 4. Missing data visualization
            if df.isnull().sum().sum() > 0:
                viz_path = self._plot_missing_data(df)
                if viz_path:
                    viz_paths.append(viz_path)

        except Exception as e:
            print(f"Visualization error: {e}")

        return viz_paths

    def _plot_distributions(self, df: pd.DataFrame, columns: List[str]) -> Optional[str]:
        """Plot distributions for numerical columns"""
        try:
            n_cols = len(columns)
            if n_cols == 0:
                return None

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()

            for idx, col in enumerate(columns):
                if idx >= 4:
                    break
                ax = axes[idx]
                df[col].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
                ax.set_title(f'Distribution: {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                ax.grid(alpha=0.3)

            # Hide unused subplots
            for idx in range(n_cols, 4):
                axes[idx].axis('off')

            plt.tight_layout()

            filepath = self.output_dir / f"distributions_{self._timestamp()}.png"
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close()

            return str(filepath)

        except Exception as e:
            print(f"Distribution plot error: {e}")
            plt.close()
            return None

    def _plot_correlation(self, df: pd.DataFrame, columns: List[str]) -> Optional[str]:
        """Plot correlation heatmap"""
        try:
            if len(columns) < 2:
                return None

            corr_matrix = df[columns].corr()

            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                        center=0, square=True, linewidths=1)
            plt.title('Correlation Heatmap')
            plt.tight_layout()

            filepath = self.output_dir / f"correlation_{self._timestamp()}.png"
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close()

            return str(filepath)

        except Exception as e:
            print(f"Correlation plot error: {e}")
            plt.close()
            return None

    def _plot_categorical(self, df: pd.DataFrame, columns: List[str]) -> Optional[str]:
        """Plot categorical value counts"""
        try:
            n_cols = len(columns)
            if n_cols == 0:
                return None

            fig, axes = plt.subplots(n_cols, 1, figsize=(12, 4 * n_cols))
            if n_cols == 1:
                axes = [axes]

            for idx, col in enumerate(columns):
                value_counts = df[col].value_counts().head(10)
                value_counts.plot(kind='bar', ax=axes[idx], color='steelblue', edgecolor='black')
                axes[idx].set_title(f'Top Values: {col}')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Count')
                axes[idx].tick_params(axis='x', rotation=45)
                axes[idx].grid(alpha=0.3)

            plt.tight_layout()

            filepath = self.output_dir / f"categorical_{self._timestamp()}.png"
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close()

            return str(filepath)

        except Exception as e:
            print(f"Categorical plot error: {e}")
            plt.close()
            return None

    def _plot_missing_data(self, df: pd.DataFrame) -> Optional[str]:
        """Visualize missing data"""
        try:
            missing = df.isnull().sum()
            missing = missing[missing > 0].sort_values(ascending=False)

            if len(missing) == 0:
                return None

            plt.figure(figsize=(10, 6))
            missing.plot(kind='bar', color='coral', edgecolor='black')
            plt.title('Missing Values by Column')
            plt.xlabel('Column')
            plt.ylabel('Missing Count')
            plt.xticks(rotation=45, ha='right')
            plt.grid(alpha=0.3)
            plt.tight_layout()

            filepath = self.output_dir / f"missing_data_{self._timestamp()}.png"
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close()

            return str(filepath)

        except Exception as e:
            print(f"Missing data plot error: {e}")
            plt.close()
            return None

    def _timestamp(self) -> str:
        """Generate timestamp for filenames"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def query_data(self, query: str) -> Dict[str, Any]:
        """
        Answer questions about the current dataset
        """
        if self.current_df is None:
            return {"error": "No data loaded"}

        query_lower = query.lower()
        df = self.current_df

        try:
            # Basic queries
            if "how many rows" in query_lower or "row count" in query_lower:
                return {"answer": f"The dataset has {len(df):,} rows"}

            elif "how many columns" in query_lower or "column count" in query_lower:
                return {"answer": f"The dataset has {len(df.columns)} columns"}

            elif "column names" in query_lower or "what columns" in query_lower:
                return {"answer": f"Columns: {', '.join(df.columns.tolist())}"}

            elif "missing" in query_lower:
                missing = df.isnull().sum()
                missing_cols = missing[missing > 0]
                if len(missing_cols) > 0:
                    return {"answer": f"Missing values in: {missing_cols.to_dict()}"}
                else:
                    return {"answer": "No missing values"}

            elif "summary" in query_lower or "describe" in query_lower:
                numeric_summary = df.describe().to_dict()
                return {"answer": "Statistical summary", "data": numeric_summary}

            else:
                return {"answer": "Try asking about: row count, columns, missing values, or summary statistics"}

        except Exception as e:
            return {"error": f"Query failed: {str(e)}"}


# ==========================================================
# Usage Example
# ==========================================================
if __name__ == "__main__":
    engine = DataAnalysisEngine()

    # Test with sample data
    sample_data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'Age': [25, 30, 35, 28, 32],
        'Salary': [50000, 60000, 75000, 55000, 70000],
        'Department': ['HR', 'IT', 'Finance', 'IT', 'HR']
    }

    df = pd.DataFrame(sample_data)
    temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)

    result = engine.analyze_file(temp_file.name)
    print("Analysis Result:", result)

    os.unlink(temp_file.name)