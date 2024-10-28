import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, OrdinalEncoder, LabelEncoder
from category_encoders import TargetEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class AutoFeatureEngineering:
    def __init__(self, target_col, null_strategy='mean', encoding_method='onehot'):
        """
        Initialize the feature engineering class with strategies for handling 
        missing values and encoding categorical features.

        Parameters:
            target_col (str): The target column for supervised learning.
            null_strategy (str): Strategy to handle missing values ('mean', 'median', 'drop').
            encoding_method (str): Encoding method for categorical features ('onehot', 'ordinal', 'target').
        """
        self.target_col = target_col
        self.null_strategy = null_strategy
        self.encoding_method = encoding_method
        self.categorical_cols = []
        self.numerical_cols = []
        self.label_encoder = LabelEncoder()  # Store the label encoder for the target column

    def detect_column_types(self, df):
        """Detect and classify columns by type."""
        self.categorical_cols = df.select_dtypes(include=['object']).columns.drop(self.target_col, errors='ignore')
        self.numerical_cols = df.select_dtypes(include=['number']).columns

    def handle_nulls(self, df):
        """Handle missing values for both features and the target column."""
        # Handle missing values for features
        if self.null_strategy == 'mean':
            df = df.fillna(df.mean(numeric_only=True))
        elif self.null_strategy == 'median':
            df = df.fillna(df.median(numeric_only=True))
        elif self.null_strategy == 'drop':
            df = df.dropna()

        # Handle missing values in the target column separately
        if df[self.target_col].isnull().sum() > 0:
            if df[self.target_col].dtype == 'object':
                # For categorical target column, fill with mode
                df[self.target_col] = df[self.target_col].fillna(df[self.target_col].mode()[0])
            else:
                # For numerical target column, fill with mean
                df[self.target_col] = df[self.target_col].fillna(df[self.target_col].mean())
        return df

    def encode_categorical(self, df):
        """Encode categorical features, including label encoding the target column."""
        # Label encode the target column if it is categorical
        if df[self.target_col].dtype == 'object':
            df[self.target_col] = self.label_encoder.fit_transform(df[self.target_col])
    
        # Encode the other categorical features based on the chosen encoding method
        categorical_features = self.categorical_cols.drop(self.target_col, errors='ignore')
    
        if self.encoding_method == 'onehot':
            df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
        elif self.encoding_method == 'ordinal':
            encoder = OrdinalEncoder()
            df[categorical_features] = encoder.fit_transform(df[categorical_features])
        elif self.encoding_method == 'target':
            encoder = TargetEncoder()
            df[categorical_features] = encoder.fit_transform(df[categorical_features], df[self.target_col])
    
        return df


    def engineer_features(self, df):
        """Generate polynomial features from numerical columns, excluding the target."""
        # Exclude the target column from polynomial transformation
        numerical_features = self.numerical_cols.drop(self.target_col, errors='ignore')
    
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_features = poly.fit_transform(df[numerical_features])
    
        # Create a DataFrame for the polynomial features
        poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(numerical_features))
    
        # Concatenate original features and polynomial features, avoiding duplicates
        df = pd.concat([df.reset_index(drop=True), poly_df.reset_index(drop=True)], axis=1)
        df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicate columns
    
        return df

    def rank_features(self, df):
        """Rank features by importance using RandomForest."""
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        if y.dtype in ['int64', 'float64'] and len(y.unique()) < 20:  # Check for discrete values
            model = RandomForestClassifier(random_state=42)
        else:  # Assume continuous if not discrete
            model = RandomForestRegressor(random_state=42)

        model.fit(X, y)
        feature_importances = pd.Series(model.feature_importances_, index=X.columns)
        return feature_importances.sort_values(ascending=False)

    def fit_transform(self, df):
        """Apply the complete feature engineering pipeline."""
        # Detect column types
        self.detect_column_types(df)

        # Handle missing values for both features and target
        df = self.handle_nulls(df)

        # Encode categorical features (excluding the target column)
        df = self.encode_categorical(df)

        # Generate polynomial features
        df = self.engineer_features(df)

        # Rank and select relevant features, ensure the target column is retained
        feature_importances = self.rank_features(df)
        relevant_features = feature_importances[feature_importances > 0.01].index

        # Ensure the target column is included
        if self.target_col not in relevant_features:
            relevant_features = relevant_features.append(pd.Index([self.target_col]))

        df = df[relevant_features]
        return df