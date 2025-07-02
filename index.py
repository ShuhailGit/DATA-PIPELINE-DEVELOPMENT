import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Step 1: Load Data
def load_data(file_path):
    """Load data from a CSV file."""
    data = pd.read_csv(file_path)
    return data

# Step 2: Preprocess Data
def preprocess_data(data, target_column):
    """Preprocess the data by handling missing values and encoding categorical variables."""
    
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Create preprocessing pipelines
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ])
    
    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)
    
    return X_processed, y

# Step 3: Split Data
def split_data(X, y, test_size=0.2, random_state=42):
    """Split the data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Step 4: Main Function to Run the Pipeline
def main():
    # Load the data
    data = load_data('data/input_data.csv')
    
    # Preprocess the data
    target_column = 'target'  # Specify your target column
    X, y = preprocess_data(data, target_column)
    
    # Split the data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Output the shapes of the resulting datasets
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, {y_test.shape}")

if __name__ == "__main__":
    main()
