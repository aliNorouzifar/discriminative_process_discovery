import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import shap
import matplotlib.pyplot as plt

# Combine positive and negative samples
def prepare_data(pivot_df_P, pivot_df_M):
    """
    Prepares the dataset by combining positive and negative samples, assigning labels, and splitting into features and target.

    Parameters:
        pivot_df_P (pd.DataFrame): Positive samples DataFrame.
        pivot_df_M (pd.DataFrame): Negative samples DataFrame.

    Returns:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target labels.
    """
    pivot_df_P = pivot_df_P.copy()
    pivot_df_M = pivot_df_M.copy()

    pivot_df_P['label'] = 1  # Label positive samples as 1
    pivot_df_M['label'] = 0  # Label negative samples as 0

    combined_df = pd.concat([pivot_df_P, pivot_df_M], ignore_index=True)

    # Separate features and target
    X = combined_df.drop(columns=['label'])
    y = combined_df['label']

    # Encode only "satisfied" and "violated" values in each column
    def encode_satisfied_violated(df):
        """
        Efficiently encodes columns containing "satisfied", "violated", and "vacsatisfied"
        into two binary columns for "satisfied" and "violated".
        """
        encoded_columns = []  # Store encoded columns to concatenate later
        for column in df.columns:
            encoded_columns.append((df[column] == "satisfied").astype(int).rename(column + "_satisfied"))
            encoded_columns.append((df[column] == "violated").astype(int).rename(column + "_violated"))
        # Concatenate all columns at once
        return pd.concat(encoded_columns, axis=1)

    X = encode_satisfied_violated(X)

    # # Convert non-numeric columns to numeric using one-hot encoding
    # X = pd.get_dummies(X, drop_first=True)

    return X, y

# Train random forest, evaluate, and explain with SHAP values
def train_and_evaluate_with_shap(pivot_df_P, pivot_df_M):
    """
    Trains a Random Forest classifier, evaluates it, and uses SHAP to visualize feature importance.

    Parameters:
        pivot_df_P (pd.DataFrame): Positive samples DataFrame.
        pivot_df_M (pd.DataFrame): Negative samples DataFrame.
    """
    X, y = prepare_data(pivot_df_P, pivot_df_M)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Train Random Forest Classifier
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # Predict on test data
    y_pred = clf.predict(X_test)

    # Evaluate using confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    # Explain the model with SHAP values
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_train)

    # shap.summary_plot(shap_values, X_train, plot_type="bar", show=True)

    # shap.summary_plot(shap_values[0].T, X_train, plot_type='dot')
    shap.summary_plot(shap_values[..., 0], X_train, plot_type='dot')


# Example usage
# train_and_evaluate_with_shap(pivot_df_P, pivot_df_M)
