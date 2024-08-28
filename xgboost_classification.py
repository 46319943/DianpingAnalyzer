import json
from collections import OrderedDict

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import xgboost as xgb
import shap
import matplotlib.pyplot as plt


# Step 1: Data Merging
def merge_data():
    # Read JSON file
    with open('Data/processed_comments_with_topics.json', 'r', encoding='UTF-8') as f:
        topics_data = json.load(f)

    # Convert to DataFrame
    topics_df = pd.DataFrame(topics_data)
    topics_df['id'] = topics_df['id'].astype(str)  # Convert id to string
    topics_df.set_index('id', inplace=True)

    # Read JSONL file
    sentiment_df = pd.read_json('Data/大众点评_with_sentiment.jsonl', lines=True)
    sentiment_df['id'] = sentiment_df['id'].astype(str)  # Convert id to string
    sentiment_df.set_index('id', inplace=True)

    # Merge dataframes
    merged_df = pd.merge(sentiment_df, topics_df[['topic_probabilities']],
                         left_index=True, right_index=True,
                         how='left', validate='1:1')

    # Save merged data
    merged_df.to_json('Data/merged_comments_for_analysis.jsonl', orient='records', lines=True)

    print("Data merged and saved to 'Data/merged_comments_for_analysis.jsonl'")
    print(f"Number of rows in merged data: {len(merged_df)}")


# Step 2: Data Preparation
def prepare_data():
    # Load merged data
    df = pd.read_json('Data/merged_comments_for_analysis.jsonl', lines=True)

    # Prepare X (features)
    X = df[['sentiment_score', 'topic_probabilities']]
    X['topic_probabilities'] = X['topic_probabilities'].apply(
        lambda x: x if isinstance(x, list) else [0] * 5)  # Assuming 5 topics

    # Expand topic probabilities and assign labels
    topic_probs = pd.DataFrame(X['topic_probabilities'].tolist(),
                               columns=[f'Topic {i + 1} Probability' for i in range(5)])

    X = pd.concat([X.drop('topic_probabilities', axis=1), topic_probs], axis=1)

    # Prepare y (target)
    def categorize_rank(rank):
        if rank <= 20:
            return 'Low Rating'
        elif rank < 40:
            return 'Medium Rating'
        else:
            return 'High Rating'

    y = df['rank'].apply(categorize_rank)

    # Print category counts
    print("Category counts:")
    print(y.value_counts())

    # Custom ordered mapping for encoding
    label_mapping = OrderedDict([
        ('Low Rating', 0),
        ('Medium Rating', 1),
        ('High Rating', 2)
    ])

    # Encode target variable
    y_encoded = y.map(label_mapping)

    return X, y_encoded, label_mapping


# Step 3: XGBoost Classification
# Step 3: XGBoost Classification
def xgboost_classification(X, y, label_mapping):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train XGBoost classifier
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, n_jobs=-1)  # n_jobs=-1 for multicore
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Print evaluation metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=list(label_mapping.keys())))

    return model, X_test


# Step 4: SHAP Analysis
def shap_analysis(model, X_test):
    # Compute SHAP values using the TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    shap_value_array = shap_values.values
    # Convert the last dimension to list
    shap_values_list = []
    for i in range(shap_value_array.shape[-1]):
        shap_values_list.append(shap_value_array[:, :, i])

    # Summary Plot
    shap.summary_plot(shap_values_list, X_test, plot_type="bar", show=False)
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png')
    plt.close()

    print()





# Main execution
if __name__ == "__main__":
    merge_data()
    X, y, label_mapping = prepare_data()
    model, X_test = xgboost_classification(X, y, label_mapping)
    shap_analysis(model, X_test)
