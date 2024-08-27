import json
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
        lambda x: x if isinstance(x, list) else [0] * 10)  # Assuming 10 topics

    # Expand topic probabilities and assign labels
    topic_probs = pd.DataFrame(X['topic_probabilities'].tolist(),
                               columns=[f'Topic {i + 1} Probability' for i in range(10)])

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

    # Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return X, y_encoded, le


# Step 3: XGBoost Classification
def xgboost_classification(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train XGBoost classifier
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, n_jobs=-1)  # n_jobs=-1 for multicore
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Print evaluation metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low Rating', 'Medium Rating', 'High Rating']))

    return model, X_test


# Step 4: SHAP Analysis
def shap_analysis(model, X_test):
    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # 1. Summary plot (bar)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('Data/shap_summary_bar.png')
    plt.close()

    # 2. Summary plot (dot)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig('Data/shap_summary_dot.png')
    plt.close()

    # 3. Beeswarm plot
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_values[1], show=False)  # Assuming class 1 (index 0 is for class 0)
    plt.tight_layout()
    plt.savefig('Data/shap_beeswarm.png')
    plt.close()

    # 4. Force plot for a single prediction
    plt.figure(figsize=(20, 3))
    shap.force_plot(explainer.expected_value[1], shap_values[1][0, :], X_test.iloc[0, :], show=False, matplotlib=True)
    plt.tight_layout()
    plt.savefig('Data/shap_force_plot.png')
    plt.close()

    # 5. Decision plot
    plt.figure(figsize=(10, 6))
    shap.decision_plot(explainer.expected_value[1], shap_values[1], X_test, show=False)
    plt.tight_layout()
    plt.savefig('Data/shap_decision_plot.png')
    plt.close()

    print("SHAP visualizations saved in Data folder")


# Main execution
if __name__ == "__main__":
    merge_data()
    X, y, le = prepare_data()
    model, X_test = xgboost_classification(X, y)
    shap_analysis(model, X_test)
