import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt


def setup_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = "uer/roberta-base-finetuned-dianping-chinese"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    return device, tokenizer, model


def analyze_sentiment(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    positive_score = probabilities[0][1].item()
    return positive_score


def visualize_sentiment_scores(sentiment_scores):
    plt.figure(figsize=(12, 5))

    # Box plot
    plt.subplot(1, 2, 1)
    plt.boxplot(sentiment_scores)
    plt.title("Box Plot of Sentiment Scores")
    plt.ylabel("Sentiment Score")

    # Histogram
    plt.subplot(1, 2, 2)
    plt.hist(sentiment_scores, bins=20, edgecolor='black')
    plt.title("Histogram of Sentiment Scores")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("Output/sentiment_visualization.png")
    print("Visualization saved as sentiment_visualization.png")


def main():
    device, tokenizer, model = setup_model()

    input_file = "Data/大众点评.jsonl"
    output_file = "Data/大众点评_with_sentiment.jsonl"

    sentiment_scores = []

    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for i, line in enumerate(infile):
            comment = json.loads(line)
            sentiment_score = analyze_sentiment(comment["text"], tokenizer, model, device)
            comment["sentiment_score"] = sentiment_score
            json.dump(comment, outfile, ensure_ascii=False)
            outfile.write("\n")

            sentiment_scores.append(sentiment_score)

            # Print progress every 100 comments
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} comments")

    print("Sentiment analysis completed. Results saved in", output_file)

    # Visualize the sentiment scores
    visualize_sentiment_scores(sentiment_scores)


if __name__ == "__main__":
    main()