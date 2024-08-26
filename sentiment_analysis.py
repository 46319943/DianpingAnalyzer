import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer
model_name = "uer/roberta-base-finetuned-dianping-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)


def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    positive_score = probabilities[0][1].item()
    return positive_score


# Process the data
input_file = "Data/大众点评.jsonl"
output_file = "Data/大众点评_with_sentiment.jsonl"

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for i, line in enumerate(infile):
        comment = json.loads(line)
        sentiment_score = analyze_sentiment(comment["text"])
        comment["sentiment_score"] = sentiment_score
        json.dump(comment, outfile, ensure_ascii=False)
        outfile.write("\n")

        # Print progress every 100 comments
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} comments")

print("Sentiment analysis completed. Results saved in", output_file)