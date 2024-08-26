import json
import pkuseg
from typing import List, Dict
from gensim import corpora
from gensim.models import LdaMulticore, CoherenceModel
import matplotlib.pyplot as plt
import pyLDAvis.gensim_models
import numpy as np


# Step 1: Data Loading
def load_data(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]


# Step 2: Text Preprocessing
def preprocess_text(text: str, seg_model: pkuseg.pkuseg, valid_tags: List[str], stop_words: List[str]) -> List[str]:
    segmented = seg_model.cut(text)
    return [word for word, tag in segmented if tag in valid_tags and word not in stop_words]


def process_comments(comments: List[Dict]) -> List[Dict]:
    seg_model = pkuseg.pkuseg(postag=True)
    valid_tags = ['a', 'ad', 'j', 'l', 'n', 'ns', 'nt', 'nz', 'v', 'vd', 'vn']
    stop_words = ['黄鹤楼']  # Add more stop words as needed

    processed_comments = []
    for comment in comments:
        processed_text = preprocess_text(comment['text'], seg_model, valid_tags, stop_words)
        if processed_text:
            comment['processed_text'] = processed_text
            processed_comments.append(comment)

    # Save processed comments
    with open('Data/processed_comments.json', 'w', encoding='utf-8') as f:
        json.dump(processed_comments, f, ensure_ascii=False, indent=2)

    return processed_comments


# Step 3: Corpus and Dictionary Generation
def create_corpus_and_dictionary(processed_comments: List[Dict]):
    texts = [comment['processed_text'] for comment in processed_comments]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Save dictionary and corpus
    dictionary.save('Data/comment_dictionary.gensim')
    corpora.MmCorpus.serialize('Data/comment_corpus.mm', corpus)

    return dictionary, corpus


# Step 4 & 5: LDA Model Training and Evaluation
def compute_coherence_values(corpus, dictionary, texts, start, limit, step):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=100,
                             workers=3, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


def plot_coherence_values(start, limit, step, coherence_values):
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.title("Coherence Scores by Number of Topics")
    plt.savefig('Data/coherence_plot.png')
    plt.close()


# Step 6: Topic Visualization
def visualize_topics(lda_model, corpus, dictionary):
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, 'Data/lda_visualization.html')


# New function to assign topic probabilities to comments
def assign_topic_probabilities(lda_model, corpus, comments):
    topic_probabilities = lda_model.get_document_topics(corpus, minimum_probability=0)
    for comment, probs in zip(comments, topic_probabilities):
        comment['topic_probabilities'] = {str(topic): prob for topic, prob in probs}
    return comments


# Main execution
if __name__ == "__main__":
    # Load data
    comments = load_data('Data/大众点评.jsonl')
    print(f"Loaded {len(comments)} comments")

    # Process comments
    processed_comments = process_comments(comments)
    print(f"Processed {len(processed_comments)} non-empty comments")

    # Create and save corpus and dictionary
    dictionary, corpus = create_corpus_and_dictionary(processed_comments)
    print(f"Created dictionary with {len(dictionary)} words")
    print(f"Created corpus with {len(corpus)} documents")

    # Compute coherence values and train models
    start, limit, step = 5, 50, 5
    model_list, coherence_values = compute_coherence_values(corpus, dictionary,
                                                            [comment['processed_text'] for comment in
                                                             processed_comments],
                                                            start, limit, step)

    # Plot coherence values
    plot_coherence_values(start, limit, step, coherence_values)
    print("Coherence values computed and plotted")

    # Find the model with the highest coherence score
    best_model = model_list[np.argmax(coherence_values)]
    print(f"Optimal number of topics: {best_model.num_topics}")

    # Save the best model
    best_model.save('Data/best_lda_model.gensim')
    print("Best LDA model saved")

    # Visualize the topics
    visualize_topics(best_model, corpus, dictionary)
    print("Topic visualization saved")

    # Assign topic probabilities to comments
    processed_comments = assign_topic_probabilities(best_model, corpus, processed_comments)

    # Save updated processed comments with topic probabilities
    with open('Data/processed_comments_with_topics.json', 'w', encoding='utf-8') as f:
        json.dump(processed_comments, f, ensure_ascii=False, indent=2)
    print("Updated processed comments saved with topic probabilities")

    print("LDA analysis complete")