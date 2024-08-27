import json
import logging
import os
import re
from typing import Dict, List, Tuple
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='inference.log',
                    filemode='a')
logger = logging.getLogger(__name__)

# Add a stream handler to also log to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Updated hard-coded parameters
INPUT_FILE = 'Data/大众点评.jsonl'
OUTPUT_FILE = 'Data/大众点评_with_sensory_experiences.jsonl'
LORA_PATH = 'PiaoYangHF/llama-3-1-8b-chat-lora-MEE'
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SYSTEM_PROMPT_FILE = 'Data/role_play.txt'
TASK_PROMPT_FILE = 'Data/task_prompt.txt'
BATCH_SIZE = 1000


def setup_model_and_tokenizer(base_model_name: str) -> Tuple[LLM, AutoTokenizer]:
    llm = LLM(model=base_model_name, dtype="bfloat16", gpu_memory_utilization=0.9, enable_lora=True,
              max_num_batched_tokens=8192, max_model_len=8192)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return llm, tokenizer


def read_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def read_txt(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def generate_input(system_prompt: str, task_prompt: str, text: str) -> List[Dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{task_prompt}\n\n{text}"}
    ]


def generate_response_batch(llm: LLM, tokenizer: AutoTokenizer, batch_messages: List[List[Dict]], lora_path: str) -> \
List[str]:
    # Prepare input texts
    texts = [tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) for messages in
             batch_messages]

    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=4096,
        seed=0,
    )

    # Generate responses with LoRA
    outputs = llm.generate(
        texts,
        sampling_params,
        lora_request=LoRARequest("custom_lora", 1, lora_path)
    )

    # Extract generated texts
    return [output.outputs[0].text for output in outputs]


def extract_annotations(text: str) -> List[str]:
    pattern = r'\[(.*?)\]'
    matches = re.findall(pattern, text)
    valid_annotations = []
    for match in matches:
        try:
            if '：' in match and '<' in match and '>' in match:
                valid_annotations.append(match)
            else:
                raise ValueError(f"Invalid annotation format: {match}")
        except Exception as e:
            logger.error(f"Error processing annotation: {str(e)}")
    return valid_annotations


def parse_annotations(annotations: str) -> Dict:
    key_mapping = {
        '感官': 'sense',
        '刺激物': 'stimulus',
        '方面': 'aspect',
        '感知': 'perception',
        '情感': 'sentiment'
    }
    sense_mapping = {
        '视觉': 'Sight',
        '听觉': 'Sound',
        '嗅觉': 'Smell',
        '味觉': 'Taste',
        '触觉': 'Touch'
    }
    sentiment_mapping = {
        '积极': 'Positive',
        '中立': 'Neutral',
        '消极': 'Negative'
    }

    result = {}
    try:
        for item in annotations.split('><'):
            key, value = item.strip('<>').split('：')
            eng_key = key_mapping.get(key, key)
            if eng_key == 'sense':
                value = sense_mapping.get(value, value)
            elif eng_key == 'sentiment':
                value = sentiment_mapping.get(value, value)
            result[eng_key] = value
    except Exception as e:
        logger.error(f"Error parsing annotation {annotations}: {str(e)}")
        return {}
    return result


def process_response(response: str) -> List[Dict]:
    extracted = extract_annotations(response)
    return [parse_annotations(ann) for ann in extracted]


def save_progress(output_file: str, results: List[Dict]):
    with open(output_file, 'a', encoding='utf-8') as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')


def load_progress(output_file: str) -> int:
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    return 0


def process_batch(llm: LLM, tokenizer: AutoTokenizer, batch: List[Dict], lora_path: str, system_prompt: str,
                  task_prompt: str) -> List[Dict]:
    batch_messages = [generate_input(system_prompt, task_prompt, item['text']) for item in batch]
    responses = generate_response_batch(llm, tokenizer, batch_messages, lora_path)
    results = []
    for item, response in zip(batch, responses):
        annotations = process_response(response)
        results.append({
            'text': item['text'],
            'response': response,
            'annotations': annotations
        })
    return results


def visualize_sense_proportion(annotations: List[Dict]):
    senses = [ann['sense'] for result in annotations for ann in result['annotations'] if 'sense' in ann]
    sense_counts = Counter(senses)

    plt.figure(figsize=(10, 10))
    plt.pie(sense_counts.values(), labels=sense_counts.keys(), autopct='%1.1f%%')
    plt.title('Proportion of Senses')
    plt.savefig('Output/sense_proportion.png')
    plt.close()


def create_word_cloud(words: List[str], title: str, output_file: str):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(words))

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.savefig(output_file)
    plt.close()


def visualize_word_clouds(annotations: List[Dict]):
    stimuli = [ann['stimulus'] for result in annotations for ann in result['annotations'] if 'stimulus' in ann]
    perceptions = [ann['perception'] for result in annotations for ann in result['annotations'] if 'perception' in ann]

    create_word_cloud(stimuli, 'Stimulus Word Cloud', 'Data/stimulus_wordcloud.png')
    create_word_cloud(perceptions, 'Perception Word Cloud', 'Data/perception_wordcloud.png')


def visualize_aspect_statistics(annotations: List[Dict]):
    aspects = [ann['aspect'] for result in annotations for ann in result['annotations'] if 'aspect' in ann]
    aspect_counts = Counter(aspects)

    plt.figure(figsize=(12, 6))
    plt.bar(aspect_counts.keys(), aspect_counts.values())
    plt.title('Aspect Statistics')
    plt.xlabel('Aspect')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('Data/aspect_statistics.png')
    plt.close()


def main():
    logger.info("Starting inference process")

    llm, tokenizer = setup_model_and_tokenizer(BASE_MODEL)
    logger.info(f"Model and tokenizer set up with base model: {BASE_MODEL}")

    system_prompt = read_txt(SYSTEM_PROMPT_FILE)
    task_prompt = read_txt(TASK_PROMPT_FILE)
    logger.info("Loaded system and task prompts")

    input_data = read_jsonl(INPUT_FILE)
    logger.info(f"Loaded {len(input_data)} items from {INPUT_FILE}")

    processed_count = load_progress(OUTPUT_FILE)
    logger.info(f"Loaded {processed_count} existing results")

    all_results = []

    for i in range(processed_count, len(input_data), BATCH_SIZE):
        batch = input_data[i:i + BATCH_SIZE]
        logger.info(f"Processing batch {i // BATCH_SIZE + 1}, items {i} to {min(i + BATCH_SIZE, len(input_data))}")

        batch_results = process_batch(llm, tokenizer, batch, LORA_PATH, system_prompt, task_prompt)
        save_progress(OUTPUT_FILE, batch_results)
        all_results.extend(batch_results)
        logger.info(f"Saved progress. Total processed: {i + len(batch_results)}")

    logger.info("Inference process completed")

    # Visualizations
    logger.info("Generating visualizations")
    visualize_sense_proportion(all_results)
    visualize_word_clouds(all_results)
    visualize_aspect_statistics(all_results)
    logger.info("Visualizations completed")


if __name__ == "__main__":
    main()
