import json
import os
import argparse
from openai import OpenAI
from tqdm import tqdm


def align_check(client, model_reason, reference_reason):
    prompt = f"""Given two statements are reasons why one UI is better than another in terms of guiding user behavior effectively.

Statement 1:
{model_reason}

Statement 2:
{reference_reason}

Your task is to check whether the first statement contains the same reasoning as the second statement.
If it does, answer with 'Yes'. Otherwise, answer with 'No'.
Just answer with 'Yes' or 'No'."""

    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def run_align_check(api_key, results_dir, data_path, models, methods):
    client = OpenAI(api_key=api_key)

    with open(data_path) as f:
        wiserui_data = json.load(f)

    for model in models:
        model_dir = os.path.join(results_dir, model.replace('-', '_'))
        for method in methods:
            input_path = os.path.join(model_dir, f'{method}.json')
            if not os.path.exists(input_path):
                print(f'Skipping {model}/{method}: file not found')
                continue

            print(f'Processing: {model}, {method}')
            with open(input_path) as f:
                model_answers = json.load(f)

            output_path = os.path.join(model_dir, f'align_check_{method}.json')
            if os.path.exists(output_path):
                with open(output_path) as f:
                    new_data = json.load(f)
            else:
                new_data = []

            done = {(d['index'], d['r_index']) for d in new_data}

            for i, item in tqdm(enumerate(wiserui_data), total=len(wiserui_data)):
                model_reason = model_answers[i]['reason']

                for j, rationale in enumerate(item['rationale']):
                    if (i, j) in done:
                        continue

                    reference_reason = rationale['reason']
                    result = align_check(client, model_reason, reference_reason)

                    new_data.append({
                        'index': i,
                        'r_index': j,
                        'model_reason': model_reason,
                        'reference_reason': reference_reason,
                        'align_check_result': result,
                        'law': rationale['law'],
                    })
                    done.add((i, j))

                    with open(output_path, 'w') as f:
                        json.dump(new_data, f, indent=4, ensure_ascii=False)

            print(f'Saved: {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run GPT-4o alignment check for Task 2 reasons')
    parser.add_argument('--api-key', required=True, help='OpenAI API key')
    parser.add_argument('--models', nargs='+', required=True, help='Model name(s), e.g. gpt_4o o1 qwen2_5_vl_7b')
    parser.add_argument('--methods', nargs='+', default=['zero_shot'], help='Method name(s)')
    parser.add_argument('--results', default='../results_task2', help='Path to results_task2 directory')
    parser.add_argument('--data', default='../WiserUI_Bench.json', help='Path to WiserUI_Bench.json')
    args = parser.parse_args()

    run_align_check(args.api_key, args.results, args.data, args.models, args.methods)
