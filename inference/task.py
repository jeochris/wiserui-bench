import json
import os
from tqdm import tqdm
import sys
from datasets import load_dataset

from methods import METHODS

# for zero-shot, cocot, ddcot
def get_answer(num, first_file, second_file):

    # check if already done
    if os.path.exists(f'{TASK_FOLDER}/{MODEL.replace("-","_")}/{METHOD}_{first_file}_{second_file}.json'):
        with open(f'{TASK_FOLDER}/{MODEL.replace("-","_")}/{METHOD}_{first_file}_{second_file}.json', 'r') as f:
            all_save_data = json.load(f)
    else:
        all_save_data = []
    
    for data in all_save_data:
        if data['index'] == num:
            return 'Already done'

    answer = method.run(num, first_file, second_file)
    # gives full package: [[answer, answer, ...], [tokens, tokens, ...]]
    
    save_data = {}
    save_data['index'] = num
    if TASK == 1:
        save_data['full_answer'] = answer[0]
        save_data['only_answer'] = answer[0][-1].split('More effective:')[1].strip()
    else:
        save_data['reason'] = answer[0]
    save_data['tokens'] = answer[1]
    
    all_save_data.append(save_data)
    with open(f'{TASK_FOLDER}/{MODEL.replace("-","_")}/{METHOD}_{first_file}_{second_file}.json', 'w') as f:
        json.dump(all_save_data, f, indent=4, ensure_ascii=False)

# for self-refine
def get_answer_self_refine(num, first_file, second_file):

    if os.path.exists(f'{TASK_FOLDER}/{MODEL.replace("-","_")}/{METHOD}_{first_file}_{second_file}.json'):
        with open(f'{TASK_FOLDER}/{MODEL.replace("-","_")}/{METHOD}_{first_file}_{second_file}.json', 'r') as f:
            all_save_data = json.load(f)
    else:
        all_save_data = []

    with open(f'{TASK_FOLDER}/{MODEL.replace("-","_")}/cocot_{first_file}_{second_file}.json', 'r') as f:
        cocot_data = json.load(f)
    
    found_data = None
    found_data_cnt = 0

    for data in cocot_data:
        if data['index'] == num:
            cocot_data = data
            found_data_cnt = 1
            break
    for data in all_save_data:
        if data['index'] == num:
            # if 2 or 1
            found_data = data
            found_data_cnt = len(data['full_answer'])
            # if already 3, done
            if 'only_answer' in data:
                return 'Already done'
            break
    
    if found_data_cnt == 0:
        raise ValueError(f'cocot data not found for {num}, {first_file}, {second_file}')
    
    if not found_data:
        start_data = [cocot_data['full_answer'], cocot_data['tokens']]
    else:
        start_data = [found_data['full_answer'], found_data['tokens']]

    for i in range(found_data_cnt, 3):
        print(f'Self-refining {i+1}...')
        answer = method.run(num, first_file, second_file, start_data=start_data)
        
        save_data = {}
        save_data['index'] = num
        save_data['full_answer'] = answer[0]
        save_data['tokens'] = answer[1]
        start_data = answer.copy()

        if i == 2:
            save_data['only_answer'] = answer[0][-1].split('More effective:')[1].strip()
            del all_save_data[-1]
        
        all_save_data.append(save_data)

        with open(f'{TASK_FOLDER}/{MODEL.replace("-","_")}/{METHOD}_{first_file}_{second_file}.json', 'w') as f:
            json.dump(all_save_data, f, indent=4, ensure_ascii=False)

# for MAD debate
def get_answer_mad_debate(num, first_file, second_file):
    
    if os.path.exists(f'{TASK_FOLDER}/{MODEL.replace("-","_")}/{METHOD}_{first_file}_{second_file}.json'):
        with open(f'{TASK_FOLDER}/{MODEL.replace("-","_")}/{METHOD}_{first_file}_{second_file}.json', 'r') as f:
            all_save_data = json.load(f)
    else:
        all_save_data = []

    found_data = None
    found_data_cnt = 0
    for data in all_save_data:
        if data['index'] == num:
            found_data = data
            found_data_cnt = len(data['full_answer'])
            if found_data_cnt == 6:
                return 'Already done'
            break
    
    if not found_data:
        start_data = [[], []]
    else:
        start_data = [found_data['full_answer'], found_data['tokens']]
    
    for i in range(found_data_cnt, 6):
        print(f'MAD debate {i+1}...')
        answer = method.run(num, first_file, second_file, start_data=start_data)

        save_data = {}
        save_data['index'] = num
        save_data['full_answer'] = answer[0]
        save_data['tokens'] = answer[1]
        start_data = answer.copy()
        if i > 0:
            del all_save_data[-1]
        all_save_data.append(save_data)
        with open(f'{TASK_FOLDER}/{MODEL.replace("-","_")}/{METHOD}_{first_file}_{second_file}.json', 'w') as f:
            json.dump(all_save_data, f, indent=4, ensure_ascii=False)
        
# for MAD moderate extractive
def get_answer_mad_moderate(num, first_file, second_file, round_for_mad):
    if os.path.exists(f'{TASK_FOLDER}/{MODEL.replace("-","_")}/{METHOD}_{round_for_mad}_{first_file}_{second_file}.json'):
        with open(f'{TASK_FOLDER}/{MODEL.replace("-","_")}/{METHOD}_{round_for_mad}_{first_file}_{second_file}.json', 'r') as f:
            all_save_data = json.load(f)
    else:
        all_save_data = []
    
    for data in all_save_data:
        if data['index'] == num:
            return 'Already done'
    
    with open(f'{TASK_FOLDER}/{MODEL.replace("-","_")}/mad_each_debate_{first_file}_{second_file}.json', 'r') as f:
        mad_data = json.load(f)
    
    mad_found = False
    for data in mad_data:
        if data['index'] == num:
            mad_data = data
            mad_found = True
            break
    if not mad_found:
        raise ValueError(f'MAD data not found for {num}, {first_file}, {second_file}')
    if len(mad_data['full_answer']) < round_for_mad * 2:
        raise ValueError(f'MAD data not complete for {num}, {first_file}, {second_file}')
    start_data = [mad_data['full_answer'][:round_for_mad * 2], mad_data['tokens'][:round_for_mad * 2]]
    answer = method.run(num, first_file, second_file, start_data=start_data)

    save_data = {}
    save_data['index'] = num
    save_data['full_answer'] = answer[0]
    save_data['only_answer'] = answer[0][-1].split('More effective:')[1].strip()
    save_data['tokens'] = answer[1]

    all_save_data.append(save_data)
    with open(f'{TASK_FOLDER}/{MODEL.replace("-","_")}/{METHOD}_{round_for_mad}_{first_file}_{second_file}.json', 'w') as f:
        json.dump(all_save_data, f, indent=4, ensure_ascii=False)


if len(sys.argv) > 1:
    MODEL = sys.argv[1]
    METHOD = sys.argv[2]
    ROUND_FOR_MAD = int(sys.argv[3])
    TASK = int(sys.argv[4])
    API_KEY = sys.argv[5]
    GPU_COUNT = int(sys.argv[6])
else:
    MODEL = 'gpt-4o'  # Change this to the model you want to use
    METHOD = 'zero_shot'  # Change this to the method you want to use
    ROUND_FOR_MAD = 1 # Change this to the number of rounds for MAD moderation
    TASK = 1 # Change this to the task number
    API_KEY = 'your_api_key_here'
    GPU_COUNT = 1

TASK_FOLDER = f'results_task{TASK}'
ds = load_dataset("jeochris/WiserUI-Bench", split='test')

method = METHODS(MODEL, TASK, METHOD, ds, API_KEY, GPU_COUNT)

print(f'Using model: {MODEL}, method: {METHOD}, moderation round for MAD: {ROUND_FOR_MAD}, task: {TASK}')

os.makedirs(TASK_FOLDER, exist_ok=True)
if not os.path.exists(f'{TASK_FOLDER}/{MODEL.replace("-","_")}'):
    os.makedirs(f'{TASK_FOLDER}/{MODEL.replace("-","_")}')

for folder in tqdm(range(0, len(ds))): ####
    for file in [('lose', 'win'), ('win', 'lose')]: ####
        if TASK == 2 and file == ('win', 'lose'):
            continue

        print(folder, file)
        
        if METHOD in ['zero_shot', 'cocot', 'ddcot']:
            get_answer(folder, file[0], file[1])
        elif METHOD == 'self_refine':
            get_answer_self_refine(folder, file[0], file[1])
        elif METHOD == 'mad_each_debate':
            get_answer_mad_debate(folder, file[0], file[1])
        elif METHOD == 'mad_moderate_extractive':
            get_answer_mad_moderate(folder, file[0], file[1], ROUND_FOR_MAD)
        print('-------------------')