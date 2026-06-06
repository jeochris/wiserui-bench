import json
import os
import argparse


INTERACTIVE_CONTROL = {'Button', 'Input Field', 'Link', 'Dropdown'}
CONTENT_STATE = {'Text', 'Image', 'Icon', 'Media', 'Video', 'Progress Bar', 'Timeline', 'Calendar'}
CONTAINER_LAYOUT = {'Card', 'Tile', 'Navigation', 'Banner', 'Background', 'Sidebar', 'Footer', 'Table'}

# Law index order used during evaluation
LAW_INDEX_LIST = [1, 2, 3, 4, 5, 6, 7, 10, 8, 9, 11, 12]


def categorize_ui_type(ui_change):
    keys = set(ui_change.keys())
    has_i = bool(keys & INTERACTIVE_CONTROL)
    has_c = bool(keys & CONTENT_STATE)
    has_l = bool(keys & CONTAINER_LAYOUT)
    n = sum([has_i, has_c, has_l])
    if n == 0:
        return 'Unknown'
    if n > 1:
        return 'Mixed'
    if has_i:
        return 'INTERACTIVE_CONTROL-only'
    if has_c:
        return 'CONTENT_STATE-only'
    return 'CONTAINER_LAYOUT-only'


def run_eval(results_dir, data_path):
    with open(data_path) as f:
        wiser_data = json.load(f)
    ui_change_map = {item['index']: item['ui_change'] for item in wiser_data}

    models = sorted(
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
    )

    for model in models:
        model_dir = os.path.join(results_dir, model)
        align_files = [f for f in os.listdir(model_dir) if f.startswith('align_check_')]

        for fname in align_files:
            method = fname.replace('align_check_', '').replace('.json', '')
            align_path = os.path.join(model_dir, fname)

            with open(align_path) as f:
                align_result = json.load(f)

            correct_count = [0] * len(LAW_INDEX_LIST)
            total_count = [0] * len(LAW_INDEX_LIST)
            data_wise = [0] * 300

            ui_type_correct = {}
            ui_type_total = {}

            for result in align_result:
                law_num = result['law']['num']
                if law_num not in LAW_INDEX_LIST:
                    print(f"Unknown law num {law_num} at index {result['index']}")
                    continue

                add_idx = LAW_INDEX_LIST.index(law_num)

                check = result['align_check_result']
                if 'Yes' in check and 'No' in check:
                    print(f"Ambiguous result at index={result['index']} r_index={result['r_index']}: {check}")
                    continue

                is_correct = 'Yes' in check

                if is_correct:
                    correct_count[add_idx] += 1
                    data_wise[result['index']] = 1
                total_count[add_idx] += 1

                ui_change = ui_change_map.get(result['index'], {})
                ui_cat = categorize_ui_type(ui_change)
                ui_type_correct.setdefault(ui_cat, 0)
                ui_type_total.setdefault(ui_cat, 0)
                if is_correct:
                    ui_type_correct[ui_cat] += 1
                ui_type_total[ui_cat] += 1

            print('=' * 80)
            print(f'Model: {model}, Method: {method}')
            print('=' * 80)

            print('\n[Law-level Statistics]')
            for i, law_num in enumerate(LAW_INDEX_LIST):
                if total_count[i] == 0:
                    print(f'  Law {law_num}: No data')
                else:
                    pct = correct_count[i] / total_count[i] * 100
                    print(f'  Law {law_num}: {correct_count[i]} / {total_count[i]}, {pct:.2f}%')
                if i % 4 == 3:
                    dim = i // 4 + 1
                    d_corr = sum(correct_count[i-3:i+1])
                    d_total = sum(total_count[i-3:i+1])
                    d_pct = d_corr / d_total * 100 if d_total else 0
                    print(f'  -- Dimension {dim}: {d_corr} / {d_total}, {d_pct:.2f}%')

            total_corr = sum(correct_count)
            total_tot = sum(total_count)
            total_pct = total_corr / total_tot * 100 if total_tot else 0
            dw_corr = sum(data_wise)
            print(f'\nTotal: {total_corr} / {total_tot}, {total_pct:.2f}%')
            print(f'Data-wise: {dw_corr} / 300, {dw_corr/300*100:.2f}%')

            print('\n[Statistics by UI Element Type]')
            for cat in ['INTERACTIVE_CONTROL-only', 'CONTENT_STATE-only', 'CONTAINER_LAYOUT-only', 'Mixed', 'Unknown']:
                if cat in ui_type_total and ui_type_total[cat] > 0:
                    c = ui_type_correct[cat]
                    t = ui_type_total[cat]
                    print(f'  {cat}: {c} / {t}, {c/t*100:.2f}%')

            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Task 2 alignment check results')
    parser.add_argument('--results', default='../results_task2', help='Path to results_task2 directory')
    parser.add_argument('--data', default='../WiserUI_Bench.json', help='Path to WiserUI_Bench.json')
    args = parser.parse_args()

    run_eval(args.results, args.data)
