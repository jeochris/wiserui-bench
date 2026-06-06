import json
import os
import argparse


def load_metadata(path):
    with open(path) as f:
        data = json.load(f)
    return {item['index']: item for item in data}


def get_file_path(results_dir, model, method, ordering):
    base = f'{method}_{ordering[0]}_{ordering[1]}'
    return os.path.join(results_dir, model, f'{base}.json')


def extract_answer(item):
    ans = item.get('only_answer', '')
    ans = ans.split('\n')[0].split('(')[0]
    ans = ans.strip('* ').strip('<> ').strip('Version ').strip('version ')
    return ans


def classify_ui_change_by_key(ui_change):
    return '+'.join(sorted(ui_change.keys()))


def classify_ui_change_by_value(ui_change):
    vals = []
    for v in ui_change.values():
        vals.extend(v)
    return '+'.join(sorted(set(vals)))


def classify_element_by_group(ui_change):
    INTERACTIVE = {'Button', 'Input Field', 'Link', 'Dropdown'}
    CONTENT = {'Text', 'Image', 'Icon', 'Media', 'Video', 'Progress Bar', 'Timeline', 'Calendar'}
    CONTAINER = {'Card', 'Tile', 'Navigation', 'Banner', 'Background', 'Sidebar', 'Footer', 'Table'}
    keys = set(ui_change.keys())
    groups = []
    if keys & INTERACTIVE:
        groups.append('Interactive Control')
    if keys & CONTENT:
        groups.append('Content & State Display')
    if keys & CONTAINER:
        groups.append('Container & Layout Structure')
    if len(groups) == 1:
        return groups[0] + '-only'
    return 'Other' if not groups else 'Mixed'


def classify_value_by_group(ui_change):
    EXISTENCE = {'Presence', 'Count'}
    SEMANTIC = {'Text Content', 'Image', 'Icon'}
    VISUAL = {'Color', 'Font Size', 'Border', 'Opacity', 'Hover State', 'Padding'}
    SPATIAL = {'Position', 'Size', 'Layout'}
    vals = set()
    for v in ui_change.values():
        vals.update(v)
    groups = []
    if vals & EXISTENCE:
        groups.append('Existence & Quantity Changes')
    if vals & SEMANTIC:
        groups.append('Semantic Content Changes')
    if vals & VISUAL:
        groups.append('Visual Styling Changes')
    if vals & SPATIAL:
        groups.append('Spatial & Structural Changes')
    if len(groups) == 1:
        return groups[0] + '-only'
    return 'Other' if not groups else 'Mixed'


def calc_metrics(group):
    lw = group['lose_win']
    wl = group['win_lose']
    aa_total = len(lw) + len(wl)
    aa_count = sum(lw) + sum(wl)
    ca_total = min(len(lw), len(wl))
    ca_count = sum(1 for i in range(ca_total) if lw[i] == 1 and wl[i] == 1)
    return {
        'aa': (aa_count, aa_total, aa_count / aa_total * 100 if aa_total else 0),
        'ca': (ca_count, ca_total, ca_count / ca_total * 100 if ca_total else 0),
    }


def print_dim(label, dim_data, out):
    out.write(f'\n--- {label} ---\n')
    for k in sorted(dim_data):
        m = calc_metrics(dim_data[k])
        out.write(f'{k}:\n')
        out.write(f'  AA: {m["aa"][0]}/{m["aa"][1]} ({m["aa"][2]:.2f}%)\n')
        out.write(f'  CA: {m["ca"][0]}/{m["ca"][1]} ({m["ca"][2]:.2f}%)\n')


def run_eval(results_dir, metadata_path, output_path):
    metadata = load_metadata(metadata_path)

    models = sorted(
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
    )
    methods = ['zero_shot', 'cocot', 'ddcot', 'self_refine', 'mad_each_debate', 'mad_moderate_extractive']

    errors = []
    summary = []

    with open(output_path, 'w', encoding='utf-8') as out:
        for model in models:
            for method in methods:
                check_path = get_file_path(results_dir, model, method, ('lose', 'win'))
                if not os.path.exists(check_path):
                    continue

                out.write(f'Model: {model}, Method: {method}\n')
                eval_data = [[], []]  # [lose_win correctness, win_lose correctness]

                dims = {k: {} for k in [
                    'page_type', 'industry_domain',
                    'ui_change_key', 'ui_change_value', 'ui_change_key_value',
                    'element_group', 'value_group', 'element_value_cross'
                ]}

                for ordering in [('lose', 'win'), ('win', 'lose')]:
                    fpath = get_file_path(results_dir, model, method, ordering)
                    with open(fpath) as f:
                        items = json.load(f)

                    is_lose_first = (ordering[0] == 'lose')
                    list_idx = 0 if is_lose_first else 1

                    for item in items:
                        idx = item['index']
                        ans = extract_answer(item)

                        if ans not in ['First', 'Second']:
                            errors.append(f'{fpath} index={idx}: {repr(ans)}')

                        if is_lose_first:
                            correct = 1 if ans == 'Second' else 0
                        else:
                            correct = 1 if ans == 'First' else 0
                        eval_data[list_idx].append(correct)

                        if idx not in metadata:
                            continue
                        meta = metadata[idx]

                        def track(dim_key, group_key):
                            if group_key not in dims[dim_key]:
                                dims[dim_key][group_key] = {'lose_win': [], 'win_lose': []}
                            target = 'lose_win' if is_lose_first else 'win_lose'
                            dims[dim_key][group_key][target].append(correct)

                        track('page_type', meta['page_type'])
                        track('industry_domain', meta['industry_domain'])
                        track('ui_change_key', classify_ui_change_by_key(meta['ui_change']))
                        track('ui_change_value', classify_ui_change_by_value(meta['ui_change']))

                        if len(meta['ui_change']) == 1:
                            key = list(meta['ui_change'].keys())[0]
                            for val in meta['ui_change'][key]:
                                track('ui_change_key_value', f'{key}-{val}')

                        eg = classify_element_by_group(meta['ui_change'])
                        vg = classify_value_by_group(meta['ui_change'])
                        track('element_group', eg)
                        track('value_group', vg)
                        track('element_value_cross', f'{eg} | {vg}')

                    cnt = len(items)
                    sub_correct = sum(eval_data[list_idx])
                    out.write(f'{ordering[0]}_{ordering[1]}: {sub_correct} / {cnt}, {sub_correct/cnt*100:.2f}%\n')

                lw, wl = eval_data[0], eval_data[1]
                aa_total = len(lw) + len(wl)
                aa_count = sum(lw) + sum(wl)
                ca_total = min(len(lw), len(wl))
                ca_count = sum(1 for i in range(ca_total) if lw[i] == 1 and wl[i] == 1)

                out.write(f'AA: {aa_count} / {aa_total}, {aa_count/aa_total*100:.2f}%\n')
                out.write(f'CA: {ca_count} / {ca_total}, {ca_count/ca_total*100:.2f}%\n')
                if aa_total == 600:
                    out.write('Done! ✅\n')
                out.write('\n')

                out.write('='*60 + '\n')
                out.write('DIMENSION BREAKDOWNS\n')
                out.write('='*60 + '\n')

                print_dim('Page Type', dims['page_type'], out)
                print_dim('Industry Domain', dims['industry_domain'], out)
                print_dim('UI Change (Key-based)', dims['ui_change_key'], out)
                print_dim('UI Change (Value-based)', dims['ui_change_value'], out)
                print_dim('UI Change (Key-Value Combination)', dims['ui_change_key_value'], out)
                print_dim('Element Group', dims['element_group'], out)
                print_dim('Value Group', dims['value_group'], out)

                out.write('\n--- Element Group x Value Group Cross Analysis ---\n')
                elem_groups = sorted(set(k.split(' | ')[0] for k in dims['element_value_cross']))
                for eg in elem_groups:
                    out.write(f'{eg}:\n')
                    for cross_key in sorted(dims['element_value_cross']):
                        if cross_key.startswith(eg + ' | '):
                            vg = cross_key.split(' | ')[1]
                            m = calc_metrics(dims['element_value_cross'][cross_key])
                            out.write(f'  -> {vg}:\n')
                            out.write(f'     AA: {m["aa"][0]}/{m["aa"][1]} ({m["aa"][2]:.2f}%)\n')
                            out.write(f'     CA: {m["ca"][0]}/{m["ca"][1]} ({m["ca"][2]:.2f}%)\n')

                out.write('='*60 + '\n\n')

                summary.append({
                    'model': model, 'method': method,
                    'aa_count': aa_count, 'aa_total': aa_total,
                    'aa_pct': round(aa_count / aa_total * 100, 2),
                    'ca_count': ca_count, 'ca_total': ca_total,
                    'ca_pct': round(ca_count / ca_total * 100, 2),
                })

            out.write('-'*50 + '\n')

        if errors:
            out.write('\n' + '='*60 + '\n')
            out.write('PARSE ERRORS\n')
            out.write('='*60 + '\n')
            for e in errors:
                out.write(e + '\n')

    print(f'Results saved to: {output_path}')

    summary_path = output_path.replace('.txt', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Summary saved to: {summary_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Task 1 results (AA/CA metrics)')
    parser.add_argument('--results', default='../results_task1', help='Path to results_task1 directory')
    parser.add_argument('--data', default='../WiserUI_Bench.json', help='Path to WiserUI_Bench.json')
    parser.add_argument('--output', default='task1_scores.txt', help='Output text file path')
    args = parser.parse_args()

    run_eval(args.results, args.data, args.output)
