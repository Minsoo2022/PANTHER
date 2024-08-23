import os
import argparse
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--exp_dir', type=str, default='/data/12ff78/shared/ms.lee/clip/logs_eval_new/TCGA-GTEx/LUAD/dino_dataaug_rdup', help='')
    parser.add_argument('--num_folds', type=int, default=5, help='')
    return parser.parse_args()

def process_sub_dir(args, sub_dir):
    seed_dirs = os.listdir(os.path.join(args.exp_dir, sub_dir))
    SEEDS =  [int(i[2:]) for i in seed_dirs if i.startswith('_s')]
    # SEEDS = [1, 42, 47, 100]
    num_folds = args.num_folds

    results = {'avg': []}
    results.update({f'e{f}': [] for f in range(args.num_folds)})

    for s in SEEDS:
        seed_result_list = []
        for n in range(num_folds):
            s_path = os.path.join(args.exp_dir, sub_dir, f'_s{s}', f'{n}', 'summary.csv')
            df_s = pd.read_csv(s_path)
            target_col = [i for i in df_s.keys() if i.startswith('c_index_test')][0]
            auc = df_s[target_col].item()
            seed_result_list.append(auc)
        seed_result_list = pd.Series(seed_result_list)
        # s_path = os.path.join(args.exp_dir, sub_dir, f'_s{s}', 'summary.csv')
        # df_s = pd.read_csv(s_path)
        # df_auc = df_s['test_auc']
        df_auc = seed_result_list
        results['avg'].append(df_auc.mean())
        for f in range(args.num_folds):
            results[f'e{f}'].append(df_auc.iloc[f])

    auc_avg = []
    auc_max = []
    for f in range(args.num_folds):
        avg_val = np.array(results[f'e{f}']).mean()
        val = np.array(results[f'e{f}']).max()
        results[f'e{f}'].append(avg_val)
        results[f'e{f}'].append(val)
        auc_avg.append(avg_val)
        auc_max.append(val)
    results['avg'].append(np.array(auc_avg).mean())
    results['avg'].append(np.array(auc_max).mean())

    return pd.DataFrame(results, index=['S' + str(s) for s in SEEDS] + ['AVG'] + ['MAX'])

def extract_task_info(sub_dir):
    parts = sub_dir.split('::')
    task_list = ['TCGA_LUAD_overall_survival']
    task_candidate = f'{parts[0]}_{parts[1]}'
    for i in task_list:
        if i in task_candidate:
            task = i
    # lr = float(parts[-2].replace('lr', ''))
    # wd = float(parts[-1].replace('wd', ''))
    return task # , lr

def format_scientific(value):
    return "{:.0e}".format(value)

def main():
    args = parse_args()
    sub_dir_list = os.listdir(args.exp_dir)
    sub_dir_list.sort()

   #  data = {'Task': [], 'LR': [], 'AVG': [], 'MAX': []}
    data = {'Task': [], 'AVG': [], 'MAX': []}


    for sub_dir in sub_dir_list:
        try:
            # task, lr = extract_task_info(sub_dir)
            task = extract_task_info(sub_dir)

            df_sum = process_sub_dir(args, sub_dir)
            data['Task'].append(task)
            # data['LR'].append(lr)
            # data['WD'].append(wd)
            data['AVG'].append(df_sum.loc['AVG', 'avg'])
            data['MAX'].append(df_sum.loc['MAX', 'avg'])
        except Exception as e:
            print(f"Error processing {sub_dir}: {e}")

    df_final = pd.DataFrame(data)
    # df_final = df_final.sort_values(by=['Task', 'LR'], ascending=[True, True])
    df_final = df_final.sort_values(by=['Task'], ascending=[True])


    # df_final['LR'] = df_final['LR'].apply(format_scientific)
    # df_final['WD'] = df_final['WD'].apply(format_scientific)

    print(df_final)

if __name__ == "__main__":
    main()