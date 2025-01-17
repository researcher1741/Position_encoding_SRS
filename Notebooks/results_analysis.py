from os import listdir, getcwd
from os.path import isfile, join, isdir
import statistics
import re


def get_files(folder):
    #### path
    cwd = getcwd()
    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
    onlyfiles = [x for x in onlyfiles if ".txt" in x]
    if len(onlyfiles) != 0:
        results = [join(cwd, folder, onlyfiles[i]) for i in range(len(onlyfiles))]
    else:
        print("No text file")
        return None
    #### Extraction
    lines = []
    for result in results:
        with open(result) as f:
            lines.append(f.readlines())
    return lines, onlyfiles


def convert_to_float(s):
    try:
        # Try converting the entire string to a float
        number = float(s)
    except ValueError:
        # If ValueError occurs, try to remove any non-numeric characters from the end and then convert
        numeric_part = ''.join(filter(str.isdigit, s))
        try:
            number = float(numeric_part)
        except ValueError:
            # If it still fails, return None
            number = None
    return number


def seed_filter(text):
    if "beau" in text:
        base = "_beau"
    elif "fashi" in text:
        base = "_fashi"
    elif "video" in text:
        base = "_video"
    elif "game" in text:
        base = "_game"
    elif "men" in text:
        base = "_men"

    # Define a list of patterns to replace
    patterns = r"_\d\d\d{0,2}" + base

    # Compile the regex patterns
    regex = re.compile(patterns)

    # Apply the regex substitution to subfolder
    text = regex.sub(base, text)
    return text


def clean_name(text):
    text = seed_filter(text)
    text = text.replace("Val_", "").replace("Test_", "").replace("Train_", "")
    text = text.replace(".png", "").replace(".txt", "").replace(".txt", "")
    return text


def get_hyper(folder, show=False):
    alllines, onlyfiles = get_files(folder)
    list_dictt = []
    for lines in alllines:
        parameters = False
        dictt = {}
        for line in lines:
            if parameters:
                name, value = line.split("\t")
                name = name.split(":")[0]
                value = value.split("\n")[0]
                if show:
                    print(name, value)
                try:
                    value = round(float(value) * 100, 10)
                except:
                    pass
                dictt[name] = value
                if name == "saving":
                    parameters = False
            if 'Training configuration' in line:
                parameters = True
        list_dictt.append(dictt)
    return list_dictt


def compare(ref1, ref2, printerror=False):
    for k, v in ref1.items():
        if k in ref2.keys():
            if ref1[k] != ref2[k]:
                if isinstance(ref1[k], str):
                    if "leaky" in ref1[k] and "leaky" in ref2[k]:
                        pass
                    else:
                        print(f"Difference at 2 {k}: ", ref1[k], ref2[k])
                elif k == "batch_size":
                    pass


                else:
                    print(f"Difference at {k}: ", ref1[k], ref2[k])
        elif printerror:
            print(f"The hyper {k} do not exist at in the second dictionary")
    if printerror:
        print(f"The hypers {[x for x in ref2.keys() if x not in ref1.keys()]} do not exist at in the first dictionary")


def print_it(main, start=0):
    """
    Extracts information from training folders, checks coherence, and generates a DataFrame.

    Parameters:
    - main (str): The main directory containing subdirectories with training results.
    - start (int, optional): The index to start considering the training folders. Default is 0.

    Returns:
    - pd.DataFrame: A DataFrame containing information about the training cases, including activation, encoding, nmax,
    ntype, mean_hit, dev_hit, mean_ndcg, and dev_ndcg.
    """
    nmax, act, ntype, mean_hit, dev_hit, mean_ndcg, dev_ndcg, encoding, runs = [], [], [], [], [], [], [], [], []
    no_list = [".ipynb_checkpoints", "Ignore"]
    folders = [join(main, x) for x in listdir(join(getcwd(), main)) if x not in no_list and isdir(join(main, x))]

    # Iterate through training folders
    for folder in folders:
        # statistics = [0, 0, 0, 0, 0]
        # print(f"folder {folder}")
        # Check if the training case is coherent
        print(f"The trainings are coherent? {check_case_coherence(folder)}")

        # Get the best case and its statistics
        case, statistics = get_best(folder, start)
        case = case.replace("None", "nan").replace("__", "_")

        # Extract information from the case string
        encoding.append(case.split("_")[0].replace("nmax", ""))
        act_case = "silu" if "silu" in folder else "leaky" if "leakyrelu" in folder else ""
        if act_case == "":
            print(f"NOOOOOO 111 {folder}")
        try:
            act.append(act_case)
            nmax.append(float(case.replace("None", "nan").split("max_")[-1].split("_")[0]))
            ntype.append(float(case.replace("None", "nan").split("ntype_")[-1].split("_")[0]))
        except:
            print(f"NOOOOOO 222 {folder}, case {case}, act_case {act_case}")
        mean_hit.append(statistics[0])
        dev_hit.append(statistics[1])
        mean_ndcg.append(statistics[2])
        dev_ndcg.append(statistics[3])
        runs.append(statistics[4])

    # Create a DataFrame from the extracted information
    df = pd.DataFrame({
        'activation': act,
        'encoding': encoding,
        'nmax': nmax,
        'ntype': ntype,
        'mean_hit': mean_hit,
        'dev_hit': dev_hit,
        'mean_ndcg': mean_ndcg,
        'dev_ndcg': dev_ndcg,
        'runs': runs
    })

    return df


def check_case_coherence(case):
    mydicts = get_hyper(case)
    Right = True
    for x, y in list(itertools.product(mydicts, mydicts)):
        if compare(x, y) is not None:
            Right = False
    return Right


def bold_over_65(value, bound):
    if value > bound:
        return 'font-weight: bold'
    else:
        return ''


##############################################################


import numpy as np
from scipy import stats


def confidence_interval(n, mu, dev):
    z_score = stats.norm.ppf(0.975)  # 95% confidence interval, two-tailed
    interval = z_score * (dev / np.sqrt(n))
    lower_bound = mu - interval
    upper_bound = mu + interval
    return round(lower_bound, 2), round(upper_bound, 2)


import itertools
import pandas as pd
import numpy as np
import math


def check_combinations(df, activations, encodings, nmax_values, ntype_values):
    # Generate all possible combinations
    all_combinations = list(itertools.product(activations, encodings, nmax_values, ntype_values))

    # Convert combinations to DataFrame for comparison
    combinations_df = pd.DataFrame(all_combinations, columns=['activation', 'encoding', 'nmax', 'ntype'])

    # Function to handle NaN comparison
    def compare_nan(val1, val2):
        if isinstance(val1, float) and isinstance(val2, float):
            if math.isnan(val1) and math.isnan(val2):
                return True
        return val1 == val2

    # Check if all combinations are in the DataFrame
    missing_combinations = []
    for idx, row in combinations_df.iterrows():
        match_found = False
        for _, df_row in df.iterrows():
            if all(compare_nan(row[col], df_row[col]) for col in combinations_df.columns):
                match_found = True
                break
        if not match_found:
            missing_combinations.append(row)

    # Print results
    if missing_combinations:
        print("Missing combinations:")
        for combination in missing_combinations:
            print(combination.to_dict())
    else:
        print("All combinations are included in the DataFrame.")


# Dictionary for renaming
def dictrenamer(df, prefix='\\'):
    """
    Renames the values in the 'encoding' column of the DataFrame based on a mapping.
    A specified prefix is prepended to each replacement string.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the 'encoding' column.
    - prefix (str): The prefix to prepend to each replacement string (default is '\\').

    Returns:
    - pd.DataFrame: A new DataFrame with the 'encoding' column updated.
    """
    # Define the base mapping without prefixes
    base_mapping = {
        'conlearnt': 'LearntCon',
        'con': 'AbsCon',
        'conROPE': 'RotatoryCon',
        'conRotatory': 'RotatoryCon',
        'learnt': 'Learnt',
        'nocon': 'Abs',
        'RMHA4': 'RMHA',
        'RMHA': 'RMHA',
        'LongerRMHA4': 'LongerRMHA',
        'LongerconROPE': 'LongRotatoryCon',
        'LongerconRotatory': 'LongRotatoryCon',
        'LongerROPE': 'LongRotatory',
        'ROPEMHAONE': 'RMHAROPEONE',
        'ROPEONE': 'RMHAROPEONE',
        'RMHAROPEONE': 'RMHAROPEONE',
        'LongROPE': 'LongRotatory',
        'LongerRotatory': 'LongRotatory',
        'ROPEMHA': 'ROPEMHA',
        'LongerROPEMHA': 'LongROPEMHA',
        '1200ROPEMHA': 'LongROPEMHA',
        '1400ROPEMHA': 'LongROPEMHA',
        'ROPE': 'ROPEMHA',
        'Rotatory': 'Rotatory',
        'None': 'None',
        'none': 'None',
        'carca': 'None',
    }

    # Prepend the prefix to each replacement string
    encoding_mapping = {key: f"{prefix}{value}" for key, value in base_mapping.items()}

    # Create a copy of the DataFrame to avoid modifying the original
    new_df = df.copy()

    # Replace the 'encoding' column based on the mapping
    new_df['encoding'] = new_df['encoding'].replace(encoding_mapping)

    # Reset the index of the new DataFrame
    new_df = new_df.reset_index(drop=True)

    return new_df


renamer = lambda x: dictrenamer(x, prefix='\\')
renamer2 = lambda x: dictrenamer(x, prefix='')



def filter_dataframe(df, apply_min=False, No_nan=False, encoding=False,
                     pseudo=False, nosas=False, add_dataset=None,
                     add_confidence=False, max_apply=[], ntype_apply=[]):
    if apply_min: df = df[df['runs'] > 2]

    if No_nan: df = df[~df['nmax'].isna()]

    if encoding: df = df[(df['ntype'] == 2.0)]

    if max_apply: df = df[(df['nmax'].isin(max_apply))]

    if pseudo:
        df = df[df['ntype'] != 2.0]
        df = df[(df['ntype'].isin(ntype_apply))]

    if nosas: df = df[~(df['encoding'] == "SASRec")]

    if add_dataset: df['dataset'] = add_dataset[0].upper() + add_dataset[1:]

    if add_confidence:
        try:
            df['CI'] = df.apply(lambda row: confidence_interval(row['runs'], row['mean_hit'], row['dev_hit']), axis=1)
            df['CI_length'] = df['CI'].apply(lambda x: x[1] - x[0] if x else None)
        except:
            pass

    return renamer(df)


######################################################
# BEST CODE

def get_best(folder, start=0):
    alllines, onlyfiles = get_files(folder)
    counter = 0
    total_ndcg, total_hit = [], []
    for lines in alllines:
        test = [x.replace('#### test Acc: 0, ', '') for x in lines if '#### test Acc:' in x]
        val = [x.replace('#### val Acc: 0, ', '') for x in lines if '#### val Acc:' in x]
        ndcg_test, hit_test = [], []
        ndcg_val, hit_val = [], []
        #         try:
        for y in val[start:]:
            if "Epoch" in y:
                y = y.split("Epoch")[0]
            ndcg, hit = y.split(' HIT: ') if 'HIT:' in y else y + ' HIT: 0\n'.split(' HIT: ')
            ndcg_val.append(convert_to_float(ndcg.replace('NDCG: ', '')))
            hit_val.append(convert_to_float(hit.replace('\n', '')))
        for x in test[start:]:
            if "Epoch" in y:
                x = x.split("Epoch")[0]
            try:
                ndcg, hit = x.split(' HIT: ') if 'HIT:' in x else x + ' HIT: 0\n'.split(' HIT: ')
            except:
                print(f"wwwwwwwwwwwwwwwwwwwwww {x}")
            ndcg_test.append(convert_to_float(ndcg.replace('NDCG: ', '')))
            hit_test.append(convert_to_float(hit.replace('\n', '')))
        #         except:
        #             print(f"lines {lines}")
        case = onlyfiles[counter].replace(".txt", "")
        # print(case)
        NDCG = round(max(ndcg_test) * 100, 2)
        HIT = round(max(hit_test) * 100, 2)
        # print(f"\t\tNDCG: {NDCG}, \t\tHIT: {HIT}")
        total_ndcg.append(max(ndcg_test) * 100)
        total_hit.append(max(hit_test) * 100)
        counter += 1

    if len(total_hit) >= 1:
        runs, ndcgs = len(total_hit), len(total_ndcg)
        if runs == 1: total_hit.append(total_hit[0])
        if ndcgs == 1: total_ndcg.append(total_ndcg[0])
        mean_hit, dev_hit = round(statistics.mean(total_hit), 2), round(statistics.stdev(total_hit), 2)
        mean_ndcg, dev_ndcg = round(statistics.mean(total_ndcg), 2), round(statistics.stdev(total_ndcg), 2)
        # print(f"Statistics hit: mean {mean_hit} std {dev_hit}")
        # print(f"Statistics ndcg: mean {mean_ndcg} std {dev_ndcg}")
        # print(f"for {runs} runs")
    case = clean_name(case)
    #     print("\n")
    return case, (mean_hit, dev_hit, mean_ndcg, dev_ndcg, runs)


def cleanit_for_position(df):
    """ Cleans the dataframes, deleting the columns which are not going to be used in the tables """
    df = df.drop(columns=['ntype'])
    df = df.drop(columns=['dataset'])
    df = df.reset_index(drop=True)
    df.index = df.index + 1
    return df


best = lambda df, maximum: df[df['dev_hit'] < maximum].sort_values(by='mean_hit', ascending=False)


def get_deviation(df):
    df = df[df['encoding'] == 'None']
    weighted_average_hit = (df['dev_hit'] * df['runs']).sum() / df['runs'].sum()
    weighted_average_ndcg = (df['dev_ndcg'] * df['runs']).sum() / df['runs'].sum()
    CI = (df['confidence_interval_length'] * df['runs']).sum() / df['runs'].sum()
    return round(weighted_average_hit, 6), round(weighted_average_ndcg, 6), round(CI, 6)


mylist = ["dataset", "activation", "encoding", "nmax", "mean_hit", "dev_hit", "mean_ndcg", "dev_ndcg"]
mylist_all = ["activation", "encoding", "nmax", "mean_hit", "dev_hit", "mean_ndcg", "dev_ndcg"]
