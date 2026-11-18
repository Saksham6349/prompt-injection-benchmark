import json

path = 'notebooks/train_classifier_colab.ipynb'
nb = json.load(open(path, encoding='utf-8'))

# Update Cell 2 (install dependencies + upgrade sympy)
nb['cells'][2]['source'] = [
    '# 1. Upgrade sympy to resolve logic attribute error and install dependencies\n',
    '!pip install -q --upgrade sympy\n',
    '!pip install -q transformers datasets accelerate scikit-learn evaluate pandas pyarrow\n',
    'import torch, os, random, hashlib, json\n',
    'import numpy as np\n',
    '\n',
    'SEED = 42\n',
    'random.seed(SEED)\n',
    'np.random.seed(SEED)\n',
    'torch.manual_seed(SEED)\n',
    'if torch.cuda.is_available():\n',
    '    torch.cuda.manual_seed_all(SEED)\n',
    '\n',
    "print('GPU Available:', torch.cuda.is_available())\n",
    'if torch.cuda.is_available():\n',
    "    print('Device Name :', torch.cuda.get_device_name(0))\n"
]

# Update Cell 4 (file loader handling both normal and (1) duplicate names)
nb['cells'][4]['source'] = [
    'import pandas as pd, os\n',
    '\n',
    'def find_file(candidates):\n',
    '    for c in candidates:\n',
    '        if os.path.exists(c):\n',
    '            return c\n',
    '    return candidates[0]\n',
    '\n',
    "train_f = find_file(['train.csv', 'train (1).csv'])\n",
    "cal_f = find_file(['calibration.csv', 'calibration (1).csv'])\n",
    "test_f = find_file(['test.csv', 'test (1).csv'])\n",
    "bench_f = find_file(['benchmark_cases.jsonl', 'benchmark_cases (1).jsonl'])\n",
    '\n',
    'train_df = pd.read_csv(train_f)\n',
    'cal_df = pd.read_csv(cal_f)\n',
    'test_df = pd.read_csv(test_f)\n',
    "print(f'Splits loaded successfully: Train={len(train_df)}, Cal={len(cal_df)}, Test={len(test_df)}')\n"
]

# Save
json.dump(nb, open(path, 'w', encoding='utf-8'), indent=1)
print('Notebook patched successfully!')
