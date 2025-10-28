# Evaluation Datasets

Created with `tools/create_eval_datasets.py`

## Datasets

### random_tokens.txt
- **Purpose**: Test basic model behavior on completely random data
- **Tokens**: 5000
- **Vocab size**: 10000
- **Pattern**: Uniform random distribution

### structured_tokens.txt
- **Purpose**: Simulate natural language token distribution
- **Tokens**: 5000
- **Vocab size**: 10000
- **Pattern**: 70% common tokens (0-99), 30% rare tokens

### repetitive_tokens.txt
- **Purpose**: Test model behavior on repeating patterns
- **Tokens**: 5000
- **Vocab size**: 10000
- **Pattern**: 100-token sequence repeated

### arithmetic.txt
- **Purpose**: Sanity test for basic reasoning
- **Format**: Simple arithmetic expressions with results
- **Token mapping**: 0-9=digits, 10=+, 11=-, 12=*, 13==, 14=SEP

## Usage

Evaluate perplexity:
```bash
python ch16/perplexity_eval.py eval_datasets/random_tokens.txt --output-json results.json
```

Compare precision modes:
```bash
python tools/compare_precision_accuracy.py --dataset eval_datasets/structured_tokens.txt
```
