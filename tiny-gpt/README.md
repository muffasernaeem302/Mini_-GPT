# Tiny GPT

A small character-level GPT (Transformer) built from scratch in PyTorch, following
Andrej Karpathy's ["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) video.
It learns to predict the next character, then generates new text one character at a time.

## What's inside
- Token and position embeddings
- Masked multi-head self-attention
- Feed-forward layers, residual connections, LayerNorm
- Training loop with train/validation loss
- Text generation

## Run it
```bash
pip install -r requirements.txt
python train.py --data data/my_stories.txt --steps 3000
```
Options: `--data` (any UTF-8 .txt file), `--steps`, `--prompt "STORY 1"`, `--length 600`.

It uses a GPU automatically if available (bigger model), otherwise a smaller CPU-sized one.

## Use your own text
Put any `.txt` file in `data/` and run `python train.py --data data/yourfile.txt`.
Around 300 KB to 1 MB of text works well.

## Data
`data/my_stories.txt` is synthetic: short template-based stories made by `generate_data.py`.
Regenerate it with `python generate_data.py`.

## Results
| Data | Final val loss |
|---|---|
| Shakespeare (CPU, small model) | ~1.9 |
| (add your own results here) | |

## Credits
Based on Andrej Karpathy's Zero to Hero series and the "Attention Is All You Need" paper.
