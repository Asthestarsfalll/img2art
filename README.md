Convert image/gif/video to ascii art. If you inputs have plenty of frames, you can specify `--fast` to use torch accelerating the peocess.

## Screen Shot

![example](./assets/example.gif)
![example1](./assets/example2.gif)

## Installation

requirements: typer[all], opencv-python, numpy

```
pip install img2art
```

## Usage

```
img2art --help
```

result:

```
 Usage: img2art [OPTIONS] SOURCE

╭─ Arguments ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    source      TEXT  Path to image [default: None] [required]                                                                │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --with-color    --no-with-color                                  Whether use color [default: no-with-color]                    │
│ --scale                            FLOAT                         Scale applied to image [default: 1.0]                         │
│ --threshold                        INTEGER                       Threshold applied to image, default to OSTU [default: -1]     │
│ --save-raw                         TEXT                          Whether to save the raw data [default: None]                  │
│ --bg-color                         <INTEGER INTEGER INTEGER>...  Backgound color, (-1, -1, -1) for none [default: -1, -1, -1]  │
│ --fast          --no-fast                                        Whether use torch to accelerate when you inputs have plenty   │
│                                                                  of frames.                                                    │
│                                                                  [default: no-fast]                                            │
│ --chunk-size                       INTEGER                       Chunk size of Videos or Gifs when using torch.                │
│                                                                  [default: 1024]                                               │
│ --help                                                           Show this message and exit.                                   │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

```
img2art path/to/image --scale 0.5 --with-color --threshold 127 --bg-color 255, 255, 255 --save-raw path/to/save.txt
```

## Reference

[bobibo](https://github.com/orzation/bobibo)
