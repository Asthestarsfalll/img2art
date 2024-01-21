Convert image to ascii art.

## Screen Shot

![example](./asset/example.gif)

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
                                                                                                                                 
╭─ Arguments ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    source      TEXT  Path to image [default: None] [required]                                                               │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --with-color    --no-with-color                                  Whether use color [default: no-with-color]                   │
│ --scale                            FLOAT                         Scale applied to image [default: 1.0]                        │
│ --threshold                        INTEGER                       Threshold applied to image, default to OSTU [default: -1]    │
│ --save-raw                         TEXT                          Whether to save the raw data [default: None]                 │
│ --bg-color                         <INTEGER INTEGER INTEGER>...  Backgound color, (-1, -1, -1) for none [default: -1, -1, -1] │
│ --help                                                           Show this message and exit.                                  │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```

```
img2art path/to/image --scale 0.5 --with-color --threshold 127 --bg-color 255, 255, 255 --save-raw path/to/save.txt
```

## Reference

[bobibo](https://github.com/orzation/bobibo)
