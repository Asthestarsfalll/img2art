import os.path as osp
from typing import Callable, Optional, Tuple, TypeVar

import typer
from typing_extensions import Annotated

from .utils import convert

T = TypeVar("T")


def _generate_check_func(check_func: Callable[[T], T]) -> Callable[[T], T]:
    def _check(arg: T) -> T:
        if check_func(arg):
            raise ValueError(arg)
        return arg

    return _check


def _is_rgb(x):
    if x is None:
        return False
    for i in x:
        if i > 255 or i < -1:
            return True
    return False


def main(
    source: Annotated[
        str,
        typer.Argument(
            help="Path to image",
            callback=_generate_check_func(lambda x: not osp.exists(x)),
        ),
    ],
    with_color: Annotated[
        bool,
        typer.Option(
            help="Whether use color. If you specify alpha, with-color will be forcely set to True."
        ),
    ] = False,
    scale: Annotated[
        float,
        typer.Option(
            help="Scale applied to image",
            callback=_generate_check_func(lambda x: x <= 0.0 or x > 1.0),
        ),
    ] = 1.0,
    threshold: Annotated[
        int,
        typer.Option(
            help="Threshold applied to image, default to OSTU",
            callback=_generate_check_func(lambda x: x < -1 or x > 255),
        ),
    ] = -1,
    save_raw: Annotated[
        Optional[str], typer.Option(help="Whether to save the raw data")
    ] = None,
    # typer bug
    bg_color: Annotated[
        Tuple[int, int, int],
        typer.Option(
            help="Backgound color, (-1, -1, -1) for none",
            callback=_generate_check_func(_is_rgb),
        ),
    ] = (-1, -1, -1),
    fast: Annotated[
        bool,
        typer.Option(
            help="Whether use torch to accelerate when you inputs have plenty of frames."
        ),
    ] = False,
    chunk_size: Annotated[
        int, typer.Option(
            help="Chunk size of Videos or Gifs when using torch.")
    ] = 1024,
    alpha: Annotated[
        bool, typer.Option(help="Whether generating lua code for alpha-nvim.")
    ] = False,
    quant: Annotated[
        int,
        typer.Option(
            help="Apply color quantization.",
            callback=_generate_check_func(lambda x: x >= 256),
        ),
    ] = -1,
    mapping: Annotated[
        str,
        typer.Option(
            help="User-define ascii characters, need to be from light to dark. The quant will be forcely set to length of mapping.",
            callback=_generate_check_func(lambda x: len(x) >= 256),
        ),
    ] = "",
    loop: Annotated[
        bool,
        typer.Option(
            help="Loop the output when input is GIF or Video, use Ctrl-C to end this."
        ),
    ] = False,
    interval: Annotated[
        float,
        typer.Option(
            help="Interval when playing GIF or Video output.",
            callback=_generate_check_func(lambda x: x < 0),
        ),
    ] = 0.05,
):
    if alpha and not with_color:
        with_color = True

    if bg_color == (-1, -1, -1):
        bg_color = None

    if mapping and len(mapping) < 255:
        quant = len(mapping)

    if mapping:
        scale = [scale, scale / 2]

    convert(
        source,
        with_color,
        scale,
        threshold,
        save_raw,
        bg_color,
        fast,
        chunk_size,
        alpha,
        quant,
        mapping,
        loop,
        interval,
    )


def launch():
    typer.run(main)


if __name__ == "__main__":
    launch()
