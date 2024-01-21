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
    with_color: Annotated[bool, typer.Option(help="Whether use color")] = False,
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
):
    convert(source, with_color, scale, threshold, save_raw, bg_color)


def launch():
    typer.run(main)


if __name__ == "__main__":
    launch()
