import os.path as osp
from typing import Optional, Tuple

import cv2
import numpy as np

__all__ = ["convert"]

BRAILLE = (
    "⠀⢀⡀⣀⠠⢠⡠⣠⠄⢄⡄⣄⠤⢤⡤⣤"
    "⠐⢐⡐⣐⠰⢰⡰⣰⠔⢔⡔⣔⠴⢴⡴⣴⠂⢂⡂⣂⠢⢢⡢⣢⠆⢆⡆⣆⠦⢦⡦⣦⠒⢒⡒⣒⠲⢲⡲⣲⠖⢖⡖⣖⠶⢶⡶⣶"
    "⠈⢈⡈⣈⠨⢨⡨⣨⠌⢌⡌⣌⠬⢬⡬⣬⠘⢘⡘⣘⠸⢸⡸⣸⠜⢜⡜⣜⠼⢼⡼⣼⠊⢊⡊⣊⠪⢪⡪⣪⠎⢎⡎⣎⠮⢮⡮⣮⠚⢚⡚⣚⠺⢺⡺⣺⠞⢞⡞⣞⠾⢾⡾⣾"
    "⠁⢁⡁⣁⠡⢡⡡⣡⠅⢅⡅⣅⠥⢥⡥⣥⠑⢑⡑⣑⠱⢱⡱⣱⠕⢕⡕⣕⠵⢵⡵⣵⠃⢃⡃⣃⠣⢣⡣⣣⠇⢇⡇⣇⠧⢧⡧⣧⠓⢓⡓⣓⠳⢳⡳⣳⠗⢗⡗⣗⠷⢷⡷⣷"
    "⠉⢉⡉⣉⠩⢩⡩⣩⠍⢍⡍⣍⠭⢭⡭⣭⠙⢙⡙⣙⠹⢹⡹⣹⠝⢝⡝⣝⠽⢽⡽⣽⠋⢋⡋⣋⠫⢫⡫⣫⠏⢏⡏⣏⠯⢯⡯⣯⠛⢛⡛⣛⠻⢻⡻⣻⠟⢟⡟⣟⠿⢿⡿⣿"
)


def apply_threshold(data: np.ndarray, threshold: int) -> np.ndarray:
    if threshold == -1:
        threshold = cv2.threshold(
            data, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[0]
    return data > threshold


def convert2braille(data: np.ndarray, y: int, x: int, h: int, w: int) -> str:
    idx = 0
    for i in range(4):
        for j in range(2):
            dy = 4 * y + i
            dx = 2 * x + j
            if dx < w and dy < h:
                if data[dy, dx]:
                    idx += 1
                if i != 3 or j != 1:
                    idx *= 2
    return BRAILLE[idx]


def render(
    color_mat: np.ndarray,
    y: int,
    x: int,
    c: str,
    bg_color: Optional[Tuple[int, int, int]] = None,
) -> str:
    fore_color = list(color_mat[y, x, :])
    if bg_color:
        return "\033[38;2;{};{};{};48;2;{};{};{}m{}".format(*fore_color, *bg_color, c)
    return "\033[38;2;{};{};{}m{}".format(*fore_color, c)


def show(img):
    cv2.imshow("x", img)
    cv2.waitKey(5000)


def convert(
    source: str,
    with_color: bool = False,
    scale: float = 1.0,
    threshold: int = -1,
    save_raw: Optional[str] = None,
    bg_color: Optional[Tuple[int, int, int]] = None,
):
    ext = osp.splitext(source)[1][1:]
    if ext in ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"]:
        # TODO: detect number of channels to handle alpha in png
        bgr_data = cv2.imread(source)
    else:
        # TODO: Support for more types (gif, video)
        raise NotImplementedError()

    if scale != 1.0:
        bgr_data = cv2.resize(
            bgr_data,
            (0, 0),
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST,
        )

    gray_data = cv2.cvtColor(bgr_data, cv2.COLOR_BGR2GRAY)
    bool_map = apply_threshold(gray_data, threshold)

    oh, ow = bool_map.shape
    h = oh // 4
    w = ow // 2
    raw = []
    color_raw = []
    for i in range(h):
        converted = [convert2braille(bool_map, i, j, oh, ow) for j in range(w)]
        print("".join(converted))
        raw.append(converted)

    if with_color:
        if bg_color == (-1, -1, -1):
            bg_color = None
        bool_map = ~bool_map
        color = cv2.resize(bgr_data, (w, h), interpolation=cv2.INTER_LINEAR)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        for i, line in enumerate(raw):
            converted = [render(color, i, j, c, bg_color) for j, c in enumerate(line)]
            print("".join(converted))
            color_raw.append(converted)

    if save_raw:
        with open(save_raw, "w", encoding="UTF-8") as f:
            for line in color_raw or raw:
                f.writelines(r"".join(line))
                f.write("\n")
