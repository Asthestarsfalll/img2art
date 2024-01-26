import os.path as osp
from math import ceil
from time import sleep
from typing import List, Optional, Tuple

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


def convert2braille(bool_map: np.ndarray, y: int, x: int, h: int, w: int) -> str:
    idx = 0
    for i in range(4):
        for j in range(2):
            dy = 4 * y + i
            dx = 2 * x + j
            if dx < w and dy < h:
                if bool_map[dy, dx]:
                    idx += 1
                if i != 3 or j != 1:
                    idx *= 2
    return BRAILLE[idx]


def _fast_convert2braille(bool_maps: List[np.ndarray], chunk_size: int):
    try:
        import torch
    except:
        raise ModuleNotFoundError("Can not find torch")
    with torch.no_grad():
        ch = len(bool_maps)
        inp = torch.tensor(np.array([bool_maps])).type(torch.float32)

        if ch <= chunk_size:
            loader = [inp]
        else:
            loader = torch.chunk(inp, ceil(ch / chunk_size))

        kernel = (
            torch.tensor([2**i for i in range(7, -1, -1)])
            .reshape((1, 1, 4, 2))
            .type(torch.float32)
        )
        if torch.cuda.is_available():
            inp = inp.cuda()
            kernel = kernel.cuda()
        res = []
        for x in loader:
            ker = kernel.repeat(x.shape[1], 1, 1, 1)
            res.append(
                torch.nn.functional.conv2d(
                    x, ker, padding=0, stride=(4, 2), groups=x.shape[1]
                )
            )
        res = torch.concat(res, 1)
        if torch.cuda.is_available():
            res = res.cpu()
        index_maps = res.numpy().astype(np.uint8)
        index_maps = np.split(index_maps, ch, 1)
        return [i[0, 0] for i in index_maps]


def _get_braille(index_map: np.ndarray) -> List:
    h, w = index_map.shape
    raw = []
    for i in range(h):
        str_list = []
        for j in range(w):
            str_list.append(BRAILLE[int(index_map[i, j])])
        raw.append("".join(str_list))
    return raw


def render(
    color_mat: np.ndarray,
    y: int,
    x: int,
    c: str,
    bg_color: Optional[Tuple[int, int, int]] = None,
) -> str:
    fore_color = list(color_mat[y, x, :])
    if bg_color:
        return "\033[38;2;{};{};{};48;2;{};{};{}m{}\033[0m".format(
            *fore_color, *bg_color, c
        )
    return "\033[38;2;{};{};{}m{}".format(*fore_color, c)


def show(img):
    cv2.imshow("x", img)
    cv2.waitKey(5000)


def _resize(img, scale, inter_type):
    if inter_type == "nearest":
        inter = cv2.INTER_NEAREST
    else:
        inter = cv2.INTER_LINEAR

    return cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=inter)


def print_converted(data: np.ndarray, interval: float = 0.05) -> None:
    for idx, d in enumerate(data):
        print("\033[2J")
        for i in d:
            print(i)
        if idx != 0:
            sleep(interval)


def _read_video_or_gif(path: str) -> List[np.ndarray]:
    data = cv2.VideoCapture(path)
    rets = []
    while True:
        ret, frame = data.read()
        if not ret:
            break
        rets.append(frame)
    data.release()
    return rets


def _apply_convert(bool_maps: np.ndarray) -> List[List[str]]:
    oh, ow = bool_maps[0].shape
    h = oh // 4
    w = ow // 2
    raw = [[] for _ in range(len(bool_maps))]
    for idx, b in enumerate(bool_maps):
        for i in range(h):
            converted = [convert2braille(b, i, j, oh, ow) for j in range(w)]
            raw[idx].append(r"".join(converted))
    return raw


def _apple_color(raw, bg_color, bgr_data):
    oh, ow = bgr_data[0].shape[:2]
    h = oh // 4
    w = ow // 2
    color_raw = [[] for _ in range(len(raw))]
    colors = [cv2.resize(i, (w, h), interpolation=cv2.INTER_LINEAR)
              for i in bgr_data]
    colors = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in colors]
    for idx, r in enumerate(raw):
        for i, line in enumerate(r):
            converted = [
                render(colors[idx], i, j, c, bg_color) for j, c in enumerate(line)
            ]
            color_raw[idx].append(r"".join(converted))
    return color_raw


def convert(
    source: str,
    with_color: bool = False,
    scale: float = 1.0,
    threshold: int = -1,
    save_raw: Optional[str] = None,
    bg_color: Optional[Tuple[int, int, int]] = None,
    fast: bool = False,
    chunk_size: int = False,
):
    ext = osp.splitext(source)[1][1:]
    try:
        if ext in ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"]:
            # TODO: detect number of channels to handle alpha in png
            bgr_data = cv2.imread(source)
        else:
            bgr_data = _read_video_or_gif(source)
    except:
        raise RuntimeError(f"Not support for {ext} file.")
    if not isinstance(bgr_data, list):
        bgr_data = [bgr_data]

    if scale != 1.0:
        bgr_data = [_resize(i, scale, "nearest") for i in bgr_data]

    gray_data = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in bgr_data]
    bool_maps = [apply_threshold(i, threshold) for i in gray_data]
    
    if fast:
        index_maps = _fast_convert2braille(bool_maps, chunk_size)
        raw = [_get_braille(x) for x in index_maps]
    else:
        raw = _apply_convert(bool_maps)
    color_raw = [[] for _ in range(len(gray_data))]

    if not with_color:
        print_converted(raw)

    if with_color:
        if bg_color == (-1, -1, -1):
            bg_color = None
        color_raw = _apple_color(raw, bg_color, bgr_data)

        print_converted(color_raw)

    if save_raw:
        with open(save_raw, "w", encoding="UTF-8") as f:
            for i in color_raw or raw:
                for line in i:
                    f.writelines(line)
                    f.write("\n")
                f.write("\n")
