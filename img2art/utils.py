import os.path as osp
from math import ceil
from time import sleep
from typing import List, Optional, Tuple, Union

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

HL_NAME = "I2A"
NVIM_HL_TEMP = "{%s,%d,%d}, "
HL_TEMP = 'vim.api.nvim_set_hl(0,"%s",{%s})'
HL_MAPPER = {}
HL_IDX = 0
ALPHA_HEADER_TEMP = """
local header = { 
    type='text',
    opts={
        position='center',
        hl = {
%s
        },
        val = {
%s
        }
    }
}
return header
-- dashboard.section.header = header
"""


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
    fore_color = tuple(color_mat[y, x, :])
    if bg_color:
        return "\033[38;2;{};{};{};48;2;{};{};{}m{}\033[0m".format(
            *fore_color, *bg_color, c
        )
    return "\033[38;2;{};{};{}m{}".format(*fore_color, c)


def _generate_hl(
    color_mat: np.ndarray,
    y: int,
    x: int,
    bg_color: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    global HL_IDX
    fore_color = "{:02x}{:02x}{:02x}".format(*color_mat[y, x, :])
    hl_name = HL_MAPPER.get(fore_color, None)
    res = None
    if not hl_name:
        hl_name = HL_NAME + str(HL_IDX)
        HL_IDX += 1
        HL_MAPPER[fore_color] = hl_name
        res = HL_TEMP % (
            hl_name,
            f'fg="#{fore_color}"' + (f', bg="#{bg_color}"' if bg_color else ""),
        )
    code = NVIM_HL_TEMP % (f'"{hl_name}"', 3 * x, 3 * x + 3)
    return res, code


def show(img):
    cv2.imshow("x", img)
    cv2.waitKey(5000)


def _resize(img, scale, inter_type):
    if inter_type == "nearest":
        inter = cv2.INTER_NEAREST
    else:
        inter = cv2.INTER_LINEAR
    if isinstance(scale, list):
        scalex, scaley = scale
    else:
        scalex = scaley = scale

    return cv2.resize(img, (0, 0), fx=scalex, fy=scaley, interpolation=inter)


def print_converted(
    data: np.ndarray, interval: float = 0.05, loop: bool = False
) -> None:
    while True:
        for idx, d in enumerate(data):
            print("\033[2J")
            for i in d:
                print(i)
            if idx != 0:
                sleep(interval)
        if not loop:
            break


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


def _convert_nvim_hl(rgb_data, bg_color):
    if bg_color:
        bg_color = "{:02x}{:02x}{:02x}".format(*bg_color)
    h = rgb_data.shape[0]
    w = rgb_data.shape[1]
    hl = []
    code = []
    for i in range(h):
        generated = [_generate_hl(rgb_data, i, j, bg_color) for j in range(w)]
        hl.extend([item[0] for item in generated if item[0]])
        code.append("{%s}," % ("".join([item[1] for item in generated])))
    return hl, "\n".join(code)


def _apply_color(raw, bg_color, rgb_data):
    color_raw = [[] for _ in range(len(raw))]
    for idx, r in enumerate(raw):
        for i, line in enumerate(r):
            converted = [
                render(rgb_data[idx], i, j, c, bg_color) for j, c in enumerate(line)
            ]
            color_raw[idx].append(r"".join(converted))
    return color_raw


def _convert_color(bgr_data, to_resize):
    if to_resize:
        oh, ow = bgr_data[0].shape[:2]
        h = oh // 4
        w = ow // 2
        bgr_data = [
            cv2.resize(i, (w, h), interpolation=cv2.INTER_LINEAR) for i in bgr_data
        ]
    colors = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in bgr_data]
    return colors


def _save(raw_data: List[List[str]], path: str):
    with open(path, "w", encoding="UTF-8") as f:
        for i in raw_data:
            for line in i:
                f.writelines(line)
                f.write("\n")
            f.write("\n")


def _convert_to_lua_str_fmt(data: List[str]):
    return "\n".join([f"[[{d}]]," for d in data])


def _quant(img, k: int):
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, palette = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    palette = palette.astype(np.uint8)
    quantized = palette[labels.flatten()]
    quantized_img = quantized.reshape((img.shape))
    return quantized_img


def _convert_mapping(gray_img, mapping):
    uni = np.unique(gray_img[0]).tolist()
    mapping = {u: mapping[i] for i, u in enumerate(uni)}
    vectorized_mapping = np.vectorize(lambda x: mapping[x])
    res = []
    for frame in gray_img:
        merge_lines = []
        r = vectorized_mapping(frame).tolist()
        for i in r:
            merge_lines.append("".join(i))
        res.append(merge_lines)
    return res


def convert(
    source: str,
    with_color: bool = False,
    scale: Union[float, List[float]] = 1.0,
    threshold: int = -1,
    save_raw: Optional[str] = None,
    bg_color: Optional[Tuple[int, int, int]] = None,
    fast: bool = False,
    chunk_size: int = False,
    alpha: bool = False,
    quant: int = -1,
    mapping: str = "",
    loop: bool = False,
    interval: float = 0.05,
):
    ext = osp.splitext(source)[1][1:]
    try:
        if ext in ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"]:
            # TODO: detect number of channels to handle alpha in png
            bgr_data = cv2.imread(source)
        else:
            assert not alpha
            bgr_data = _read_video_or_gif(source)
    except:
        if alpha:
            raise RuntimeError("Do not support convert video in alpha mode.")
        else:
            raise RuntimeError(f"Not support for {ext} file.")
    if not isinstance(bgr_data, list):
        bgr_data = [bgr_data]

    if scale != 1.0:
        bgr_data = [_resize(i, scale, "nearest") for i in bgr_data]

    if quant > 0:
        bgr_data = [_quant(frame, quant) for frame in bgr_data]

    gray_data = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in bgr_data]

    if mapping:
        raw = _convert_mapping(gray_data, mapping)
    else:
        bool_maps = [apply_threshold(i, threshold) for i in gray_data]
        if fast:
            index_maps = _fast_convert2braille(bool_maps, chunk_size)
            raw = [_get_braille(x) for x in index_maps]
        else:
            raw = _apply_convert(bool_maps)

    color_raw = None
    hl_data = None

    if with_color:
        resized_rgb_data = _convert_color(bgr_data, mapping == "")
        color_raw = _apply_color(raw, bg_color, resized_rgb_data)
        if alpha:
            hl_data, code = _convert_nvim_hl(resized_rgb_data[0], bg_color)
            hl_data = [
                [*hl_data, ALPHA_HEADER_TEMP % (code, _convert_to_lua_str_fmt(raw[0]))]
            ]

    print_converted(color_raw or raw, interval=interval, loop=loop)

    if save_raw:
        _save(hl_data or color_raw or raw, save_raw)
