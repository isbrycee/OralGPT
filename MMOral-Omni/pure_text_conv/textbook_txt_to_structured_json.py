#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict


def load_tokenizer(model_id: Optional[str] = None):
    """
    加载 Qwen 2.5 系列的 tokenizer。默认优先使用 Qwen2.5-VL 的 tokenizer。
    """
    from transformers import AutoTokenizer

    tried = []
    if model_id:
        candidate_ids = [model_id]
    else:
        candidate_ids = [
            "Qwen/Qwen2.5-VL-7B-Instruct",  # 首选：VL 2.5
            "Qwen/Qwen2.5-7B",              # 回退：纯文本 2.5
            "Qwen/Qwen2-VL-7B-Instruct",    # 备选：VL 2.0
        ]

    last_err = None
    for mid in candidate_ids:
        try:
            print(f"Loading tokenizer: {mid}")
            tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True, use_fast=False)
            return tok, mid
        except Exception as e:
            tried.append(mid)
            last_err = e
            print(f"Failed to load tokenizer {mid}: {e}")

    raise RuntimeError(f"Failed to load any tokenizer. Tried: {tried}. Last error: {last_err}")

def read_text_file(path: Path) -> str:
    """
    尝试用常见编码读取文本文件，优先 utf-8。
    """
    encodings = ["utf-8", "utf-8-sig", "gb18030", "utf-16", "latin-1"]
    for enc in encodings:
        try:
            with path.open("r", encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    # 最终尝试二进制读取并解码失败字符
    with path.open("rb") as f:
        data = f.read()
    return data.decode("utf-8", errors="ignore")

def split_text_by_dashes(text: str) -> List[str]:
    """
    使用仅由连字符组成的一整行作为分隔符（例如 '---', '-----'），进行分段。
    - 只匹配整行的分隔线（两端可有空白）
    - 分割后会 strip 段落首尾空白
    - 过滤空段
    """
    # 标准化换行
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 使用多行模式，将仅包含若干 '-' 的行作为分隔符
    pattern = re.compile(r"^[ \t]*-{3,}[ \t]*$", flags=re.MULTILINE)
    parts = re.split(pattern, text)
    # 清理空段
    cleaned = []
    for p in parts:
        q = p.strip()
        if q:
            cleaned.append(q)
    return cleaned

def count_tokens(tokenizer, text: str) -> int:
    """
    使用指定 tokenizer 统计文本 token 数（不加特殊 token）。
    """
    # 有些 tokenizer 的 __call__/encode 行为不同，这里统一用 encode
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return len(ids)
    except TypeError:
        # 某些 tokenizer 需要作为 __call__ 调用
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        return len(ids)

def split_into_two_by_newline(text: str) -> List[str]: 
    """ 将文本按最接近中点的换行符拆成两段；若没有换行，则按字符中点拆。 返回两个去除首尾空白的子串。 """ 
    newlines = [i for i, ch in enumerate(text) if ch == "\n"] 
    if not newlines: 
        mid = len(text) // 2 
        left = text[:mid].strip()
        right = text[mid:].strip() 
        return [s for s in [left, right] if s]

    target = len(text) // 2
    split_idx = min(newlines, key=lambda i: abs(i - target))

    left = text[:split_idx].rstrip("\n").strip()
    right = text[split_idx + 1 :].strip()

    # 避免极端情况下产生空段，回退到字符中点
    if not left or not right:
        mid = len(text) // 2
        left = text[:mid].strip()
        right = text[mid:].strip()

    return [s for s in [left, right] if s]

def filter_text(text: str) -> str: 
    """ 过滤规则： 
    1) 删除包含 Figure 和 : 的行（大小写不敏感，Figure 需为完整单词） 
    2) 删除包含常见图片扩展名或图片描述（.jpg/.png/...、<img>、Markdown 图片、data:image/...）的行 返回保留的行，使用原始的换行符拼接。 
    """ 

    import re
    # 匹配常见图片扩展名（末尾可跟非字母数字字符或行尾），大小写不敏感
    img_ext_re = re.compile(
        r'\.(?:jpg|jpeg|png|gif|bmp|webp|tiff?|svg|heic|heif)(?=[^\w]|$)',
        re.IGNORECASE
    )
    # Figure 与 : 同行（Figure 为完整单词）
    figure_word_re = re.compile(r'\bfigure\b', re.IGNORECASE)

    # 常见图片相关描述：HTML img 标签、Markdown 图片、data:image/...
    html_img_re = re.compile(r'<img\b', re.IGNORECASE)
    md_img_re = re.compile(r'!$[^$]*$$[^)]+?$', re.IGNORECASE)
    data_img_re = re.compile(r'data:image/(?:png|jpeg|gif|webp|svg|bmp|tiff?)', re.IGNORECASE)

    lines = text.split('\n')
    kept = []

    for line in lines:
        check = line.strip()

        # 规则 1：Figure 与冒号同一行
        if ':' in check and figure_word_re.search(check):
            continue

        # 规则 2：包含图片扩展名
        if img_ext_re.search(check):
            continue

        # 规则 2 扩展：常见图片标记
        if html_img_re.search(check) or md_img_re.search(check) or data_img_re.search(check):
            continue

        kept.append(line)

    return '\n'.join(kept)

def process_directory(
    input_dir: Path,
    output_jsonl: Path,
    tokenizer,
    bin_size: int = 2000,
    split_threshold: int = 8000,
) -> Tuple[int, int, int, int, float]:
    """
    处理目录下的所有 .txt 文件，结果写入 JSONL。
    返回统计信息：文件数、段落数、总 token、最大段 token、平均 token。
    """
    txt_files = sorted([p for p in input_dir.glob("*.txt") if p.is_file()])
    if not txt_files:
        raise FileNotFoundError(f"目录中未找到 .txt 文件: {input_dir}")

    total_files = 0
    total_segments = 0
    total_tokens = 0
    max_tokens = 0
    min_tokens: Optional[int] = None
    bin_counts: Dict[int, int] = {}

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as fout:
        for fp in txt_files:
            total_files += 1
            content = read_text_file(fp)
            segments = split_text_by_dashes(content)
            if not segments:
                print(f"警告：文件无有效分段（或全部为空）: {fp}")
                continue

            for seg in segments:
                seg = filter_text(seg)
                n_tok = count_tokens(tokenizer, seg)

                # 如果超过阈值，按换行符拆成两个近似等长的段
                final_segs = [seg]
                if n_tok > split_threshold:
                    sub_segs = split_into_two_by_newline(seg)
                    if len(sub_segs) >= 2:
                        final_segs = sub_segs
                    # 若只得到一个有效子段（极少数情况），则仍用原段

                for s in final_segs:
                    s_tok = count_tokens(tokenizer, s)

                    total_segments += 1
                    total_tokens += s_tok
                    max_tokens = max(max_tokens, s_tok)
                    if min_tokens is None:
                        min_tokens = s_tok
                    else:
                        min_tokens = min(min_tokens, s_tok)

                    # 更新分布统计
                    bin_idx = s_tok // bin_size if s_tok >= 0 else 0
                    bin_counts[bin_idx] = bin_counts.get(bin_idx, 0) + 1

                    # 写入 JSONL
                    record = {"text": s}
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    avg_tokens = float(total_tokens) / total_segments if total_segments > 0 else 0.0
    if min_tokens is None:
        min_tokens = 0
    return total_files, total_segments, total_tokens, max_tokens, min_tokens, avg_tokens, bin_counts

def main():
    parser = argparse.ArgumentParser(description="按 '---' 分隔 txt 并统计 token，输出为 JSONL")
    parser.add_argument("--input_dir", type=str, default='/home/jinghao/projects/x-ray-VLM/RGB/pure_text_conv_data/textbook_txt', help="输入的 txt 文件夹路径")
    parser.add_argument("-o", "--output", type=str, default=None, help="输出的 jsonl 文件路径，默认在输入目录下生成 segments.jsonl")
    parser.add_argument("--model-id", type=str, default=None, help="自定义 tokenizer 的模型 ID（可选）")
    parser.add_argument("--bin-size", type=int, default=2000, help="token 分布区间大小（默认 2000）")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"输入路径不是有效目录: {input_dir}")

    output_path = Path(args.output) if args.output else (input_dir / "segments.jsonl")

    tokenizer, used_model = load_tokenizer(args.model_id)
    print(f"Using tokenizer: {used_model}")

    files, segs, toks, max_tok, min_tok, avg_tok, bin_counts = process_directory(input_dir, output_path, tokenizer)

    print("处理完成：")
    print(f"- 使用 tokenizer: {used_model}")
    print(f"- 输入目录: {str(input_dir)}")
    print(f"- 输出文件: {str(output_path)}")
    print(f"- 文件数: {files}")
    print(f"- 段落数: {segs}")
    print(f"- Token 总数: {toks}")
    print(f"- 段落最大 token: {max_tok}")
    print(f"- 段落最小 token: {min_tok}")
    print(f"- 段落平均 token: {avg_tok:.2f}")

    # 打印分布统计
    if bin_counts:
        print(f"- Token 分布（每 {args.bin_size} tokens 一个区间）:")
        for bin_idx in sorted(bin_counts.keys()):
            start = bin_idx * args.bin_size
            end = start + args.bin_size - 1
            count = bin_counts[bin_idx]
            print(f"  {start}-{end}: {count}")
    else:
        print("- 无可统计的分布数据（未产生有效段落）")

if __name__ == "__main__":
    main()
