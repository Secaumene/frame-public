#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""语言策略机械检查脚本(language policy checker)。

用途:
    按 docs/standards/language-policy.md 的 LANG-001 / LANG-002 / LANG-003
    执行三项可机械判定的检查:
    1. 文件/目录名:含非 ASCII 字符一律 FAIL;基名须匹配 ^[a-z0-9_.-]+$。
       豁免:生态惯例文件白名单(CMakeLists.txt、README.md 等,见
       NAME_WHITELIST)、所有点开头的文件与目录(含 .github)。
    2. 源码(.h/.cpp/.cu/.py/.cmake/CMakeLists.txt)剥离注释与字符串后,
       剩余代码不得含汉字(标识符纯英文)。
    3. 注释块级中文判定:连续整行注释构成一个注释块,每块须含至少一个
       汉字。行级豁免:shebang、纯 URL/路径/代码行、NOLINT 与 clang-format
       指令行、合法 FRAME 待办标签行自身;块级豁免:许可证头。

用法:
    python3 scripts/check_language_policy.py [repo_root]
    repo_root 缺省为本脚本所在目录的上一级。优先扫描 git 追踪文件
    (含未被忽略的未跟踪文件);非 git 环境回退为全仓遍历,
    排除 .git 与 build* 目录。

输出:
    全部通过打印 OK 并以 0 退出;否则逐条打印
    "FAIL <文件>:<行> <原因>" 并以 1 退出(文件级问题行号记 0)。

规则出处:docs/standards/language-policy.md(LANG-001、LANG-002、LANG-003)。
"""

import os
import re
import subprocess
import sys

# 生态惯例文件白名单(LANG-002):这些名字允许大写字母
NAME_WHITELIST = {
    "CMakeLists.txt",
    "README.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "PULL_REQUEST_TEMPLATE.md",
    "CMakePresets.json",
    "Dockerfile",
}

BASENAME_RE = re.compile(r"^[a-z0-9_.-]+$")
# 汉字判定范围:CJK 统一表意文字基本区 U+4E00..U+9FFF
HAN_RE = re.compile("[一-鿿]")
# 合法待办标签正则;关键词以字符串拼接写出,
# 避免本脚本自身被 check_iron_rules.sh 的标签格式检查命中
TAG_RE = re.compile("TO" "DO" r"\(FRAME-(IMPL|DESIGN|TEST|DOC|PERF|DEP)\):")

C_EXTS = {".h", ".cpp", ".cu"}
EXCLUDE_DIR_NAMES = {".git", "__pycache__", ".pytest_cache", ".cache"}


def is_build_dir(name):
    """判定是否为构建产物目录(全仓遍历回退模式下排除)。"""
    return name == "build" or name.startswith(("build-", "build_", "cmake-build"))


def path_excluded(rel):
    """判定相对路径是否落在排除目录内。"""
    parts = rel.split("/")
    return any(p in EXCLUDE_DIR_NAMES or is_build_dir(p) for p in parts[:-1])


def list_repo_files(root):
    """列出待检文件的相对路径。

    优先使用 git(追踪文件 + 未被 .gitignore 忽略的未跟踪文件);
    以 -z 取原始字节并以 surrogateescape 解码,保证非 ASCII 文件名
    也能被检查到。非 git 环境回退为 os.walk 全仓遍历。
    """
    try:
        out = subprocess.run(
            ["git", "-C", root, "ls-files", "-z",
             "--cached", "--others", "--exclude-standard"],
            capture_output=True, check=True)
        files = []
        for raw in out.stdout.split(b"\0"):
            if not raw:
                continue
            rel = raw.decode("utf-8", errors="surrogateescape")
            if path_excluded(rel):
                continue
            if os.path.isfile(os.path.join(root, rel)):
                files.append(rel)
        return sorted(files)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass

    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in EXCLUDE_DIR_NAMES and not is_build_dir(d)]
        for name in filenames:
            rel = os.path.relpath(os.path.join(dirpath, name), root)
            files.append(rel.replace(os.sep, "/"))
    return sorted(files)


def check_names(files):
    """检查 1:文件与目录命名(LANG-002)。逐路径分量判定,去重报告。"""
    fails = []
    seen = set()
    for rel in files:
        parts = rel.split("/")
        for i, comp in enumerate(parts):
            sub = "/".join(parts[: i + 1])
            if sub in seen:
                continue
            seen.add(sub)
            if comp.startswith("."):
                continue  # 点开头文件/目录豁免(含 .github)
            if comp in NAME_WHITELIST:
                continue
            if not comp.isascii():
                fails.append((sub, 0,
                              "non-ASCII file or directory name (LANG-002)"))
            elif not BASENAME_RE.match(comp):
                fails.append((sub, 0,
                              "name does not match ^[a-z0-9_.-]+$ (LANG-002)"))
    return fails


def split_c(text):
    """C/C++/CUDA 源码分离器。

    按字符状态机将文本切分为「代码」与「注释」两个以行号为键的字典;
    字符串与字符字面量被整体剥离(既不计入代码也不计入注释)。
    """
    code = {}
    comments = {}
    line = 1
    i = 0
    n = len(text)
    state = "code"
    quote = ""
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if c == "\n":
            line += 1
            i += 1
            if state == "line_comment":
                state = "code"
            continue
        if state == "code":
            if c == "/" and nxt == "/":
                state = "line_comment"
                comments.setdefault(line, "")
                i += 2
                continue
            if c == "/" and nxt == "*":
                state = "block_comment"
                comments.setdefault(line, "")
                i += 2
                continue
            if c in "\"'":
                state = "string"
                quote = c
                i += 1
                continue
            code[line] = code.get(line, "") + c
            i += 1
            continue
        if state == "line_comment":
            comments[line] = comments.get(line, "") + c
            i += 1
            continue
        if state == "block_comment":
            if c == "*" and nxt == "/":
                state = "code"
                i += 2
                continue
            comments[line] = comments.get(line, "") + c
            i += 1
            continue
        # 此处 state == "string":字符串字面量状态
        if c == "\\":
            if nxt == "\n":
                line += 1
            i += 2
            continue
        if c == quote:
            state = "code"
        i += 1
    return code, comments


def split_py(text):
    """Python 源码分离器:# 注释、单/双引号与三引号字符串。"""
    code = {}
    comments = {}
    line = 1
    i = 0
    n = len(text)
    state = "code"
    quote = ""
    while i < n:
        c = text[i]
        if c == "\n":
            line += 1
            i += 1
            if state == "comment":
                state = "code"
            continue
        if state == "code":
            if c == "#":
                state = "comment"
                comments.setdefault(line, "")
                i += 1
                continue
            if c in "\"'":
                if text[i:i + 3] == c * 3:
                    state = "tstring"
                    quote = c * 3
                    i += 3
                else:
                    state = "string"
                    quote = c
                    i += 1
                continue
            code[line] = code.get(line, "") + c
            i += 1
            continue
        if state == "comment":
            comments[line] = comments.get(line, "") + c
            i += 1
            continue
        if state == "string":
            if c == "\\":
                if text[i + 1:i + 2] == "\n":
                    line += 1
                i += 2
                continue
            if c == quote:
                state = "code"
            i += 1
            continue
        # 此处 state == "tstring":三引号字符串状态
        if c == "\\":
            if text[i + 1:i + 2] == "\n":
                line += 1
            i += 2
            continue
        if text[i:i + 3] == quote:
            state = "code"
            i += 3
            continue
        i += 1
    return code, comments


def split_cmake(text):
    """CMake 源码分离器:# 行注释、#[[ ]] 块注释、双引号字符串。"""
    code = {}
    comments = {}
    line = 1
    i = 0
    n = len(text)
    state = "code"
    while i < n:
        c = text[i]
        if c == "\n":
            line += 1
            i += 1
            if state == "comment":
                state = "code"
            continue
        if state == "code":
            if c == "#":
                if text[i + 1:i + 3] == "[[":
                    state = "block_comment"
                    comments.setdefault(line, "")
                    i += 3
                else:
                    state = "comment"
                    comments.setdefault(line, "")
                    i += 1
                continue
            if c == '"':
                state = "string"
                i += 1
                continue
            code[line] = code.get(line, "") + c
            i += 1
            continue
        if state == "comment":
            comments[line] = comments.get(line, "") + c
            i += 1
            continue
        if state == "block_comment":
            if text[i:i + 2] == "]]":
                state = "code"
                i += 2
                continue
            comments[line] = comments.get(line, "") + c
            i += 1
            continue
        # 此处 state == "string":字符串字面量状态
        if c == "\\":
            if text[i + 1:i + 2] == "\n":
                line += 1
            i += 2
            continue
        if c == '"':
            state = "code"
        i += 1
    return code, comments


SPLITTERS = {"c": split_c, "py": split_py, "cmake": split_cmake}


def classify(rel):
    """按扩展名判定源码类别;非源码返回 None(只做命名检查)。"""
    base = os.path.basename(rel)
    ext = os.path.splitext(base)[1]
    if base == "CMakeLists.txt" or ext == ".cmake":
        return "cmake"
    if ext == ".py":
        return "py"
    if ext in C_EXTS:
        return "c"
    return None


def comment_line_exempt(content, raw_line):
    """检查 3 的行级豁免判定。content 为剥离注释定界符后的文本。"""
    t = content.strip().lstrip("*!/=- \t").strip()
    if not t:
        return True  # 空注释行(装饰行)
    if raw_line.lstrip().startswith("#!"):
        return True  # shebang
    if "://" in t:
        return True  # URL
    if "NOLINT" in content or "clang-format" in content:
        return True  # 工具指令行
    if TAG_RE.search(content):
        return True  # 合法 FRAME 待办标签行自身
    if "-*-" in t:
        return True  # 编码声明等元指令
    if re.fullmatch(r"\S+", t):
        return True  # 单一 token:路径/标识符/文件名
    if t.isascii() and (t.endswith((";", "{", "}", ")", ",", "\\"))
                        or t.startswith(("#include", "#pragma", "#define",
                                         "$", ">>>"))):
        return True  # 代码示例行
    return False


def block_is_license(texts):
    """检查 3 的块级豁免:许可证头。"""
    joined = " ".join(t for _, t in texts).lower()
    return ("copyright" in joined or "spdx" in joined
            or "license" in joined or "licence" in joined)


def check_source(rel, text, kind, fails):
    """对单个源码文件执行检查 2 与检查 3。"""
    code, comments = SPLITTERS[kind](text)
    raw_lines = text.split("\n")

    # ---- 检查 2:剥离注释与字符串后代码不得含任何非 ASCII 字符(LANG-001)----
    # LANG-001 要求标识符仅含 [A-Za-z0-9_]:只查汉字会放过西里尔/希腊/带附标
    # 字母等非 ASCII 标识符,故按整行 ASCII 判定(注释与字符串已剥离,不误伤)。
    for ln in sorted(code):
        if not code[ln].isascii():
            fails.append((rel, ln,
                          "non-ASCII character in code outside comments/strings"
                          " (LANG-001)"))

    # ---- 检查 3:注释块级中文判定(LANG-003)----
    # 整行注释 = 该行代码部分为空白且存在注释文本
    full_lines = sorted(ln for ln in comments if not code.get(ln, "").strip())
    blocks = []
    for ln in full_lines:
        if blocks and ln == blocks[-1][-1] + 1:
            blocks[-1].append(ln)
        else:
            blocks.append([ln])
    for block in blocks:
        texts = [(ln, comments[ln]) for ln in block]
        if any(HAN_RE.search(t) for _, t in texts):
            continue  # 块内已含汉字
        if block_is_license(texts):
            continue  # 许可证头豁免
        if all(comment_line_exempt(t, raw_lines[ln - 1] if ln - 1 < len(raw_lines) else "")
               for ln, t in texts):
            continue  # 全部行命中行级豁免
        fails.append((rel, block[0],
                      "comment block contains no Chinese (LANG-003)"))


def main(argv):
    if len(argv) > 1:
        root = os.path.abspath(argv[1])
    else:
        root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

    files = list_repo_files(root)
    fails = []
    fails.extend(check_names(files))

    for rel in files:
        kind = classify(rel)
        if kind is None:
            continue
        try:
            with open(os.path.join(root, rel), encoding="utf-8",
                      errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue  # 无法读取(如损坏的符号链接)则跳过内容检查
        check_source(rel, text, kind, fails)

    if fails:
        for path, line, reason in fails:
            print("FAIL {}:{} {}".format(path, line, reason))
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
