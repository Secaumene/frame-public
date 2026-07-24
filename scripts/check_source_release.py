#!/usr/bin/env python3
"""源码发布检查器：确保待发布树只包含合规的文本源码。"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - 仅旧版 Python 会进入此分支。
    tomllib = None


EXPECTED_LICENSE_SHA256 = (
    "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986"
)
ALLOWED_GIT_MODES = {"100644", "100755"}
REJECTED_SUFFIXES = (
    ".tar.gz", ".tar.bz2", ".tar.xz", ".tar.zst", ".tgz", ".tbz", ".tbz2",
    ".txz", ".tar.lz", ".zip", ".7z", ".rar", ".tar", ".gz", ".bz2", ".xz",
    ".zst", ".lz", ".lzma", ".whl", ".egg", ".gem", ".deb", ".rpm", ".apk",
    ".msi", ".dmg", ".pkg", ".conda", ".cab",
    ".o", ".obj", ".a", ".lib", ".so", ".dylib", ".dll", ".exe", ".out",
    ".pyc", ".pyo", ".pyd", ".class", ".jar", ".wasm", ".bc", ".cubin", ".ptx",
    ".bin", ".onnx", ".pt", ".pth", ".tflite", ".safetensors", ".ckpt", ".pb",
    ".mlmodel", ".engine", ".plan", ".trt",
)
WORKFLOW_PATTERNS = (
    (re.compile(r"actions/upload-artifact(?:@|/)", re.IGNORECASE),
     "workflow upload-artifact action is forbidden"),
    (re.compile(r"\bgh\s+release\s+upload\b", re.IGNORECASE),
     "workflow GitHub release upload is forbidden"),
    (re.compile(r"\b(?:python\s+-m\s+)?twine\s+upload\b", re.IGNORECASE),
     "workflow PyPI upload is forbidden"),
    (re.compile(r"\bpoetry\s+publish\b", re.IGNORECASE),
     "workflow Poetry publish is forbidden"),
    (re.compile(r"\bpdm\s+publish\b", re.IGNORECASE),
     "workflow PDM publish is forbidden"),
    (re.compile(r"\bflit\s+publish\b", re.IGNORECASE),
     "workflow Flit publish is forbidden"),
    (re.compile(r"pypa/gh-action-pypi-publish", re.IGNORECASE),
     "workflow PyPI publish action is forbidden"),
    (re.compile(r"\b(?:docker|podman|nerdctl|buildah|crane|oras)(?:\s+image)?\s+push\b", re.IGNORECASE),
     "workflow container push is forbidden"),
    (re.compile(r"docker/build-push-action", re.IGNORECASE),
     "workflow container push action is forbidden"),
)


@dataclass(frozen=True)
class SourceFile:
    """保存单个待检查文件的来源、路径与字节内容。"""

    path: str
    data: bytes


class FailureParser(argparse.ArgumentParser):
    """将命令行参数错误统一输出为英文 FAIL 状态。"""

    def error(self, message: str) -> None:
        """终止参数解析并保留门禁可识别的失败前缀。"""
        self.exit(2, f"FAIL {message}\n")


def run_git(root: Path, arguments: list[str]) -> bytes | None:
    """在目标目录执行 Git 命令，失败时返回空值。"""
    try:
        result = subprocess.run(
            ["git", "-C", os.fspath(root), *arguments],
            check=False,
            capture_output=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def parse_ls_tree(output: bytes) -> list[tuple[str, str, str]] | None:
    """解析 NUL 分隔的 ls-tree 记录。"""
    records: list[tuple[str, str, str]] = []
    for record in output.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, kind, oid = metadata.decode("ascii").split(" ")
            path = raw_path.decode("utf-8", "surrogateescape")
        except (UnicodeDecodeError, ValueError):
            return None
        records.append((mode, kind, oid + "\0" + path))
    return records


def collect_git_ref(root: Path, reference: str | None, failures: list[str]) -> list[SourceFile] | None:
    """从指定 Git 引用读取完整树，完全忽略工作区状态。"""
    if not reference:
        failures.append("git-ref mode requires --ref")
        return None
    output = run_git(root, ["ls-tree", "-rz", "--full-tree", "-r", reference])
    if output is None:
        failures.append("git ls-tree failed")
        return None
    entries = parse_ls_tree(output)
    if entries is None:
        failures.append("git ls-tree returned malformed output")
        return None
    files: list[SourceFile] = []
    for mode, kind, packed in entries:
        oid, path = packed.split("\0", 1)
        if kind != "blob" or mode not in ALLOWED_GIT_MODES:
            failures.append(f"{path}: unsafe git tree entry mode {mode} type {kind}")
            continue
        data = run_git(root, ["cat-file", "blob", oid])
        if data is None:
            failures.append(f"{path}: git cat-file failed")
            continue
        files.append(SourceFile(path, data))
    return files


def parse_stage_modes(output: bytes) -> dict[str, str] | None:
    """解析 stage 0 索引模式；冲突索引条目视为不安全。"""
    modes: dict[str, str] = {}
    for record in output.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, _oid, stage = metadata.decode("ascii").split(" ")
            path = raw_path.decode("utf-8", "surrogateescape")
        except (UnicodeDecodeError, ValueError):
            return None
        if stage != "0" or path in modes:
            return None
        modes[path] = mode
    return modes


def collect_working_tree(root: Path, failures: list[str]) -> list[SourceFile] | None:
    """读取未忽略工作区文件，并同时核对索引模式和实际节点类型。"""
    listed = run_git(root, ["ls-files", "--cached", "--others", "--exclude-standard", "-z"])
    staged = run_git(root, ["ls-files", "--stage", "-z"])
    if listed is None or staged is None:
        failures.append("git ls-files failed")
        return None
    modes = parse_stage_modes(staged)
    if modes is None:
        failures.append("git index contains malformed or conflicted entries")
        return None
    paths = [item.decode("utf-8", "surrogateescape") for item in listed.split(b"\0") if item]
    files: list[SourceFile] = []
    for path in paths:
        if modes.get(path) not in (None, *ALLOWED_GIT_MODES):
            failures.append(f"{path}: unsafe cached mode {modes[path]}")
            continue
        candidate = root / path
        try:
            node = candidate.lstat()
        except FileNotFoundError:
            continue  # 已删除的工作区项不属于待发布集合。
        except OSError:
            failures.append(f"{path}: lstat failed")
            continue
        if stat.S_ISLNK(node.st_mode):
            failures.append(f"{path}: symlink is forbidden")
            continue
        if not stat.S_ISREG(node.st_mode):
            failures.append(f"{path}: non-regular file is forbidden")
            continue
        try:
            data = candidate.read_bytes()
        except OSError:
            failures.append(f"{path}: file read failed")
            continue
        files.append(SourceFile(path, data))
    return files


def collect_tree(root: Path, failures: list[str]) -> list[SourceFile] | None:
    """递归读取无 Git 目录树中的全部常规文件，不跟随符号链接。"""
    files: list[SourceFile] = []
    try:
        root_stat = root.lstat()
    except OSError:
        failures.append("tree root lstat failed")
        return None
    if not stat.S_ISDIR(root_stat.st_mode):
        failures.append("tree root is not a directory")
        return None
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in list(dirnames):
            candidate = current / name
            try:
                node = candidate.lstat()
            except OSError:
                failures.append(f"{candidate.relative_to(root)}: lstat failed")
                dirnames.remove(name)
                continue
            if stat.S_ISLNK(node.st_mode):
                failures.append(f"{candidate.relative_to(root)}: symlink is forbidden")
                dirnames.remove(name)
        for name in filenames:
            candidate = current / name
            relative = candidate.relative_to(root).as_posix()
            try:
                node = candidate.lstat()
            except OSError:
                failures.append(f"{relative}: lstat failed")
                continue
            if stat.S_ISLNK(node.st_mode):
                failures.append(f"{relative}: symlink is forbidden")
                continue
            if not stat.S_ISREG(node.st_mode):
                failures.append(f"{relative}: non-regular file is forbidden")
                continue
            try:
                data = candidate.read_bytes()
            except OSError:
                failures.append(f"{relative}: file read failed")
                continue
            files.append(SourceFile(relative, data))
    return files


def has_binary_control_bytes(data: bytes) -> bool:
    """识别任意路径中不属于文本空白的二进制控制字节。"""
    return any(byte == 0x7F or (byte < 0x20 and byte not in (9, 10, 12, 13)) for byte in data)


def load_project_metadata(data: bytes, failures: list[str]) -> dict[str, object] | None:
    """读取项目元数据；Python 3.9/3.10 使用受限的标准库后备解析。"""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        failures.append("pyproject.toml: UTF-8 decoding failed")
        return None
    if tomllib is not None:
        try:
            project = tomllib.loads(text).get("project", {})
        except ValueError:
            failures.append("pyproject.toml: TOML parsing failed")
            return None
        if not isinstance(project, dict):
            failures.append("pyproject.toml: project table is invalid")
            return None
        return project

    section = re.search(
        r"(?ms)^\[project\][ \t]*(?:#.*)?\n(.*?)(?=^\[|\Z)",
        text,
    )
    if section is None:
        failures.append("pyproject.toml: project table is missing")
        return None
    body = section.group(1)
    license_matches = re.findall(
        r"(?m)^[ \t]*license[ \t]*=[ \t]*([\"'])([^\"']+)\1[ \t]*(?:#.*)?$",
        body,
    )
    license_files_matches = re.findall(
        r"(?m)^[ \t]*license-files[ \t]*=[ \t]*\[[ \t]*([\"'])([^\"']+)\1[ \t]*\]"
        r"[ \t]*(?:#.*)?$",
        body,
    )
    if len(license_matches) != 1 or len(license_files_matches) != 1:
        failures.append("pyproject.toml: fallback metadata parsing failed")
        return None
    return {
        "license": license_matches[0][1],
        "license-files": [license_files_matches[0][1]],
    }


def validate_content(files: list[SourceFile], failures: list[str]) -> None:
    """执行文本、后缀、工作流和发布元数据的完整检查。"""
    by_path = {item.path: item.data for item in files}
    for item in files:
        lowered = item.path.lower()
        if lowered.endswith(REJECTED_SUFFIXES):
            failures.append(f"{item.path}: compiled, package, or archive suffix is forbidden")
        if b"\0" in item.data:
            failures.append(f"{item.path}: NUL byte is forbidden")
        try:
            text = item.data.decode("utf-8")
        except UnicodeDecodeError:
            failures.append(f"{item.path}: UTF-8 decoding failed")
            continue
        if has_binary_control_bytes(item.data):
            failures.append(f"{item.path}: binary control byte is forbidden")
        if item.path.startswith(".github/workflows/") and item.path.lower().endswith((".yml", ".yaml")):
            for pattern, reason in WORKFLOW_PATTERNS:
                if pattern.search(text):
                    failures.append(f"{item.path}: {reason}")

    license_data = by_path.get("LICENSE")
    if license_data is None:
        failures.append("LICENSE: root LICENSE file is missing")
    elif hashlib.sha256(license_data).hexdigest() != EXPECTED_LICENSE_SHA256:
        failures.append("LICENSE: SHA256 does not match GNU GPLv3 text")

    pyproject_data = by_path.get("pyproject.toml")
    if pyproject_data is None:
        failures.append("pyproject.toml: file is missing")
    else:
        project = load_project_metadata(pyproject_data, failures)
        if project is not None:
            if project.get("license") != "GPL-3.0-or-later":
                failures.append("pyproject.toml: project.license must be GPL-3.0-or-later")
            if project.get("license-files") != ["LICENSE"]:
                failures.append("pyproject.toml: project.license-files must be [\"LICENSE\"]")


def check_root(root: Path, mode: str, reference: str | None, safety_only: bool) -> list[str]:
    """按指定模式收集文件并返回全部失败项。"""
    failures: list[str] = []
    if mode == "git-ref":
        files = collect_git_ref(root, reference, failures)
    elif mode == "working-tree":
        files = collect_working_tree(root, failures)
    elif mode == "tree":
        files = collect_tree(root, failures)
    else:
        failures.append(f"unknown mode: {mode}")
        return failures
    if files is None:
        return failures
    if not files:
        failures.append("source file collection is empty")
        return failures
    if not failures and not safety_only:
        validate_content(files, failures)
    return failures


def main(argv: list[str] | None = None) -> int:
    """解析命令行并输出稳定的英文检查状态。"""
    parser = FailureParser(description="source-only release checker")
    parser.add_argument("positional_mode", nargs="?", metavar="mode", help="legacy mode argument")
    parser.add_argument("--mode", choices=("git-ref", "working-tree", "tree"), help="check mode")
    parser.add_argument("--root", help="source tree root")
    parser.add_argument("--ref", help="Git reference required by git-ref mode")
    parser.add_argument("--safety-only", action="store_true", help="skip text and metadata checks")
    args = parser.parse_args(argv)
    if args.mode and args.positional_mode and args.mode != args.positional_mode:
        print("FAIL mode conflict between positional mode and --mode")
        return 1
    mode = args.mode or args.positional_mode
    if mode is None:
        print("FAIL mode is required")
        return 1
    root = Path(args.root) if args.root else Path.cwd()
    if not root.exists():
        print(f"FAIL root path does not exist: {root}")
        return 1
    if not root.is_dir():
        print(f"FAIL root path is not a directory: {root}")
        return 1
    failures = check_root(root, mode, args.ref, args.safety_only)
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    suffix = " (safety-only)" if args.safety_only else ""
    print(f"OK source-only release: {mode}{suffix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
