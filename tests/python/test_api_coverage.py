"""机械化核对 python/frame/_core.pyi 导出清单与 tests/python/ 测试源文本的
引用覆盖(BUILD-021:"绑定层每个暴露 API 至少 1 个用例",判定方法="对 .pyi
导出清单与 tests/python/ 中被引用符号求差集,差集应为空")。

判定口径(用可执行断言固化,与 code-reviewer 人工核对方法同构):
- 导出符号集合 = `.pyi` 顶层 class/def 名字,以及每个 class 体内的非 dunder
  方法/属性名 + 简单赋值目标名(用 `ast` 解析 `.pyi`——它是合法 Python 语法,
  不手写正则扫描,避免正则本身遗漏语法变体)。DType 的三个枚举成员
  (float32/float16/bfloat16)按"class 体内简单赋值"同一逻辑纳入。
- 覆盖判定:该符号名以整词边界(`\\b`)出现在除本文件外的其余
  `tests/python/test_*.py` 源文本中即视为"被引用"。
- 取舍:整词匹配无法区分"真实调用/属性访问"与"巧合同名英文单词"(如
  `Graph.name` 与其余代码里独立出现的单词 "name"),存在假阳性放行的风险;
  本测试选择宁可假阳性放行也不假阴性拦截真实覆盖(不使用更严格的
  `\\.symbol\\b` 属性访问模式,因为顶层类名/函数名的常见调用形态是
  `core.Graph(...)` 这种前缀不含 `.symbol` 紧邻场景之外的多种写法,严格模式
  反而容易漏报)。本仓库其余 test_*.py 内每个导出符号均有对应的真实调用
  (人工核对,见各测试文件的具体用例),故本测试的宽松整词匹配在当前测试
  套件下不产生实际的假阳性掩盖。
- 已知例外:`Node` 类当前在 `python/src/*.cpp` 中未绑定任何构造函数,且没有
  任何导出函数返回 `Node` 实例(公开算子绑定与 `add_graph_input` 均返回 `Value`,
  见 python/src/bind_ops.cpp、bind_graph.cpp)——纯 Python 侧无法真正构造/
  获得一个 `Node` 对象,只能验证符号存在
  (见 test_graph_ops.py::test_node_type_is_exported_but_currently_unreachable)。
  这不是本测试放宽验收线,而是该符号本身的可测试面就止于"存在性";记入
  test-writer 报告的"疑似实现问题"。
"""

import ast
import pathlib
import re

_PYI_PATH = pathlib.Path(__file__).resolve().parents[2] / "python" / "frame" / "_core.pyi"
_TEST_DIR = pathlib.Path(__file__).resolve().parent
_SELF_FILE_NAME = pathlib.Path(__file__).name


def _collect_exported_symbols() -> set:
    """解析 .pyi,收集顶层 class/def 名字 + class 体内非 dunder 方法名/简单
    赋值目标名(涵盖 DType 的枚举成员)。"""
    tree = ast.parse(_PYI_PATH.read_text(encoding="utf-8"))
    symbols = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            symbols.add(node.name)
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not item.name.startswith("__"):
                        symbols.add(item.name)
                elif isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name):
                            symbols.add(target.id)
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    symbols.add(item.target.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.add(node.name)
    return symbols


def _collect_other_test_source_text() -> str:
    """拼接除本文件外全部 test_*.py 的源文本,作为覆盖判定的检索目标。"""
    chunks = []
    for path in sorted(_TEST_DIR.glob("test_*.py")):
        if path.name == _SELF_FILE_NAME:
            continue
        chunks.append(path.read_text(encoding="utf-8"))
    return "\n".join(chunks)


def test_pyi_export_surface_has_no_uncovered_symbol():
    """.pyi 导出清单与其余 test_*.py 引用符号求差集,BUILD-021 判据:差集为
    空。"""
    exported_symbols = _collect_exported_symbols()
    assert exported_symbols, "sanity check: _core.pyi 解析出的导出符号集合不应为空"

    haystack = _collect_other_test_source_text()
    uncovered = {
        name for name in exported_symbols if re.search(r"\b" + re.escape(name) + r"\b", haystack) is None
    }
    assert uncovered == set(), (
        "symbols exported in python/frame/_core.pyi but not referenced by any "
        f"other tests/python/test_*.py file: {sorted(uncovered)}"
    )
