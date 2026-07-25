"""快速入门章节与共用示例源码的同步回归测试。"""

import re
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HELP_PATH = REPOSITORY_ROOT / "help" / "01-quickstart" / "README.md"
EXAMPLE_PATH = REPOSITORY_ROOT / "examples" / "00_quickstart" / "main.cpp"
EXAMPLES_CMAKE_PATH = REPOSITORY_ROOT / "examples" / "CMakeLists.txt"
INSTALL_CONSUMER_CMAKE_PATH = (
    REPOSITORY_ROOT
    / "tests"
    / "cmake"
    / "install_consumer"
    / "project"
    / "CMakeLists.txt"
)


class HelpQuickstartTest(unittest.TestCase):
    """验证章节代码、仓内示例和安装消费测试使用同一源码。"""

    def test_cpp_block_matches_quickstart_example(self):
        """章节必须仅含一个 C++ 代码块，且与示例源码逐字一致。"""
        help_text = HELP_PATH.read_text(encoding="utf-8")
        cpp_blocks = re.findall(
            r"^```cpp[ \t]*\r?\n(.*?)^```[ \t]*$",
            help_text,
            flags=re.MULTILINE | re.DOTALL,
        )

        self.assertEqual(1, len(cpp_blocks))
        documented_source = cpp_blocks[0].strip("\r\n")
        example_source = EXAMPLE_PATH.read_text(encoding="utf-8").rstrip("\r\n")
        self.assertEqual(example_source, documented_source)

    def test_quickstart_example_is_shared_by_both_cmake_tests(self):
        """仓内 CTest 与安装后仓外消费必须引用同一快速入门源码。"""
        example_relative_path = "00_quickstart/main.cpp"
        examples_cmake = EXAMPLES_CMAKE_PATH.read_text(encoding="utf-8")
        install_consumer_cmake = INSTALL_CONSUMER_CMAKE_PATH.read_text(
            encoding="utf-8"
        )

        self.assertIn(example_relative_path, examples_cmake)
        self.assertIn(example_relative_path, install_consumer_cmake)


if __name__ == "__main__":
    unittest.main()
