"""源码发布检查器的 unittest 覆盖。"""

import contextlib
import hashlib
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "check_source_release.py"
MODULE_SPEC = importlib.util.spec_from_file_location("check_source_release", SCRIPT_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = CHECKER
MODULE_SPEC.loader.exec_module(CHECKER)


class SourceReleaseCheckTest(unittest.TestCase):
    """为三种收集模式和完整内容检查构造独立临时树。"""

    def setUp(self):
        """创建每例独享的临时目录。"""
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        """释放临时目录。"""
        self.temporary.cleanup()

    def git(self, *arguments):
        """在临时仓执行 Git 命令。"""
        return subprocess.run(
            ["git", "-C", os.fspath(self.root), *arguments],
            check=True,
            capture_output=True,
        )

    def init_git(self):
        """初始化可提交测试树的 Git 仓。"""
        self.git("init", "-q")
        self.git("config", "user.name", "Source Release Test")
        self.git("config", "user.email", "source-release@example.invalid")

    def write(self, relative, data):
        """写入字节文件并按需创建父目录。"""
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def run_check(self, mode, reference=None, safety_only=False):
        """在临时树调用检查器核心并得到失败列表。"""
        return CHECKER.check_root(self.root, mode, reference, safety_only)

    @contextlib.contextmanager
    def synthetic_license_hash(self, data=b"synthetic GPLv3 text"):
        """临时替换哈希常量，使最小树测试无需复制长许可证正文。"""
        previous = CHECKER.EXPECTED_LICENSE_SHA256
        CHECKER.EXPECTED_LICENSE_SHA256 = hashlib.sha256(data).hexdigest()
        try:
            yield data
        finally:
            CHECKER.EXPECTED_LICENSE_SHA256 = previous

    def write_valid_minimum_tree(self, license_data):
        """写入通过完整检查的最小文本树。"""
        self.write("LICENSE", license_data)
        self.write(
            "pyproject.toml",
            b"[project]\nlicense = \"GPL-3.0-or-later\"\nlicense-files = [\"LICENSE\"]\n",
        )
        self.write("README.md", b"Frame source tree\n")

    def test_git_ref_head_symlink_and_index_delete_are_both_checked(self):
        """HEAD 符号链接与索引删除必须分别在两个模式中失败。"""
        self.init_git()
        self.write("target", b"target\n")
        (self.root / "linked").symlink_to("target")
        self.git("add", "target", "linked")
        self.git("commit", "-qm", "initial")
        self.git("rm", "-q", "linked")
        self.assertTrue(any("unsafe git tree entry mode 120000" in item for item in self.run_check("git-ref", "HEAD", safety_only=True)))
        self.assertEqual([], self.run_check("working-tree", safety_only=True))

    def test_git_ref_head_regular_and_index_staged_symlink_are_both_checked(self):
        """HEAD 常规文件不能掩盖索引中暂存的符号链接。"""
        self.init_git()
        self.write("regular", b"content\n")
        self.git("add", "regular")
        self.git("commit", "-qm", "initial")
        (self.root / "regular").unlink()
        (self.root / "regular").symlink_to("target")
        self.git("add", "regular")
        self.assertEqual([], self.run_check("git-ref", "HEAD", safety_only=True))
        self.assertTrue(any("unsafe cached mode 120000" in item for item in self.run_check("working-tree", safety_only=True)))

    def test_gitlink_mode_fails(self):
        """Gitlink 索引项必须被安全检查拒绝。"""
        self.init_git()
        self.write("submodule", b"not used\n")
        self.git("add", "submodule")
        self.git("commit", "-qm", "initial")
        commit_oid = self.git("rev-parse", "HEAD").stdout.decode("ascii").strip()
        self.git("update-index", "--cacheinfo", "160000," + commit_oid + ",submodule")
        failures = self.run_check("working-tree", safety_only=True)
        self.assertTrue(any("unsafe cached mode 160000" in item for item in failures))

    def test_tree_symlink_fails(self):
        """无 Git 树中的符号链接不得被跟随。"""
        self.write("target", b"text\n")
        (self.root / "linked").symlink_to("target")
        self.assertTrue(any("linked: symlink is forbidden" == item for item in self.run_check("tree", safety_only=True)))

    def test_non_utf8_fails(self):
        """非 UTF-8 文件必须在完整检查中失败。"""
        self.write("bad.txt", b"\xff")
        self.assertTrue(any("bad.txt: UTF-8 decoding failed" == item for item in self.run_check("tree")))

    def test_nul_fails(self):
        """包含 NUL 的文本文件必须失败。"""
        self.write("bad.txt", b"text\0more")
        self.assertTrue(any("bad.txt: NUL byte is forbidden" == item for item in self.run_check("tree")))

    def test_tar_gz_fails(self):
        """双扩展名归档必须被拒绝。"""
        self.write("source.tar.gz", b"text")
        self.assertTrue(any("source.tar.gz: compiled, package, or archive suffix is forbidden" == item for item in self.run_check("tree")))

    def test_extensionless_binary_control_byte_fails(self):
        """无扩展名文件中的二进制控制字节必须被拒绝。"""
        self.write("program", b"\x01binary")
        self.assertTrue(any("program: binary control byte is forbidden" == item for item in self.run_check("tree")))

    def test_suffixed_binary_control_byte_fails(self):
        """带扩展名路径也不能借控制字节绕过二进制检查。"""
        self.write("payload.dat", b"\x01binary")
        self.assertTrue(any("payload.dat: binary control byte is forbidden" == item for item in self.run_check("tree")))

    def test_compiled_package_and_model_suffixes_fail(self):
        """新增的编译、模型和包格式后缀必须逐个被拒绝。"""
        suffixes = (".cubin", ".ptx", ".onnx", ".pt", ".pth", ".tflite", ".conda", ".pyd", ".bin")
        for suffix in suffixes:
            with self.subTest(suffix=suffix):
                filename = "payload" + suffix
                self.write(filename, b"text")
                failures = self.run_check("tree")
                self.assertTrue(any(f"{filename}: compiled, package, or archive suffix is forbidden" == item for item in failures))
                (self.root / filename).unlink()

    def test_workflow_publish_command_fails(self):
        """工作流上传或发布命令必须失败。"""
        self.write(".github/workflows/release.yml", b"steps:\n  - run: gh release upload v1 file\n")
        self.assertTrue(any("workflow GitHub release upload is forbidden" in item for item in self.run_check("tree")))

    def test_workflow_upload_artifact_action_fails(self):
        """工作流 artifact 上传 action 必须失败。"""
        self.write(".github/workflows/release.yml", b"uses: actions/upload-artifact@v4\n")
        self.assertTrue(any("workflow upload-artifact action is forbidden" in item for item in self.run_check("tree")))

    def test_workflow_package_publish_commands_fail(self):
        """各 Python 包发布命令必须逐个被拒绝。"""
        commands = (
            (b"twine upload dist/*", "workflow PyPI upload is forbidden"),
            (b"poetry publish", "workflow Poetry publish is forbidden"),
            (b"pdm publish", "workflow PDM publish is forbidden"),
            (b"flit publish", "workflow Flit publish is forbidden"),
        )
        for command, expected in commands:
            with self.subTest(command=command):
                self.write(".github/workflows/release.yml", b"steps:\n  - run: " + command + b"\n")
                self.assertTrue(any(expected in item for item in self.run_check("tree")))

    def test_workflow_pypi_publish_action_fails(self):
        """PyPI 发布 action 必须失败。"""
        self.write(".github/workflows/release.yml", b"uses: pypa/gh-action-pypi-publish@release/v1\n")
        self.assertTrue(any("workflow PyPI publish action is forbidden" in item for item in self.run_check("tree")))

    def test_workflow_container_push_commands_fail(self):
        """各容器推送命令必须逐个被拒绝。"""
        commands = (b"docker push image", b"podman push image", b"buildah push image", b"nerdctl push image", b"crane push image", b"oras push image")
        for command in commands:
            with self.subTest(command=command):
                self.write(".github/workflows/release.yml", b"steps:\n  - run: " + command + b"\n")
                self.assertTrue(any("workflow container push is forbidden" in item for item in self.run_check("tree")))

    def test_workflow_container_push_action_fails(self):
        """容器推送 action 必须失败。"""
        self.write(".github/workflows/release.yml", b"uses: docker/build-push-action@v6\n")
        self.assertTrue(any("workflow container push action is forbidden" in item for item in self.run_check("tree")))

    def test_workflow_release_creation_without_attachment_is_allowed(self):
        """仅创建标签或 Release 的人工流程不得被发布规则误拒。"""
        self.write(".github/workflows/release.yml", b"steps:\n  - run: gh release create v1 --generate-notes\n")
        failures = self.run_check("tree")
        self.assertFalse(any("workflow " in item for item in failures))

    def test_tree_without_git_succeeds(self):
        """无 Git 目录的最小合规树必须通过。"""
        with self.synthetic_license_hash() as license_data:
            self.write_valid_minimum_tree(license_data)
            self.assertEqual([], self.run_check("tree"))

    def test_empty_tree_fails(self):
        """空树不能作为发布源。"""
        self.assertEqual(["source file collection is empty"], self.run_check("tree", safety_only=True))

    def test_git_command_failure_fails(self):
        """非 Git 目录中的 git-ref 模式必须报告命令失败。"""
        self.write("source.txt", b"text\n")
        self.assertEqual(["git ls-tree failed"], self.run_check("git-ref", "HEAD", safety_only=True))

    def test_license_byte_drift_fails(self):
        """许可证任意字节偏差必须失败。"""
        with self.synthetic_license_hash(b"expected"):
            self.write_valid_minimum_tree(b"expected changed")
            self.assertTrue(any("LICENSE: SHA256 does not match GNU GPLv3 text" == item for item in self.run_check("tree")))

    def test_required_release_metadata_files_fail_when_missing(self):
        """根许可证和项目元数据任一缺失都必须失败。"""
        with self.synthetic_license_hash() as license_data:
            self.write("pyproject.toml", b"[project]\nlicense = \"GPL-3.0-or-later\"\nlicense-files = [\"LICENSE\"]\n")
            self.assertIn("LICENSE: root LICENSE file is missing", self.run_check("tree"))
            self.write("LICENSE", license_data)
            (self.root / "pyproject.toml").unlink()
            self.assertIn("pyproject.toml: file is missing", self.run_check("tree"))

    def test_pyproject_license_fields_fail_when_inconsistent(self):
        """SPDX 与许可证文件声明必须逐项保持一致。"""
        with self.synthetic_license_hash() as license_data:
            self.write_valid_minimum_tree(license_data)
            self.write(
                "pyproject.toml",
                b"[project]\nlicense = \"MIT\"\nlicense-files = [\"COPYING\"]\n",
            )
            failures = self.run_check("tree")
            self.assertIn(
                "pyproject.toml: project.license must be GPL-3.0-or-later",
                failures,
            )
            self.assertIn(
                "pyproject.toml: project.license-files must be [\"LICENSE\"]",
                failures,
            )

    def test_pyproject_malformed_toml_fails(self):
        """损坏的 TOML 不能绕过发布元数据检查。"""
        with self.synthetic_license_hash() as license_data:
            self.write_valid_minimum_tree(license_data)
            self.write("pyproject.toml", b"[project\nlicense = \"GPL-3.0-or-later\"\n")
            self.assertIn("pyproject.toml: TOML parsing failed", self.run_check("tree"))

    def test_python_39_metadata_fallback_succeeds(self):
        """缺少 tomllib 时受限后备解析仍支持项目的最低 Python 版本。"""
        with self.synthetic_license_hash() as license_data:
            self.write_valid_minimum_tree(license_data)
            previous = CHECKER.tomllib
            CHECKER.tomllib = None
            try:
                self.assertEqual([], self.run_check("tree"))
            finally:
                CHECKER.tomllib = previous

    def test_valid_minimum_tree_succeeds(self):
        """完整检查应接受所有元数据合规的最小树。"""
        with self.synthetic_license_hash() as license_data:
            self.write_valid_minimum_tree(license_data)
            output = io.StringIO()
            previous = Path.cwd()
            try:
                os.chdir(self.root)
                with contextlib.redirect_stdout(output):
                    result = CHECKER.main(["tree"])
            finally:
                os.chdir(previous)
            self.assertEqual(0, result)
            self.assertEqual("OK source-only release: tree\n", output.getvalue())

    def test_cli_root_and_named_mode_succeed(self):
        """CLI 必须支持显式 --root 与 --mode 组合。"""
        with self.synthetic_license_hash() as license_data:
            self.write_valid_minimum_tree(license_data)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = CHECKER.main(["--root", os.fspath(self.root), "--mode", "tree"])
            self.assertEqual(0, result)
            self.assertEqual("OK source-only release: tree\n", output.getvalue())

    def test_cli_mode_conflict_fails(self):
        """位置 mode 与 --mode 不一致时必须失败。"""
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = CHECKER.main(["tree", "--mode", "working-tree"])
        self.assertEqual(1, result)
        self.assertEqual("FAIL mode conflict between positional mode and --mode\n", output.getvalue())

    def test_cli_missing_mode_fails(self):
        """未提供两种 mode 形式时必须失败。"""
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = CHECKER.main(["--root", os.fspath(self.root)])
        self.assertEqual(1, result)
        self.assertEqual("FAIL mode is required\n", output.getvalue())

    def test_cli_missing_root_path_fails(self):
        """--root 缺少路径参数时必须失败。"""
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            with self.assertRaises(SystemExit) as raised:
                CHECKER.main(["--mode", "tree", "--root"])
        self.assertEqual(2, raised.exception.code)
        self.assertIn("FAIL argument --root: expected one argument", output.getvalue())


if __name__ == "__main__":
    unittest.main()
