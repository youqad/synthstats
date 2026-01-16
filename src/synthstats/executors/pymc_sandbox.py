"""PyMC Sandbox Executor - safe execution of PyMC code.

Provides AST-based security checks and subprocess isolation for running
untrusted PyMC probabilistic programs.
"""

import ast
import os
import subprocess
import sys
import tempfile

from synthstats.core.executor import ToolResult
from synthstats.core.types import ToolCall

# modules that could enable file system, network, or shell access
BLOCKED_MODULES = frozenset({
    "os",
    "sys",
    "subprocess",
    "socket",
    "shutil",
    "pathlib",
    "ctypes",
    "multiprocessing",
    "threading",
    "signal",
    "pty",
    "tty",
    "termios",
    "fcntl",
    "resource",
    "syslog",
    "platform",
    "webbrowser",
    "http",
    "urllib",
    "ftplib",
    "smtplib",
    "poplib",
    "imaplib",
    "nntplib",
    "telnetlib",
    "xmlrpc",
    "asyncio",
    "concurrent",
    # dynamic import modules (gadget chain enablers)
    "importlib",
    "pkgutil",
    "types",
    "inspect",
    # I/O modules that bypass open()
    "io",
    "builtins",
    "code",
    "codeop",
    "runpy",
    # network and file bypass modules
    "requests",
    "tempfile",
    # memory introspection
    "gc",
    # persistence via exit hooks
    "atexit",
    # memory mapping bypass
    "mmap",
    # file creation bypasses
    "sqlite3",
    "logging",
    # crypto (reduces attack surface)
    "ssl",
    # serialization (RCE via __reduce__, defense in depth)
    "pickle",
    "_pickle",
})

# built-in functions that allow code execution or file access
BLOCKED_FUNCTIONS = frozenset({
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "hasattr",
    "type",
    "input",
    "breakpoint",
})

# dangerous methods on allowed modules (pandas, numpy, scipy, torch)
# that bypass open() by implementing their own file I/O
BLOCKED_METHODS = frozenset({
    # pandas file I/O - read
    "read_csv", "read_pickle", "read_json", "read_excel", "read_parquet",
    "read_html", "read_xml", "read_sql", "read_table", "read_fwf",
    "read_stata", "read_sas", "read_spss", "read_feather", "read_orc",
    # pandas file I/O - write
    "to_csv", "to_pickle", "to_json", "to_excel", "to_parquet",
    "to_html", "to_sql", "to_stata", "to_feather",
    # numpy file I/O
    "load", "loadtxt", "genfromtxt", "fromfile",
    "save", "savetxt", "savez", "savez_compressed", "tofile",
    # scipy file I/O
    "loadmat", "savemat", "wavread", "wavwrite",
    # torch file I/O (pickle-based, RCE risk)
    "load_state_dict", "save_state_dict",
    # ctypes/shared library loading
    "load_library", "CDLL", "WinDLL", "PyDLL", "OleDLL",
    # pickle operations (RCE)
    "loads", "dumps", "dump",
})

# environment variables safe to pass to sandboxed subprocess
# explicitly allowlist to prevent leaking API keys, credentials, etc.
# NOTE: PYTHONPATH intentionally excluded - could allow module injection
# from shared directories on multi-tenant systems
SAFE_ENV_VARS = frozenset({
    "PATH",
    "HOME",
    "TMPDIR",
    "TEMP",
    "TMP",
    "LANG",
    "LC_ALL",
})


class ASTSecurityChecker(ast.NodeVisitor):
    """Visitor that detects dangerous code patterns.

    Uses defense-in-depth: blocks known dangerous patterns AND uses allowlist
    for attribute access to prevent gadget chain attacks.
    """

    # allowlisted module prefixes for attribute access
    ALLOWED_MODULES = frozenset({
        "pm", "pymc", "np", "numpy", "math", "scipy", "arviz", "az",
        "xarray", "pandas", "pd", "jax", "numpyro", "torch",
    })

    def __init__(self):
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            module_name = alias.name.split(".")[0]
            if module_name in BLOCKED_MODULES:
                self.violations.append(f"Blocked import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            module_name = node.module.split(".")[0]
            if module_name in BLOCKED_MODULES:
                self.violations.append(f"Blocked import: from {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # check for direct function calls like open(), exec(), loadmat()
        # also catches `from scipy.io import loadmat; loadmat('file')`
        if isinstance(node.func, ast.Name):
            if node.func.id in BLOCKED_FUNCTIONS:
                self.violations.append(f"Blocked function call: {node.func.id}()")
            elif node.func.id in BLOCKED_METHODS:
                self.violations.append(
                    f"Blocked file I/O function: {node.func.id}() - use PyMC for modeling only"
                )
        # check for method calls - block both BLOCKED_FUNCTIONS and BLOCKED_METHODS
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in BLOCKED_FUNCTIONS:
                self.violations.append(f"Blocked function call: {node.func.attr}()")
            elif node.func.attr in BLOCKED_METHODS:
                self.violations.append(
                    f"Blocked file I/O method: {node.func.attr}() - use PyMC for modeling only"
                )
        # CRITICAL: block calls on complex expressions (gadget chains)
        # e.g., ().__class__.__base__.__subclasses__()[0]()
        elif isinstance(node.func, ast.Subscript):
            self.violations.append(
                "Blocked: calls on subscript expressions are not allowed"
            )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # block dunder attribute access (used in gadget chains)
        if node.attr.startswith("__") and node.attr.endswith("__"):
            # only allow __name__ and __doc__ - needed for introspection
            # __init__ and __call__ are NOT needed for modeling and increase attack surface
            if node.attr not in {"__name__", "__doc__"}:
                self.violations.append(
                    f"Blocked dunder access: {node.attr}"
                )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        # block loading of dangerous function names (prevents aliasing bypass)
        # e.g., `imp = __import__; imp("os")` - blocks loading __import__ into imp
        if isinstance(node.ctx, ast.Load):
            if node.id in BLOCKED_FUNCTIONS:
                self.violations.append(
                    f"Blocked: cannot reference '{node.id}' (prevents aliasing bypass)"
                )
        self.generic_visit(node)


def check_code_safety(code: str) -> tuple[bool, str | None]:
    """Check if code is safe to execute.

    Returns:
        (is_safe, error_message) tuple
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    checker = ASTSecurityChecker()
    checker.visit(tree)

    if checker.violations:
        return False, f"Blocked: {'; '.join(checker.violations)}"

    return True, None


class PyMCExecutor:
    """Executor for PyMC probabilistic programs.

    Runs code in a subprocess with:
    - AST-based security checks before execution
    - Timeout enforcement
    - Stdout/stderr capture
    """

    name: str = "pymc"

    def execute(self, payload: ToolCall, *, timeout_s: float) -> ToolResult:
        """Execute PyMC code safely.

        Args:
            payload: ToolCall with 'code' in input dict.
            timeout_s: Maximum execution time in seconds.

        Returns:
            ToolResult with output, success flag, and optional error.
        """
        code = payload.input.get("code")
        if code is None:
            return ToolResult(
                output="",
                success=False,
                error="Missing 'code' key in input",
            )

        # AST security check before execution
        is_safe, error = check_code_safety(code)
        if not is_safe:
            return ToolResult(
                output="",
                success=False,
                error=error,
            )

        # execute in subprocess for isolation
        return self._run_subprocess(code, timeout_s)

    def _run_subprocess(self, code: str, timeout_s: float) -> ToolResult:
        """Run code in isolated subprocess.

        Uses stdin to pass code directly, avoiding temp file TOCTOU vulnerability.
        """
        try:
            result = subprocess.run(
                [sys.executable, "-"],  # read code from stdin
                input=code,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=tempfile.gettempdir(),
                # use allowlist to prevent leaking API keys/credentials to sandbox
                env={
                    **{k: os.environ[k] for k in SAFE_ENV_VARS if k in os.environ},
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                return ToolResult(
                    output=result.stdout,
                    success=False,
                    error=error_msg,
                )

            return ToolResult(
                output=result.stdout,
                success=True,
                error=None,
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                output="",
                success=False,
                error=f"Timeout: code execution exceeded {timeout_s}s",
            )
