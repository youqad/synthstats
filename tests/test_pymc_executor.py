"""Tests for PyMC Sandbox Executor - WRITTEN FIRST per TDD.

Tests cover:
1. Protocol compliance
2. AST security checks (blocking dangerous imports/functions)
3. Subprocess execution with timeout
4. Output capture and ELPD extraction
"""

from synthstats.core.executor import Executor
from synthstats.core.types import ToolCall


class TestPyMCExecutorProtocol:
    def test_executor_implements_protocol(self):
        """PyMCExecutor must implement the Executor protocol."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        assert isinstance(executor, Executor)
        assert executor.name == "pymc"

    def test_executor_has_required_attributes(self):
        """Executor must have name attribute."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        assert hasattr(executor, "name")
        assert hasattr(executor, "execute")


class TestPyMCExecutorSafeCode:
    def test_safe_pymc_code_runs(self):
        """Safe PyMC code should execute successfully."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "print('hello from pymc')"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is True
        assert "hello from pymc" in result.output
        assert result.error is None

    def test_safe_math_code_runs(self):
        """Basic Python math should work."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "x = 2 + 2\nprint(f'result: {x}')"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is True
        assert "result: 4" in result.output

    def test_captures_output(self):
        """Executor should capture stdout."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "for i in range(3): print(f'line {i}')"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is True
        assert "line 0" in result.output
        assert "line 1" in result.output
        assert "line 2" in result.output


class TestPyMCExecutorASTSecurityBlocks:
    """Test that dangerous code patterns are blocked by AST analysis."""

    def test_blocks_os_import(self):
        """Importing os module should be blocked."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "import os\nos.system('echo pwned')"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is False
        assert result.error is not None
        assert "blocked" in result.error.lower() or "forbidden" in result.error.lower()

    def test_blocks_os_from_import(self):
        """from os import ... should be blocked."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "from os import system\nsystem('echo pwned')"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is False
        assert "blocked" in result.error.lower() or "forbidden" in result.error.lower()

    def test_blocks_subprocess(self):
        """Importing subprocess should be blocked."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "import subprocess\nsubprocess.run(['ls'])"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is False
        assert "blocked" in result.error.lower() or "forbidden" in result.error.lower()

    def test_blocks_sys(self):
        """Importing sys should be blocked."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "import sys\nsys.exit(1)"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is False
        assert "blocked" in result.error.lower() or "forbidden" in result.error.lower()

    def test_blocks_socket(self):
        """Importing socket should be blocked."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "import socket\ns = socket.socket()"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is False
        assert "blocked" in result.error.lower() or "forbidden" in result.error.lower()

    def test_blocks_shutil(self):
        """Importing shutil should be blocked."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "import shutil\nshutil.rmtree('/')"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is False
        assert "blocked" in result.error.lower() or "forbidden" in result.error.lower()

    def test_blocks_pathlib(self):
        """Importing pathlib should be blocked."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "from pathlib import Path\nPath('/').iterdir()"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is False
        assert "blocked" in result.error.lower() or "forbidden" in result.error.lower()

    def test_blocks_open_function(self):
        """Using open() should be blocked."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "f = open('/etc/passwd', 'r')\nprint(f.read())"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is False
        assert "blocked" in result.error.lower() or "forbidden" in result.error.lower()

    def test_blocks_exec_eval(self):
        """Using exec() should be blocked."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "exec('import os')"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is False
        assert "blocked" in result.error.lower() or "forbidden" in result.error.lower()

    def test_blocks_eval_function(self):
        """Using eval() should be blocked."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "eval('__import__(\"os\")')"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is False
        assert "blocked" in result.error.lower() or "forbidden" in result.error.lower()

    def test_blocks_compile_function(self):
        """Using compile() should be blocked."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "code = compile('import os', '<string>', 'exec')"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is False
        assert "blocked" in result.error.lower() or "forbidden" in result.error.lower()

    def test_blocks_dunder_import(self):
        """Using __import__() should be blocked."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "__import__('os').system('echo pwned')"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is False
        assert "blocked" in result.error.lower() or "forbidden" in result.error.lower()


class TestPyMCExecutorTimeout:
    def test_timeout_enforcement(self):
        """Code that runs too long should timeout."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        # infinite loop
        action = ToolCall(
            name="pymc",
            input={"code": "while True: pass"},
            raw="",
        )
        result = executor.execute(action, timeout_s=1.0)

        assert result.success is False
        assert result.error is not None
        assert "timeout" in result.error.lower()

    def test_long_running_code_within_timeout_succeeds(self):
        """Code that finishes before timeout should succeed."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "import time\ntime.sleep(0.1)\nprint('done')"},
            raw="",
        )
        result = executor.execute(action, timeout_s=5.0)

        assert result.success is True
        assert "done" in result.output


class TestPyMCExecutorErrorHandling:
    def test_syntax_error_handled(self):
        """Syntax errors should be caught and reported."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "def broken(:\n    pass"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is False
        assert result.error is not None

    def test_runtime_error_handled(self):
        """Runtime errors should be caught and reported."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "x = 1 / 0"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is False
        assert result.error is not None
        assert "division" in result.error.lower() or "zero" in result.error.lower()

    def test_missing_code_key(self):
        """Missing 'code' key should return error."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={},  # missing 'code'
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is False
        assert result.error is not None


class TestPyMCExecutorGadgetChainBlocks:
    """Test blocking of Python gadget chain attacks via dunder access."""

    def test_blocks_class_dunder(self):
        """Accessing __class__ should be blocked."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "().__class__"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is False
        assert "blocked" in result.error.lower()

    def test_blocks_subclasses_dunder(self):
        """Accessing __subclasses__ should be blocked."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "object.__subclasses__()"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is False
        assert "blocked" in result.error.lower()

    def test_blocks_base_dunder(self):
        """Accessing __base__ should be blocked."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "().__class__.__base__"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is False
        assert "blocked" in result.error.lower()

    def test_blocks_mro_dunder(self):
        """Accessing __mro__ should be blocked."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "str.__mro__"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is False
        assert "blocked" in result.error.lower()

    def test_blocks_globals_dunder(self):
        """Accessing __globals__ should be blocked."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "(lambda: 0).__globals__"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is False
        assert "blocked" in result.error.lower()

    def test_blocks_subscript_call(self):
        """Calling result of subscript (gadget pattern) should be blocked."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        # this pattern: [c for c in ...][0]() is common in gadgets
        action = ToolCall(
            name="pymc",
            input={"code": "[1, 2, 3][0]()"},  # simplified example
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is False
        assert "blocked" in result.error.lower()

    def test_allows_safe_dunder_access(self):
        """__name__ and __doc__ should be allowed for normal use."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "print(int.__name__)"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is True
        assert "int" in result.output

    def test_blocks_init_dunder(self):
        """__init__ should be blocked (not needed for modeling)."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "str.__init__"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is False
        assert "blocked" in result.error.lower()

    def test_blocks_call_dunder(self):
        """__call__ should be blocked (not needed for modeling)."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "int.__call__"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is False
        assert "blocked" in result.error.lower()

    def test_blocks_importlib(self):
        """importlib should be blocked to prevent dynamic imports."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "import importlib\nimportlib.import_module('os')"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is False
        assert "blocked" in result.error.lower()

    def test_blocks_io_module(self):
        """io module should be blocked."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "import io\nio.open('/etc/passwd')"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is False
        assert "blocked" in result.error.lower()


class TestPyMCExecutorAliasingBypass:
    """Test that aliasing dangerous functions is blocked."""

    def test_blocks_import_alias(self):
        """Aliasing __import__ should be blocked."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "imp = __import__; imp('os')"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is False
        assert "blocked" in result.error.lower()

    def test_blocks_open_alias(self):
        """Aliasing open should be blocked."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "o = open; o('/etc/passwd')"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is False
        assert "blocked" in result.error.lower()

    def test_blocks_eval_alias(self):
        """Aliasing eval should be blocked."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "e = eval; e('1+1')"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is False
        assert "blocked" in result.error.lower()

    def test_blocks_globals_reference(self):
        """Referencing globals should be blocked."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "g = globals"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is False
        assert "blocked" in result.error.lower()


class TestPyMCExecutorFileIOBypass:
    """Test blocking of file I/O on allowed modules (pandas, numpy, scipy, torch).

    These modules are allowed for modeling but have their own file I/O that
    bypasses the open() blocklist.
    """

    def test_blocks_pandas_read_csv(self):
        """pandas read_csv should be blocked."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "import pandas as pd\npd.read_csv('/etc/passwd')"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is False
        assert "blocked" in result.error.lower()
        assert "read_csv" in result.error

    def test_blocks_pandas_read_pickle(self):
        """pandas read_pickle should be blocked (RCE risk)."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "import pandas as pd\npd.read_pickle('malicious.pkl')"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is False
        assert "blocked" in result.error.lower()

    def test_blocks_pandas_to_csv(self):
        """pandas to_csv should be blocked."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "import pandas as pd\ndf = pd.DataFrame()\ndf.to_csv('/tmp/exfil.csv')"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is False
        assert "blocked" in result.error.lower()

    def test_blocks_numpy_load(self):
        """numpy load should be blocked (pickle-based, RCE risk)."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "import numpy as np\nnp.load('malicious.npy', allow_pickle=True)"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is False
        assert "blocked" in result.error.lower()

    def test_blocks_numpy_loadtxt(self):
        """numpy loadtxt should be blocked."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "import numpy as np\nnp.loadtxt('/etc/passwd')"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is False
        assert "blocked" in result.error.lower()

    def test_blocks_numpy_save(self):
        """numpy save should be blocked."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={
                "code": (
                    "import numpy as np\n"
                    "arr = np.array([1,2,3])\n"
                    "np.save('/tmp/data.npy', arr)"
                )
            },
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is False
        assert "blocked" in result.error.lower()

    def test_blocks_torch_load(self):
        """torch load should be blocked (pickle-based, RCE risk)."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "import torch\ntorch.load('malicious.pt')"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is False
        assert "blocked" in result.error.lower()

    def test_blocks_scipy_loadmat(self):
        """scipy loadmat should be blocked."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "from scipy.io import loadmat\nloadmat('data.mat')"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is False
        assert "blocked" in result.error.lower()

    def test_blocks_pickle_loads(self):
        """pickle loads/dumps should be blocked (RCE)."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "data = b'...'\nimport pickle\npickle.loads(data)"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        # blocked at import level since pickle isn't in allowed list
        assert result.success is False

    def test_allows_numpy_array_operations(self):
        """Regular numpy array operations should still work."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={"code": "import numpy as np\narr = np.array([1,2,3])\nprint(arr.mean())"},
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is True
        assert "2.0" in result.output

    def test_allows_pandas_dataframe_operations(self):
        """Regular pandas DataFrame operations should still work."""
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executor = PyMCExecutor()
        action = ToolCall(
            name="pymc",
            input={
                "code": (
                    "import pandas as pd\n"
                    "df = pd.DataFrame({'a': [1,2,3]})\n"
                    "print(df.sum())"
                )
            },
            raw="",
        )
        result = executor.execute(action, timeout_s=10.0)

        assert result.success is True
