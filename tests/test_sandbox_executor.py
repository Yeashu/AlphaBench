"""Tests for alphabench.sandbox_executor."""

import pytest
from alphabench.sandbox_executor import SandboxExecutor


@pytest.fixture
def sandbox():
    return SandboxExecutor(timeout=10, max_output_chars=4000)


def test_basic_eda_with_injected_globals(sandbox):
    """Agent code using injected df/pd/np should work without imports."""
    import pandas as pd
    import numpy as np
    # Create a DatetimeIndex DataFrame as DatasetService would provide
    dates = pd.date_range("2021-01-01", periods=3, freq="D", tz="UTC")
    df = pd.DataFrame({"close": [100.0, 101.0, 99.0]}, index=dates)
    df.index.name = "timestamp"
    result = sandbox.run("print(len(df))", globals={"pd": pd, "np": np, "df": df})
    assert result.returncode == 0, f"Expected success but got stderr: {result.stderr}"
    assert "3" in result.stdout


def test_df_describe_runs(sandbox):
    import pandas as pd
    import numpy as np
    df = pd.DataFrame({"close": list(range(100))})
    result = sandbox.run("print(df.describe())", globals={"pd": pd, "np": np, "df": df})
    assert result.returncode == 0
    assert "mean" in result.stdout


def test_disallowed_import_rejected(sandbox):
    """import os in agent code should be blocked by ImportChecker."""
    import pandas as pd
    import numpy as np
    df = pd.DataFrame({"close": [1.0]})
    result = sandbox.run("import os\nprint(os.getcwd())", globals={"pd": pd, "np": np, "df": df})
    assert result.returncode == 1
    assert "Disallowed" in result.stderr


def test_syntax_error_returns_nonzero(sandbox):
    import pandas as pd
    import numpy as np
    df = pd.DataFrame({"close": [1.0]})
    result = sandbox.run("def broken(:\n    pass", globals={"pd": pd, "np": np, "df": df})
    assert result.returncode == 1


def test_timeout_enforced():
    """Code that runs a very long loop (no import needed) should trigger timed_out."""
    slow_sandbox = SandboxExecutor(timeout=1)
    import pandas as pd
    import numpy as np
    df = pd.DataFrame({"close": [1.0]})
    # Use a pure-Python busy loop to burn time without importing anything
    result = slow_sandbox.run(
        "x = 0\nwhile True:\n    x += 1",
        globals={"pd": pd, "np": np, "df": df},
    )
    assert result.timed_out is True


def test_matplotlib_runs_headless(sandbox):
    """matplotlib should work with MPLBACKEND=Agg (no display needed)."""
    import pandas as pd
    import numpy as np
    df = pd.DataFrame({"close": list(range(50))})
    code = (
        "import matplotlib.pyplot as plt\n"
        "plt.plot(df['close'])\n"
        "plt.title('test')\n"
        "print('plot ok')"
    )
    result = sandbox.run(code, globals={"pd": pd, "np": np, "df": df})
    assert result.returncode == 0
    assert "plot ok" in result.stdout


def test_allowed_import_passes(sandbox):
    import pandas as pd
    import numpy as np
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
    code = "import math\nprint(math.sqrt(df['close'].mean()))"
    result = sandbox.run(code, globals={"pd": pd, "np": np, "df": df})
    assert result.returncode == 0
