"""Device selection and parallel worker processes."""
import subprocess

from joblib import Parallel, delayed


def select_device() -> str:
    """Return the CUDA device with the most free memory, or "cpu"."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "cpu"
    gpus = [tuple(int(v) for v in line.split(",")) for line in out.strip().splitlines()]
    idx, _ = max(gpus, key=lambda g: g[1])
    return f"cuda:{idx}"


def _guarded(func, name, args):
    try:
        func(*args)
        return name, None
    except Exception as e:  # one failed item must not kill the others
        return name, e


def run_parallel(func, items: dict, n_proc: int = 4) -> dict:
    """Run func(*args) for each {name: args} in n_proc worker processes (joblib/loky).

    Returns {name: None | Exception}; failures are printed as they occur.
    """
    results = Parallel(n_jobs=n_proc, backend="loky", return_as="generator")(
        delayed(_guarded)(func, name, args) for name, args in items.items()
    )
    errors = {}
    for name, err in results:
        errors[name] = err
        if err is not None:
            print(f"{name}: FAILED ({type(err).__name__}: {err})")
    return errors
