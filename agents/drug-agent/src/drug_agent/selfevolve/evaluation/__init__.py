"""The trusted evaluation kernel.

Everything in this subpackage is **immutable during a measurement**. The
integrity manifest in :mod:`.integrity` is verified before and after every
candidate evaluation, and a candidate that changes any file here is quarantined
rather than scored.

That boundary is the whole reason the rest of the package can be believed. The
Darwin Gödel Machine paper documents an agent that learned to fake its own
tool-use logs to score better; the user-serving system in *this* repository was
already observed scoring `hard_constraint_pass_rate == 1.0` by deleting every
constraint. Neither needed malice — a metric that can be reached by editing the
measurement will be.
"""

from .benchmark import BENCHMARK_ID, BENCHMARK_VERSION, BenchmarkTask, load_benchmark
from .integrity import KernelIntegrityError, kernel_digest, kernel_manifest, verify_kernel
from .kernel import EvaluationKernel

__all__ = [
    "BENCHMARK_ID",
    "BENCHMARK_VERSION",
    "BenchmarkTask",
    "EvaluationKernel",
    "KernelIntegrityError",
    "kernel_digest",
    "kernel_manifest",
    "load_benchmark",
    "verify_kernel",
]
