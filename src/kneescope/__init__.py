"""kneescope — measure the persistence boundary of Muon-style optimizer momentum.

Public API:

- :class:`SnapshotWriter` — write snapshot directories (format v1),
- :func:`load_snapshot` — read and validate them,
- :func:`analyze` — run the persistence analysis, returning per-shape-group
  :class:`BandReport`s with sigma*, sigma*_upper and the recommended knee50,
- :func:`analyze_matrix` — the core per-matrix estimator.

Optional: :mod:`kneescope.mp_diag` (Marchenko-Pastur diagnostic,
assumption-laden), :mod:`kneescope.capture.torch` (PyTorch adapter),
:meth:`AnalysisResult.plot` (matplotlib).
"""

__version__ = "0.1.0"

from .bands import (
    DEFAULT_BIN_EDGES,
    AnalysisResult,
    BandReport,
    BinStats,
    BoundaryStatus,
    analyze,
    bin_directions,
    detect_band,
)
from .estimator import MatrixEstimate, analyze_matrix
from .formats import Snapshot, SnapshotFormatError, load_snapshot
from .writer import SnapshotWriter

__all__ = [
    "__version__",
    "analyze",
    "analyze_matrix",
    "load_snapshot",
    "SnapshotWriter",
    "Snapshot",
    "SnapshotFormatError",
    "AnalysisResult",
    "BandReport",
    "BinStats",
    "BoundaryStatus",
    "MatrixEstimate",
    "DEFAULT_BIN_EDGES",
    "bin_directions",
    "detect_band",
]
