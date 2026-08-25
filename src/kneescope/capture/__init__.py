"""Optional framework capture adapters.

Adapters live in per-framework submodules (e.g. :mod:`kneescope.capture.torch`)
and import their framework lazily, so the base package stays numpy-only.
Nothing here is imported by the top-level package; import the adapter you use
explicitly, e.g. ``from kneescope.capture.torch import TorchKneeProbe``.
"""
