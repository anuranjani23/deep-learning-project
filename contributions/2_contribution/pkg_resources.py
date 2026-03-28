"""
Minimal stub for pkg_resources to satisfy lightning_fabric import.
Only implements declare_namespace as a no-op.
"""


def declare_namespace(_name: str) -> None:
    return
