"""Native MINER baseline integration.

The implementation is adapted from vishwa91/MINER at commit
f8f6ed442048f883cf70541117f0754ad437cc69 under the MIT license.  It is
self-contained and never imports the external reference checkout.
"""

from .model import BlockSiren

__all__ = ["BlockSiren"]
