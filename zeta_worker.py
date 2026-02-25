
from mpmath import mp
def fetch_single_zero(n):
    # Precision is set inside the worker
    mp.dps = 25
    return float(mp.zetazero(n).imag)
