"""Process-local NumPy 2 bridge for unmodified pinned LigandMPNN/OpenFold."""
import runpy
import sys
from pathlib import Path
import numpy as np

if __name__=='__main__':
    # Upstream np.int meant Python int; no dtype/numeric behavior is changed.
    if 'int' not in np.__dict__: np.int=int
    script=Path(sys.argv[1]);sys.argv=sys.argv[1:]
    sys.path.insert(0,str(script.parent))
    runpy.run_path(str(script),run_name='__main__')
