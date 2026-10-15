from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from provguard.study import main
if __name__ == '__main__':
    main()
    if '--inspect' not in sys.argv:
        from provguard.secondary import run
        run()
