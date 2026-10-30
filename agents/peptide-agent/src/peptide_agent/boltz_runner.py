"""One interpreter, fixed BoltzGen entrypoint, no external runtime provisioning."""
import sys
from pytorch_lightning import seed_everything
from boltzgen.cli.boltzgen import main

if __name__ == "__main__":
    seed = int(sys.argv.pop(1))
    seed_everything(seed, workers=True)
    main()
