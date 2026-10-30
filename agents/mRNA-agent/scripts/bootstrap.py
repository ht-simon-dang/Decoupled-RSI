"""Install into MRNA_DATA_DIR, reusing the peptide-agent base environment."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

PROJECT=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get("MRNA_DATA_DIR",str(Path.home()/".cache/decoupled-rsi/mrna"))).expanduser().resolve()
PINNED={
    "GEMORNA":("RainaBio/GEMORNA","3bba88e4c0a32377dcfac8caa492709b954f9d9e"),
    "LinearDesign":("LinearDesignSoftware/LinearDesign","f0126ca89a8b853088b4bccfd2cc8c378d3678be"),
    "RiboNN":("Sanofi-Public/RiboNN","dc896e1d1f78bf6b4e9bb7525c7852fcf25532e8"),
    "Smart5UTR":("deepomicslab/Smart5UTR","0085dd6982e112257ee4b0a7225b5ef3b6db59e9"),
    "parade":("autosome-ru/parade","f8e3e02688918e40d85eff0976a984810ee04372"),
}


def check_space(reserve):
    used=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink())
    if used+reserve>50_000_000_000:raise RuntimeError("50 GB quota: insufficient remaining allocation")


def run(args,**kwargs):subprocess.run([str(x) for x in args],check=True,**kwargs)


def main(stage="all"):
    if stage == "optional":
        python=ROOT/'venv/bin/python'
        if not python.is_file():raise RuntimeError('Install the runtime/core stage first')
        run([python,PROJECT/'scripts/bootstrap_optional.py'])
        return
    ROOT.mkdir(parents=True,exist_ok=True)
    base=Path(os.environ.get("PEPTIDE_PYTHON",str(PROJECT.parent/"peptide-agent/.venv/bin/python")))
    if not base.is_file():raise RuntimeError("Set PEPTIDE_PYTHON to the existing peptide-agent Python 3.11 interpreter")
    check_space(6_000_000_000)
    if not (ROOT/'venv/bin/python').is_file():run([base,'-m','venv','--system-site-packages',ROOT/'venv'])
    python=ROOT/'venv/bin/python'
    run([python,'-m','pip','install','--no-cache-dir','-r',PROJECT/'requirements.txt'])
    run([python,'-m','pip','install','-e',PROJECT,'--no-deps'])
    if stage == 'runtime':
        print('Runtime installed; scientific sources and model assets are separate stages.')
        return
    (ROOT/'external').mkdir(exist_ok=True)
    for name,(repo,revision) in PINNED.items():
        path=ROOT/'external'/name
        if not path.is_dir():
            check_space(1_000_000_000)
            run(['git','clone','--depth','1','https://github.com/'+repo,str(path)],env=dict(os.environ,GIT_LFS_SKIP_SMUDGE='1'))
        current=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
        if current!=revision:
            run(['git','-C',path,'fetch','--depth','1','origin',revision],env=dict(os.environ,GIT_LFS_SKIP_SMUDGE='1'))
            run(['git','-C',path,'checkout','--detach',revision],env=dict(os.environ,GIT_LFS_SKIP_SMUDGE='1'))
    legacy=ROOT/'envs/gemorna'
    if not (legacy/'bin/python').is_file():
        conda=shutil.which('conda')
        if not conda:raise RuntimeError('Install conda and expose it on PATH for the Python 3.10 GEMORNA environment')
        run([conda,'create','-p',legacy,'python=3.10','pip','-y'])
        run([legacy/'bin/pip','install','--no-cache-dir','torch==2.5.1','--index-url','https://download.pytorch.org/whl/cpu'])
        run([legacy/'bin/pip','install','--no-cache-dir','numpy==1.26.4','pandas==2.2.3','biopython==1.85','torchtext==0.6.0','tensorflow-cpu==2.15.1','scikit-learn==1.5.2'])
    run(['make','-C',ROOT/'external/LinearDesign'])
    check_space(2_000_000_000)
    run([python,PROJECT/'scripts/fetch_assets.py'])
    run([python,PROJECT/'scripts/fetch_references.py'])
    if stage == 'all':
        run([python,PROJECT/'scripts/bootstrap_optional.py'])
    run([python,'-m','mrna_agent.cli','doctor'])


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['runtime','core','optional','all'],default='all')
    main(parser.parse_args().stage)
