"""Install all deployable optional models with quota checks and real inference gates."""
from pathlib import Path
import sys,os,json,subprocess,zipfile
from bootstrap import ROOT,PROJECT,check_space,run


def main():
    check_space(12_000_000_000)
    pins=json.loads(Path(__file__).with_name('optional-pins.json').read_text())
    for name,(repo,revision) in pins.items():
        p=ROOT/'external'/name
        if not p.exists():run(['git','clone','--depth','1','--no-checkout','https://github.com/'+repo,p],env=dict(os.environ,GIT_LFS_SKIP_SMUDGE='1'))
        run(['git','-C',p,'fetch','--depth','1','origin',revision],env=dict(os.environ,GIT_LFS_SKIP_SMUDGE='1'))
        run(['git','-C',p,'checkout','--detach',revision],env=dict(os.environ,GIT_LFS_SKIP_SMUDGE='1'))
    python=ROOT/'venv/bin/python';compat=ROOT/'envs/transformers4'
    if not (compat/'bin/python').exists():run([python,'-m','venv','--system-site-packages',compat])
    run([compat/'bin/pip','install','--no-cache-dir','transformers==4.44.2','ml-collections==1.1.0','xgboost==3.2.0'])
    run([python,'-m','pip','install','--no-cache-dir','remotezip==0.12.6'])
    archive=ROOT/'external/RNAdegformer-Webapp/MultiRNAFold.zip';target=archive.parent
    with zipfile.ZipFile(archive) as z:
        check_space(sum(i.file_size for i in z.infolist()))
        for i in z.infolist():
            if not (target/i.filename).resolve().is_relative_to(target):raise ValueError('Unsafe archive path')
        z.extractall(target)
    (target/'MultiRNAFold/simfold_pf').chmod(0o755);(ROOT/'tmp').mkdir(exist_ok=True)
    run([python,PROJECT/'scripts/fetch_optional.py'])
    print('Optional assets installed. Readiness validation is still required; no readiness record was created.')

if __name__=='__main__':main()
