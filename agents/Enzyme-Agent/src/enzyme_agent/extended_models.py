"""Pinned upstream distributions for the isolated scientific runtimes."""
import json
import os
import tarfile
from pathlib import Path

from .config import atomic_bytes, atomic_json, check_space, data_root, quota_lock
from .models import download_file, file_sha

FOUNDRY_REVISION = 'd078b24d6d0e6f8607e9a0311128f4a897f83953'
THERMO_REVISION = '370f76ec62bd929f7425e311d8df04a0d094990f'
CHECKPOINT_URL = 'https://files.ipd.uw.edu/pub/rfd3/rfd3_foundry_2025_12_01_remapped.ckpt'


def source_archive(repo, revision, destination, include):
    archive = destination.parent / (destination.name + '.tar.gz')
    if not archive.exists():
        download_file(f'https://codeload.github.com/{repo}/tar.gz/{revision}', archive)
    files = {}
    try:
        with quota_lock(), tarfile.open(archive) as tar:
            members = [(m, Path(*Path(m.name).parts[1:])) for m in tar.getmembers() if m.isfile()]
            members = [(m,p) for m,p in members if '..' not in p.parts and include(str(p))]
            check_space(sum(m.size for m,p in members))
            for member,path in members:
                target = destination / path
                target.parent.mkdir(parents=True,exist_ok=True)
                temporary = target.with_suffix(target.suffix+'.partial')
                temporary.write_bytes(tar.extractfile(member).read())
                os.replace(temporary,target)
                files[str(path)] = {'sha256': file_sha(target), 'size': member.size}
    finally:
        archive.unlink(missing_ok=True)
    return files


def download_extended(name):
    existing=data_root()/'models'/name/'manifest.json'
    if existing.exists():
        manifest=json.loads(existing.read_text())
        expected={'rfdiffusion3':FOUNDRY_REVISION,'thermompnn':THERMO_REVISION}.get(name)
        intact=manifest.get('revision')==expected and all((existing.parent/'source'/p).is_file() and (existing.parent/'source'/p).stat().st_size==info['size'] for p,info in manifest.get('source_files',{}).items())
        if name=='rfdiffusion3':intact=intact and (existing.parent/'rfd3_latest.ckpt').is_file() and (existing.parent/'rfd3_latest.ckpt').stat().st_size==2690316669
        if intact:return manifest
    if name == 'rfdiffusion3':
        root = data_root() / 'models/rfdiffusion3'
        root.mkdir(parents=True, exist_ok=True)
        files = source_archive('RosettaCommons/foundry', FOUNDRY_REVISION, root/'source',
            lambda p: p.endswith(('.py', '.yaml', '.json', '.toml', '.md')) and not any(x in p for x in ('tests/', '.assets/', 'tutorials/')))
        checksum = download_file(CHECKPOINT_URL, root/'rfd3_latest.ckpt', 2690316669)
        manifest = {'repo':'RosettaCommons/foundry', 'revision':FOUNDRY_REVISION, 'license':'BSD-3-Clause',
            'checkpoint':{'url':CHECKPOINT_URL, 'sha256':checksum, 'size':2690316669}, 'source_files':files}
    elif name == 'thermompnn':
        root = data_root() / 'models/thermompnn'
        root.mkdir(parents=True, exist_ok=True)
        keep = {'transfer_model.py','protein_mpnn_utils.py','model_utils.py','LICENSE','README.md',
                'models/thermoMPNN_default.pt','vanilla_model_weights/v_48_020.pt'}
        files = source_archive('Kuhlman-Lab/ThermoMPNN', THERMO_REVISION, root/'source', lambda p:p in keep)
        manifest = {'repo':'Kuhlman-Lab/ThermoMPNN','revision':THERMO_REVISION,'license':'MIT','source_files':files}
    else:
        raise ValueError('Unknown extended model')
    atomic_json(root/'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    import sys
    for name in sys.argv[1:]:
        result = download_extended(name)
        print(name, result['revision'], flush=True)
