"""End-to-end process protocol validation with explicitly synthetic backends."""

import json
import sys

from antibody_agent.config import atomic_json
from antibody_agent.delivery import public_run
from antibody_agent.exports import export
from antibody_agent.store import Store
from antibody_agent.workflow import Engine

BRIDGE = r"""
import json, sys
from Bio.SeqUtils import seq3
request=json.load(open(sys.argv[1]))
tool,args=request['tool'],request['arguments']
def pdb_for(binder,state=None):
    lines=[];serial=1
    chains=[('H',binder['chains'][0]['sequence'],0)]
    if state:chains.append(('A',state['sequence'],5))
    for chain,seq,y in chains:
        for i,aa in enumerate(seq,1):
            lines.append(f'ATOM  {serial:5d}  CA  {seq3(aa).upper():>3} {chain}{i:4d}    {i*3.8:8.3f}{y:8.3f}{0:8.3f}{1:6.2f}{80:6.2f}           C')
            serial+=1
    return '\n'.join(lines)+'\n'
if tool.startswith('generation.'):
    binder=args['brief']['reference']
    binder['chains'][0]['sequence']='Y'+binder['chains'][0]['sequence'][1:]
    binder['pairing_source']='model_proposed'
    binder['source']='SYNTHETIC PROTOCOL FIXTURE, NOT REAL INFERENCE'
    data={'binders':[binder]}
elif tool.startswith('fold.'):
    data={'pdb':pdb_for(args['binder'])}
elif tool.startswith('profile.'):
    data={'profile':'tnp','applicable':True,'severe_flags':[],'metrics':{'synthetic_fixture_only':1}}
else:
    data={'pdb':pdb_for(args['binder'],args['state']), 'context_id':args['context_id'],
          'antigen_iptm':.8,'complex_plddt':.9,'templates_used':False,'forced_contacts':False}
json.dump({'tool':tool,'status':'ok','data':data},open(sys.argv[2],'w'))
"""


def test_generate_evaluate_deliver_and_resume_cache(tmp_path, monkeypatch, brief, binder):
    # The fixture program is never installed or registered in production.
    script = tmp_path / "synthetic_bridge.py"
    script.write_text(BRIDGE)
    config = {
        "command": [sys.executable, str(script)],
        "version": "SYNTHETIC_TEST_ONLY",
        "weights_version": "NO_SCIENTIFIC_WEIGHTS",
    }
    path = tmp_path / "runtime.json"
    atomic_json(path, {name: config for name in ("antifold", "immunebuilder", "tnp", "boltz2")})
    monkeypatch.setenv("ANTIBODY_RUNTIMES", str(path))
    values = brief.model_dump()
    values.update(
        route="optimize",
        generators=["antifold"],
        reference=binder.model_dump(),
        library=[],
        constraints={"editable": {"VHH": ["1"]}, "max_mutations": 1},
    )
    with Store(tmp_path / "runs") as store:
        result = Engine(store).run(values)
        assert result["state"] == "complete", json.dumps(result["payload"], indent=2)
        public = public_run(result)
        assert public["delivery"]["state"] == "awaiting_human_review", public
        assert len(public["candidates"]) == 1
        assert public["candidates"][0]["mutations"] == ["VHH:A1Y"]
        assert store.reviews(result["id"]) == []
        assert "variable_domain" in export(result, "fasta")[0]
        used = store.cost(result["id"])["tool_calls"]
        assert used == 4
        store.update(result["id"], "interrupted", result["payload"])
        Engine(store).request_resume(result["id"])
        resumed = Engine(store).resume(result["id"])
        assert resumed["state"] == "complete"
        assert store.cost(result["id"])["tool_calls"] == used
