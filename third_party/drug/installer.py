#!/usr/bin/env python3
"""Install one declared Drug Agent dependency stage, or inspect it offline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DRUG = REPO / "agents" / "drug-agent"
PHASES = ("source", "environment", "assets")
_ENVIRONMENTS = None


class Refusal(RuntimeError):
    pass


def conda() -> str:
    return os.environ.get("CONDA_EXE", "conda")


def environment_paths() -> list[Path]:
    global _ENVIRONMENTS
    if _ENVIRONMENTS is None:
        _ENVIRONMENTS = []
        if shutil.which(conda()):
            try:
                result = subprocess.run(
                    [conda(), "env", "list", "--json"], capture_output=True,
                    text=True, check=True, timeout=20,
                )
                _ENVIRONMENTS = [Path(p) for p in json.loads(result.stdout)["envs"]]
            except (OSError, ValueError, KeyError, subprocess.SubprocessError):
                pass
    return _ENVIRONMENTS


def interpreter(names: list[str], override: str | None = None) -> str:
    if override:
        return os.path.abspath(os.path.expanduser(override)) if "/" in override else override
    for name in names:
        for path in environment_paths():
            candidate = path / "bin" / "python"
            if path.name == name and candidate.is_file():
                return str(candidate)
    return sys.executable


def context(component: dict, *, discover_python: bool = False) -> dict[str, str]:
    external = Path(os.environ.get("DRUGAGENT_EXTERNAL_DIR", str(DRUG / ".external"))).expanduser()
    if not external.is_absolute():
        external = DRUG / external
    variables = component["runtime"]["configuration_variables"]
    root_override = os.environ.get(variables.get("root", ""))
    values = {
        "repo": str(REPO), "drug": str(DRUG), "external": str(external),
        "python": sys.executable, "conda": conda(),
        "boltz_cache": os.path.expanduser(os.environ.get("BOLTZ_CACHE", "~/.boltz")),
        "protenix_cache": os.path.expanduser(root_override or os.environ.get("PROTENIX_ROOT_DIR", "~/.protenix")),
    }
    source = component["layout"].get("source_root")
    values["source"] = source.format_map(values) if source else ""
    root = component["layout"].get("runtime_root")
    # Boltz's adapter reads BOLTZ_CACHE, not DRUGAGENT_BOLTZ_ROOT.
    if component["id"] == "boltz":
        root_override = None
    values["runtime_root"] = os.path.expanduser(root_override) if root_override else (root.format_map(values) if root else "")
    if values["runtime_root"] and not Path(values["runtime_root"]).is_absolute():
        values["runtime_root"] = str(DRUG / values["runtime_root"])
    names = component["runtime"]["conda_env_candidates"]
    override = os.environ.get(variables.get("python", ""))
    core_name = os.environ.get("DRUGAGENT_CONDA_ENV", "drug-agent")
    values["core_env"] = core_name
    names = [core_name if name == "drug-agent" else name for name in names]
    if discover_python:
        values["runtime_python"] = interpreter(names, override)
        values["core_python"] = interpreter([core_name])
    else:
        values["runtime_python"] = override or ("<python in " + ", ".join(names) + ">" if names else sys.executable)
        values["core_python"] = "<python in " + core_name + ">"
    return values


def expand(value, values):
    if isinstance(value, str):
        # Only substitute our path tokens; preserve braces inside Python snippets.
        for key, replacement in values.items():
            value = value.replace("{" + key + "}", replacement)
        return value
    if isinstance(value, list):
        return [expand(item, values) for item in value]
    if isinstance(value, dict):
        return {key: expand(item, values) for key, item in value.items()}
    return value


def plan(component: dict, phase: str | None) -> dict:
    values = context(component)
    return {
        "component": component["id"],
        "mode": "PLAN_ONLY",
        "layout": expand(component["layout"], values),
        "runtime": component["runtime"],
        "resolved_runtime_root": values["runtime_root"],
        "source_reference": component["source_checkout"],
        "stages": {name: expand(component["stages"][name], values) for name in ([phase] if phase else PHASES)},
        "note": "Planning runs no package manager, fetch command or interpreter probe.",
    }


def usable_file(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    if path.is_file():
        try:
            with path.open("rb") as stream:
                head = stream.read(150)
            if head.startswith(b"version https://git-lfs.github.com/spec/v1"):
                return False, "Git LFS pointer; actual asset bytes are absent"
            if path.stat().st_size == 0:
                return False, "empty file"
        except OSError as exc:
            return False, str(exc)
    return True, "present"


def module_probe(python: str, modules: list[str], package: dict | None = None) -> dict:
    # PathFinder inspects import paths without importing torch or a model package.
    code = """
import importlib.machinery,json,pathlib,sys
request=json.loads(sys.argv[1]); missing=[]; locations={}
for name in request['modules']:
    top=name.split('.')[0]
    spec=importlib.machinery.PathFinder.find_spec(top)
    if spec is None: missing.append(name)
    elif spec.submodule_search_locations: locations[top]=list(spec.submodule_search_locations)
package=request.get('package'); assets=[]
if package:
    spec=importlib.machinery.PathFinder.find_spec(package['module'])
    for root in (spec.submodule_search_locations or []) if spec else []:
        assets.extend(str(p) for p in pathlib.Path(root).glob(package['pattern']) if p.is_file())
print(json.dumps({'missing_modules':missing,'package_asset_count':len(assets)}))
"""
    try:
        result = subprocess.run(
            [python, "-B", "-c", code, json.dumps({"modules": modules, "package": package})],
            capture_output=True, text=True, timeout=20, check=True,
        )
        return json.loads(result.stdout)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"missing_modules": modules, "package_asset_count": 0, "probe_error": str(exc)}


def check(component: dict, phase: str) -> dict:
    stage = component["stages"][phase]
    values = context(component, discover_python=phase in {"environment", "assets"})
    missing = []
    checked = []
    result = {"component": component["id"], "phase": phase, "check_scope": "offline file/module presence; no scientific execution"}
    if stage["support"] == "not_required":
        return {**result, "status": "NOT_REQUIRED", "reason": stage["reason"], "missing": []}

    def record_path(path: Path):
        ok, detail = usable_file(path)
        checked.append({"path": str(path), "status": detail})
        if not ok:
            missing.append(str(path) + ": " + detail)

    if phase == "source":
        source = component["source_checkout"]
        if source:
            root = Path(values["source"])
            try:
                current = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=True).stdout.strip()
                if not current.startswith(source["revision"]):
                    missing.append("checkout revision differs: " + current + " expected " + source["revision"])
                checked.append({"path": str(root), "git_revision": current})
            except (OSError, subprocess.SubprocessError):
                missing.append("pinned Git checkout: " + str(root))
        elif values["runtime_root"]:
            for relative in component["layout"]["source_sentinels"]:
                record_path(Path(values["runtime_root"]) / relative)
            if not component["layout"]["source_sentinels"]:
                missing.append(stage.get("unblock", "No source-readiness contract is declared"))
        else:
            missing.append(stage.get("unblock", "No source-readiness contract is declared"))
    elif phase == "environment":
        modules = component["runtime"]["modules"]
        python = values["runtime_python"]
        if modules:
            probe = module_probe(python, modules)
            missing.extend("Python module: " + name for name in probe["missing_modules"])
            if probe.get("probe_error"):
                missing.append(probe["probe_error"])
            checked.append({"interpreter": python, "modules_checked_without_import": modules})
        binary = component["runtime"].get("binary")
        if binary:
            var = component["runtime"]["configuration_variables"].get("binary", "")
            configured = os.environ.get(var)
            found = configured or shutil.which(binary)
            if not found:
                for env in environment_paths():
                    if env.name in component["runtime"]["conda_env_candidates"] and (env / "bin" / binary).is_file():
                        found = str(env / "bin" / binary)
                        break
            if not found or not os.access(found, os.X_OK):
                missing.append("executable: " + binary)
            else:
                checked.append({"executable": found})
        if not modules and not binary:
            missing.append("No complete environment probe is declared; follow the recorded unblock instructions")
    else:
        root = Path(values["runtime_root"]) if values["runtime_root"] else None
        relatives = list(component["layout"]["required_artifacts"])
        source_root = values["source"]
        # E.g. AiZynth's data files are sentinels under a data root, not its code checkout.
        if root and str(root) != source_root:
            relatives += component["layout"]["source_sentinels"]
        for relative in dict.fromkeys(relatives):
            if root:
                var = component["layout"].get("artifact_env_overrides", {}).get(relative)
                selected = os.environ.get(var, relative) if var else relative
                selected_path = Path(selected).expanduser()
                record_path(selected_path if selected_path.is_absolute() else root / selected_path)
            else:
                missing.append("Runtime root is unspecified for " + relative)
        for path in component["layout"].get("additional_asset_paths", []):
            record_path(Path(expand(path, values)))
        package = component["layout"].get("package_artifact_probe")
        if package:
            probe = module_probe(values["runtime_python"], [], package)
            checked.append({"package_asset_probe": package, **probe})
            if probe["package_asset_count"] == 0:
                missing.append("No packaged model assets found for " + package["module"])
        if not checked:
            missing.append(stage.get("unblock", "No complete asset-presence contract is declared"))
    return {**result, "status": "MISSING" if missing else "PRESENT", "checked": checked, "missing": missing,
            "unblock": stage.get("unblock", component.get("integration", {}).get("install_hint", "") if component.get("integration") else "Consult the pinned source instructions"),
            "request_inputs_not_checked": component["layout"]["additional_inputs"]}


def execute(component: dict, phase: str) -> None:
    global _ENVIRONMENTS
    stage = component["stages"][phase]
    if stage["support"] != "supported":
        raise Refusal(stage.get("unblock", stage.get("reason", "This stage is not automated")))
    if ":" in str(REPO):
        raise Refusal("Use a checkout path without ':' for installation. Offline --plan and --check work in this publication-review directory.")
    values = context(component, discover_python=True)
    recipe = expand(stage, values)
    variables = component["runtime"]["configuration_variables"]
    root_override = os.environ.get(variables.get("root", ""))
    # Shared legacy downloaders use DRUGAGENT_EXTERNAL_DIR. Do not write to one
    # root and then misleadingly verify a different per-tool runtime root.
    uses_shared_fetch = any(
        "fetch_weights.py" in arg or "fetch_pockets.py" in arg
        for argv in recipe.get("commands", []) for arg in argv
    )
    if root_override and (phase == "source" or uses_shared_fetch):
        default_root = component["layout"].get("runtime_root")
        expected_root = expand(default_root, {**values, "runtime_root": ""}) if default_root else None
        if expected_root and Path(values["runtime_root"]).resolve() != Path(expected_root).resolve():
            raise Refusal("This source/asset recipe uses DRUGAGENT_EXTERNAL_DIR. Remove the conflicting per-tool ROOT override, or install the alternate location manually; --check can inspect it.")
    env = os.environ.copy()
    env["DRUGAGENT_EXTERNAL_DIR"] = values["external"]
    env.update(recipe.get("environment", {}))
    if recipe.get("create_runtime_root"):
        Path(values["runtime_root"]).mkdir(parents=True, exist_ok=True)
    if recipe["kind"] == "commands":
        commands = recipe["commands"]
    elif recipe["kind"] in {"conda_environment", "conda_packages"}:
        name = recipe["environment_name"]
        exists = any(path.name == name for path in environment_paths())
        if recipe["kind"] == "conda_environment":
            env_file = Path(recipe["file"])
            if not env_file.is_file():
                raise Refusal("Dependency file is absent: " + str(env_file) + ". Fetch the source/bundle first.")
            commands = [[conda(), "env", "update" if exists else "create", "--name", name, "--file", str(env_file)]]
        else:
            commands = [[conda(), "install" if exists else "create", "-y", "--name", name,
                         *[part for channel in recipe["channels"] for part in ("-c", channel)], *recipe["packages"]]]
    else:
        raise Refusal("Unsupported recipe kind: " + recipe["kind"])
    cwd = Path(values["source"]) if recipe.get("upstream_recipe") else DRUG
    if not cwd.is_dir():
        raise Refusal("Required working directory is absent: " + str(cwd))
    for argv in commands:
        print(json.dumps({"executing": argv, "cwd": str(cwd)}), flush=True)
        subprocess.run(argv, cwd=cwd, env=env, check=True)
    if phase in {"source", "assets"}:
        result = check(component, phase)
        print(json.dumps(result, indent=2))
        if result["status"] == "MISSING":
            raise Refusal("The command finished, but required files are still missing; installation is incomplete.")
    else:
        print("Environment commands completed. Run --check and the runtime capability report; model assets and scientific readiness are separate.")
        # An isolated upstream environment may not be one of the adapter's
        # legacy discovery names. Print the explicit bridge; do not imply that
        # a child process can export configuration into its caller's shell.
        python_var = variables.get("python")
        env_name = recipe.get("environment_name")
        if python_var and env_name:
            _ENVIRONMENTS = None
            paths = [path for path in environment_paths() if path.name == env_name]
            if paths:
                executable = paths[0] / "bin" / "python"
                print("Configure the Task Agent in your shell:")
                print("export " + python_var + "=" + shlex.quote(str(executable)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, epilog="No arguments show help. Supply COMPONENT and STAGE to install. 'all' is accepted only with --plan or --check. Exit 0: requested operation/check succeeded; 1: files/modules missing; 2: refused/unsupported/invalid stage; other nonzero: upstream command failed.")
    parser.add_argument("component", nargs="?", help="Component ID from --list, case-insensitive")
    parser.add_argument("phase", nargs="?", choices=PHASES)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="Print paths and stage recipes; no subprocess or network")
    mode.add_argument("--check", action="store_true", help="Offline presence checks; no model imports, downloads or installation")
    parser.add_argument("--list", action="store_true", help="List component IDs and stage support")
    args = parser.parse_args()
    manifest = json.loads((HERE / "dependencies.json").read_text())
    components = {item["id"].lower(): item for item in manifest["components"]}
    if args.list:
        for key, component in components.items():
            print(key + "\t" + "\t".join(phase + "=" + component["stages"][phase]["support"] for phase in PHASES))
        return 0
    if not args.component:
        parser.print_help()
        return 0
    key = args.component.lower()
    if key == "all" and not (args.plan or args.check):
        parser.error("Bulk installation is not supported. Choose one component and stage explicitly.")
    if key != "all" and key not in components:
        parser.error("Unknown component; use --list")
    selected = list(components.values()) if key == "all" else [components[key]]
    if args.plan:
        print(json.dumps([plan(component, args.phase) for component in selected], indent=2))
        return 0
    if args.check:
        results = [check(component, phase) for component in selected for phase in ([args.phase] if args.phase else PHASES)]
        print(json.dumps(results, indent=2))
        return 1 if any(item["status"] == "MISSING" for item in results) else 0
    if not args.phase:
        parser.error("Installation requires an explicit stage: source, environment or assets")
    try:
        execute(selected[0], args.phase)
    except Refusal as exc:
        print("REFUSED: " + str(exc), file=sys.stderr)
        return 2
    except (OSError, subprocess.CalledProcessError) as exc:
        print("INSTALLATION FAILED: " + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
