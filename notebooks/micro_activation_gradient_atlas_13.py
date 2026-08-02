"""DG-02: activation, gate, rank and linear-probe atlas on DG-01 windows."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import csv, hashlib, io, json, math, os, shutil, sys, time, zipfile

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("HAY_DG02_OUTPUT", "/tmp/hay_dg02")).parent / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(values, **kwargs): del kwargs; return values


def find_repository() -> Path:
    candidates = [Path(__file__).resolve().parents[1]]
    candidates += [p.parent for p in Path("/kaggle/working").glob("**/pyproject.toml")]
    candidates += [p.parent for p in Path("/kaggle/input").glob("**/pyproject.toml")]
    for root in candidates:
        if (root / "src/hay_single_compartment").is_dir(): return root.resolve()
    raise FileNotFoundError("LearningSingleCompartiment repository not found")


REPO_ROOT = find_repository(); sys.path.insert(0, str(REPO_ROOT / "src"))
from hay_single_compartment import InputOnlyConvGRU, InputOnlyGRU, MICRO_STATE_NAMES  # noqa: E402
from hay_single_compartment.failure_atlas import sha256_file  # noqa: E402
from hay_single_compartment.loss_landscape import clone_hidden  # noqa: E402

RUNS = ("gru_mse", "gru_mrstft", "causal_conv_gru_mse", "causal_conv_gru_mrstft")


@dataclass(frozen=True)
class Config:
    temporal_bin: int = 5
    context_chunk_steps: int = 256
    phase_radius_steps: int = 8
    # With only three independent event trajectories, weakly regularized
    # 200-dimensional probes are an interpolation trap. This strong fixed
    # penalty is preregistered for every representation and checkpoint.
    ridge_alpha: float = 100.0
    seed: int = 20260802


CFG = Config()
OUTPUT = Path(os.environ.get("HAY_DG02_OUTPUT", "/kaggle/working/hay_micro_activation_gradient_atlas_13"))
if not Path("/kaggle").exists() and "HAY_DG02_OUTPUT" not in os.environ: OUTPUT = REPO_ROOT / "artifacts/activation_gradient_atlas_13"
OUTPUT.mkdir(parents=True, exist_ok=True); FIGURES = OUTPUT / "figures"; FIGURES.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def valid_dataset(path: Path) -> bool:
    try:
        with h5py.File(path, "r") as h: return "validation/inputs" in h and "validation/burnin_inputs" in h
    except (OSError, ValueError): return False


def valid_source(path: Path, suffixes: tuple[str, ...]) -> bool:
    if path.is_dir(): return all(any(p.as_posix().endswith(s) for p in path.rglob("*") if p.is_file()) for s in suffixes)
    try:
        with zipfile.ZipFile(path) as z: return all(any(n.endswith("/" + s) or n == s for n in z.namelist()) for s in suffixes)
    except (OSError, zipfile.BadZipFile): return False


def discover(override: str, predicate, patterns: tuple[str, ...]) -> Path:
    value = os.environ.get(override)
    if value:
        path = Path(value).expanduser().resolve()
        if not predicate(path): raise ValueError(f"invalid {override}: {path}")
        return path
    roots = [p for p in (Path("/kaggle/input"), Path("/kaggle/working"), REPO_ROOT.parent) if p.exists()]
    found: dict[str, Path] = {}
    for root in roots:
        for pattern in patterns:
            for path in root.glob("**/" + pattern):
                if predicate(path): found[str(path.resolve())] = path.resolve()
    if not found: raise FileNotFoundError(f"cannot discover {override}")
    return sorted(found.values(), key=str)[0]


DATASET = discover("HAY_DG02_DATASET", valid_dataset, ("*.h5",))
FACTORIAL = discover("HAY_DG02_FACTORIAL", lambda p: valid_source(p, tuple(f"checkpoints/{r}.pt" for r in RUNS)), ("*.zip", "hay_micro_orthogonal_factorial_11"))
DG01 = discover("HAY_DG02_DG01", lambda p: valid_source(p, ("decision.json", "validation_window_manifest.csv")), ("*.zip", "hay_micro_loss_landscape_12"))
print("repository:", REPO_ROOT); print("device:", DEVICE); print("dataset:", DATASET); print("factorial:", FACTORIAL); print("DG-01:", DG01); print("output:", OUTPUT)
print("[contract] frozen models; probe labels only; validation only; test closed", flush=True)


def source_bytes(source: Path, suffix: str) -> bytes:
    normalized = suffix.replace("\\", "/").lstrip("/")
    if source.is_dir():
        matches = [p for p in source.rglob("*") if p.is_file() and (p.as_posix().endswith("/" + normalized) or p.as_posix().endswith(normalized))]
        if len(matches) != 1: raise FileNotFoundError((source, suffix, matches))
        return matches[0].read_bytes()
    with zipfile.ZipFile(source) as z:
        matches = [n for n in z.namelist() if n == normalized or n.endswith("/" + normalized)]
        if len(matches) != 1: raise FileNotFoundError((source, suffix, matches))
        return z.read(matches[0])


def checkpoint(run: str): return torch.load(io.BytesIO(source_bytes(FACTORIAL, f"checkpoints/{run}.pt")), map_location="cpu", weights_only=False)
CHECKPOINTS = {run: checkpoint(run) for run in RUNS}
DATASET_HASH = sha256_file(DATASET)
if {c["dataset_sha256"] for c in CHECKPOINTS.values()} != {DATASET_HASH}: raise RuntimeError("dataset/checkpoint hash mismatch")
decision = json.loads(source_bytes(DG01, "decision.json")); expected_manifest_hash = decision["manifest_sha256"]
manifest_bytes = source_bytes(DG01, "validation_window_manifest.csv")
if hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_hash: raise RuntimeError("DG-01 manifest hash mismatch")
MANIFEST = list(csv.DictReader(io.StringIO(manifest_bytes.decode("utf-8"))))


def pack(values: np.ndarray) -> np.ndarray:
    usable = values.shape[1] // CFG.temporal_bin * CFG.temporal_bin
    return values[:, :usable].reshape(values.shape[0], usable // CFG.temporal_bin, -1).astype(np.float32)


with h5py.File(DATASET, "r") as h:
    inputs = pack(h["validation/inputs"][:]); burnin = pack(h["validation/burnin_inputs"][:])
    states = h["validation/states"][:, ::CFG.temporal_bin][:, 1:inputs.shape[1] + 1].astype(np.float32)
    raw_spikes = h["validation/spikes"][:]
spikes = raw_spikes.reshape(raw_spikes.shape[0], -1, CFG.temporal_bin).max(2).astype(bool)
SOMA = list(MICRO_STATE_NAMES).index("soma.v_mV")


def build_model(c: Mapping[str, Any]) -> nn.Module:
    s = c["model_spec"]
    if s["class"] == "InputOnlyGRU": model = InputOnlyGRU(s["input_dim"], s["state_dim"], hidden_dim=s["hidden_dim"], layers=s["layers"], decoder_dim=s["decoder_dim"])
    else: model = InputOnlyConvGRU(s["input_dim"], s["state_dim"], hidden_dim=s["hidden_dim"], conv_channels=s["conv_channels"], dilations=tuple(s["dilations"]), kernel_size=s["kernel_size"], decoder_dim=s["decoder_dim"])
    model.load_state_dict(c["model_state_dict"]); return model.to(DEVICE).eval()


def advance(model, values, hidden=None):
    with torch.no_grad():
        for start in range(0, len(values), CFG.context_chunk_steps): _, hidden = model(torch.as_tensor(values[None, start:start + CFG.context_chunk_steps], device=DEVICE), hidden); hidden = clone_hidden(hidden)
    return hidden


def history_cache(model):
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in MANIFEST: grouped.setdefault(int(row["trajectory"]), []).append(row)
    result = {}
    for trajectory, rows in sorted(grouped.items()):
        hidden = advance(model, burnin[trajectory]); position = 0
        for row in sorted(rows, key=lambda x:int(x["start_step"])):
            start = int(row["start_step"]); hidden = advance(model, inputs[trajectory, position:start], hidden)
            result[row["window_id"]] = clone_hidden(hidden); position = start
    return result


def replay(gru: nn.GRU, features: torch.Tensor, hidden: torch.Tensor | None):
    h = features.new_zeros(features.shape[0], gru.hidden_size) if hidden is None else hidden[-1].clone()
    wir,wiz,win=gru.weight_ih_l0.chunk(3); whr,whz,whn=gru.weight_hh_l0.chunk(3)
    bir,biz,bin_=gru.bias_ih_l0.chunk(3); bhr,bhz,bhn=gru.bias_hh_l0.chunk(3)
    records={k:[] for k in ("reset","update","candidate","hidden")}
    for step in range(features.shape[1]):
        x=features[:,step]; r=torch.sigmoid(F.linear(x,wir,bir)+F.linear(h,whr,bhr)); z=torch.sigmoid(F.linear(x,wiz,biz)+F.linear(h,whz,bhz)); n=torch.tanh(F.linear(x,win,bin_)+r*F.linear(h,whn,bhn)); h=(1-z)*n+z*h
        for key,value in (("reset",r),("update",z),("candidate",n),("hidden",h)): records[key].append(value)
    return {k:torch.stack(v,1) for k,v in records.items()}


def internals(model, values: torch.Tensor, hidden):
    if isinstance(model, InputOnlyGRU): recurrent_hidden=hidden; features=model.input_encoder(values); frontend=features
    else:
        recurrent_hidden, cache = hidden
        combined=torch.cat((cache,values),1); frontend=model.frontend(combined.transpose(1,2)).transpose(1,2); features=frontend
    sequence,next_hidden=model.recurrent(features,recurrent_hidden)
    gates=replay(model.recurrent,features,recurrent_hidden)
    normalized=model.decoder.network[0](sequence); linear=model.decoder.network[1](normalized); decoded=model.decoder.network[2](linear); prediction=model.decoder.network[3](decoded)
    return {
        "recurrent_input":frontend, "hidden":sequence,
        "decoder":decoded, "reset":gates["reset"],
        "update":gates["update"], "candidate":gates["candidate"],
        "replayed_hidden":gates["hidden"], "prediction":prediction,
    }, next_hidden


def nearest_phase(trajectory: int, steps_array: np.ndarray) -> np.ndarray:
    locations=np.flatnonzero(spikes[trajectory]); result=np.full(len(steps_array),999,dtype=np.int32)
    for i,step in enumerate(steps_array):
        if len(locations): result[i]=int(locations[np.argmin(np.abs(locations-step))]-step)
    return result


def write_rows(path, rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w",newline="",encoding="utf-8") as h: w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)


def stats(values):
    flat=values.reshape(-1).astype(np.float64); mean=flat.mean(); std=flat.std()+1e-12
    return {"mean":mean,"std":std,"q01":np.quantile(flat,.01),"q50":np.quantile(flat,.5),"q99":np.quantile(flat,.99),"near_zero_fraction":np.mean(np.abs(flat)<1e-3),"skewness":np.mean(((flat-mean)/std)**3),"excess_kurtosis":np.mean(((flat-mean)/std)**4)-3}


def rank_stats(values):
    x=values.reshape(-1,values.shape[-1]).astype(np.float64); x-=x.mean(0); s=np.linalg.svd(x,compute_uv=False); energy=s*s; p=energy/max(energy.sum(),1e-12); cumulative=np.cumsum(p)
    return {"rank90":int(np.searchsorted(cumulative,.9)+1),"rank99":int(np.searchsorted(cumulative,.99)+1),"participation_ratio":float(1/max(np.sum(p*p),1e-12))}


def auc(y,score):
    y=np.asarray(y,bool); pos=score[y]; neg=score[~y]
    return float(((pos[:,None]>neg).mean()+.5*(pos[:,None]==neg).mean())) if len(pos) and len(neg) else float("nan")


def correlation(a,b):
    return float(np.corrcoef(a,b)[0,1]) if len(a)>2 and np.std(a)>0 and np.std(b)>0 else float("nan")


def ridge_oof(features, target, trajectories, mask, classification=False):
    predictions=np.full(len(target),np.nan,np.float64); folds=[]
    for held in sorted(set(trajectories[mask])):
        train=mask&(trajectories!=held); test=mask&(trajectories==held)
        if train.sum()<10 or test.sum()==0: continue
        mean=features[train].mean(0); std=np.maximum(features[train].std(0),1e-5); x=(features-mean)/std
        xa=np.concatenate((x,np.ones((len(x),1))),1); weights=np.ones(train.sum())
        if classification:
            yy=target[train]>0.5; weights=np.where(yy,.5/max(yy.mean(),1e-6),.5/max((~yy).mean(),1e-6))
        xt=xa[train]*np.sqrt(weights)[:,None]; yt=target[train]*np.sqrt(weights)
        beta=np.linalg.solve(xt.T@xt+CFG.ridge_alpha*np.eye(xt.shape[1]),xt.T@yt); predictions[test]=xa[test]@beta; folds.append(int(held))
    valid=mask&np.isfinite(predictions); return predictions,valid,folds


activation_rows=[]; rank_rows=[]; gate_rows=[]; probe_rows=[]; gradient_rows=[]; replay_rows=[]
for run_index,run in enumerate(RUNS,1):
    print(f"[{run_index}/4] {run}: contexts and activations",flush=True); c=CHECKPOINTS[run]; model=build_model(c); cache=history_cache(model)
    collected={k:[] for k in ("raw_input","recurrent_input","hidden","decoder","reset","update","candidate")}; metadata=[]; prediction_values=[]; max_replay=0.0
    for row in MANIFEST:
        tr,start,stop=int(row["trajectory"]),int(row["start_step"]),int(row["stop_step"]); values=torch.as_tensor(inputs[tr:tr+1,start:stop],device=DEVICE)
        with torch.no_grad():
            internal,_=internals(model,values,cache[row["window_id"]])
            max_replay=max(max_replay,float((internal["hidden"]-internal["replayed_hidden"]).abs().max()))
        collected["raw_input"].append(values.squeeze(0).cpu().numpy())
        for name in collected:
            if name!="raw_input": collected[name].append(internal[name].squeeze(0).cpu().numpy())
        prediction_values.append(internal["prediction"].squeeze(0).cpu().numpy())
        phase=nearest_phase(tr,np.arange(start,stop)); metadata += [(tr,step,row["sampling_view"],int(ph)) for step,ph in zip(range(start,stop),phase)]
    arrays={k:np.concatenate(v) for k,v in collected.items()}; predictions=np.concatenate(prediction_values); meta=np.array(metadata,object); trajectory=meta[:,0].astype(int); absolute_step=meta[:,1].astype(int); sampling=meta[:,2].astype(str); phase=meta[:,3].astype(int); support=np.abs(phase)<=CFG.phase_radius_steps
    truth_voltage=np.array([states[t,s,SOMA] for t,s in zip(trajectory,absolute_step)]); mean=np.asarray(c["state_mean"]); std=np.asarray(c["state_std"]); predicted_voltage=predictions[:,SOMA]*std[SOMA]+mean[SOMA]; residual=truth_voltage-predicted_voltage
    # Activation distributions and ranks retain the exact manifest multiplicity.
    for name,array in arrays.items():
        for view in sorted(set(sampling)):
            selected=array[sampling==view]; row={"run":run,"activation":name,"view":view,"samples":len(selected),**stats(selected),**rank_stats(selected)}
            if name in {"reset","update"}: row.update({"low_saturation_fraction":float((selected<.05).mean()),"high_saturation_fraction":float((selected>.95).mean()),"binary_entropy":float(np.mean(-(np.clip(selected,1e-7,1-1e-7)*np.log(np.clip(selected,1e-7,1-1e-7))+(1-np.clip(selected,1e-7,1-1e-7))*np.log(1-np.clip(selected,1e-7,1-1e-7)))) )})
            activation_rows.append(row)
        rank_rows.append({"run":run,"activation":name,"view":"all_manifest",**rank_stats(array)})
    for gate in ("reset","update"):
        for view in sorted(set(sampling)):
            selected=arrays[gate][sampling==view]; gate_rows.append({"run":run,"gate":gate,"view":view,"mean":float(selected.mean()),"low_fraction":float((selected<.05).mean()),"high_fraction":float((selected>.95).mean()),"retention_tau_proxy_steps":float(np.median(-1/np.log(np.clip(selected,.001,.999)))) if gate=="update" else float("nan")})
    # Remove duplicate overlapping coordinates before trajectory-held-out probes.
    keep=[]; seen=set()
    for i,key in enumerate(zip(trajectory,absolute_step)):
        if key not in seen: seen.add(key); keep.append(i)
    keep=np.array(keep); tr=trajectory[keep]; ph=phase[keep]; sup=support[keep]; target_residual=residual[keep]
    representations={name:array[keep] for name,array in arrays.items() if name in {"raw_input","recurrent_input","hidden","decoder"}}
    for name,features in representations.items():
        score,valid,folds=ridge_oof(features,sup.astype(float),tr,np.ones(len(tr),bool),True)
        probe_rows.append({"run":run,"representation":name,"task":"event_support","samples":int(valid.sum()),"held_out_trajectories":json.dumps(folds),"roc_auc":auc(sup[valid],score[valid]),"correlation":correlation(sup[valid].astype(float),score[valid])})
        phase_score,phase_valid,folds=ridge_oof(features,ph.astype(float),tr,sup,False)
        probe_rows.append({"run":run,"representation":name,"task":"exact_phase","samples":int(phase_valid.sum()),"held_out_trajectories":json.dumps(folds),"mae_steps":float(np.mean(np.abs(phase_score[phase_valid]-ph[phase_valid]))) if phase_valid.any() else float("nan"),"correlation":correlation(ph[phase_valid],phase_score[phase_valid])})
        amp_score,amp_valid,folds=ridge_oof(features,target_residual,tr,sup,False)
        baseline=float(np.sqrt(np.mean(target_residual[amp_valid]**2))) if amp_valid.any() else float("nan"); corrected=float(np.sqrt(np.mean((target_residual[amp_valid]-amp_score[amp_valid])**2))) if amp_valid.any() else float("nan")
        probe_rows.append({"run":run,"representation":name,"task":"event_voltage_residual","samples":int(amp_valid.sum()),"held_out_trajectories":json.dumps(folds),"baseline_rmse_mV":baseline,"probe_rmse_mV":corrected,"relative_improvement":(baseline-corrected)/baseline if baseline>0 else float("nan"),"correlation":correlation(target_residual[amp_valid],amp_score[amp_valid])})
    # Activation gradients use the same windows and no parameter update.
    model.train()
    for view in ("event","subthreshold"):
        selected=[row for row in MANIFEST if row["sampling_view"]==view]
        for row in selected:
            tr0,start,stop=int(row["trajectory"]),int(row["start_step"]),int(row["stop_step"]); values=torch.as_tensor(inputs[tr0:tr0+1,start:stop],device=DEVICE); model.zero_grad(set_to_none=True); internal,_=internals(model,values,cache[row["window_id"]]); internal["hidden"].retain_grad(); internal["decoder"].retain_grad(); target=torch.as_tensor((states[tr0:tr0+1,start:stop]-mean)/std,device=DEVICE); loss=(internal["prediction"][...,SOMA]-target[...,SOMA]).square().mean(); loss.backward()
            for name in ("hidden","decoder"):
                grad=internal[name].grad.detach(); gradient_rows.append({"run":run,"view":view,"window_id":row["window_id"],"activation":name,"loss":float(loss.detach()),"gradient_norm":float(torch.linalg.vector_norm(grad)),"near_zero_fraction":float((grad.abs()<1e-8).float().mean())})
    model.eval(); replay_rows.append({"run":run,"gate_replay_max_abs_error":max_replay}); print(f"[{run}] complete",flush=True)


write_rows(OUTPUT/"activation_summary.csv",activation_rows); write_rows(OUTPUT/"rank_summary.csv",rank_rows); write_rows(OUTPUT/"gate_summary.csv",gate_rows); write_rows(OUTPUT/"probe_summary.csv",probe_rows); write_rows(OUTPUT/"activation_gradients.csv",gradient_rows); write_rows(OUTPUT/"replay_verification.csv",replay_rows); (OUTPUT/"validation_window_manifest.csv").write_bytes(manifest_bytes)

# Conservative decision: use median hidden/decoder held-out capability.
def task_values(rep,task,key): return [float(r[key]) for r in probe_rows if r["representation"]==rep and r["task"]==task and r.get(key,"") not in ("",None) and math.isfinite(float(r[key]))]
hidden_phase=np.median(task_values("hidden","exact_phase","correlation")); decoder_phase=np.median(task_values("decoder","exact_phase","correlation")); hidden_amp=np.median(task_values("hidden","event_voltage_residual","relative_improvement")); decoder_amp=np.median(task_values("decoder","event_voltage_residual","relative_improvement")); hidden_support=np.median(task_values("hidden","event_support","roc_auc"))
if hidden_phase>=.5 and hidden_amp>=.2 and (decoder_phase<.5 or decoder_amp<.2): branch="readout_contract"
elif hidden_phase>=.5 and hidden_amp>=.2 and decoder_phase>=.5 and decoder_amp>=.2: branch="objective_exposure_factorial"
elif hidden_phase<.2 and hidden_amp<.05 and hidden_support>=.7: branch="information_contract_or_scaffold"
else: branch="inconclusive_capability_bench"
result={"experiment_id":"DG-02","decision":branch,"requires_human_review":True,"median_hidden_support_auc":float(hidden_support),"median_hidden_phase_correlation":float(hidden_phase),"median_decoder_phase_correlation":float(decoder_phase),"median_hidden_event_residual_improvement":float(hidden_amp),"median_decoder_event_residual_improvement":float(decoder_amp),"manifest_sha256":expected_manifest_hash,"dataset_sha256":DATASET_HASH,"test_split_opened":False,"checkpoint_optimizer_steps":0,"limitations":["linear probes only","one checkpoint seed","six event windows from three trajectories","overlapping coordinates removed before probes"]}
(OUTPUT/"decision.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); (OUTPUT/"provenance.json").write_text(json.dumps({"created_utc":datetime.now(timezone.utc).isoformat(),"repository":str(REPO_ROOT),"dataset":str(DATASET),"factorial":str(FACTORIAL),"dg01":str(DG01),"device":str(DEVICE),"config":asdict(CFG),**result},indent=2),encoding="utf-8")

# Compact diagnostic figures.
fig,axes=plt.subplots(1,2,figsize=(14,5)); reps=("raw_input","recurrent_input","hidden","decoder"); x=np.arange(len(reps)); width=.18
for i,run in enumerate(RUNS):
    support_rows={r["representation"]:r for r in probe_rows if r["run"]==run and r["task"]=="event_support"}; amp_rows={r["representation"]:r for r in probe_rows if r["run"]==run and r["task"]=="event_voltage_residual"}; axes[0].bar(x+(i-1.5)*width,[float(support_rows[r]["roc_auc"]) for r in reps],width,label=run); axes[1].bar(x+(i-1.5)*width,[float(amp_rows[r]["relative_improvement"]) for r in reps],width,label=run)
for ax,title in zip(axes,("Held-out event-support AUROC","Held-out event residual RMSE improvement")): ax.set_xticks(x,reps,rotation=25,ha="right");ax.set_title(title);ax.grid(axis="y",alpha=.25)
axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(FIGURES/"probe_capabilities.png",dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(13,5));
for run in RUNS:
    rr=[r for r in gate_rows if r["run"]==run and r["gate"]=="update"]; axes[0].plot([r["view"] for r in rr],[float(r["mean"]) for r in rr],marker="o",label=run); axes[1].plot([r["view"] for r in rr],[float(r["retention_tau_proxy_steps"]) for r in rr],marker="o",label=run)
axes[0].set_title("Mean GRU update gate");axes[1].set_title("Update-gate retention proxy");
for ax in axes: ax.grid(alpha=.25);ax.tick_params(axis="x",rotation=25)
axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(FIGURES/"gate_dynamics.png",dpi=180);plt.close(fig)
(OUTPUT/"README.md").write_text("# DG-02 Activation and Gradient Atlas\n\nFrozen-checkpoint activation, gate, rank, trajectory-held-out linear-probe and activation-gradient diagnostics on the exact DG-01 manifest. No test data or checkpoint update.\n",encoding="utf-8")
ZIP_PATH=Path(shutil.make_archive(str(OUTPUT.parent/"hay_micro_activation_gradient_atlas_13_complete"),"zip",root_dir=OUTPUT.parent,base_dir=OUTPUT.name));print(json.dumps(result,indent=2));print("archive:",ZIP_PATH,f"({ZIP_PATH.stat().st_size/2**20:.1f} MiB)")
