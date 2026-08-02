"""SR-01: parameter-identical GRU information-contract preflight.

The only causal factor is the normalized physical-state channel: always zero,
initialization only, or recursively filled with the model's own prediction.
Teacher state is never consumed after the first complete-trajectory step.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import csv, hashlib, io, json, math, os, sys, time, zipfile

import h5py
import numpy as np
import torch
from torch import nn
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("HAY_SR01_OUTPUT", "/tmp/hay_micro_state_feedback_preflight_14")).parent / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(values=None, total=None, desc="", **kwargs):
        del kwargs
        if values is not None: return values
        class Progress:
            def update(self, value=1): del value
            def set_postfix(self, **values): print(desc, values, flush=True)
            def close(self): pass
        return Progress()


def find_repository() -> Path:
    candidates = [Path(__file__).resolve().parents[1]]
    candidates += [p.parent for p in Path("/kaggle/working").glob("**/pyproject.toml")]
    candidates += [p.parent for p in Path("/kaggle/input").glob("**/pyproject.toml")]
    for root in candidates:
        if (root / "src/hay_single_compartment").is_dir(): return root.resolve()
    raise FileNotFoundError("LearningSingleCompartiment repository not found")


REPO_ROOT = find_repository(); sys.path.insert(0, str(REPO_ROOT / "src"))
from hay_single_compartment import (  # noqa: E402
    MICRO_EVENT_NAMES, MICRO_STATE_NAMES, StateContextGRU,
    StratifiedWindowSampler, classify_micro_events, count_trainable_parameters,
)
from hay_single_compartment.failure_atlas import (  # noqa: E402
    is_slow_state, match_spikes, residual_spectrum, sha256_file,
)


@dataclass(frozen=True)
class Config:
    temporal_bin: int = 5
    batch_trajectories: int = 6
    chunk_steps: int = 256
    epochs: int = 20
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    stratified_windows_per_epoch: int = 48
    stratified_window_steps: int = 256
    hidden_dim: int = 200
    decoder_dim: int = 200
    seed: int = 20260803


CFG = Config(
    epochs=int(os.environ.get("HAY_SR01_EPOCHS", "20")),
    batch_trajectories=int(os.environ.get("HAY_SR01_BATCH_TRAJECTORIES", "6")),
    stratified_windows_per_epoch=int(os.environ.get("HAY_SR01_WINDOWS_PER_EPOCH", "48")),
)
if CFG.epochs < 1 or CFG.batch_trajectories < 1:
    raise ValueError("epochs and batch size must be positive")
OUTPUT = Path(os.environ.get("HAY_SR01_OUTPUT", "/kaggle/working/hay_micro_state_feedback_preflight_14"))
if not Path("/kaggle").exists() and "HAY_SR01_OUTPUT" not in os.environ:
    OUTPUT = REPO_ROOT / "artifacts/state_feedback_preflight_14"
CHECKPOINTS = OUTPUT / "checkpoints"; FIGURES = OUTPUT / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True); CHECKPOINTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARMS = {
    "no_state_context": "none",
    "initial_state_only": "initial_only",
    "predicted_state_feedback": "predicted_feedback",
}


def valid_dataset(path: Path) -> bool:
    try:
        with h5py.File(path, "r") as h:
            return all(k in h for k in (
                "train/inputs", "train/states", "train/burnin_inputs", "train/burnin_states",
                "validation/inputs", "validation/states", "validation/burnin_inputs",
                "validation/burnin_states", "validation/spikes", "validation/regimes",
            )) and "state_names_json" in h.attrs
    except (OSError, ValueError, KeyError): return False


def discover_dataset() -> Path:
    value = os.environ.get("HAY_SR01_DATASET")
    if value:
        path = Path(value).expanduser().resolve()
        if not valid_dataset(path): raise ValueError(f"invalid HAY_SR01_DATASET: {path}")
        return path
    roots = [p for p in (Path("/kaggle/input"), Path("/kaggle/working"), REPO_ROOT.parent) if p.exists()]
    found = {str(p.resolve()): p.resolve() for root in roots for p in root.glob("**/*.h5") if valid_dataset(p)}
    if not found: raise FileNotFoundError("Attach hay_micro_4c_event_enriched_v2.h5 or set HAY_SR01_DATASET")
    return sorted(found.values(), key=lambda p: (0 if "event_enriched_v2" in p.name else 1, str(p)))[0]


DATASET = discover_dataset()
print("repository:", REPO_ROOT); print("device:", DEVICE); print("dataset:", DATASET); print("output:", OUTPUT)
print("[contract] validation selection only; test closed; teacher state only at trajectory initialization", flush=True)
print("[integrity] hashing dataset ...", flush=True); DATASET_HASH = sha256_file(DATASET); print("[integrity]", DATASET_HASH, flush=True)
EXPECTED_DATASET_HASH = "1fd0eaf7ffc6bbd5e8eb2db64ba4bcc67289048ef0be9367760088ff1739a3bf"
if DATASET_HASH != EXPECTED_DATASET_HASH: raise RuntimeError(f"dataset hash mismatch: {DATASET_HASH}")


def pack(values: np.ndarray) -> np.ndarray:
    usable = values.shape[1] // CFG.temporal_bin * CFG.temporal_bin
    return values[:, :usable].reshape(values.shape[0], usable // CFG.temporal_bin, -1).astype(np.float32)


def binary(values: np.ndarray) -> np.ndarray:
    usable = values.shape[1] // CFG.temporal_bin * CFG.temporal_bin
    return values[:, :usable].reshape(values.shape[0], usable // CFG.temporal_bin, CFG.temporal_bin).max(2)


def regimes(values: np.ndarray) -> np.ndarray:
    usable = values.shape[1] // CFG.temporal_bin * CFG.temporal_bin
    return values[:, :usable].reshape(values.shape[0], usable // CFG.temporal_bin, CFG.temporal_bin)[:, :, -1]


def load_split(name: str, maximum: int | None = None) -> dict[str, np.ndarray]:
    with h5py.File(DATASET, "r") as h:
        config = json.loads(h.attrs["config_json"]); count = int(h[f"{name}/inputs"].shape[0])
        if maximum is not None: count = min(count, maximum)
        burnin = pack(h[f"{name}/burnin_inputs"][:count]); inputs = pack(h[f"{name}/inputs"][:count])
        burnin_states = h[f"{name}/burnin_states"][:count, ::CFG.temporal_bin].astype(np.float32)
        states = h[f"{name}/states"][:count, ::CFG.temporal_bin].astype(np.float32)
        spikes = binary(h[f"{name}/spikes"][:count]); regime = regimes(h[f"{name}/regimes"][:count])
    burnin_states = burnin_states[:, :burnin.shape[1] + 1]; states = states[:, :inputs.shape[1] + 1]
    events = classify_micro_events(states[:, 1:], spikes, MICRO_STATE_NAMES, float(config["dt_ms"]) * CFG.temporal_bin)
    return {"burnin":burnin,"inputs":inputs,"burnin_states":burnin_states,"states":states,"spikes":spikes,"regimes":regime,"events":events}


max_train = int(os.environ.get("HAY_SR01_MAX_TRAIN_TRAJECTORIES", "0")) or None
max_validation = int(os.environ.get("HAY_SR01_MAX_VALIDATION_TRAJECTORIES", "0")) or None
print("[data] loading train + validation (never test) ...", flush=True)
train = load_split("train", max_train); validation = load_split("validation", max_validation)
STATE_NAMES = list(MICRO_STATE_NAMES)
with h5py.File(DATASET, "r") as h:
    if json.loads(h.attrs["state_names_json"]) != STATE_NAMES: raise RuntimeError("state order mismatch")
    input_names = json.loads(h.attrs["input_names_json"]); dataset_config = json.loads(h.attrs["config_json"])
normalization = np.concatenate((train["burnin_states"], train["states"][:, 1:]), axis=1)
STATE_MEAN = normalization.reshape(-1, len(STATE_NAMES)).mean(0).astype(np.float32)
STATE_STD = np.maximum(normalization.reshape(-1, len(STATE_NAMES)).std(0), 1e-6).astype(np.float32)
del normalization
train_burnin_n = (train["burnin_states"] - STATE_MEAN) / STATE_STD
train_states_n = (train["states"] - STATE_MEAN) / STATE_STD
validation_burnin_n = (validation["burnin_states"] - STATE_MEAN) / STATE_STD
INPUT_DIM = train["inputs"].shape[-1]; STATE_DIM = len(STATE_NAMES)
DT_MS = float(dataset_config["dt_ms"]) * CFG.temporal_bin


def build(arm: str) -> StateContextGRU:
    torch.manual_seed(CFG.seed); np.random.seed(CFG.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(CFG.seed)
    return StateContextGRU(INPUT_DIM, STATE_DIM, CFG.hidden_dim, CFG.decoder_dim, ARMS[arm])


def state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


parameter_counts = {arm: count_trainable_parameters(build(arm)) for arm in ARMS}
initial_hashes = {arm: state_hash(build(arm)) for arm in ARMS}
if len(set(parameter_counts.values())) != 1 or len(set(initial_hashes.values())) != 1:
    raise RuntimeError(f"arms are not exactly matched: parameters={parameter_counts}, hashes={initial_hashes}")
print("[control] parameters:", parameter_counts, "| shared initialization:", next(iter(initial_hashes.values())), flush=True)


def tensor(values: np.ndarray, indices: np.ndarray | None = None) -> torch.Tensor:
    if indices is not None: values = values[indices]
    return torch.as_tensor(values, device=DEVICE)


def detach(hidden): return None if hidden is None else tuple(x.detach() for x in hidden)
def clone(hidden): return None if hidden is None else tuple(x.detach().clone() for x in hidden)
def stack_hidden(values): return tuple(torch.cat([value[i] for value in values], dim=0) for i in range(2))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


@torch.no_grad()
def predict_validation(model: StateContextGRU) -> np.ndarray:
    model.eval(); results = []
    for offset in range(0, len(validation["inputs"]), CFG.batch_trajectories):
        stop = min(len(validation["inputs"]), offset + CFG.batch_trajectories)
        hidden = None; initial = tensor(validation_burnin_n[offset:stop, 0])
        burnin = tensor(validation["burnin"][offset:stop]); values = tensor(validation["inputs"][offset:stop])
        for start in range(0, burnin.shape[1], CFG.chunk_steps):
            _, hidden = model(burnin[:, start:start+CFG.chunk_steps], hidden, initial_state=initial if hidden is None else None)
            hidden = detach(hidden)
        outputs = []
        for start in range(0, values.shape[1], CFG.chunk_steps):
            prediction, hidden = model(values[:, start:start+CFG.chunk_steps], hidden); hidden = detach(hidden); outputs.append(prediction)
        normalized = torch.cat(outputs, 1).cpu().numpy(); results.append(normalized * STATE_STD + STATE_MEAN)
    return np.concatenate(results)


FAST_EVENTS = ("isolated_spike", "burst_spike", "rapid_fire", "tuft_plateau", "spike_with_tuft_plateau")
SOMA = STATE_NAMES.index("soma.v_mV")
SLOW = [i for i,n in enumerate(STATE_NAMES) if is_slow_state(n)]
SYNAPTIC = [i for i,n in enumerate(STATE_NAMES) if any(t in n for t in ("ampa_","nmda_","gabaa_"))]


def metrics(prediction: np.ndarray) -> dict[str, float | int]:
    truth = validation["states"][:, 1:prediction.shape[1]+1]; error = prediction - truth
    state_nrmse = np.sqrt(np.mean(np.square(error / STATE_STD), axis=(0,1)))
    event_indices = [MICRO_EVENT_NAMES.index(n) for n in FAST_EVENTS]
    event = validation["events"][..., event_indices].any(-1); sub = validation["events"][...,0].astype(bool)
    spike_report, _ = match_spikes(truth[...,SOMA], prediction[...,SOMA], DT_MS)
    spectrum, _ = residual_spectrum(truth[...,SOMA], prediction[...,SOMA], DT_MS, ((0,50),(50,200),(200,1000)))
    bands = {f"power_ratio_{int(x['low_hz'])}_{int(x['high_hz'])}_Hz":x["prediction_to_teacher_ratio"] for x in spectrum["bands"]}
    result: dict[str,float|int] = {
        "mean_state_nrmse":float(state_nrmse.mean()), "worst_state_nrmse":float(state_nrmse.max()),
        "slow_state_nrmse":float(np.mean(state_nrmse[SLOW])), "synaptic_state_nrmse":float(np.mean(state_nrmse[SYNAPTIC])),
        "soma_rmse_mV":float(np.sqrt(np.mean(np.square(error[...,SOMA])))),
        "event_soma_rmse_mV":float(np.sqrt(np.mean(np.square(error[...,SOMA][event])))),
        "subthreshold_soma_rmse_mV":float(np.sqrt(np.mean(np.square(error[...,SOMA][sub])))),
        "truth_spikes":int(spike_report["truth_spikes"]), "predicted_spikes":int(spike_report["predicted_spikes"]),
        "matched_spikes":int(spike_report["matched_spikes"]), "spike_precision":float(spike_report["precision"]),
        "spike_recall":float(spike_report["recall"]), "spike_f1":float(spike_report["f1"]), "log_psd_rmse":float(spectrum["log_psd_rmse"]), **bands,
    }
    return result


complete_inputs = np.concatenate((train["burnin"], train["inputs"]), axis=1)


@torch.no_grad()
def contexts_for_rows(model: StateContextGRU, rows: np.ndarray):
    model.eval(); wanted: dict[int,list[int]] = {}
    burnin_steps = train["burnin"].shape[1]
    for trajectory, start in rows: wanted.setdefault(int(trajectory), []).append(burnin_steps + int(start))
    cache = {}
    for trajectory, stops in wanted.items():
        hidden = None; position = 0; sequence = tensor(complete_inputs[trajectory:trajectory+1]); initial = tensor(train_burnin_n[trajectory:trajectory+1,0])
        for stop in sorted(set(stops)):
            while position < stop:
                end = min(stop, position + CFG.chunk_steps)
                _, hidden = model(sequence[:,position:end], hidden, initial_state=initial if hidden is None else None)
                hidden = detach(hidden); position = end
            cache[(trajectory, stop - burnin_steps)] = clone(hidden)
    return stack_hidden([cache[(int(t),int(s))] for t,s in rows])


def updates_per_epoch() -> int:
    batches = math.ceil(len(train["inputs"]) / CFG.batch_trajectories)
    chunks = math.ceil(train["burnin"].shape[1]/CFG.chunk_steps) + math.ceil(train["inputs"].shape[1]/CFG.chunk_steps)
    return batches * chunks + math.ceil(CFG.stratified_windows_per_epoch / CFG.batch_trajectories)


def train_arm(arm: str):
    model = build(arm).to(DEVICE); optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type=="cuda")
    last_path = CHECKPOINTS/f"{arm}_last.pt"; best_path = CHECKPOINTS/f"{arm}.pt"; summary_path=OUTPUT/f"{arm}_summary.json"
    history=[]; best=float("inf"); start_epoch=1
    if last_path.exists():
        saved=torch.load(last_path,map_location=DEVICE,weights_only=False)
        if saved.get("dataset_sha256")==DATASET_HASH and saved.get("config")==asdict(CFG) and saved.get("arm")==arm:
            model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); scheduler.load_state_dict(saved["scheduler"]); scaler.load_state_dict(saved["scaler"])
            history=list(saved["history"]); best=float(saved["best"]); start_epoch=int(saved["epoch"])+1; print(f"[{arm}] resume after epoch {start_epoch-1}",flush=True)
    if summary_path.exists() and best_path.exists():
        summary=json.loads(summary_path.read_text());
        if summary.get("completed_epochs")==CFG.epochs and summary.get("dataset_sha256")==DATASET_HASH:
            model.load_state_dict(torch.load(best_path,map_location=DEVICE,weights_only=False)["model_state_dict"]); print(f"[{arm}] completed run reused",flush=True); return model,history,summary["best_validation"]
    started=time.perf_counter(); total=updates_per_epoch()
    for epoch in range(start_epoch,CFG.epochs+1):
        model.train(); progress=tqdm(total=total,desc=f"{arm} epoch {epoch}/{CFG.epochs}",leave=True); totals=0.0; updates=0
        order=np.random.default_rng(CFG.seed+epoch).permutation(len(train["inputs"]))
        for offset in range(0,len(order),CFG.batch_trajectories):
            idx=order[offset:offset+CFG.batch_trajectories]; hidden=None; initial=tensor(train_burnin_n,idx)[:,0]
            phases=((tensor(train["burnin"],idx),tensor(train_burnin_n,idx)),(tensor(train["inputs"],idx),tensor(train_states_n,idx)))
            for phase_x,phase_y in phases:
                for start in range(0,phase_x.shape[1],CFG.chunk_steps):
                    chunk=phase_x[:,start:start+CFG.chunk_steps]; target=phase_y[:,start+1:start+1+chunk.shape[1]]; before=detach(hidden)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda",dtype=torch.float16,enabled=DEVICE.type=="cuda"):
                        prediction,_=model(chunk,before,initial_state=initial if before is None else None); loss=torch.mean(torch.square(prediction-target))
                    scaler.scale(loss).backward(); scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(),CFG.gradient_clip); scaler.step(optimizer); scaler.update()
                    with torch.no_grad(): _,hidden=model(chunk,before,initial_state=initial if before is None else None)
                    hidden=detach(hidden); totals+=float(loss.detach()); updates+=1; progress.update()
        sampler=StratifiedWindowSampler(train["events"],CFG.stratified_window_steps,seed=CFG.seed+epoch); sampled=sampler.sample(CFG.stratified_windows_per_epoch)
        for offset in range(0,len(sampled),CFG.batch_trajectories):
            rows=sampled[offset:offset+CFG.batch_trajectories]; hidden=contexts_for_rows(model,rows)
            windows=tensor(np.stack([train["inputs"][int(t),int(s):int(s)+CFG.stratified_window_steps] for t,s in rows]))
            target=tensor(np.stack([train_states_n[int(t),int(s)+1:int(s)+1+CFG.stratified_window_steps] for t,s in rows]))
            model.train(); optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16,enabled=DEVICE.type=="cuda"):
                prediction,_=model(windows,hidden); loss=torch.mean(torch.square(prediction-target))
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(),CFG.gradient_clip); scaler.step(optimizer); scaler.update()
            totals+=float(loss.detach()); updates+=1; progress.update()
        scheduler.step(); prediction=predict_validation(model); observed=metrics(prediction); elapsed=time.perf_counter()-started; completed=epoch-start_epoch+1; eta=elapsed/completed*max(0,CFG.epochs-epoch)
        row={"arm":arm,"epoch":epoch,"train_mse":totals/max(1,updates),"learning_rate":optimizer.param_groups[0]["lr"],**{f"validation_{k}":v for k,v in observed.items()},"elapsed_s":elapsed,"eta_s":eta,"optimizer_updates":updates}
        history.append(row); score=float(observed["event_soma_rmse_mV"]); improved=score<best
        if improved:
            best=score; torch.save({"format_version":1,"experiment_id":"SR-01","arm":arm,"mode":ARMS[arm],"model_state_dict":model.state_dict(),"model_spec":{"class":"StateContextGRU","input_dim":INPUT_DIM,"state_dim":STATE_DIM,"hidden_dim":CFG.hidden_dim,"decoder_dim":CFG.decoder_dim,"mode":ARMS[arm],"parameters":parameter_counts[arm]},"state_mean":STATE_MEAN,"state_std":STATE_STD,"state_names":STATE_NAMES,"input_names":input_names,"dataset_sha256":DATASET_HASH,"config":asdict(CFG),"epoch":epoch,"best_validation":observed,"teacher_state_after_initialization":False},best_path)
            # Export traces at float16 only after all float32 metrics are computed.
            np.savez_compressed(OUTPUT/f"{arm}_validation_prediction.npz",prediction=prediction.astype(np.float16))
        torch.save({"arm":arm,"dataset_sha256":DATASET_HASH,"config":asdict(CFG),"epoch":epoch,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),"history":history,"best":best},last_path)
        progress.set_postfix(event_rmse=f"{observed['event_soma_rmse_mV']:.3f}",recall=f"{observed['spike_recall']:.3f}",eta_min=f"{eta/60:.1f}"); progress.close()
        write_rows(OUTPUT/f"{arm}_history.csv",history)
    best_checkpoint=torch.load(best_path,map_location=DEVICE,weights_only=False); model.load_state_dict(best_checkpoint["model_state_dict"]); summary={"arm":arm,"mode":ARMS[arm],"parameters":parameter_counts[arm],"initial_state_hash":initial_hashes[arm],"dataset_sha256":DATASET_HASH,"completed_epochs":CFG.epochs,"best_epoch":best_checkpoint["epoch"],"best_validation":best_checkpoint["best_validation"],"history":history}
    summary_path.write_text(json.dumps(summary,indent=2),encoding="utf-8"); return model,history,best_checkpoint["best_validation"]


all_history=[]; best_metrics={}
for index,arm in enumerate(ARMS,1):
    print(f"\n[{index}/{len(ARMS)}] training {arm}",flush=True); _,history,observed=train_arm(arm); all_history.extend(history); best_metrics[arm]=observed
write_rows(OUTPUT/"training_history.csv",all_history)
write_rows(OUTPUT/"validation_comparison.csv",[{"arm":arm,"mode":ARMS[arm],"parameters":parameter_counts[arm],**value} for arm,value in best_metrics.items()])
write_rows(OUTPUT/"checkpoint_manifest.csv",[
    {"arm":arm,"file":f"checkpoints/{arm}.pt","sha256":sha256_file(CHECKPOINTS/f"{arm}.pt"),"included_in_default_zip":False}
    for arm in ARMS
])
np.savez_compressed(OUTPUT/"validation_reference.npz",truth=validation["states"][:,1:].astype(np.float16),spikes=validation["spikes"],events=validation["events"],regimes=validation["regimes"])

control=best_metrics["initial_state_only"]; candidate=best_metrics["predicted_state_feedback"]; anchor=best_metrics["no_state_context"]
event_gain=(control["event_soma_rmse_mV"]-candidate["event_soma_rmse_mV"])/control["event_soma_rmse_mV"]
initial_gain=(anchor["event_soma_rmse_mV"]-control["event_soma_rmse_mV"])/anchor["event_soma_rmse_mV"]
guard_names=("mean_state_nrmse","slow_state_nrmse","synaptic_state_nrmse","subthreshold_soma_rmse_mV")
guard_ratios={name:candidate[name]/control[name] for name in guard_names}; guards_pass=all(value<=1.10 for value in guard_ratios.values())
if candidate["matched_spikes"]>0 and event_gain>=0.20 and guards_pass: branch="promote_to_three_seed_replication"
elif control["matched_spikes"]>0 and initial_gain>=0.20 and event_gain<0.10: branch="initial_observability_only"
elif candidate["matched_spikes"]==0 and event_gain<0.10: branch="close_direct_feedback_branch"
else: branch="tradeoff_or_inconclusive"
decision={"experiment_id":"SR-01","decision":branch,"requires_human_review":True,"event_rmse_improvement_vs_initial_only":event_gain,"initialization_event_rmse_improvement_vs_no_state":initial_gain,"candidate_matched_spikes":candidate["matched_spikes"],"candidate_spike_recall":candidate["spike_recall"],"guardrail_ratios":guard_ratios,"guardrails_pass":guards_pass,"parameters_identical":len(set(parameter_counts.values()))==1,"initial_weights_identical":len(set(initial_hashes.values()))==1,"teacher_state_after_initialization":False,"test_split_opened":False,"dataset_sha256":DATASET_HASH}
(OUTPUT/"decision.json").write_text(json.dumps(decision,indent=2),encoding="utf-8")
(OUTPUT/"provenance.json").write_text(json.dumps({"created_utc":datetime.now(timezone.utc).isoformat(),"repository":str(REPO_ROOT),"dataset":str(DATASET),"device":str(DEVICE),"config":asdict(CFG),"arms":ARMS,"parameter_counts":parameter_counts,"initial_hashes":initial_hashes,**decision},indent=2),encoding="utf-8")

history_frame={arm:[row for row in all_history if row["arm"]==arm] for arm in ARMS}
fig,axes=plt.subplots(2,2,figsize=(14,9)); plot_metrics=("validation_event_soma_rmse_mV","validation_spike_recall","validation_mean_state_nrmse","validation_subthreshold_soma_rmse_mV")
for ax,metric in zip(axes.ravel(),plot_metrics):
    for arm,rows in history_frame.items(): ax.plot([r["epoch"] for r in rows],[r[metric] for r in rows],marker="o",label=arm)
    ax.set_title(metric); ax.set_xlabel("epoch"); ax.grid(alpha=.25)
axes[0,0].legend(); fig.tight_layout(); fig.savefig(FIGURES/"learning_curves.png",dpi=170); plt.close(fig)

truth=validation["states"][:,1:]; spike_positions=np.argwhere(validation["spikes"])
if len(spike_positions):
    trajectory,center=map(int,spike_positions[0]); before=int(round(25/DT_MS)); after=int(round(50/DT_MS)); start=max(0,center-before); stop=min(truth.shape[1],center+after)
    fig,ax=plt.subplots(figsize=(14,5)); axis=np.arange(start,stop)*DT_MS; ax.plot(axis,truth[trajectory,start:stop,SOMA],label="teacher",linewidth=2,color="black")
    for arm in ARMS:
        with np.load(OUTPUT/f"{arm}_validation_prediction.npz") as archive: pred=archive["prediction"]
        ax.plot(axis,pred[trajectory,start:stop,SOMA],label=arm)
    ax.axvline(center*DT_MS,color="gray",linestyle="--",alpha=.5); ax.set(xlabel="time (ms)",ylabel="soma V (mV)",title="First validation teacher spike"); ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(FIGURES/"event_trace.png",dpi=170); plt.close(fig)

(OUTPUT/"README.md").write_text("# SR-01 state-feedback preflight\n\nParameter-identical standard GRUs differing only in physical-state information contract. Teacher state enters at most once at complete-trajectory initialization. Validation only; test closed. Start with `decision.json`, `validation_comparison.csv`, and `figures/`.\n",encoding="utf-8")
ZIP_PATH=OUTPUT.parent/"hay_micro_state_feedback_preflight_14_complete.zip"
include_checkpoints=os.environ.get("HAY_SR01_DOWNLOAD_CHECKPOINTS","0")=="1"
with zipfile.ZipFile(ZIP_PATH,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for path in sorted(OUTPUT.rglob("*")):
        relative=path.relative_to(OUTPUT)
        checkpoint_file=bool(relative.parts and relative.parts[0]=="checkpoints")
        if path.is_file() and not path.name.endswith("_last.pt") and (include_checkpoints or not checkpoint_file):
            archive.write(path,path.relative_to(OUTPUT.parent))
print(json.dumps(decision,indent=2)); print("archive:",ZIP_PATH,f"({ZIP_PATH.stat().st_size/2**20:.1f} MiB)","| checkpoints included:",include_checkpoints)
