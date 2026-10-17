"""Masked reconstruction pretraining for the canonical streaming source."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping
import numpy as np
import torch
from .encoder import NumericPatchEncoder
from .windows import WindowBatch
from .streaming import EpochSource, WindowStreamConfig, WindowSamplerState

@dataclass(frozen=True)
class PretrainConfig:
    encoder_backend: str = "official_patchtst"
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    mask_ratio: float = 0.30
    grad_clip_norm: float = 5.0
    amp: bool = True
    amp_dtype: str = "float16"
    seed: int = 17
    early_stopping_patience: int | None = None
    def __post_init__(self):
        if self.encoder_backend != "official_patchtst": raise ValueError("pretraining production backend must be official_patchtst")
        if self.epochs < 0 or self.batch_size <= 0 or self.learning_rate <= 0 or not (0 < self.mask_ratio < 1): raise ValueError("invalid pretraining settings")
        if self.amp_dtype not in {"float16", "bfloat16", "fp16", "bf16"}: raise ValueError("amp_dtype must be float16 or bfloat16")
        if self.early_stopping_patience is not None and self.early_stopping_patience <= 0: raise ValueError("early_stopping_patience must be positive")
    def as_dict(self):
        return {"encoder_backend": self.encoder_backend, "epochs": self.epochs, "batch_size": self.batch_size,
                "learning_rate": self.learning_rate, "weight_decay": self.weight_decay, "mask_ratio": self.mask_ratio,
                "grad_clip_norm": self.grad_clip_norm, "amp": self.amp, "amp_dtype": self.amp_dtype, "seed": self.seed,
                "early_stopping_patience": self.early_stopping_patience}

@dataclass
class PretrainResult:
    history: list[float]
    sampler_state: WindowSamplerState
    progress: dict[str, Any]
    mask_states: list[np.ndarray] = field(default_factory=list)

def _select_masks(observed, q, p, ratio, rng):
    patch_obs = observed[..., 0].reshape(len(observed), q, p).any(-1)
    selected = np.zeros((len(observed), q), dtype=bool); traces = []
    for row in range(len(observed)):
        eligible = np.flatnonzero(patch_obs[row].detach().cpu().numpy())
        if len(eligible) >= 2:
            count = min(len(eligible)-1, max(1, int(np.ceil(len(eligible)*ratio))))
            selected[row, rng.choice(eligible, size=count, replace=False)] = True
        traces.append(selected[row].copy())
    return torch.as_tensor(selected, dtype=torch.bool, device=observed.device), traces

def _visible_inputs(model, batch, target, device):
    values = torch.as_tensor(batch.values, dtype=torch.float32, device=device)
    observed = torch.as_tensor(batch.observed, dtype=torch.bool, device=device)
    calendar = torch.as_tensor(batch.calendar, dtype=torch.float32, device=device)
    magnitude = torch.as_tensor(batch.magnitude, dtype=torch.float32, device=device)
    if magnitude.ndim != 2 or magnitude.shape[1] < 2 or not bool(torch.isfinite(magnitude[:, :2]).all()) or bool((magnitude[:, 1] <= 0).any()):
        raise ValueError("WindowBatch magnitude contains invalid normalization statistics")
    q, p = model.n_patches, model.patch_length
    physical = values[..., 0] * magnitude[:, 1:2] + magnitude[:, 0:1]
    raw = physical.reshape(-1, q, p)
    obs = observed[..., 0].reshape(-1, q, p)
    target_points = target.unsqueeze(-1)
    visible = obs & ~target_points
    centers=[]; scales=[]
    for row, keep in zip(raw, visible.reshape(len(raw), -1)):
        genuine = row.reshape(-1)[keep]
        if genuine.numel() == 0: centers.append(row.new_tensor(0.)); scales.append(row.new_tensor(1.))
        else:
            med = genuine.median()
            mad = (genuine - med).abs().median()
            std = genuine.std()
            s = mad if mad > 0.01 else (std if std > 0.01 else row.new_tensor(0.05))
            centers.append(med)
            scales.append(s.clamp_min(0.05))
    center=torch.stack(centers)[:,None]; scale=torch.stack(scales)[:,None]
    normalized=((raw.reshape(len(raw),-1)-center)/scale).clamp(-50.0, 50.0).reshape(-1,q,p)
    masked = normalized.masked_fill(~visible, 0.)
    return masked.reshape(-1,q*p,1), (obs & ~target_points).reshape(-1,q*p,1), calendar, normalized, obs

def _batch_loss(model, batch, ratio, rng, device, autocast):
    observed=torch.as_tensor(batch.observed,dtype=torch.bool,device=device)
    target, traces=_select_masks(observed,model.n_patches,model.patch_length,ratio,rng)
    valid_rows=target.any(-1)
    if not bool(valid_rows.any()): return None,traces
    ix=valid_rows.detach().cpu().numpy(); target=target[valid_rows]
    sub=WindowBatch(batch.values[ix],batch.observed[ix],batch.calendar[ix],batch.magnitude[ix],
        [x for x,k in zip(batch.customer_ids,ix) if k],[x for x,k in zip(batch.center_dates,ix) if k],batch.start_epochs[ix])
    masked, masked_obs, calendar, normalized, obs = _visible_inputs(model,sub,target,device)
    patch_obs=obs.any(-1)
    with autocast:
        output,_=model.token_outputs(masked,masked_obs,calendar,padding_mask=~patch_obs)
        prediction=model.reconstruction_head(output).reshape(-1,model.n_patches,model.patch_length)
        valid=target.unsqueeze(-1)&obs
        if not bool(valid.any()): return None,traces
        loss=((prediction-normalized)**2)[valid].mean()
    return loss,traces

def pretrain_encoder(model: NumericPatchEncoder, batch: WindowBatch, *, epochs=5, batch_size=32, learning_rate=.001, mask_ratio=.3, seed=0, device="cpu") -> list[float]:
    """Masked pretraining for existing in-memory development experiments."""
    if epochs<0 or batch_size<=0 or not (0<mask_ratio<1): raise ValueError("invalid pretraining settings")
    if len(batch.values)==0: raise ValueError("cannot pretrain on empty WindowBatch")
    if epochs==0:return []
    model.to(device); model.train(); opt=torch.optim.Adam(model.parameters(),lr=learning_rate); rng=np.random.default_rng(seed); history=[]
    for _ in range(epochs):
        losses=[]
        for order in np.array_split(rng.permutation(len(batch.values)), max(1,int(np.ceil(len(batch.values)/batch_size)))):
            if len(order)==0:continue
            sub=WindowBatch(batch.values[order],batch.observed[order],batch.calendar[order],batch.magnitude[order],[batch.customer_ids[i] for i in order],[batch.center_dates[i] for i in order],batch.start_epochs[order])
            opt.zero_grad(set_to_none=True); loss,_=_batch_loss(model,sub,mask_ratio,rng,device,torch.autocast(device_type=device,enabled=False))
            if loss is None:continue
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();losses.append(float(loss.detach().cpu()))
        if not losses:raise ValueError("no valid reconstruction targets")
        history.append(float(np.mean(losses)))
    return history

def pretrain_encoder_stream(model: torch.nn.Module, source: EpochSource, stream: WindowStreamConfig, config: PretrainConfig, *, device="cpu", resume: Mapping[str,Any]|None=None, checkpoint: Callable[[Mapping[str,Any]],None]|None=None, validation_source: EpochSource|None=None, validation_stream: WindowStreamConfig|None=None, max_steps: int|None=None) -> PretrainResult:
    if device not in {"cpu","cuda"}:raise ValueError("device must be cpu or cuda")
    if device=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA requested but unavailable")
    if getattr(model,"backend",None)!="official_patchtst":raise ValueError("streaming pretraining requires official_patchtst")
    model.to(device);model.train(); amp_enabled=bool(config.amp and device=="cuda"); dtype=torch.float16 if config.amp_dtype in {"float16","fp16"} else torch.bfloat16
    if amp_enabled and dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():raise ValueError("CUDA bfloat16 AMP is unsupported")
    opt=torch.optim.Adam(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    scaler=torch.amp.GradScaler("cuda",enabled=amp_enabled and dtype is torch.float16)
    index=source.index_windows(stream); mask_rng=np.random.default_rng(config.seed)
    sampler=WindowSamplerState(source_order_sha256=index.source_order_sha256,index_sha256=index.index_sha256)
    progress={"epoch":0,"global_step":0,"windows_seen":0,"batches_seen":0,"running_loss":0.,"last_completed_step":0,"validation_history":[],"pending_validation":False,"unique_customers_seen":0,"best_validation_loss":float("inf"),"best_validation_step":0,"best_validation_epoch":0,"patience_counter":0}; history=[]
    customers_seen = set()
    if resume is not None:
        state=resume.get("state",resume); model.load_state_dict(state["model"],strict=True);opt.load_state_dict(state["optimizer"])
        if state.get("scaler") is not None:scaler.load_state_dict(state["scaler"])
        sampler=WindowSamplerState.from_dict(state["sampler"]);progress.update(state.get("progress",{}));history=list(state.get("history",[]))
        customers_seen = set(progress.get("customers_seen_list", []))
        if state.get("rng") is not None:
            from .checkpoint import restore_rng_state
            restore_rng_state(state["rng"],mask_rng)
    best_model_state: dict[str, torch.Tensor] | None = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()} if resume is not None and progress.get("best_validation_loss", float("inf")) < float("inf") else None
    def emit_checkpoint(is_best: bool = False):
        if checkpoint is not None:
            from .checkpoint import capture_rng_state
            checkpoint({"schema":"patchtst-unlabelled-pretrain-v1","is_best":is_best,"state":{"model":model.state_dict(),"optimizer":opt.state_dict(),"scaler":scaler.state_dict() if scaler.is_enabled() else None,"sampler":sampler.as_dict(),"progress":dict(progress),"rng":capture_rng_state(mask_rng),"history":list(history)}})

    def validate_pending():
        if not progress["pending_validation"]:
            return
        if validation_source is not None:
            model.eval(); validation_rng=np.random.default_rng(918273); val_losses=[]
            for val_batch,_ in validation_source.iter_window_batches(validation_stream or stream):
                with torch.no_grad():
                    val_loss,_=_batch_loss(model,val_batch,config.mask_ratio,validation_rng,device,torch.autocast(device_type=device,dtype=dtype,enabled=amp_enabled))
                if val_loss is not None:val_losses.append(float(val_loss.detach().cpu()))
            if not val_losses:raise ValueError("validation source had no valid reconstruction targets")
            val_mean = float(np.mean(val_losses))
            progress["validation_history"].append(val_mean)
            nonlocal best_model_state, stopped
            if val_mean < progress["best_validation_loss"]:
                progress["best_validation_loss"] = val_mean
                progress["best_validation_step"] = progress["global_step"]
                progress["best_validation_epoch"] = progress["epoch"]
                progress["patience_counter"] = 0
                best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                emit_checkpoint(is_best=True)
            else:
                progress["patience_counter"] += 1
                if config.early_stopping_patience is not None and progress["patience_counter"] >= config.early_stopping_patience:
                    progress["status"] = "early_stopped"
                    stopped = True
            model.train()
        progress["pending_validation"] = False
    validate_pending()
    stopped=False
    for epoch in range(int(progress.get("epoch",0)),config.epochs):
        losses=[]
        for batch,next_sampler in source.iter_window_batches(stream,state=sampler):
            opt.zero_grad(set_to_none=True);loss,traces=_batch_loss(model,batch,config.mask_ratio,mask_rng,device,torch.autocast(device_type=device,dtype=dtype,enabled=amp_enabled));sampler=next_sampler
            customers_seen.update(batch.customer_ids)
            progress["unique_customers_seen"] = len(customers_seen)
            progress["customers_seen_list"] = sorted(customers_seen)
            progress["epoch"]=sampler.epoch;progress["batches_seen"]+=1;progress["windows_seen"]+=len(batch.values)
            progress["pending_validation"] = sampler.epoch > epoch
            if loss is not None:
                if scaler.is_enabled():scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(opt.param_groups[0]["params"],config.grad_clip_norm);scaler.step(opt);scaler.update()
                else:loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),config.grad_clip_norm);opt.step()
                value=float(loss.detach().cpu());losses.append(value);history.append(value);progress["global_step"]+=1;progress["last_completed_step"]=progress["global_step"];progress["running_loss"]=value
                emit_checkpoint()
                if max_steps is not None and progress["global_step"] >= max_steps:
                    stopped=True;break
        if not losses and not stopped:raise ValueError("epoch had no valid reconstruction targets")
        validate_pending()
        emit_checkpoint()
        if stopped:break
        progress["epoch"]=epoch+1
    emit_checkpoint()
    model.eval();return PretrainResult(history,sampler,progress)
