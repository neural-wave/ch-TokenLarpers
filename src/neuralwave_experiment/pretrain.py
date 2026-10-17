"""Offline CLI for unlabelled-only streaming PatchTST pretraining."""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
from typing import Any
import numpy as np
import torch
from .encoder import EncoderConfig
from .encoder_factory import build_encoder
from .streaming import EpochSource, WindowConfig, WindowStreamConfig, assert_unlabelled_only
from .pretraining import PretrainConfig, pretrain_encoder_stream
from .checkpoint import save_pretrain_checkpoint, load_pretrain_checkpoint, export_encoder_artifact

DEFAULTS={"encoder_backend":"official_patchtst","context_length":672,"patch_length":16,"d_model":64,"n_heads":4,"n_layers":2,"dropout":.2,"stride":96,"min_observed_fraction":.75,"max_windows_per_customer":128,"windows_per_epoch":4096,"season_quota":"balanced","min_windows_per_available_season":1,"epochs":5,"batch_size":64,"learning_rate":.001,"weight_decay":0.,"mask_ratio":.3,"grad_clip_norm":5.,"amp":True,"amp_dtype":"float16","seed":17}

def _ids(path: Path)->set[str]:
    if path.suffix.lower() in {".parquet",".pq"}:
        import pyarrow.parquet as pq
        names=set(pq.ParquetFile(path).schema.names); col="customer_id" if "customer_id" in names else None
        if col is None: raise ValueError("exclude-labels parquet has no customer_id")
        found=set()
        for b in pq.ParquetFile(path).iter_batches(columns=[col],batch_size=8192): found.update(str(x) for x in b.column(0).to_pylist() if x is not None)
        return found
    import pandas as pd
    frame=pd.read_csv(path,sep=None,engine="python",dtype={"customer_id":str})
    if "customer_id" not in frame: raise ValueError("exclude-labels file has no customer_id")
    return set(frame["customer_id"].dropna().astype(str))

def _source(path: Path, payload: dict[str,Any], labels: set[str], split: str)->EpochSource:
    selected=[]
    for item in payload["sources"]:
        if not isinstance(item,dict) or item.get("role")!="unlabelled_measurements":continue
        item_split=item.get("split",item.get("partition",item.get("purpose")))
        if item_split==split:selected.append(item)
    if not selected: raise ValueError(f"manifest has no unlabelled_measurements source split={split!r}")
    paths=[]
    for item in selected:
        raw=item.get("paths",item.get("path")); raw=[raw] if isinstance(raw,str) else raw
        if not isinstance(raw,list) or not raw:raise ValueError("manifest source path is missing")
        for value in raw:
            p=Path(value); p=(path.parent/p if not p.is_absolute() else p).resolve()
            if not p.is_file():raise FileNotFoundError(f"manifest source does not exist: {p}")
            paths.append(p)
    return EpochSource(paths,role="unlabelled_measurements",labelled_ids=labels,manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def _canonical_ids(ids):return "\n".join(sorted(map(str,ids))).encode()
def _json(path,value):path.write_text(json.dumps(value,sort_keys=True,indent=2,default=str)+"\n")

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest",required=True,type=Path);ap.add_argument("--exclude-labels",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path);ap.add_argument("--config",required=True,type=Path)
    ap.add_argument("--device",choices=["cpu","cuda"],required=True);ap.add_argument("--resume",type=Path);ap.add_argument("--max-steps",type=int)
    args=ap.parse_args(argv)
    if args.max_steps is not None and args.max_steps <= 0:raise ValueError("--max-steps must be positive")
    if args.out.exists() and any(args.out.iterdir()) and args.resume is None:raise FileExistsError("output directory must be new and empty")
    args.out.mkdir(parents=True,exist_ok=True)
    config=dict(DEFAULTS);config.update(json.loads(args.config.read_text()))
    labels=_ids(args.exclude_labels); payload=json.loads(args.manifest.read_text())
    if not isinstance(payload,dict) or not isinstance(payload.get("sources"),list):raise ValueError("manifest must contain sources")
    train_source=_source(args.manifest,payload,labels,"train"); val_source=_source(args.manifest,payload,labels,"validation")
    assert_unlabelled_only(train_source,labels);assert_unlabelled_only(val_source,labels)
    train_index_cfg=WindowStreamConfig(WindowConfig(int(config["context_length"]),int(config["stride"]),float(config["min_observed_fraction"])),int(config["max_windows_per_customer"]),int(config["windows_per_epoch"]),int(config["batch_size"]),str(config["season_quota"]),int(config["min_windows_per_available_season"]),int(config["seed"]))
    train_index=train_source.index_windows(train_index_cfg);val_index=val_source.index_windows(train_index_cfg)
    if set(train_index._customer_ids)&set(val_index._customer_ids):raise ValueError("training and validation unlabelled customers overlap")
    enc_cfg=EncoderConfig(context_length=int(config["context_length"]),patch_length=int(config["patch_length"]),d_model=int(config["d_model"]),n_heads=int(config["n_heads"]),n_layers=int(config["n_layers"]),dropout=float(config["dropout"]))
    train_cfg=PretrainConfig(epochs=int(config["epochs"]),batch_size=int(config["batch_size"]),learning_rate=float(config["learning_rate"]),weight_decay=float(config["weight_decay"]),mask_ratio=float(config["mask_ratio"]),grad_clip_norm=float(config["grad_clip_norm"]),amp=bool(config["amp"]),amp_dtype=str(config["amp_dtype"]),seed=int(config["seed"]))
    random.seed(train_cfg.seed);np.random.seed(train_cfg.seed);torch.manual_seed(train_cfg.seed);model=build_encoder(enc_cfg,backend="official_patchtst")
    resume=None
    if args.resume:
        resume=load_pretrain_checkpoint(args.resume,model=model,expected_metadata={"config":config,"source_manifest_sha256":train_source.manifest_sha256,"source_bytes_sha256":train_source.source_bytes_sha256,"source_order_sha256":train_index.source_order_sha256,"index_sha256":train_index.index_sha256,"validation_source_bytes_sha256":val_source.source_bytes_sha256,"validation_source_order_sha256":val_index.source_order_sha256,"validation_index_sha256":val_index.index_sha256,"device":args.device,"amp":bool(train_cfg.amp and args.device=="cuda"),"amp_dtype":train_cfg.amp_dtype})
    def sink(payload):
        state=payload.get("state",payload); step=int(state.get("progress",{}).get("global_step",0)); meta={"config":config,"source_manifest_sha256":train_source.manifest_sha256,"source_bytes_sha256":train_source.source_bytes_sha256,"source_order_sha256":train_index.source_order_sha256,"index_sha256":train_index.index_sha256,"validation_source_bytes_sha256":val_source.source_bytes_sha256,"validation_source_order_sha256":val_index.source_order_sha256,"validation_index_sha256":val_index.index_sha256,"device":args.device,"amp":bool(train_cfg.amp and args.device=="cuda"),"amp_dtype":train_cfg.amp_dtype}
        save_pretrain_checkpoint(args.out/f"checkpoint_step_{step:08d}.pt",state,meta);save_pretrain_checkpoint(args.out/"checkpoint_latest.pt",state,meta)
    result=pretrain_encoder_stream(model,train_source,train_index_cfg,train_cfg,device=args.device,resume=resume,checkpoint=sink,validation_source=val_source,validation_stream=train_index_cfg,max_steps=args.max_steps)
    if args.max_steps is not None and result.progress.get("epoch",0)<train_cfg.epochs:
        _json(args.out/"run.json",{"status":"paused","history":result.history,"validation_history":result.progress.get("validation_history",[]),"progress":result.progress,"source_role":"unlabelled_measurements","labels_used":False})
        return 0
    source_info={"population":"unlabelled_only","labels_used":False,"train_customer_count":train_index.n_customers,"val_customer_count":val_index.n_customers,"train_exclusion_hash":hashlib.sha256(_canonical_ids(labels)).hexdigest(),"val_exclusion_hash":hashlib.sha256(_canonical_ids(labels)).hexdigest(),"manifest_sha256":train_source.manifest_sha256}
    export_encoder_artifact(args.out/"encoder",model,encoder_config={"context_length":enc_cfg.context_length,"patch_length":enc_cfg.patch_length,"d_model":enc_cfg.d_model,"n_heads":enc_cfg.n_heads,"n_layers":enc_cfg.n_layers,"dropout":enc_cfg.dropout},window_config={"context_length":train_index_cfg.window.context_length,"stride":train_index_cfg.window.stride,"min_observed_fraction":train_index_cfg.window.min_observed_fraction,"max_windows_per_customer":train_index_cfg.window.max_windows_per_customer},source=source_info)
    _json(args.out/"run.json",{"history":result.history,"validation_history":result.progress.get("validation_history",[]),"progress":result.progress,"source_role":"unlabelled_measurements","labels_used":False})
    return 0

if __name__=="__main__":raise SystemExit(main())
