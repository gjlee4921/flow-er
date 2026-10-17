"""Train the gating U-Nets on the cache and write predictions (docs/unito-spec.md).

    # UNITO as published, one model per gate (all ALS gates when --gates is left out)
    uv run python scripts/train_gate.py unito --evaluation als --gates "<gate key>" \\
        --train-roles train finetune --val-roles val --grid --predict-roles test
    # shared template-refining U-Net: pretrain on the pool, then fine-tune on ALS
    uv run python scripts/train_gate.py shared --stage pretrain --predict-roles test  # zero-shot
    uv run python scripts/train_gate.py shared --stage finetune --evaluation als \\
        --train-roles train finetune --val-roles val --predict-roles test
    # zero-shot predictions of the pretrained shared model
    uv run python scripts/train_gate.py predict --model-dir <models>/pretrain/shared/shared \\
        --evaluation als --method shared_zero_shot --roles test
"""

import argparse
from pathlib import Path

import torch

from flow_er.ml import cache
from flow_er.ml.unet import dataset as ds
from flow_er.ml.unet import infer, train
from flow_er.ml.unet.model import shared_arch, unito_arch


def common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--evaluation", default="als")
    p.add_argument("--gates", nargs="*", help="gate keys (default: every gate with training files)")
    p.add_argument("--exclude", help="regex on gate keys to leave out")
    p.add_argument("--train-roles", nargs="+", default=["train"])
    p.add_argument("--val-roles", nargs="+", default=["val"])
    p.add_argument("--grid", action="store_true", help="lr {1e-3, 3e-4} x batch {4, 8}")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--max-minutes", type=float, help="time budget per grid point")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--predict-roles", nargs="*", default=[])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common(u := sub.add_parser("unito", help="per-gate U-Net, UNITO as published"))
    u.add_argument("--model", default="unito")
    common(s := sub.add_parser("shared", help="shared template-refining U-Net"))
    s.add_argument("--stage", choices=["pretrain", "finetune"], required=True)
    s.add_argument("--no-unstim", action="store_true", help="1ch variant: [density, template]")
    s.add_argument("--init", type=Path, help="pretrained model dir (default: the pretrain run)")
    s.add_argument("--no-augment", action="store_true")
    s.add_argument("--model", help="default shared[_1ch] / shared_finetuned[_1ch]")
    pr = sub.add_parser("predict", help="write predictions of a saved model")
    pr.add_argument("--model-dir", type=Path, help="shared model dir")
    pr.add_argument("--per-gate-root", type=Path, help="<models>/<evaluation>/unito")
    pr.add_argument("--evaluation", default="als")
    pr.add_argument("--method", required=True)
    pr.add_argument("--roles", nargs="+", default=["test"])
    pr.add_argument("--gates", nargs="*")
    pr.add_argument("--exclude")
    pr.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    if a.cmd == "predict":
        infer.predict_roles(a.evaluation, a.method, a.roles, a.model_dir, a.per_gate_root,
                            a.gates or None, a.exclude)
        return

    run = {"train_roles": tuple(a.train_roles), "val_roles": tuple(a.val_roles),
           "grid": a.grid, "epochs": a.epochs, "patience": a.patience, "threads": a.threads,
           "seed": a.seed, "exclude": a.exclude,
           "max_seconds": a.max_minutes * 60 if a.max_minutes else None}
    if a.cmd == "unito":
        gates = a.gates or ds.gate_keys_of(ds.select(a.evaluation, a.train_roles,
                                                     exclude=a.exclude))
        for gk in gates:
            try:
                train.train(train.Config(evaluation=a.evaluation, model=a.model,
                                         arch=unito_arch(), channels=ds.UNITO_CHANNELS,
                                         gate_keys=[gk], out_name=cache.gate_id(gk), **run))
            except ValueError as e:
                print(f"skipped {gk}: {e}")
        if a.predict_roles:
            infer.predict_roles(a.evaluation, a.model, a.predict_roles,
                                per_gate_root=train.model_dir(a.evaluation, a.model, ""),
                                gate_keys=gates, exclude=a.exclude)
        return

    suffix = "_1ch" if a.no_unstim else ""
    channels = ds.SHARED_1CH_CHANNELS if a.no_unstim else ds.SHARED_CHANNELS
    if a.stage == "pretrain":
        evaluation, model, init = "pretrain", a.model or f"shared{suffix}", None
        run.update(train_roles=("train",), val_roles=("val",), balance_datasets=True)
    else:
        evaluation, model = a.evaluation, a.model or f"shared_finetuned{suffix}"
        init = a.init or train.model_dir("pretrain", f"shared{suffix}", "shared")
    out = train.train(train.Config(
        evaluation=evaluation, model=model, channels=channels,
        arch=shared_arch(len(channels), None if init else "imagenet"),
        gate_keys=a.gates or None, augment=None if a.no_augment else ds.Augment(),
        init=str(init) if init else None, **run))
    if a.predict_roles:
        method = model if a.stage == "finetune" else f"shared_zero_shot{suffix}"
        infer.predict_roles(a.evaluation, method, a.predict_roles, model_dir=out,
                            gate_keys=a.gates or None, exclude=a.exclude)


if __name__ == "__main__":
    main()
