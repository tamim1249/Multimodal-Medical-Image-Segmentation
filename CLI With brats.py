# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p=argparse.ArgumentParser("BMT-HARNet++ BraTS2020 Final")
    p.add_argument("--mode",         choices=["train","infer","sanity"], default="train")
    p.add_argument("--dataset_root", default=CFG.DATASET_ROOT)
    p.add_argument("--epochs",       type=int,   default=CFG.EPOCHS)
    p.add_argument("--batch_size",   type=int,   default=CFG.BATCH_SIZE)
    p.add_argument("--patch_size",   nargs=3, type=int, default=list(CFG.PATCH_SIZE))
    p.add_argument("--base_filters", type=int,   default=CFG.BASE_FILTERS)
    p.add_argument("--lr",           type=float, default=CFG.LR)
    p.add_argument("--checkpoint",   default=CFG.CHECKPOINT)
    p.add_argument("--amp",          action="store_true")
    p.add_argument("--no_amp",       action="store_true")
    p.add_argument("--infer_dir",    default=None)
    p.add_argument("--output_dir",   default="inference_output")
    p.add_argument("--threshold",    type=float, default=CFG.THRESHOLD)
    p.add_argument("--tta",          action="store_true")
    p.add_argument("--max_patients", type=int,   default=CFG.MAX_PATIENTS)
    p.add_argument("--num_workers",  type=int,   default=CFG.NUM_WORKERS)
    args, unknown = p.parse_known_args()
    return args


def apply_args(args):
    CFG.DATASET_ROOT = args.dataset_root; CFG.EPOCHS=args.epochs
    CFG.BATCH_SIZE=args.batch_size; CFG.PATCH_SIZE=tuple(args.patch_size)
    CFG.BASE_FILTERS=args.base_filters; CFG.LR=args.lr
    CFG.CHECKPOINT=args.checkpoint; CFG.MAX_PATIENTS=args.max_patients
    CFG.NUM_WORKERS=args.num_workers
    if args.no_amp: CFG.AMP=False
    elif args.amp:  CFG.AMP=True


def main():
    args=parse_args(); apply_args(args); seed_everything(CFG.SEED)
    if   args.mode=="sanity": sanity_check()
    elif args.mode=="train":  train(CFG)
    elif args.mode=="infer":
        if not args.infer_dir: raise ValueError("--infer_dir required")
        infer_patient(args.checkpoint, args.infer_dir, args.output_dir, args.threshold, args.tta)


if __name__=="__main__":
    main()
