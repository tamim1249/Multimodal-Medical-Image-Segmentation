 Training loop
# =============================================================================
def train_one_epoch(model, loader, optimizer, scaler, device, epoch):
    model.train()
    total, n = 0.0, 0
    dice_sum = torch.zeros(3)
    pbar = tqdm(loader, desc=f"Ep {epoch+1} Train", leave=False)
    for img, lab, bnd in pbar:
        img,lab,bnd = img.to(device), lab.to(device), bnd.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=CFG.AMP):
            seg,bd,ds = model(img)
            loss,comps = combined_loss(seg,bd,ds,lab,bnd)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)
        scaler.step(optimizer); scaler.update()
        total += loss.item()
        with torch.no_grad():
            d,_,_,_,_ = compute_metrics(seg, lab)
            dice_sum  += d.cpu()
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", wt=f"{d[0].item():.3f}", et=f"{d[2].item():.3f}")
    n = max(n,1)
    return total/n, (dice_sum/n).tolist()


@torch.no_grad()
def validate(model, loader, device, compute_hd=False):
    model.eval()
    total, n = 0.0, 0
    sums = {k: torch.zeros(3) for k in ["dice","iou","prec","rec","spec"]}
    hd_sum = np.zeros(3)
    patient_dice = []
    for img, lab, bnd in tqdm(loader, desc="Val", leave=False):
        img,lab,bnd = img.to(device),lab.to(device),bnd.to(device)
        with autocast(enabled=CFG.AMP):
            seg,bd,ds = model(img)
            loss,_ = combined_loss(seg,bd,ds,lab,bnd)
        total += loss.item()
        d,i,p,r,s = compute_metrics(seg, lab)
        sums["dice"]+=d.cpu(); sums["iou"]+=i.cpu()
        sums["prec"]+=p.cpu(); sums["rec"]+=r.cpu(); sums["spec"]+=s.cpu()
        patient_dice.append(d.cpu().tolist())
        if compute_hd:
            hd_sum += np.array(hd95_per_class(seg, lab))
        n += 1
    n = max(n,1)
    out = {k:(v/n).tolist() for k,v in sums.items()}
    out["loss"]         = total/n
    out["hd95"]         = (hd_sum/n).tolist() if compute_hd else [None]*3
    out["patient_dice"] = patient_dice
    return out


def train(cfg=CFG):
    seed_everything(cfg.SEED)
    device = get_device()
    logger.info(f"Device: {device}")

    # Architecture figure — generate before training
    plot_architecture()

    dirs = scan_dataset(cfg.DATASET_ROOT)
    if cfg.MAX_PATIENTS > 0:
        dirs = dirs[:cfg.MAX_PATIENTS]
    tr_dirs, va_dirs = train_test_split(dirs, test_size=cfg.VAL_SPLIT, random_state=cfg.SEED)
    logger.info(f"Train: {len(tr_dirs)} | Val: {len(va_dirs)}")

    train_ds = BraTSDataset(tr_dirs, cfg.PATCH_SIZE, augment=True,  ppv=1)
    val_ds   = BraTSDataset(va_dirs, cfg.PATCH_SIZE, augment=False, ppv=1)
    nw = cfg.NUM_WORKERS
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              num_workers=nw, pin_memory=(device.type=="cuda"),
                              persistent_workers=(nw>0), drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False,
                              num_workers=nw, pin_memory=(device.type=="cuda"),
                              persistent_workers=(nw>0))

    model = BMTHARNetPP(4, cfg.NUM_CLASSES, cfg.BASE_FILTERS, cfg.TRANSFORMER_DEPTH,
                        cfg.NUM_HEADS, cfg.DROPOUT, cfg.USE_DEEP_SUPERVISION).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    scaler    = GradScaler(enabled=cfg.AMP)

    history   = History()
    all_patient_dice = []
    best, bad = -1.0, 0

    for epoch in range(cfg.EPOCHS):
        t0 = time.time()
        tr_loss, tr_dice = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch)
        compute_hd = ((epoch+1) % 10 == 0) or (epoch == cfg.EPOCHS-1)
        va = validate(model, val_loader, device, compute_hd=compute_hd)
        scheduler.step(epoch+1)

        history.update_train(tr_loss, tr_dice)
        history.update_val(va)
        all_patient_dice.extend(va["patient_dice"])

        mean_dice = float(np.mean(va["dice"]))
        logger.info(
            f"Ep {epoch+1:03d}/{cfg.EPOCHS} | "
            f"tr={tr_loss:.4f} val={va['loss']:.4f} | "
            f"Dice WT={va['dice'][0]:.4f} TC={va['dice'][1]:.4f} ET={va['dice'][2]:.4f} | "
            f"IoU  WT={va['iou'][0]:.4f} TC={va['iou'][1]:.4f} ET={va['iou'][2]:.4f} | "
            f"HD95={[f'{v:.1f}' if v else 'N/A' for v in va['hd95']]} | "
            f"{time.time()-t0:.1f}s"
        )

        # Save curves every 10 epochs
        if (epoch+1) % 10 == 0 and len(history.tr_loss) >= 2:
            plot_loss_curves(history)
            plot_dice_curves(history)

        if mean_dice > best:
            best, bad = mean_dice, 0
            torch.save({
                "epoch": epoch+1, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "best_mean_dice": best,
                "cfg": {k:v for k,v in cfg.__dict__.items() if k.isupper()},
            }, cfg.CHECKPOINT)
            logger.info(f"✓ Checkpoint saved | mean Dice={best:.4f}")
        else:
            bad += 1
            if bad >= cfg.EARLY_STOP_PATIENCE:
                logger.info(f"Early stop at epoch {epoch+1}"); break
