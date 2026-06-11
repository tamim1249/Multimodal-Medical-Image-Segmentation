# =============================================================================
# Inference
# =============================================================================
def load_image_only(patient_dir):
    mods, ref = [], None
    for m in CFG.MODALITIES:
        vol,nii = load_nifti(patient_dir, m)
        if ref is None: ref=nii
        mods.append(zscore_nonzero(vol))
    return np.stack(mods,0).astype(np.float32), ref


@torch.no_grad()
def sliding_window_infer(model, image, patch_size, device, overlap=0.5, tta=False):
    model.eval()
    c,h0,w0,d0 = image.shape; ph,pw,pd = patch_size
    pad_h,pad_w,pad_d = max(ph-h0,0),max(pw-w0,0),max(pd-d0,0)
    if pad_h or pad_w or pad_d:
        image=np.pad(image,[(0,0),(0,pad_h),(0,pad_w),(0,pad_d)],mode="constant")
    _,h,w,d = image.shape
    sh,sw,sd = max(1,int(ph*(1-overlap))),max(1,int(pw*(1-overlap))),max(1,int(pd*(1-overlap)))
    hs=sorted(set(list(range(0,h-ph+1,sh))+[h-ph]))
    ws=sorted(set(list(range(0,w-pw+1,sw))+[w-pw]))
    ds=sorted(set(list(range(0,d-pd+1,sd))+[d-pd]))
    prob=np.zeros((3,h,w,d),dtype=np.float32); count=np.zeros((1,h,w,d),dtype=np.float32)
    for z in tqdm(hs,desc="SW",leave=False):
        for y in ws:
            for x in ds:
                patch=image[:,z:z+ph,y:y+pw,x:x+pd]
                pt=torch.from_numpy(patch[None]).to(device)
                with autocast(enabled=CFG.AMP):
                    pp=predict_tta(model,pt,tta)
                prob[:,z:z+ph,y:y+pw,x:x+pd]+=pp[0].cpu().numpy()
                count[:,z:z+ph,y:y+pw,x:x+pd]+=1
    return (prob/np.maximum(count,1))[:,:h0,:w0,:d0]


def infer_patient(checkpoint, patient_dir, output_dir="inference_output", threshold=0.5, tta=False):
    device=get_device()
    ckpt=torch.load(checkpoint, map_location=device)
    model=BMTHARNetPP(4,CFG.NUM_CLASSES,CFG.BASE_FILTERS,CFG.TRANSFORMER_DEPTH,CFG.NUM_HEADS,CFG.DROPOUT,False).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    logger.info(f"Checkpoint: epoch={ckpt.get('epoch')} dice={ckpt.get('best_mean_dice'):.4f}")
    image,ref = load_image_only(patient_dir)
    prob = sliding_window_infer(model,image,CFG.PATCH_SIZE,device,CFG.SW_OVERLAP,tta)
    seg  = remove_small_cc((prob>threshold).astype(np.float32))
    os.makedirs(output_dir, exist_ok=True)
    for i,name in enumerate(["WT","TC","ET"]):
        nib.save(nib.Nifti1Image(seg[i].astype(np.uint8),ref.affine,ref.header),
                 os.path.join(output_dir,f"pred_{name}.nii.gz"))
    brats=np.zeros(seg.shape[1:],dtype=np.uint8)
    brats[seg[0]>0.5]=2; brats[seg[1]>0.5]=1; brats[seg[2]>0.5]=4
    nib.save(nib.Nifti1Image(brats,ref.affine,ref.header),
             os.path.join(output_dir,"pred_brats_labelmap.nii.gz"))
    logger.info(f"Saved to: {output_dir}")
    return seg, prob
