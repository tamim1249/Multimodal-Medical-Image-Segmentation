class ConvBNAct(nn.Module):
    def __init__(self, a,b,k=3,s=1,p=1):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(a,b,k,s,p,bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True))
    def forward(self,x): return self.net(x)
class ResBlock(nn.Module):
    def __init__(self,a,b):
        super().__init__(); self.c1=ConvBNAct(a,b); self.c2=nn.Sequential(nn.Conv2d(b,b,3,1,1,bias=False), nn.BatchNorm2d(b)); self.proj=nn.Identity() if a==b else nn.Conv2d(a,b,1); self.act=nn.ReLU(inplace=True)
    def forward(self,x): return self.act(self.c2(self.c1(x))+self.proj(x))
class Transformer2D(nn.Module):
    def __init__(self, dim, heads=4, mlp=4, drop=0.1):
        super().__init__(); self.n1=nn.LayerNorm(dim); self.attn=nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True); self.n2=nn.LayerNorm(dim); self.ff=nn.Sequential(nn.Linear(dim,dim*mlp),nn.GELU(),nn.Dropout(drop),nn.Linear(dim*mlp,dim))
    def forward(self,x):
        b,c,h,w=x.shape; t=x.flatten(2).transpose(1,2); z=self.n1(t); a,_=self.attn(z,z,z,need_weights=False); t=t+a; t=t+self.ff(self.n2(t)); return t.transpose(1,2).reshape(b,c,h,w)
class ChannelAttention(nn.Module):
    def __init__(self,c,r=8):
        super().__init__(); m=max(c//r,4); self.net=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(c,m,1),nn.ReLU(True),nn.Conv2d(m,c,1),nn.Sigmoid())
    def forward(self,x): return x*self.net(x)
class SpatialAttention(nn.Module):
    def __init__(self): super().__init__(); self.c=nn.Sequential(nn.Conv2d(2,1,7,padding=3,bias=False),nn.Sigmoid())
    def forward(self,x): return x*self.c(torch.cat([x.mean(1,keepdim=True),x.max(1,keepdim=True).values],1))
class BoundaryHARB(nn.Module):
    def __init__(self,a,b):
        super().__init__(); self.red=ConvBNAct(a,b,1,1,0); self.ca=ChannelAttention(b); self.sa=SpatialAttention(); self.gate=nn.Sequential(ConvBNAct(b,max(b//2,4)),nn.Conv2d(max(b//2,4),1,1),nn.Sigmoid()); self.ref=ResBlock(b,b)
    def forward(self,x):
        x=self.red(x); x=self.ca(x); x=self.sa(x); g=self.gate(x); return self.ref(x*(1+g)), g
class MultiScaleAgg(nn.Module):
    def __init__(self, chs, out):
        super().__init__(); self.proj=nn.ModuleList([ConvBNAct(c,out,1,1,0) for c in chs]); self.fuse=nn.Sequential(ConvBNAct(out*len(chs),out),ResBlock(out,out))
    def forward(self, feats, size):
        xs=[]
        for f,p in zip(feats,self.proj):
            z=p(f)
            if z.shape[-2:]!=size: z=F.interpolate(z,size=size,mode='bilinear',align_corners=False)
            xs.append(z)
        return self.fuse(torch.cat(xs,1))

class BMTHARNet2D(nn.Module):
    def __init__(self, base=24, depth=1, heads=4, deep=True):
        super().__init__(); self.deep=deep; F0=base
        self.stem=ResBlock(3,F0); self.e1=ResBlock(F0,F0); self.d1=ConvBNAct(F0,F0*2,3,2,1); self.e2=ResBlock(F0*2,F0*2)
        self.d2=ConvBNAct(F0*2,F0*4,3,2,1); self.e3=ResBlock(F0*4,F0*4); self.d3=ConvBNAct(F0*4,F0*8,3,2,1); self.e4=ResBlock(F0*8,F0*8)
        self.d4=ConvBNAct(F0*8,F0*16,3,2,1); self.bot=nn.Sequential(ResBlock(F0*16,F0*16), *[Transformer2D(F0*16,heads=heads) for _ in range(depth)])
        self.up4=nn.ConvTranspose2d(F0*16,F0*8,2,2); self.h4=BoundaryHARB(F0*16,F0*8)
        self.up3=nn.ConvTranspose2d(F0*8,F0*4,2,2); self.h3=BoundaryHARB(F0*8,F0*4)
        self.up2=nn.ConvTranspose2d(F0*4,F0*2,2,2); self.h2=BoundaryHARB(F0*4,F0*2)
        self.up1=nn.ConvTranspose2d(F0*2,F0,2,2); self.h1=BoundaryHARB(F0*2,F0)
        self.agg=MultiScaleAgg([F0,F0*2,F0*4,F0*8],F0); self.seg=nn.Conv2d(F0,1,1); self.bnd=nn.Conv2d(F0,1,1)
        self.ds2=nn.Conv2d(F0*2,1,1); self.ds3=nn.Conv2d(F0*4,1,1); self.ds4=nn.Conv2d(F0*8,1,1)
    def forward(self,x):
        s1=self.e1(self.stem(x)); s2=self.e2(self.d1(s1)); s3=self.e3(self.d2(s2)); s4=self.e4(self.d3(s3)); z=self.bot(self.d4(s4))
        x4,_=self.h4(torch.cat([self.up4(z),s4],1)); x3,_=self.h3(torch.cat([self.up3(x4),s3],1)); x2,_=self.h2(torch.cat([self.up2(x3),s2],1)); x1,_=self.h1(torch.cat([self.up1(x2),s1],1))
        a=self.agg([x1,x2,x3,x4], x1.shape[-2:]); ds=[self.ds2(x2),self.ds3(x3),self.ds4(x4)] if self.training and self.deep else []
        return self.seg(a), self.bnd(a), ds


def dice_loss(logits,y,eps=1e-6):
    p=torch.sigmoid(logits); inter=(p*y).sum((2,3)); den=p.sum((2,3))+y.sum((2,3)); return 1-((2*inter+eps)/(den+eps)).mean()
def focal_tversky(logits,y,a=0.7,b=0.3,g=0.75,eps=1e-6):
    p=torch.sigmoid(logits); tp=(p*y).sum((2,3)); fp=(p*(1-y)).sum((2,3)); fn=((1-p)*y).sum((2,3)); tv=(tp+eps)/(tp+a*fp+b*fn+eps); return (1-tv).pow(g).mean()
def deep_loss(ds,y):
    if not ds: return torch.tensor(0., device=y.device)
    w=[0.5,0.3,0.2]; loss=0
    for d,ww in zip(ds,w): loss += ww*dice_loss(d, F.interpolate(y,size=d.shape[-2:],mode='nearest'))
    return loss
def total_loss(seg,bnd,ds,y,b):
    return dice_loss(seg,y)+0.7*F.binary_cross_entropy_with_logits(seg,y)+0.6*focal_tversky(seg,y)+0.3*F.binary_cross_entropy_with_logits(bnd,b)+0.25*deep_loss(ds,y)

@torch.no_grad()
def metrics(logits,y,thr=0.5):
    p=(torch.sigmoid(logits)>thr).float(); y=y.float()
    tp=(p*y).sum((1,2,3)); fp=(p*(1-y)).sum((1,2,3)); fn=((1-p)*y).sum((1,2,3)); tn=((1-p)*(1-y)).sum((1,2,3)); eps=1e-6
    dice=((2*tp+eps)/(2*tp+fp+fn+eps)).mean().item(); iou=((tp+eps)/(tp+fp+fn+eps)).mean().item(); prec=((tp+eps)/(tp+fp+eps)).mean().item(); rec=((tp+eps)/(tp+fn+eps)).mean().item(); spec=((tn+eps)/(tn+fp+eps)).mean().item()
    return dice,iou,prec,rec,spec

def hd95_np(pred,gt):
    pred=pred.astype(bool); gt=gt.astype(bool)
    if pred.sum()==0 and gt.sum()==0: return 0.0
    if pred.sum()==0 or gt.sum()==0: return 256.0
    pb=pred ^ ndi.binary_erosion(pred); gb=gt ^ ndi.binary_erosion(gt)
    if pb.sum()==0 or gb.sum()==0: return 256.0
    dtg=ndi.distance_transform_edt(~gb); dtp=ndi.distance_transform_edt(~pb); return float(np.percentile(np.concatenate([dtg[pb],dtp[gb]]),95))


def run_epoch(model,loader,opt,scaler,device,train=True):
    model.train(train); total=0; msum=np.zeros(5); n=0
    pbar=tqdm(loader, leave=False, desc='Train' if train else 'Val')
    for img,mask,bnd,_ in pbar:
        img,mask,bnd=img.to(device),mask.to(device),bnd.to(device)
        if train: opt.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with autocast(enabled=CFG.AMP):
                seg,bd,ds=model(img); loss=total_loss(seg,bd,ds,mask,bnd)
            if train:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        total += loss.item(); msum += np.array(metrics(seg.detach(),mask)); n += 1
        pbar.set_postfix(loss=f'{loss.item():.4f}', dice=f'{msum[0]/n:.4f}')
    return total/max(n,1), (msum/max(n,1)).tolist()

@torch.no_grad()
def test_model(model, loader, device, fig_dir, out_dir):
    os.makedirs(fig_dir,exist_ok=True); os.makedirs(out_dir,exist_ok=True); model.eval(); rows=[]; hd=[]; shown=0
    for img,mask,bnd,names in tqdm(loader, desc='Test'):
        img=img.to(device); seg,_,_=model(img); prob=torch.sigmoid(seg).cpu().numpy(); pred=(prob>CFG.THRESHOLD).astype(np.uint8); gt=mask.numpy().astype(np.uint8)
        mm=metrics(seg.cpu(),mask)
        for i,name in enumerate(names):
            h=hd95_np(pred[i,0], gt[i,0]); hd.append(h); rows.append([name,*mm,h])
            if shown<12:
                # de-normalize image roughly
                im=img[i].cpu().numpy().transpose(1,2,0); im=im*np.array([0.229,0.224,0.225])+np.array([0.485,0.456,0.406]); im=np.clip(im,0,1)
                fig,axs=plt.subplots(1,4,figsize=(14,4)); axs[0].imshow(im); axs[0].set_title('Image'); axs[1].imshow(gt[i,0],cmap='gray'); axs[1].set_title('GT'); axs[2].imshow(pred[i,0],cmap='gray'); axs[2].set_title('Pred'); axs[3].imshow(im); axs[3].imshow(pred[i,0],alpha=.45,cmap='Reds'); axs[3].set_title('Overlay')
                for ax in axs: ax.axis('off')
                plt.tight_layout(); plt.savefig(os.path.join(fig_dir,f'sample_{shown+1}_{name}.png'),dpi=160); plt.close(); shown+=1
    arr=np.array([[r[1],r[2],r[3],r[4],r[5],r[6]] for r in rows],float)
    labels=['Dice','IoU','Precision','Recall','Specificity','HD95']
    for j,l in enumerate(labels):
        plt.figure(figsize=(5,4)); plt.boxplot(arr[:,j]); plt.title(l); plt.ylabel(l); plt.tight_layout(); plt.savefig(os.path.join(fig_dir,f'boxplot_{l}.png'),dpi=160); plt.close()
    plt.figure(figsize=(7,5)); plt.bar(labels, arr.mean(0)); plt.title('CVC-ClinicDB Test Metrics'); plt.xticks(rotation=25); plt.tight_layout(); plt.savefig(os.path.join(fig_dir,'metrics_bar.png'),dpi=160); plt.close()
    import csv
    with open(os.path.join(out_dir,'test_metrics.csv'),'w',newline='') as f:
        wr=csv.writer(f); wr.writerow(['name','dice','iou','precision','recall','specificity','hd95']); wr.writerows(rows)
    print('Mean metrics:', dict(zip(labels, arr.mean(0).round(4))))
    print('Figures saved:', fig_dir)


def train(args):
    seed_all(CFG.SEED); os.makedirs(CFG.FIG_DIR,exist_ok=True); os.makedirs(CFG.OUT_DIR,exist_ok=True)
    pairs=list_pairs(args.image_dir,args.mask_dir)
    if args.max_train>0: pairs=pairs[:args.max_train]
    tr,tmp=train_test_split(pairs,test_size=0.2,random_state=CFG.SEED); va,te=train_test_split(tmp,test_size=0.5,random_state=CFG.SEED)
    print(f'Total={len(pairs)} Train={len(tr)} Val={len(va)} Test={len(te)}')
    train_loader=DataLoader(CVCDataset(tr,args.img_size,True),batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=True)
    val_loader=DataLoader(CVCDataset(va,args.img_size,False),batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=True)
    test_loader=DataLoader(CVCDataset(te,args.img_size,False),batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('Device:',device)
    model=BMTHARNet2D(args.base,args.depth,args.heads,True).to(device); print('Params(M):',sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=CFG.WEIGHT_DECAY); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs,eta_min=1e-6); scaler=GradScaler(enabled=CFG.AMP)
    hist=[]; best=-1
    for ep in range(args.epochs):
        t=time.time(); tl,tm=run_epoch(model,train_loader,opt,scaler,device,True); vl,vm=run_epoch(model,val_loader,opt,scaler,device,False); sch.step(); hist.append([ep+1,tl,vl,*tm,*vm])
        print(f'Epoch {ep+1:03d}/{args.epochs} train_loss={tl:.4f} val_loss={vl:.4f} val_dice={vm[0]:.4f} val_iou={vm[1]:.4f} time={time.time()-t:.1f}s')
        if vm[0]>best:
            best=vm[0]; torch.save({'model':model.state_dict(),'args':vars(args),'best_dice':best},args.checkpoint); print('Saved best:',args.checkpoint,best)
    # curves
    h=np.array(hist,float); names=['train_loss','val_loss','train_dice','train_iou','train_precision','train_recall','train_specificity','val_dice','val_iou','val_precision','val_recall','val_specificity']
    for idx,nm in enumerate(names, start=1):
        plt.figure(figsize=(6,4)); plt.plot(h[:,0],h[:,idx]); plt.xlabel('Epoch'); plt.ylabel(nm); plt.title(nm); plt.tight_layout(); plt.savefig(os.path.join(CFG.FIG_DIR,f'curve_{nm}.png'),dpi=160); plt.close()
    ckpt=torch.load(args.checkpoint,map_location=device); model.load_state_dict(ckpt['model']); test_model(model,test_loader,device,CFG.FIG_DIR,CFG.OUT_DIR)


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['train','test'],default='train')
    p.add_argument('--image_dir',default=CFG.IMAGE_DIR); p.add_argument('--mask_dir',default=CFG.MASK_DIR)
    p.add_argument('--epochs',type=int,default=CFG.EPOCHS); p.add_argument('--batch_size',type=int,default=CFG.BATCH_SIZE); p.add_argument('--img_size',type=int,default=CFG.IMG_SIZE)
    p.add_argument('--base',type=int,default=CFG.BASE); p.add_argument('--depth',type=int,default=CFG.DEPTH); p.add_argument('--heads',type=int,default=CFG.HEADS); p.add_argument('--lr',type=float,default=CFG.LR)
    p.add_argument('--workers',type=int,default=CFG.NUM_WORKERS); p.add_argument('--checkpoint',default=CFG.CHECKPOINT); p.add_argument('--max_train',type=int,default=CFG.MAX_TRAIN)
    args,_=p.parse_known_args(); return args
if __name__=='__main__':
    args=parse_args(); train(args)
