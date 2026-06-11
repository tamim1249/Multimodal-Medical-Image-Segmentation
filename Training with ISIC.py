--------------- Model ----------------
class ConvBlock(nn.Module):
    def __init__(self, a,b):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(a,b,3,1,1,bias=False),nn.BatchNorm2d(b),nn.ReLU(inplace=True),nn.Conv2d(b,b,3,1,1,bias=False),nn.BatchNorm2d(b),nn.ReLU(inplace=True))
    def forward(self,x): return self.net(x)
class MiniTransformer(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__(); self.n1=nn.LayerNorm(c); self.att=nn.MultiheadAttention(c, heads, batch_first=True); self.n2=nn.LayerNorm(c); self.ff=nn.Sequential(nn.Linear(c,c*2),nn.GELU(),nn.Linear(c*2,c))
    def forward(self,x):
        b,c,h,w=x.shape; t=x.flatten(2).transpose(1,2); z=self.n1(t); a,_=self.att(z,z,z,need_weights=False); t=t+a; t=t+self.ff(self.n2(t)); return t.transpose(1,2).reshape(b,c,h,w)
class BGBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__(); self.conv=ConvBlock(in_c,out_c); self.gate=nn.Sequential(nn.Conv2d(out_c,out_c//2,3,1,1),nn.ReLU(inplace=True),nn.Conv2d(out_c//2,1,1),nn.Sigmoid())
    def forward(self,x):
        f=self.conv(x); g=self.gate(f); return f*(1+g), g
class BMT2D(nn.Module):
    def __init__(self, base=16):
        super().__init__(); b=base
        self.e1=ConvBlock(3,b); self.e2=ConvBlock(b,b*2); self.e3=ConvBlock(b*2,b*4); self.e4=ConvBlock(b*4,b*8)
        self.pool=nn.MaxPool2d(2); self.bot=nn.Sequential(ConvBlock(b*8,b*16), MiniTransformer(b*16,4))
        self.u4=nn.ConvTranspose2d(b*16,b*8,2,2); self.d4=BGBlock(b*16,b*8)
        self.u3=nn.ConvTranspose2d(b*8,b*4,2,2); self.d3=BGBlock(b*8,b*4)
        self.u2=nn.ConvTranspose2d(b*4,b*2,2,2); self.d2=BGBlock(b*4,b*2)
        self.u1=nn.ConvTranspose2d(b*2,b,2,2); self.d1=BGBlock(b*2,b)
        self.agg=ConvBlock(b+b*2+b*4+b*8,b)
        self.out=nn.Conv2d(b,1,1); self.bnd=nn.Conv2d(b,1,1)
        self.ds2=nn.Conv2d(b*2,1,1); self.ds3=nn.Conv2d(b*4,1,1)
    def forward(self,x):
        s1=self.e1(x); s2=self.e2(self.pool(s1)); s3=self.e3(self.pool(s2)); s4=self.e4(self.pool(s3)); z=self.bot(self.pool(s4))
        x4,_=self.d4(torch.cat([self.u4(z),s4],1)); x3,_=self.d3(torch.cat([self.u3(x4),s3],1)); x2,_=self.d2(torch.cat([self.u2(x3),s2],1)); x1,_=self.d1(torch.cat([self.u1(x2),s1],1))
        size=x1.shape[-2:]
        agg=torch.cat([x1,F.interpolate(x2,size),F.interpolate(x3,size),F.interpolate(x4,size)],1)
        agg=self.agg(agg)
        return self.out(agg), self.bnd(agg), [self.ds2(x2), self.ds3(x3)]

# ---------------- Loss/metric ----------------
def boundary_from_mask(mask):
    mx=F.max_pool2d(mask,3,1,1); mn=-F.max_pool2d(-mask,3,1,1); return (mx-mn).clamp(0,1)
def dice_loss(logits, y, eps=1e-6):
    p=torch.sigmoid(logits); inter=(p*y).sum((2,3)); den=p.sum((2,3))+y.sum((2,3)); return 1-((2*inter+eps)/(den+eps)).mean()
def focal_tversky(logits,y,a=.7,b=.3,g=.75,eps=1e-6):
    p=torch.sigmoid(logits); tp=(p*y).sum((2,3)); fp=(p*(1-y)).sum((2,3)); fn=((1-p)*y).sum((2,3)); tv=(tp+eps)/(tp+a*fp+b*fn+eps); return ((1-tv)**g).mean()
def loss_fn(logits,bnd,ds,y):
    bg=boundary_from_mask(y)
    loss=dice_loss(logits,y)+0.7*F.binary_cross_entropy_with_logits(logits,y)+0.5*focal_tversky(logits,y)+0.25*F.binary_cross_entropy_with_logits(bnd,bg)
    for d in ds:
        yy=F.interpolate(y,d.shape[-2:],mode='nearest'); loss += 0.15*dice_loss(d,yy)
    return loss
@torch.no_grad()
def metrics(logits,y):
    p=(torch.sigmoid(logits)>0.5).float(); y=y.float()
    tp=(p*y).sum((1,2,3)); fp=(p*(1-y)).sum((1,2,3)); fn=((1-p)*y).sum((1,2,3)); tn=((1-p)*(1-y)).sum((1,2,3)); eps=1e-6
    dice=((2*tp+eps)/(2*tp+fp+fn+eps)).mean().item(); iou=((tp+eps)/(tp+fp+fn+eps)).mean().item(); prec=((tp+eps)/(tp+fp+eps)).mean().item(); rec=((tp+eps)/(tp+fn+eps)).mean().item(); spec=((tn+eps)/(tn+fp+eps)).mean().item()
    return dice,iou,prec,rec,spec

def hd95_np(pred, gt):
    pred=pred.astype(bool); gt=gt.astype(bool)
    if pred.sum()==0 and gt.sum()==0: return 0.0
    if pred.sum()==0 or gt.sum()==0: return 999.0
    st=ndi.generate_binary_structure(2,1); pb=pred ^ ndi.binary_erosion(pred,st); gb=gt ^ ndi.binary_erosion(gt,st)
    if pb.sum()==0 or gb.sum()==0: return 999.0
    dtg=ndi.distance_transform_edt(~gb); dtp=ndi.distance_transform_edt(~pb)
    return float(np.percentile(np.concatenate([dtg[pb],dtp[gb]]),95))

# ---------------- Train/test/figures ----------------
def run_epoch(model, loader, opt=None):
    train=opt is not None; model.train(train); total=0; mets=[]
    for x,y in tqdm(loader, leave=False):
        x=x.to(CFG.DEVICE).float(); y=y.to(CFG.DEVICE).float()
        if train: opt.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            o,b,ds=model(x); loss=loss_fn(o,b,ds,y)
            if train: loss.backward(); opt.step()
        total+=loss.item(); mets.append(metrics(o.detach(),y))
    return total/max(1,len(loader)), np.mean(mets,0).tolist()

def evaluate_save(model, loader, out_csv, max_hd=200):
    model.eval(); rows=[]; allm=[]; idx=0
    with torch.no_grad():
        for x,y in tqdm(loader, desc='Test'):
            x=x.to(CFG.DEVICE).float(); y=y.to(CFG.DEVICE).float(); o,_,_=model(x); m=metrics(o,y); allm.append(m)
            p=(torch.sigmoid(o)>0.5).cpu().numpy(); yy=y.cpu().numpy()
            for b in range(p.shape[0]):
                hd=hd95_np(p[b,0], yy[b,0]) if idx < max_hd else None
                rows.append({'idx':idx,'dice':m[0],'iou':m[1],'precision':m[2],'recall':m[3],'specificity':m[4],'hd95':hd}); idx+=1
    os.makedirs(os.path.dirname(out_csv),exist_ok=True)
    with open(out_csv,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    return np.mean(allm,0).tolist()

def make_figures(model, test_pairs, img_size, n=8, show=False):
    os.makedirs(CFG.FIG_DIR,exist_ok=True); model.eval()
    # training history figure created elsewhere
    chosen=test_pairs[:n]
    fig=plt.figure(figsize=(12, n*3))
    for i,(ip,mp) in enumerate(chosen):
        img=cv2.cvtColor(cv2.imread(ip),cv2.COLOR_BGR2RGB); mask=cv2.imread(mp,0)
        imgr=cv2.resize(img,(img_size,img_size)); mr=(cv2.resize(mask,(img_size,img_size),interpolation=cv2.INTER_NEAREST)>127).astype(np.float32)
        x=imgr.astype(np.float32)/255.; x=(x-np.array([.485,.456,.406]))/np.array([.229,.224,.225]); x=torch.tensor(x.transpose(2,0,1)[None]).float().to(CFG.DEVICE)
        with torch.no_grad(): pred=torch.sigmoid(model(x)[0])[0,0].cpu().numpy()
        pb=(pred>0.5).astype(np.float32); overlay=imgr.copy(); overlay[pb>0]=[255,0,0]
        for j,arr,title in [(0,imgr,'Image'),(1,mr,'GT'),(2,pb,'Prediction'),(3,overlay,'Overlay')]:
            ax=plt.subplot(n,4,i*4+j+1); ax.imshow(arr, cmap='gray' if j in [1,2] else None); ax.set_title(title); ax.axis('off')
    plt.tight_layout(); path=os.path.join(CFG.FIG_DIR,'fig1_qualitative_results.png'); plt.savefig(path,dpi=200,bbox_inches='tight')
    if show: plt.show()
    else: plt.close()

def plot_history(hist, show=False):
    os.makedirs(CFG.FIG_DIR,exist_ok=True)
    epochs=[h['epoch'] for h in hist]
    plt.figure(figsize=(10,5)); plt.plot(epochs,[h['train_loss'] for h in hist],label='Train Loss'); plt.plot(epochs,[h['val_loss'] for h in hist],label='Val Loss'); plt.legend(); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('ISIC Training Loss'); plt.tight_layout(); plt.savefig(os.path.join(CFG.FIG_DIR,'fig2_loss_curves.png'),dpi=200); plt.show() if show else plt.close()
    plt.figure(figsize=(10,5)); plt.plot(epochs,[h['val_dice'] for h in hist],label='Val Dice'); plt.plot(epochs,[h['val_iou'] for h in hist],label='Val IoU'); plt.legend(); plt.xlabel('Epoch'); plt.ylabel('Score'); plt.title('ISIC Validation Metrics'); plt.tight_layout(); plt.savefig(os.path.join(CFG.FIG_DIR,'fig3_metric_curves.png'),dpi=200); plt.show() if show else plt.close()

def show_all_figures():
    files=sorted([f for f in os.listdir(CFG.FIG_DIR) if f.endswith('.png')])
    if not files: return
    cols=2; rows=math.ceil(len(files)/cols); plt.figure(figsize=(16,rows*5))
    for i,f in enumerate(files,1):
        img=plt.imread(os.path.join(CFG.FIG_DIR,f)); ax=plt.subplot(rows,cols,i); ax.imshow(img); ax.set_title(f); ax.axis('off')
    plt.tight_layout(); plt.show()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['train','show'],default='train'); ap.add_argument('--epochs',type=int,default=15); ap.add_argument('--batch_size',type=int,default=8); ap.add_argument('--img_size',type=int,default=224); ap.add_argument('--base',type=int,default=16); ap.add_argument('--max_train',type=int,default=0); ap.add_argument('--show',action='store_true')
    args,_=ap.parse_known_args(); seed_all(CFG.SEED); CFG.IMG_SIZE=args.img_size; CFG.BATCH_SIZE=args.batch_size; CFG.EPOCHS=args.epochs; os.makedirs(CFG.OUT_DIR,exist_ok=True); os.makedirs(CFG.FIG_DIR,exist_ok=True)
    if args.mode=='show': show_all_figures(); return
    train_pairs=collect_pairs(CFG.TRAIN_IMG_DIR,CFG.TRAIN_MASK_DIR,args.max_train or None); val_pairs=collect_pairs(CFG.VAL_IMG_DIR,CFG.VAL_MASK_DIR); test_pairs=collect_pairs(CFG.TEST_IMG_DIR,CFG.TEST_MASK_DIR)
    print(f'Train={len(train_pairs)} Val={len(val_pairs)} Test={len(test_pairs)} Device={CFG.DEVICE}')
    train_loader=DataLoader(ISICDataset(train_pairs,args.img_size,True),batch_size=args.batch_size,shuffle=True,num_workers=CFG.NUM_WORKERS,pin_memory=torch.cuda.is_available())
    val_loader=DataLoader(ISICDataset(val_pairs,args.img_size,False),batch_size=args.batch_size,shuffle=False,num_workers=CFG.NUM_WORKERS,pin_memory=torch.cuda.is_available())
    test_loader=DataLoader(ISICDataset(test_pairs,args.img_size,False),batch_size=args.batch_size,shuffle=False,num_workers=CFG.NUM_WORKERS,pin_memory=torch.cuda.is_available())
    model=BMT2D(args.base).to(CFG.DEVICE); opt=torch.optim.AdamW(model.parameters(),lr=CFG.LR,weight_decay=1e-4); best=-1; hist=[]; ckpt=os.path.join(CFG.OUT_DIR,'best_isic_model.pth')
    for ep in range(args.epochs):
        tl,tm=run_epoch(model,train_loader,opt); vl,vm=run_epoch(model,val_loader,None); hist.append({'epoch':ep+1,'train_loss':tl,'val_loss':vl,'val_dice':vm[0],'val_iou':vm[1]})
        print(f'Epoch {ep+1:03d}/{args.epochs} train_loss={tl:.4f} val_loss={vl:.4f} Dice={vm[0]:.4f} IoU={vm[1]:.4f}')
        if vm[0]>best: best=vm[0]; torch.save(model.state_dict(),ckpt); print('Saved best:',ckpt)
    model.load_state_dict(torch.load(ckpt,map_location=CFG.DEVICE)); testm=evaluate_save(model,test_loader,os.path.join(CFG.OUT_DIR,'test_metrics.csv'))
    print(f'TEST Dice={testm[0]:.4f} IoU={testm[1]:.4f} Precision={testm[2]:.4f} Recall={testm[3]:.4f} Specificity={testm[4]:.4f}')
    plot_history(hist,args.show); make_figures(model,test_pairs,args.img_size,n=8,show=args.show)
    if args.show: show_all_figures()
    print('Figures:',CFG.FIG_DIR); print('CSV:',os.path.join(CFG.OUT_DIR,'test_metrics.csv'))
if __name__=='__main__': main()
