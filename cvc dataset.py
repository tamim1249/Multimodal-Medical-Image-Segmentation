class CVCDataset(Dataset):
    def __init__(self, pairs, img_size=256, train=True):
        self.pairs = pairs; self.img_size = img_size; self.train = train
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        ip, mp, name = self.pairs[idx]
        img = cv2.imread(ip, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.img_size,self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size,self.img_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)
        bnd = make_boundary(mask)
        if self.train:
            img, mask, bnd = augment(img, mask, bnd)
            mask = (mask > 0.5).astype(np.float32); bnd = (bnd > 0.5).astype(np.float32)
        img = img.astype(np.float32)/255.0
        img = (img - np.array([0.485,0.456,0.406], dtype=np.float32)) / np.array([0.229,0.224,0.225], dtype=np.float32)
        img = img.transpose(2,0,1)
        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask[None], dtype=torch.float32), torch.tensor(bnd[None], dtype=torch.float32), name

