class ISICDataset(Dataset):
    def __init__(self, pairs, img_size=224, train=True):
        self.pairs = pairs; self.img_size = img_size; self.train = train
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        img = cv2.imread(ip, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)
        if self.train:
            if random.random() < 0.5:
                img = np.fliplr(img).copy(); mask = np.fliplr(mask).copy()
            if random.random() < 0.5:
                img = np.flipud(img).copy(); mask = np.flipud(mask).copy()
            if random.random() < 0.35:
                k = random.randint(1,3); img = np.rot90(img,k).copy(); mask = np.rot90(mask,k).copy()
            if random.random() < 0.6:
                alpha = random.uniform(0.85,1.20); beta = random.randint(-18,18)
                img = np.clip(img.astype(np.float32)*alpha + beta, 0, 255).astype(np.uint8)
            if random.random() < 0.25:
                img = cv2.GaussianBlur(img, (3,3), 0)
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485,0.456,0.406])) / np.array([0.229,0.224,0.225])
        img = img.transpose(2,0,1).astype(np.float32)
        return torch.tensor(img), torch.tensor(mask[None].astype(np.float32))
