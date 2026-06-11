Sanity check
# =============================================================================
def sanity_check():
    device=get_device()
    model=BMTHARNetPP(4,3,base=8,depth=1,heads=4,deep_sup=False).to(device)
    x=torch.randn(1,4,48,48,48).to(device)
    y=torch.rand(1,3,48,48,48).to(device).round()
    b=torch.rand(1,1,48,48,48).to(device).round()
    model.train()
    seg,bd,ds=model(x); loss,comps=combined_loss(seg,bd,ds,y,b)
    print(f"seg: {seg.shape} | bnd: {bd.shape} | loss: {loss.item():.4f}")
    print(f"comps: {comps}")
    print(f"params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    # also test architecture figure
    plot_architecture()
    print(f"Architecture figure saved to: {FIGURES_DIR}/fig1_architecture.png")
    print("Sanity check PASSED ✓")
