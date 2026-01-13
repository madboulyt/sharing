def load_dfine_model_unencrypted(checkpoint_path, config_path, device):
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    if "HGNetv2" in cfg.yaml_cfg:
        try:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        except Exception:
            pass

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    elif "model" in checkpoint:
        state = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint
    cfg.model.load_state_dict(state, strict=True)
    d_model = DFineDeployModel(cfg).to(device)
    d_model.eval()

    return d_model
