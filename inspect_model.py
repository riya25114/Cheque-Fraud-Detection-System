import torch
ckpt = torch.load("siamese_signature_model.pth", map_location="cpu")

print("Keys in checkpoint:", ckpt.keys())  # show what’s inside
if "state_dict" in ckpt:
    state_dict = ckpt["state_dict"]
else:
    state_dict = ckpt

print("\nLayer names and shapes:")
for k, v in state_dict.items():
    print(k, v.shape)