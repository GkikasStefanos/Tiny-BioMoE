# Tiny‑BioMoE

a Lightweight Embedding Model for Biosignal Analysis

> **Tiny‑BioMoE v1.0** · **7.34 M parameters** · **3.04 GFLOPs** · **192‑D embeddings** · **PyTorch ≥ 2.0**

---

## Highlights

| Feature          | Description                                                                 |
| ---------------- | --------------------------------------------------------------------------- |
| **Compact**      | <8 M parameters – runs comfortably on a laptop GPU / modern CPU             |
| **Cross‑domain** | Pre‑trained on **4.4 M** ECG, EMG & EEG representations via multi‑task learning |

<br/>

<p align="center">
  <img src="docs/overview.png" alt="Tiny‑BioMoE overview" width="48%"/>
  &nbsp;
  <img src="docs/encoders.png" alt="Encoder‑1 and Encoder‑2 details" width="48%"/>
</p>

<p align="center"><b>Figure&nbsp;1.</b> Overall Tiny‑BioMoE architecture (left) and the two expert encoders (right).</p>

---

## Table of Contents

1. [Pre‑trained checkpoint](#pre-trained-checkpoint)
2. [Quick start](#quick-start)

   * [Extract embeddings](#extract-embeddings)
3. [Fine‑tuning](#fine-tuning)
4. [Citation](#citation)
5. [Licence & acknowledgements](#licence--acknowledgements)

---

## Pre‑trained checkpoint

Get the weights from the **[GitHub Releases page](https://github.com/GkikasStefanos/Tiny-BioMoE/releases)**.

| File              | Size      |
| ----------------- | --------- |
| `Tiny-BioMoE.pth` | **89 MB** |

```bash
# download the latest checkpoint
auto=https://github.com/GkikasStefanos/Tiny-BioMoE/releases/latest/download/Tiny-BioMoE.pth
curl -L -o Tiny-BioMoE.pth "$auto"
```

> Verify the file if you wish:
>
> ```bash
> sha256sum Tiny-BioMoE.pth
> ```

The checkpoint contains **only one key**:

```text
model_state_dict    # MoE backbone weights (SpectFormer‑T‑w + EfficientViT‑w)
```

---

## Quick start

> Assumes **PyTorch ≥ 2.0** and **timm ≥ 0.9** are already installed.

### Extract embeddings

```python
import torch, torch.nn as nn
from PIL import Image
from torchvision import transforms
import architectures.spectformer, architectures.efficientvit
from timm.models import create_model

# ---------------------------------------------------------------
# Setup ----------------------------------------------------------
# ---------------------------------------------------------------
emb_size, num_experts = 96, 2
final_emb_size = emb_size * num_experts  # 192‑D

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"running on {device}")

# ---------------------------------------------------------------
# Backbone -------------------------------------------------------
# ---------------------------------------------------------------
class MoE(nn.Module):
    def __init__(self, enc1, enc2):
        super().__init__()
        self.enc1, self.enc2 = enc1, enc2
        self.ln_img = nn.LayerNorm((3, 224, 224))
        self.ln_e   = nn.LayerNorm(emb_size)
        self.ln_out = nn.LayerNorm(final_emb_size)
        self.fcn = nn.Sequential(nn.ELU(), nn.Linear(emb_size, emb_size),
                                 nn.Hardtanh(0, 1))
    @torch.no_grad()
    def forward(self, x):
        x = self.ln_img(x)
        z1, *_ = self.enc1(x)
        z2 = self.enc2(x)
        z1 = self.ln_e(z1) * self.fcn(z1)
        z2 = self.ln_e(z2) * self.fcn(z2)
        return self.ln_out(torch.cat((z1, z2), 1))

enc1 = create_model('spectformer_t_w'); enc1.head = nn.Identity()
enc2 = create_model('EfficientViT_w');  enc2.head = nn.Identity()
backbone = MoE(enc1, enc2).to(device).eval()
backbone.load_state_dict(torch.load('Tiny-BioMoE.pth', map_location=device)['model_state_dict'])

# ---------------------------------------------------------------
# One image → embedding -----------------------------------------
# ---------------------------------------------------------------
tr  = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
img = Image.open('img.png').convert('RGB')
img = tr(img).unsqueeze(0).to(device)        # → tensor [1, 3, 224, 224]
feat = backbone(img).squeeze().cpu().numpy()  # 192‑D vector
print(feat[:8])
```

---

## Fine‑tuning

Add your own classification/regression head and (optionally) un‑freeze the backbone:

```python
import torch, torch.nn as nn
import architectures.spectformer, architectures.efficientvit
from timm.models import create_model

# ---------------------------------------------------------------
# Setup ----------------------------------------------------------
# ---------------------------------------------------------------
emb_size, num_experts = 96, 2
final_emb_size = emb_size * num_experts  # 192‑D

class MoE(nn.Module):
    # identical to the class in Quick‑start
    ...

enc1 = create_model('spectformer_t_w'); enc1.head = nn.Identity()
enc2 = create_model('EfficientViT_w');  enc2.head = nn.Identity()
backbone = MoE(enc1, enc2).to('cuda')
backbone.load_state_dict(torch.load('Tiny-BioMoE.pth', map_location='cpu')['model_state_dict'])

# freeze if you only need fixed embeddings
for p in backbone.parameters():
    p.requires_grad = False

head = nn.Sequential(nn.ELU(), nn.Linear(final_emb_size, num_classes)).to('cuda')
optimizer = torch.optim.Adam(head.parameters(), lr=1e‑3)
```

---

## Citation

```bibtex
@misc{tiny_biomoe,
title={Tiny-BioMoE: a Lightweight Embedding Model for Biosignal Analysis}, 
author={Stefanos Gkikas and Ioannis Kyprakis and Manolis Tsiknakis},
year={2025},
eprint={2507.21875},
archivePrefix={arXiv},
primaryClass={cs.AI}
}
```

---

## Licence & acknowledgements

* Code & weights: **MIT Licence** – see [`LICENSE`](./LICENSE)
---

### Contact

Email **Stefanos Gkikas** ([gkikas@ics.forth.gr](mailto:gkikas@ics.forth.gr)).
