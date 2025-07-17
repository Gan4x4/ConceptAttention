# tests/test_single_layer_guidance.py
import torch
import torch.nn.functional as F
import unittest
#from concept_attention.modified_double_stream_block import ModifiedDoubleStreamBlock
from concept_attention.flux.src.flux.model import Flux,  FluxParams


import torch, unittest, types

# ------------------------------------------------------------------
# 1) Monkey‑patch RoPE to NOP so we don’t hit the shape mismatch.
#    We must patch the *module‑level* symbol that the block grabbed
#    when it was imported.
# ------------------------------------------------------------------
import concept_attention.modified_double_stream_block as mdb
def _noop_rope(q, k, pe):          # keep signature identical
    return q, k
mdb.apply_rope = _noop_rope        # patch in‑place **before** using the class

from concept_attention.modified_double_stream_block import ModifiedDoubleStreamBlock
class TinyBlockSmokeTest(unittest.TestCase):
    def test_modified_double_stream_block_forward_and_backprop(self):
        B, Hf, Wf, D = 1, 8, 8, 32  # batch size, feature-map size, feature dim
        L_patches = Hf * Wf
        hidden_size = D
        num_heads = 4
        mlp_ratio = 2.0

        # Create a random GT mask at full image resolution  (e.g., 64×64)
        H_img, W_img = 64, 64
        gt = torch.randint(0, 2, (B, H_img, W_img), dtype=torch.float32)

        # Instantiate the block with ground truth
        block = ModifiedDoubleStreamBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            #gt=gt
        )

        # Dummy inputs
        #img = torch.randn(B, num_heads, L_patches, D // num_heads, requires_grad=True)
        #txt = torch.randn(B, num_heads, 1, D // num_heads, requires_grad=True)
        #concepts = txt.clone()
        #pe = None
        #cons_pe = None
        #cons_vec = concepts.clone()
        device = "cuda:0"
        block.to(device)

        B, Li, Lt, Lc, D = 1, 16, 5, 1, 32  # 4x4 img grid
        img = torch.randn(B, Li, D, device=device, requires_grad=True)
        txt = torch.randn(B, Lt, D, device=device, requires_grad=True)
        vec = torch.randn(B, D, device=device)
        pe = torch.zeros(B, 1, 1, D, device=device)  # unused
        concepts = torch.randn(B, Lc, D, device=device)
        concept_vec = torch.randn(B, D, device=device)
        concept_pe = torch.zeros_like(pe)

        # Forward pass; receive img, txt, concepts, dict and ce_loss
        img_out, txt_out, concepts_out, ca_dict  = block(
            img, txt, vec, pe, concepts, concept_vec, concept_pe
        )


    def test_full_model(self):
        #
        # 1.  Tiny config ─ keep every dimension small so the test is fast.
        #
        p = FluxParams(  # dataclass in model.py lines 11‑24 :contentReference[oaicite:4]{index=4}
            in_channels=32,  # chan/patch that goes into img stream
            vec_in_dim=32,
            context_in_dim=32,
            hidden_size=64,  # must be divisible by num_heads …
            mlp_ratio=2.0,
            num_heads=4,
            depth=1,  # one double‑stream block is enough for a smoke test
            depth_single_blocks=0,  # can be zero
            axes_dim=[16],  # Σaxes_dim must equal hidden_size//num_heads (=16)  :contentReference[oaicite:5]{index=5}
            theta=10000,
            qkv_bias=True,
            guidance_embed=False,
        )
        device = "cuda:0"
        model = Flux(p).cuda()  # real model – no RoPE hack

        #
        # 2.  Dummy batch.
        #
        B, T_img, T_txt = 1, 16, 5
        img = torch.randn(B, T_img, p.in_channels, device="cuda", requires_grad=True)
        txt = torch.randn(B, T_txt, p.context_in_dim, device="cuda")
        img_ids = torch.arange(T_img, device=device).unsqueeze(0).unsqueeze(-1).long()  # (1, 16, 1)
        txt_ids = torch.arange(T_txt, device=device).unsqueeze(0).unsqueeze(-1).long()
        timesteps = torch.tensor([10], device="cuda")  # any integer ok
        y = torch.randn(B, p.vec_in_dim, device="cuda")

        #
        # 3.  Forward → loss → backward.
        #
        out = model(img, img_ids, txt, txt_ids, timesteps, y)  # (B, T_img, in_channels)


        # 4.  Sanity assertions – gradients exist & shapes look right.
        self.assertEqual(out.shape, torch.Size([B, T_img, p.in_channels]))
        self.assertIsNotNone(img.grad)
        self.assertEqual(img.grad.shape, img.shape)

if __name__ == "__main__":
    unittest.main()