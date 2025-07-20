# tests/test_single_layer_guidance.py
import torch
import torch.nn.functional as F
import unittest
from concept_attention.modified_double_stream_block import ModifiedDoubleStreamBlock, BPDoubleStreamBlock
from concept_attention.flux.src.flux.model import Flux,  FluxParams
import torch, unittest, types
from concept_attention.modified_flux_dit import ModifiedFluxDiT

# ------------------------------------------------------------------
# 1) Monkey‑patch RoPE to NOP so we don’t hit the shape mismatch.
#    We must patch the *module‑level* symbol that the block grabbed
#    when it was imported.
# ------------------------------------------------------------------
#import concept_attention.modified_double_stream_block as mdb
#def _noop_rope(q, k, pe):          # keep signature identical
#    return q, k
#mdb.apply_rope = _noop_rope        # patch in‑place **before** using the class

#from concept_attention.modified_double_stream_block import ModifiedDoubleStreamBlock
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
        p = FluxParams(
            in_channels=32,  # channel dim for the image stream
            vec_in_dim=32,  # dim of the conditioning vector y
            context_in_dim=32,  # dim of each text token embedding
            hidden_size=64,  # must be divisible by num_heads
            mlp_ratio=2.0,
            num_heads=4,
            depth=1,  # one ModifiedDoubleStreamBlock
            depth_single_blocks=0,
            axes_dim=[16],  # len == 1 because num_heads=4, hidden//heads = 16
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        )
        device = "cuda:0"
        dtype = torch.bfloat16
        #model = Flux(p).cuda()  # real model – no RoPE hack
        model = ModifiedFluxDiT(p, attention_block_class=ModifiedDoubleStreamBlock).to(dtype)
        model.to(device)

        #
        # 2.  Dummy batch.

        B, T_img, T_txt, T_con = 1, 16, 5, 1  # 16 img‑patches, 5 text tokens, 1 concept token
        img = torch.randn(B, T_img, p.in_channels, device=device, dtype=dtype)
        txt = torch.randn(B, T_txt, p.context_in_dim, device=device, dtype=dtype)
        concepts = torch.randn(B, T_con, p.context_in_dim, device=device, dtype=dtype)

        # Pos‑ID tensors need trailing axis = n_axes (here 1)
        img_ids = torch.arange(T_img).unsqueeze(0).unsqueeze(-1).long().to(device)  # (1,16,1)
        txt_ids = torch.arange(T_txt).unsqueeze(0).unsqueeze(-1).long().to(device)  # (1, 5,1)
        concept_ids = torch.arange(T_con).unsqueeze(0).unsqueeze(-1).long().to(device)  # (1, 1,1)

        concept_vec = torch.randn(B, p.context_in_dim, device=device, dtype=dtype)
        timesteps = torch.tensor([10], device=device, dtype=torch.long)
        y = torch.randn(B, p.vec_in_dim, device=device, dtype=dtype)

        # ---------- 3. Forward pass ------------------------------------
        img_out, attn_dict = model(
            img, img_ids,
            txt, txt_ids,
            concepts, concept_ids,
            concept_vec,
            timesteps,
            y
        )
        # 4.  Sanity assertions – gradients exist & shapes look right.
        self.assertEqual(img_out.shape, torch.Size([B, T_img, p.in_channels]))
        self.assertEqual(img.grad.shape, img.shape)

    def set_inputs(self, h=64, w=64, prompt="default", concept="default", device="cuda:0", dtype=torch.bfloat16):
        """
        Helper to create dummy inputs with specified image size, prompt, and concept.
        """
        patch_size = 16
        H_patches = h // patch_size
        W_patches = w // patch_size
        T_img = H_patches * W_patches
        T_txt = 5  # arbitrary small number of tokens for prompt
        T_con = 1  # one concept token

        img = torch.randn(1, T_img, 32, device=device, dtype=dtype)
        txt = torch.full((1, T_txt, 32), hash(prompt) % 100 / 100.0, device=device, dtype=dtype)
        concepts = torch.full((1, T_con, 32), hash(concept) % 100 / 100.0, device=device, dtype=dtype)

        img_ids = torch.arange(T_img).unsqueeze(0).unsqueeze(-1).long().to(device)
        txt_ids = torch.arange(T_txt).unsqueeze(0).unsqueeze(-1).long().to(device)
        concept_ids = torch.arange(T_con).unsqueeze(0).unsqueeze(-1).long().to(device)

        concept_vec = torch.randn(1, 32, device=device, dtype=dtype)
        timesteps = torch.tensor([10], device=device, dtype=torch.long)
        y = torch.randn(1, 32, device=device, dtype=dtype)

        return img, img_ids, txt, txt_ids, concepts, concept_ids, concept_vec, timesteps, y

    def test_direct_input_setting(self):
        """
        Test setting image size, prompt, and concept directly.
        """
        p = FluxParams(
            in_channels=32,
            vec_in_dim=32,
            context_in_dim=32,
            hidden_size=64,
            mlp_ratio=2.0,
            num_heads=4,
            depth=2,
            depth_single_blocks=0,
            axes_dim=[16],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        )
        device = "cuda:0"
        dtype = torch.bfloat16
        model = ModifiedFluxDiT(p, attention_block_class=BPDoubleStreamBlock).to(dtype)
        model.to(device)
        H = 256
        W = 256
        # Example usage: set h, w, prompt, concept directly
        img, img_ids, txt, txt_ids, concepts, concept_ids, concept_vec, timesteps, y = \
            self.set_inputs(h=H, w=W, prompt="Day view of citystreet", concept="car", device=device, dtype=dtype)

        img.requires_grad_(True)
        img_out, attn_dict = model(
            img, img_ids,
            txt, txt_ids,
            concepts, concept_ids,
            concept_vec,
            timesteps,
            y
        )
        self.assertEqual(img_out.shape[1], (H // 16) * (W // 16))
        self.assertEqual(txt.shape[2], 32)
        self.assertEqual(concepts.shape[2], 32)
        #self.assertEqual(img.grad.shape, img.shape)

if __name__ == "__main__":
    unittest.main()