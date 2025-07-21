"""
    Wrapper pipeline for concept attention. 
"""
from dataclasses import dataclass
import PIL
import numpy as np
import matplotlib.pyplot as plt
from concept_attention.flux.src.flux.sampling import prepare
from concept_attention.segmentation import add_noise_to_image, encode_image
from concept_attention.utils import embed_concepts, linear_normalization
import torch
import einops
from tqdm import tqdm
from concept_attention.modified_double_stream_block import ModifiedDoubleStreamBlock, BPDoubleStreamBlock

from concept_attention.binary_segmentation_baselines.raw_cross_attention import RawCrossAttentionBaseline, RawCrossAttentionSegmentationModel
from concept_attention.binary_segmentation_baselines.raw_output_space import RawOutputSpaceBaseline, RawOutputSpaceSegmentationModel
from concept_attention.image_generator import FluxGenerator

@dataclass
class ConceptAttentionPipelineOutput():
    image: PIL.Image.Image | np.ndarray
    concept_heatmaps: list[PIL.Image.Image]
    cross_attention_maps: list[PIL.Image.Image]


def compute_heatmaps_from_vectors(
    image_vectors,
    concept_vectors,
    layer_indices: list[int],
    timesteps: list[int] = list(range(4)),
    softmax: bool = True,
    normalize_concepts: bool = False,
    w=64,
    h=64
):
    """
        Accepts image vectors and concept vectors. These can be from cross attentions or attention outputs.  
    """
    # Check if there are heads in the input 
    if len(image_vectors.shape) == 6: 
        # Collapse the had dimension
        image_vectors = einops.rearrange(
            image_vectors,
            "time layers batch head patches dim -> time layers batch patches (head dim)"
        )
        concept_vectors = einops.rearrange(
            concept_vectors,
            "time layers batch head concepts dim -> time layers batch concepts (head dim)"
        )


    # Apply linear normalization to concepts
    if normalize_concepts:
        concept_vectors = linear_normalization(concept_vectors, dim=-2)

    # Compute heatmaps 
    heatmaps = einops.einsum(
        image_vectors, 
        concept_vectors,
        "time layers batch patches dim, time layers batch concepts dim -> time layers batch concepts patches",
    )

    # Apply softmax
    if softmax:
        heatmaps = torch.nn.functional.softmax(heatmaps, dim=-2)
    # Pull out the timesteps and layers
    heatmaps = heatmaps[timesteps]
    heatmaps = heatmaps[:, layer_indices]
    # Average over the heatmaps
    heatmaps = einops.reduce(
        heatmaps,
        "time layers batch concepts patches -> batch concepts patches",
        reduction="mean"
    )
    heatmaps = einops.rearrange(
        heatmaps,
        "batch concepts (h w) -> batch concepts h w",
        h=h, #64,
        w=w #64
    )

    return heatmaps

class ConceptAttentionFluxPipeline():
    """
        This is an object that allows you to generate images with flux, and
        'encode' images with flux.  
    """

    def __init__(
        self, 
        model_name: str = "flux-schnell", 
        offload_model=False,
        device="cuda:0",
        gt = None
    ):
        self.model_name = model_name
        self.offload_model = offload_model
        # Load the generator
        if gt is not None:
            attention_block_class = BPDoubleStreamBlock
        else:
            attention_block_class = ModifiedDoubleStreamBlock
        self.flux_generator = FluxGenerator(
            model_name=model_name,
            attention_block_class=attention_block_class,
            offload=offload_model,
            device=device
        )

    @torch.no_grad()
    def generate_image(
        self, 
        prompt: str,
        concepts: list[str],
        width: int = 1024,
        height: int = 1024,
        return_cross_attention = False,
        layer_indices = list(range(15, 19)),
        return_pil_heatmaps = True,
        seed: int = 0,
        num_inference_steps: int = 4,
        guidance: float = 0.0,
        timesteps=None,
        softmax: bool = True,
        cmap="plasma"
    ) -> ConceptAttentionPipelineOutput:
        """
            Generate an image with flux, given a list of concepts.
        """
        assert return_cross_attention is False, "Not supported yet"
        assert all([layer_index >= 0 and layer_index < 19 for layer_index in layer_indices]), "Invalid layer index"
        assert height == width, "Height and width must be the same for now"

        if timesteps is None:
            timesteps = list(range(num_inference_steps))
        # Run the raw output space object
        image, concept_attention_dict = self.flux_generator.generate_image(
            width=width,
            height=height,
            prompt=prompt,
            num_steps=num_inference_steps,
            concepts=concepts,
            seed=seed,
            guidance=guidance,
        )
        
        cross_attention_maps = compute_heatmaps_from_vectors(
            concept_attention_dict["cross_attention_image_vectors"],
            concept_attention_dict["cross_attention_concept_vectors"],
            layer_indices=layer_indices,
            timesteps=timesteps,
            softmax=softmax,
            w=width//16,
            h=height//16
        )
        # Compute concept the heatmaps
        concept_heatmaps = compute_heatmaps_from_vectors(
            concept_attention_dict["output_space_image_vectors"],
            concept_attention_dict["output_space_concept_vectors"],
            layer_indices=layer_indices,
            timesteps=timesteps,
            softmax=softmax,
            w=width // 16,
            h=height // 16
        )

        concept_heatmaps = concept_heatmaps.to(torch.float32).detach().cpu().numpy()[0]
        cross_attention_maps = cross_attention_maps.to(torch.float32).detach().cpu().numpy()[0]
        # Convert the torch heatmaps to PIL images.
        if return_pil_heatmaps:
            concept_heatmaps_min = concept_heatmaps.min()
            concept_heatmaps_max = concept_heatmaps.max()
            # Convert to a matplotlib color scheme
            colored_heatmaps = []
            for concept_heatmap in concept_heatmaps:
                concept_heatmap = (concept_heatmap - concept_heatmaps_min) / (concept_heatmaps_max - concept_heatmaps_min)
                colored_heatmap = plt.get_cmap(cmap)(concept_heatmap)
                rgb_image = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
                colored_heatmaps.append(rgb_image)

            concept_heatmaps = [PIL.Image.fromarray(concept_heatmap) for concept_heatmap in colored_heatmaps]

            cross_attention_min = cross_attention_maps.min()
            cross_attention_max = cross_attention_maps.max()
            colored_cross_attention_maps = []
            for cross_attention_map in cross_attention_maps:
                cross_attention_map = (cross_attention_map - cross_attention_min) / (cross_attention_max - cross_attention_min)
                colored_cross_attention_map = plt.get_cmap(cmap)(cross_attention_map)
                rgb_image = (colored_cross_attention_map[:, :, :3] * 255).astype(np.uint8)
                colored_cross_attention_maps.append(rgb_image)

            cross_attention_maps = [PIL.Image.fromarray(cross_attention_map) for cross_attention_map in colored_cross_attention_maps]

            # gan4x4
            # First save individual heatmaps
            saved_paths = self.save_per_layer_heatmaps(
                concept_attention_dict=concept_attention_dict,
                concepts=concepts,
                layer_indices=layer_indices,
                timesteps=timesteps,
                width=width,
                height=height
            )

            # Then create grids for each concept
            for concept in concepts:
                self.create_heatmap_grid(
                    concept=concept,
                    layer_indices=layer_indices,
                    timesteps=timesteps,
                    heatmap_type="output"
                )
                """
                self.create_heatmap_grid(
                    concept=concept,
                    layer_indices=layer_indices,
                    timesteps=timesteps,
                    heatmap_type="cross"
                )
                """

        return ConceptAttentionPipelineOutput(
            image=image,
            concept_heatmaps=concept_heatmaps,
            cross_attention_maps=cross_attention_maps
        )

    def encode_image(
        self,
        image: PIL.Image.Image,
        concepts: list[str],
        prompt: str = "", # Optional
        width: int = 1024,
        height: int = 1024,
        layer_indices = list(range(15, 19)),
        num_samples: int = 1,
        num_steps: int = 4,
        noise_timestep: int = 2,
        device: str = "cuda:0",
        return_pil_heatmaps: bool = True,
        seed: int = 0,
        cmap="plasma",
        stop_after_multi_modal_attentions=True,
        softmax=True
    ) -> ConceptAttentionPipelineOutput:
        """
            Encode an image with flux, given a list of concepts.
        """
        assert all([layer_index >= 0 and layer_index < 19 for layer_index in layer_indices]), "Invalid layer index"
        assert height == width, "Height and width must be the same for now"
        print("Encoding image")

        # Encode the image into the VAE latent space
        encoded_image_without_noise = encode_image(
            image,
            self.flux_generator.ae,
            offload=self.flux_generator.offload,
            device=device,
        )
        # Do N trials
        combined_concept_attention_dict = {
            "cross_attention_image_vectors": [],
            "cross_attention_concept_vectors": [],
            # "cross_attention_maps": [],
            "output_space_image_vectors": [],
            "output_space_concept_vectors": [],
        }
        print("Sampling")
        for i in tqdm(range(num_samples)):
            # Add noise to image
            encoded_image, timesteps = add_noise_to_image(
                encoded_image_without_noise,
                num_steps=num_steps,
                noise_timestep=noise_timestep,
                seed=seed + i,
                width=width,
                height=height,
                device=device,
                is_schnell=self.flux_generator.is_schnell,
            )
            # Now run the diffusion model once on the noisy image
            # Encode the concept vectors
            
            if self.flux_generator.offload:
                self.flux_generator.t5, self.flux_generator.clip = self.flux_generator.t5.to(device), self.flux_generator.clip.to(device)
            inp = prepare(t5=self.flux_generator.t5, clip=self.flux_generator.clip, img=encoded_image, prompt=prompt)

            concept_embeddings, concept_ids, concept_vec = embed_concepts(
                self.flux_generator.clip,
                self.flux_generator.t5,
                concepts,
            )

            inp["concepts"] = concept_embeddings.to(encoded_image.device)
            inp["concept_ids"] = concept_ids.to(encoded_image.device)
            inp["concept_vec"] = concept_vec.to(encoded_image.device)
            # offload TEs to CPU, load model to gpu
            if self.flux_generator.offload:
                self.flux_generator.t5, self.flux_generator.clip = self.flux_generator.t5.cpu(), self.flux_generator.clip.cpu()
                torch.cuda.empty_cache()
                self.flux_generator.model = self.flux_generator.model.to(device)
            # Denoise the intermediate images
            guidance_vec = torch.full((encoded_image.shape[0],), 0.0, device=encoded_image.device, dtype=encoded_image.dtype)
            t_curr = timesteps[0]
            t_prev = timesteps[1]
            t_vec = torch.full((encoded_image.shape[0],), t_curr, dtype=encoded_image.dtype, device=encoded_image.device)
            _, concept_attention_dict = self.flux_generator.model(
                img=inp["img"],
                img_ids=inp["img_ids"],
                txt=inp["txt"],
                txt_ids=inp["txt_ids"],
                concepts=inp["concepts"],
                concept_ids=inp["concept_ids"],
                concept_vec=inp["concept_vec"],
                y=inp["concept_vec"],
                timesteps=t_vec,
                guidance=guidance_vec,
                stop_after_multimodal_attentions=stop_after_multi_modal_attentions, # Always true for the demo
                joint_attention_kwargs=None,
            )

            for key in combined_concept_attention_dict.keys():
                combined_concept_attention_dict[key].append(concept_attention_dict[key])

        # Pull out the concept and image vectors from each block
        for key in combined_concept_attention_dict.keys():
            combined_concept_attention_dict[key] = torch.stack(combined_concept_attention_dict[key]).squeeze(1)

        # Compute the heatmaps
        concept_heatmaps = compute_heatmaps_from_vectors(
            combined_concept_attention_dict["output_space_image_vectors"],
            combined_concept_attention_dict["output_space_concept_vectors"],
            layer_indices=layer_indices,
            timesteps=timesteps,
            softmax=softmax
        )

        cross_attention_maps = compute_heatmaps_from_vectors(
            combined_concept_attention_dict["cross_attention_image_vectors"],
            combined_concept_attention_dict["cross_attention_concept_vectors"],
            layer_indices=layer_indices,
            timesteps=timesteps,
            softmax=softmax
        )

        concept_heatmaps = concept_heatmaps.to(torch.float32).detach().cpu().numpy()[0]
        cross_attention_maps = cross_attention_maps.to(torch.float32).detach().cpu().numpy()[0]
        # Convert the torch heatmaps to PIL images.
        if return_pil_heatmaps:
            concept_heatmaps_min = concept_heatmaps.min()
            concept_heatmaps_max = concept_heatmaps.max()
            # Convert to a matplotlib color scheme
            colored_heatmaps = []
            for concept_heatmap in concept_heatmaps:
                concept_heatmap = (concept_heatmap - concept_heatmaps_min) / (concept_heatmaps_max - concept_heatmaps_min)
                colored_heatmap = plt.get_cmap(cmap)(concept_heatmap)
                rgb_image = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
                colored_heatmaps.append(rgb_image)

            concept_heatmaps = [PIL.Image.fromarray(concept_heatmap) for concept_heatmap in colored_heatmaps]

            cross_attention_min = cross_attention_maps.min()
            cross_attention_max = cross_attention_maps.max()
            colored_cross_attention_maps = []
            for cross_attention_map in cross_attention_maps:
                cross_attention_map = (cross_attention_map - cross_attention_min) / (cross_attention_max - cross_attention_min)
                colored_cross_attention_map = plt.get_cmap(cmap)(cross_attention_map)
                rgb_image = (colored_cross_attention_map[:, :, :3] * 255).astype(np.uint8)
                colored_cross_attention_maps.append(rgb_image)

            cross_attention_maps = [PIL.Image.fromarray(cross_attention_map) for cross_attention_map in colored_cross_attention_maps]


        return ConceptAttentionPipelineOutput(
            image=image,
            concept_heatmaps=concept_heatmaps,
            cross_attention_maps=cross_attention_maps
        )

#==================================Gan4x4========================================
    def save_per_layer_heatmaps(
            self,
            concept_attention_dict: dict,
            concepts: list[str],
            layer_indices: list[int],
            timesteps: list[int],
            width: int,
            height: int,
            output_dir: str = "layer_heatmaps",
            softmax: bool = True,
            cmap: str = "plasma"
    ) -> dict:
        """
        Save heatmaps for each layer and timestep separately, organized by concept.
        Returns dict with paths to saved images for grid generation.
        """
        import os
        saved_paths = {concept: [] for concept in concepts}

        # Create concept-specific folders
        for concept in concepts:
            concept_dir = os.path.join(output_dir, concept)
            os.makedirs(concept_dir, exist_ok=True)

        for timestep in timesteps:
            for layer_idx in layer_indices:
                output_heatmaps = compute_heatmaps_from_vectors(
                    concept_attention_dict["output_space_image_vectors"],
                    concept_attention_dict["output_space_concept_vectors"],
                    layer_indices=[layer_idx],
                    timesteps=[timestep],
                    softmax=softmax,
                    w=width // 16,
                    h=height // 16
                )
                """
                cross_heatmaps = compute_heatmaps_from_vectors(
                    concept_attention_dict["cross_attention_image_vectors"],
                    concept_attention_dict["cross_attention_concept_vectors"],
                    layer_indices=[layer_idx],
                    timesteps=[timestep],
                    softmax=softmax,
                    w=width // 16,
                    h=height // 16
                )
                """
                for heatmap_type, heatmaps in [
                    ("output", output_heatmaps),
                    #("cross", cross_heatmaps)
                ]:
                    heatmaps = heatmaps.to(torch.float32).detach().cpu().numpy()[0]
                    heatmaps_min = heatmaps.min()
                    heatmaps_max = heatmaps.max()

                    for concept_idx, concept in enumerate(concepts):
                        heatmap = heatmaps[concept_idx]
                        heatmap = (heatmap - heatmaps_min) / (heatmaps_max - heatmaps_min)
                        colored_heatmap = plt.get_cmap("plasma")(heatmap)  # Using default colormap
                        rgb_image = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)

                        img = PIL.Image.fromarray(rgb_image)
                        filename = f"{heatmap_type}_layer{layer_idx}_step{timestep}.png"
                        save_path = os.path.join(output_dir, concept, filename)
                        img.save(save_path)
                        saved_paths[concept].append(save_path)

        return saved_paths

    def create_heatmap_grid(
            self,
            concept: str,
            layer_indices: list[int],
            timesteps: list[int],
            input_dir: str = "layer_heatmaps",
            output_filename: str = None,
            heatmap_type: str = "output",
            border_size: int = 2,
            label_size: int = 30
    ) -> PIL.Image.Image:
        """
        Create a grid of heatmaps with borders and labels for timesteps and layers.
        """
        import os
        from PIL import Image, ImageDraw, ImageFont

        def resize4x(pil):
            # Open the image
            image = pil

            # Get the original size
            width, height = image.size

            # Calculate the new size (4 times larger)
            new_width = width * 4
            new_height = height * 4

            # Resize the image
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)


            return resized_image


        if output_filename is None:
            output_filename = f"{concept}_{heatmap_type}_grid.png"

        # Collect all images for the grid
        images = []
        for layer_idx in layer_indices:
            row_images = []
            for timestep in timesteps:
                filename = f"{heatmap_type}_layer{layer_idx}_step{timestep}.png"
                img_path = os.path.join(input_dir, concept, filename)
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    img = resize4x(img)
                    row_images.append(img)
            if row_images:
                images.append(row_images)

        if not images:
            raise ValueError("No images found for grid creation")

        # Calculate grid dimensions
        cell_width = images[0][0].width
        cell_height = images[0][0].height

        # Add space for borders and labels
        grid_width = cell_width * len(timesteps) + (len(timesteps) + 1) * border_size + label_size
        grid_height = cell_height * len(layer_indices) + (len(layer_indices) + 1) * border_size + label_size

        # Create new image with white background
        grid_img = Image.new('RGB', (grid_width, grid_height), 'white')
        draw = ImageDraw.Draw(grid_img)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
        except:
            font = ImageFont.load_default()

        # Draw timestep labels at the top
        for col_idx, timestep in enumerate(timesteps):
            x = label_size + border_size + col_idx * (cell_width + border_size) + cell_width // 2
            draw.text((x, label_size // 2), f"t={timestep}", fill='black', font=font, anchor="mm")

        # Draw layer labels on the left
        for row_idx, layer_idx in enumerate(layer_indices):
            y = label_size + border_size + row_idx * (cell_height + border_size) + cell_height // 2
            draw.text((label_size // 2, y), f"L{layer_idx}", fill='black', font=font, anchor="mm")

        # Paste images into grid
        for row_idx, row in enumerate(images):
            for col_idx, img in enumerate(row):
                x_offset = label_size + border_size + col_idx * (cell_width + border_size)
                y_offset = label_size + border_size + row_idx * (cell_height + border_size)
                grid_img.paste(img, (x_offset, y_offset))

        # Draw grid lines
        for i in range(len(timesteps) + 1):
            x = label_size + i * (cell_width + border_size)
            draw.rectangle([(x, label_size), (x + border_size - 1, grid_height)], fill='black')

        for i in range(len(layer_indices) + 1):
            y = label_size + i * (cell_height + border_size)
            draw.rectangle([(label_size, y), (grid_width, y + border_size - 1)], fill='black')

        # Save and return
        grid_img.save(output_filename)
        return grid_img