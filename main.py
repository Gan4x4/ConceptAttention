from concept_attention import ConceptAttentionFluxPipeline

pipeline = ConceptAttentionFluxPipeline(
    model_name="flux-schnell",
    device="cuda:0",
    offload_model=True,
    gt=None
)

prompt = "A dragon standing on a rock. "
concepts = ["dragon", "rock", "sky", "cloud"]

pipeline_output = pipeline.generate_image(
    prompt=prompt,
    concepts=concepts,
    width=128,
    height=128
)

image = pipeline_output.image
concept_heatmaps = pipeline_output.concept_heatmaps

image.save("image.png")
for concept, concept_heatmap in zip(concepts, concept_heatmaps):
    concept_heatmap.save(f"{concept}.png")