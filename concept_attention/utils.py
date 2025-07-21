import torch
import numpy as np
from sklearn.metrics import average_precision_score

# Utils for concept encoding
def embed_concepts(
    clip,
    t5,
    concepts: list[str],
    batch_size=1
):
    """
        Here the goal is to embed a bunch of concept vectors 
        into our text embedding space.  
    """
    # Code pulled from concept_attention.flux/sampling.py: prepare()
    # Embed each concept separately
    concept_embeddings = []
    for i, concept in enumerate(concepts): #gan4x4
        concept_embedding = t5(concept)
        # gan4x4
        torch.save(concept_embedding, f"concept_{i}_{concept}.pt")
        # Pull out the first token
        token_embedding = concept_embedding[0, 0, :] # First token of first prompt
        concept_embeddings.append(token_embedding)
    concept_embeddings = torch.stack(concept_embeddings).unsqueeze(0)
    # Add filler tokens of zeros
    concept_ids = torch.zeros(batch_size, concept_embeddings.shape[1], 3)

    # Embed the concepts to a clip vector
    prompt = " ".join(concepts)
    vec = clip(prompt)
    torch.save(vec, f"concept_clip.pt")
    vec = torch.zeros_like(vec).to(vec.device)

    return concept_embeddings, concept_ids, vec

def linear_normalization(x, dim):
    # Subtract the minimum to shift all values to non-negative range
    x_min = torch.min(x, dim=dim, keepdim=True)[0]
    x_shifted = x - x_min
    # Sum the values along the specified dimension
    x_sum = torch.sum(x_shifted, dim=dim, keepdim=True)
    # Avoid division by zero by setting sums of zero to one
    x_sum = torch.where(x_sum == 0, torch.ones_like(x_sum), x_sum)
    # Normalize by dividing by the sum
    return x_shifted / x_sum

################################## Metrics ##################################

def get_ap_scores(predict, target, ignore_index=-1):
    total = []
    for pred, tgt in zip(predict, target):
        target_expand = tgt.unsqueeze(0).expand_as(pred)
        target_expand_numpy = target_expand.data.cpu().numpy().reshape(-1)
        # Tensor process
        x = torch.zeros_like(target_expand)
        t = tgt.unsqueeze(0).clamp(min=0).long()
        target_1hot = x.scatter_(0, t, 1)
        predict_flat = pred.data.cpu().numpy().reshape(-1)
        predict_flat = np.nan_to_num(predict_flat)
        target_flat = target_1hot.data.cpu().numpy().reshape(-1)

        p = predict_flat[target_expand_numpy != ignore_index]
        t = target_flat[target_expand_numpy != ignore_index]

        total.append(np.nan_to_num(average_precision_score(t, p)))

    return total

def batch_pix_accuracy(predict, target):
    """Batch Pixel Accuracy
    Args:
        predict: input 3D tensor
        target: label 3D tensor
    """
    # _, predict = torch.max(predict, 0)
    predict = predict.cpu().numpy() + 1
    target = target.cpu().numpy() + 1
    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((predict == target) * (target > 0))
    assert pixel_correct <= pixel_labeled, \
        "Correct area should be smaller than Labeled"

    return pixel_correct, pixel_labeled


def batch_intersection_union(predict, target, nclass):
    """Batch Intersection of Union
    Args:
        predict: input 3D tensor
        target: label 3D tensor
        nclass: number of categories (int)
    """
    # _, predict = torch.max(predict, 0)
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = predict.cpu().numpy() + 1
    target = target.cpu().numpy() + 1

    predict = predict * (target > 0).astype(predict.dtype)
    intersection = predict * (predict == target)
    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), \
        "Intersection area should be smaller than Union area"
    return area_inter, area_union

"""
from concept_attention.concept_attention_pipeline import compute_heatmaps_from_vectors
import PIL
def concept_dict2pil(concept_attention_dict, layer_indices, timesteps, softmax,width, height, cmap="viridis", return_pil_heatmaps=True):
    cross_attention_maps = compute_heatmaps_from_vectors(
        concept_attention_dict["cross_attention_image_vectors"],
        concept_attention_dict["cross_attention_concept_vectors"],
        layer_indices=layer_indices,
        timesteps=timesteps,
        softmax=softmax,
        w=width // 16,
        h=height // 16
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
            cross_attention_map = (cross_attention_map - cross_attention_min) / (
                        cross_attention_max - cross_attention_min)
            colored_cross_attention_map = plt.get_cmap(cmap)(cross_attention_map)
            rgb_image = (colored_cross_attention_map[:, :, :3] * 255).astype(np.uint8)
            colored_cross_attention_maps.append(rgb_image)

        cross_attention_maps = [PIL.Image.fromarray(cross_attention_map) for cross_attention_map in
                                colored_cross_attention_maps]

    return concept_heatmaps, cross_attention_maps
"""