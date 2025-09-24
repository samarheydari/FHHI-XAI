import os
import sys
import gc
import time
import copy
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes, make_grid
import zennit.image as zimage

from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from crp.image import imgify

logger = logging.getLogger(__name__)

from LCRP.utils.crp_configs import ATTRIBUTORS, CANONIZERS, VISUALIZATIONS, COMPOSITES
from LCRP.utils.render import vis_opaque_img_border


def plot_explanations(model_name, model, dataset, sample_id, class_id, layer, prediction_num, mode, n_concepts,
                      n_refimgs, output_dir):
    # Taken from L-CRP/experiments/plot_crp_explanation.py (visualization only)
    img, t = dataset[sample_id]
    print(img.shape)

    fig = plot_one_image_explanation(
        model_name, model, img, dataset, class_id, layer, prediction_num, mode, n_concepts, n_refimgs, output_dir
    )

    plt.show()
    print("Done plotting.")


def log_memory(message="", reset_max=False):
    """Log memory usage in a readable format."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        if reset_max:
            torch.cuda.reset_max_memory_allocated()
        print(f"MEMORY {message}: {allocated:.2f}GB (max: {max_allocated:.2f}GB)")
    else:
        print(f"MEMORY {message}: CUDA not available")


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _with_model_on_cpu(model, fn):
    """Move the SAME model object to CPU for canonizer-safe work, then restore device."""
    original_device = _model_device(model)
    model.to("cpu").eval()
    try:
        return fn()
    finally:
        model.to(original_device).eval()


def _run_attr_cpu(model, attribution_cls, composite, img_or_batch, *,
                  condition=None, conditions=None, record_layer=None,
                  init_rel=1, exclude_parallel=False, take_prediction=None):
    """
    Run CRP attribution on CPU (avoids BN-folding device mismatches in canonizers),
    then restore the SAME model object back to its original device.
    """
    def _do():
        attribution_cpu = attribution_cls(model)
        if take_prediction is not None:
            setattr(attribution_cpu, "take_prediction", take_prediction)

        x_cpu = img_or_batch.detach().cpu().requires_grad_(True)
        rec = [] if record_layer is None else list(record_layer)  # never pass None

        if conditions is not None:
            return attribution_cpu(
                x_cpu, conditions, composite,
                record_layer=rec, init_rel=init_rel, exclude_parallel=exclude_parallel
            )
        else:
            return attribution_cpu(
                x_cpu, condition, composite,
                record_layer=rec, init_rel=init_rel
            )

    return _with_model_on_cpu(model, _do)


def plot_one_image_explanation_optimized(model_name, model, img, dataset, class_id, layer, prediction_num, mode,
                                         n_concepts, n_refimgs, output_dir):
    # --- device & model state ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    log_memory("STARTING")

    # Build composite after model is on the right device
    composite = COMPOSITES[model_name](canonizers=[CANONIZERS[model_name]()])
    condition = [{"y": class_id}]

    # Ensure input on same device, add batch dim
    img = img[None, ...].to(device)
    ratio = img.shape[-2] / img.shape[-1]

    # Create the figure with axes
    fig, axs = plt.subplots(
        n_concepts, 3,
        figsize=((1.6 + 1 / ratio) * n_refimgs / 4, 1.6 * n_concepts),
        gridspec_kw={'width_ratios': [4, 4, n_refimgs * ratio]},
        dpi=200
    )
    # Normalize axs shape when n_concepts == 1 (matplotlib returns 1D array)
    if n_concepts == 1:
        axs = np.array([axs])

    log_memory("AFTER FIGURE CREATION")

    # -------------------------
    # Branch: segmentation models
    # -------------------------
    if "deeplab" in model_name or "unet" in model_name or "pidnet" in model_name:
        log_memory("BEFORE SEGMENTATION ATTR")

        # Run attribution on CPU to avoid canonizer device mismatch
        attr = _run_attr_cpu(
            model, ATTRIBUTORS[model_name], composite,
            img, condition=condition, record_layer=[layer], init_rel=1
        )

        log_memory("AFTER SEGMENTATION ATTR")

        # Heatmap
        heatmap = zimage.imgify(attr.heatmap.detach().cpu(), symmetric=True)

        # Mask
        if "DLR" in output_dir:
            mask = attr.prediction[0][0]
            mask = (mask - mask.min()) / (mask.max() - mask.min())
            mask = mask > 0.5
        else:
            mask = (attr.prediction[0].argmax(dim=0) == class_id).detach().cpu()

        # Reverse augmentation for visualization on CPU
        sample_ = dataset.reverse_augmentation(img[0].detach().cpu())

        # pidnet: resize mask to sample_ spatial size (CPU ops)
        if "pidnet" in model_name:
            mask = mask.float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            resized_mask = torch.nn.functional.interpolate(
                mask, size=(sample_.shape[1], sample_.shape[2]), mode='nearest'
            )
            mask = resized_mask.bool().squeeze(0).squeeze(0)  # (H,W)

        # Apply mask to image (draw_segmentation_masks expects CPU tensors)
        if sample_.shape[0] == 3:
            vis = draw_segmentation_masks(sample_.to(torch.uint8), masks=mask.cpu(), alpha=0.5, colors=["red"])
        else:
            vis = draw_segmentation_masks(sample_[:3, :, :].to(torch.uint8), masks=mask.cpu(), alpha=0.5, colors=["red"])

        img_ = F.to_pil_image(vis.cpu())
        axs[0][0].imshow(np.asarray(img_))
        axs[0][0].contour(mask.cpu().numpy(), colors="black", linewidths=[2])

        # Clear variables
        del mask, sample_, vis
        gc.collect()

    # -------------------------
    # Branch: detection models
    # -------------------------
    elif "yolo" in model_name or "ssd" in model_name:
        log_memory("BEFORE DETECTION ATTR")

        # Attribution on CPU (canonizers happy)
        attr = _run_attr_cpu(
            model, ATTRIBUTORS[model_name], composite,
            img, condition=condition, record_layer=[layer], init_rel=1, take_prediction=prediction_num
        )

        log_memory("AFTER DETECTION ATTR")

        # Heatmap
        heatmap = np.array(zimage.imgify(attr.heatmap.detach().cpu(), symmetric=True))
        heatmap = zimage.imgify(heatmap, symmetric=True)

        # Boxes on the model's (restored) device
        img_dev = img.to(_model_device(model))
        predicted_boxes = model.predict_with_boxes(img_dev)[1][0]
        predicted_classes = attr.prediction.argmax(dim=2)[0]
        print("Predicted classes: ", torch.unique(predicted_classes).detach().cpu().numpy())
        sorted_idx = attr.prediction.max(dim=2)[0].argsort(descending=True)[0]
        predicted_classes = predicted_classes[sorted_idx]
        predicted_boxes = predicted_boxes[sorted_idx]
        predicted_boxes = [b for b, c in zip(predicted_boxes, predicted_classes) if c == class_id][prediction_num]

        boxes = torch.tensor(predicted_boxes, dtype=torch.float)[None]
        colors = ["#ffcc00" for _ in boxes]

        # Visualize boxes
        base_img = dataset.reverse_normalization(img[0].detach().cpu()).to(torch.uint8)
        result = draw_bounding_boxes(base_img, boxes.cpu(), colors=colors, width=8)

        img_ = F.to_pil_image(result.cpu())
        axs[0][0].imshow(np.asarray(img_))

        # Clear variables
        del predicted_boxes, predicted_classes, sorted_idx, boxes, result, base_img
        gc.collect()

    else:
        raise NameError(f"Unknown model family for visualization: {model_name}")

    log_memory("AFTER DETECTION/SEGMENTATION")

    # === Channel relevance ===
    if mode == "relevance":
        channel_rels = ChannelConcept().attribute(attr.relevances[layer], abs_norm=True)
    else:
        channel_rels = attr.activations[layer].detach().cpu().flatten(start_dim=2).max(2)[0]
        channel_rels = channel_rels / channel_rels.abs().sum(1)[:, None]

    # === Top concepts ===
    topk = torch.topk(channel_rels[0], n_concepts)
    topk_ind = topk.indices.detach().cpu().numpy()
    topk_rel = topk.values.detach().cpu().numpy()

    print("Concepts:", topk)

    attribution_start_ts = time.time()

    # === Conditional heatmaps ===
    conditions = [{"y": class_id, layer: c} for c in topk_ind]
    if mode == "relevance":
        log_memory("BEFORE CONDITIONAL HEATMAPS")

        # Second attribution also on CPU; pass record_layer=[layer]
        cond_heatmap, _, _, _ = _run_attr_cpu(
            model, ATTRIBUTORS[model_name], composite,
            img, conditions=conditions, record_layer=[layer], init_rel=1,
            exclude_parallel=True, take_prediction=prediction_num
        )

        log_memory("AFTER CONDITIONAL HEATMAPS")
    else:
        cond_heatmap = torch.stack([attr.activations[layer][0][t] for t in topk_ind]).detach().cpu()

    logger.debug(f"Time to compute conditional heatmaps: {time.time() - attribution_start_ts:.2f}s")

    # === Reference images (run on CPU to avoid canonizer device mismatch) ===
    print("Computing reference images...")
    log_memory("BEFORE REF IMAGES")
    ref_images_start_ts = time.time()

    def _get_refs_cpu():
        # Build a CPU attribution and FV object bound to CPU
        attribution_cpu = ATTRIBUTORS[model_name](model)
        fv_cpu = VISUALIZATIONS[model_name](
            attribution_cpu, dataset, get_layer_names(model, [torch.nn.Conv2d]),
            preprocess_fn=lambda x: x, path=output_dir, max_target="max", device="cpu"
        )
        # Call get_max_reference on CPU
        return fv_cpu.get_max_reference(
            topk_ind, layer, mode, (0, n_refimgs), composite=composite, rf=True,
            plot_fn=vis_opaque_img_border, batch_size=2
        )

    ref_imgs = _with_model_on_cpu(model, _get_refs_cpu)

    logger.debug(f"Time to compute reference images: {time.time() - ref_images_start_ts:.2f}s")
    log_memory("AFTER REF IMAGES")

    # === Plotting ===
    plotting_start_ts = time.time()
    print("Plotting...")
    resize = torchvision.transforms.Resize((150, 150))

    for r, row_axs in enumerate(axs):
        log_memory(f"PLOTTING CONCEPT {r}")

        for c, ax in enumerate(row_axs):
            if c == 0:
                if r == 0:
                    ax.set_title("input")
                elif r == 1:
                    ax.set_title("heatmap")
                    ax.imshow(heatmap)
                else:
                    ax.axis('off')

            if c == 1:
                if r == 0:
                    ax.set_title("cond. heatmap")
                ax.imshow(imgify(cond_heatmap[r], symmetric=True, cmap="bwr", padding=False))
                ax.set_ylabel(f"concept {topk_ind[r]}\n relevance: {(topk_rel[r] * 100):2.1f}%")

            elif c >= 2:
                if r == 0 and c == 2:
                    ax.set_title("concept visualizations")

                if ref_imgs and topk_ind[r] in ref_imgs:
                    resized_refs = []
                    for i in ref_imgs[topk_ind[r]]:
                        img_tensor = resize(torch.from_numpy(np.array(i)).permute((2, 0, 1)))
                        resized_refs.append(img_tensor)

                    grid = make_grid(resized_refs, nrow=int(max(1, n_refimgs // 2)), padding=0)
                    grid = np.array(zimage.imgify(grid.detach().cpu()))
                    ax.imshow(grid)
                    ax.yaxis.set_label_position("right")

                    del resized_refs, grid
                    gc.collect()

            ax.set_xticks([])
            ax.set_yticks([])

        gc.collect()

    plt.tight_layout()

    logger.debug(f"Time to plot: {time.time() - plotting_start_ts:.2f}s")
    log_memory("AFTER PLOTTING")

    # Cleanup
    torch.cuda.empty_cache()
    gc.collect()

    return fig


def plot_one_image_explanation(model_name, model, img, dataset, class_id, layer, prediction_num, mode, n_concepts,
                               n_refimgs, output_dir):
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_max_memory_allocated()

        fig = plot_one_image_explanation_optimized(
            model_name, model, img, dataset, class_id, layer, prediction_num, mode, n_concepts, n_refimgs, output_dir
        )

        gc.collect()
        torch.cuda.empty_cache()
        return fig

    except Exception as e:
        print(f"Error during explanation: {e}")
        gc.collect()
        torch.cuda.empty_cache()
        raise


def plot_one_image_explanation_old(model_name, model, img, dataset, class_id, layer, prediction_num, mode, n_concepts,
                                   n_refimgs, output_dir):
    """Known-good but VRAM-heavy baseline. Kept for reference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    composite  = COMPOSITES[model_name](canonizers=[CANONIZERS[model_name]()])
    condition  = [{"y": class_id}]

    img = img[None, ...].to(device)
    ratio = img.shape[-2] / img.shape[-1]

    fig, axs = plt.subplots(
        n_concepts, 3,
        figsize=((1.6 + 1 / ratio) * n_refimgs / 4, 1.6 * n_concepts),
        gridspec_kw={'width_ratios': [4, 4, n_refimgs * ratio]},
        dpi=200
    )
    if n_concepts == 1:
        axs = np.array([axs])

    if "deeplab" in model_name or "unet" in model_name:
        # Attribution on CPU
        attr = _run_attr_cpu(
            model, ATTRIBUTORS[model_name], composite,
            img, condition=condition, record_layer=[layer], init_rel=1
        )
        heatmap = zimage.imgify(attr.heatmap.detach().cpu(), symmetric=True)

        if "DLR" in output_dir:
            mask = attr.prediction[0][0]
            mask = (mask - mask.min()) / (mask.max() - mask.min())
            mask = mask > 0.5
        else:
            mask = (attr.prediction[0].argmax(dim=0) == class_id).detach().cpu()

        sample_ = dataset.reverse_augmentation(img[0].detach().cpu())

        if sample_.shape[0] == 3:
            vis = draw_segmentation_masks(sample_.to(torch.uint8), masks=mask.cpu(), alpha=0.5, colors=["red"])
        else:
            vis = draw_segmentation_masks(sample_[:3, :, :].to(torch.uint8), masks=mask.cpu(), alpha=0.5, colors=["red"])

        img_ = F.to_pil_image(vis.cpu())
        axs[0][0].imshow(np.asarray(img_))
        axs[0][0].contour(mask.cpu().numpy(), colors="black", linewidths=[2])

    elif "yolo" in model_name or "ssd" in model_name:
        # Attribution on CPU
        attr = _run_attr_cpu(
            model, ATTRIBUTORS[model_name], composite,
            img, condition=condition, record_layer=[layer], init_rel=1, take_prediction=prediction_num
        )
        heatmap = np.array(zimage.imgify(attr.heatmap.detach().cpu(), symmetric=True))
        heatmap = zimage.imgify(heatmap, symmetric=True)

        img_dev = img.to(_model_device(model))
        predicted_boxes  = model.predict_with_boxes(img_dev)[1][0]
        predicted_classes = attr.prediction.argmax(dim=2)[0]
        print("Predicted classes: ", torch.unique(predicted_classes).detach().cpu().numpy())
        sorted_idx = attr.prediction.max(dim=2)[0].argsort(descending=True)[0]
        predicted_classes = predicted_classes[sorted_idx]
        predicted_boxes   = predicted_boxes[sorted_idx]
        predicted_boxes   = [b for b, c in zip(predicted_boxes, predicted_classes) if c == class_id][prediction_num]
        boxes  = torch.tensor(predicted_boxes, dtype=torch.float)[None]
        colors = ["#ffcc00" for _ in boxes]

        base_img = dataset.reverse_normalization(img[0].detach().cpu()).to(torch.uint8)
        result   = draw_bounding_boxes(base_img, boxes.cpu(), colors=colors, width=8)

        img_ = F.to_pil_image(result.cpu())
        axs[0][0].imshow(np.asarray(img_))
    else:
        raise NameError

    # === Conditional heatmaps ===
    topk_dummy = torch.topk(torch.tensor([1.0, 0.0]).unsqueeze(0), 1)  # placeholder if needed elsewhere
    # Reference images on CPU (same approach as optimized path) if you still need them here:
    # ...

    plt.tight_layout()
    return fig


def fig_to_array(fig):
    """Convert a matplotlib figure to a numpy array."""
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data
