import os
import gc
import sys
import copy
import warnings
import joblib
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties
from PIL import Image
#import plotly
#import plotly.graph_objs
#import plotly.graph_objects as go


import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.utils import draw_segmentation_masks, make_grid

import zennit.image as zimage
from sklearn.mixture import GaussianMixture
from sklearn.exceptions import InconsistentVersionWarning

from crp.helper import get_layer_names
from LCRP.utils.crp_configs import ATTRIBUTORS, CANONIZERS, VISUALIZATIONS, COMPOSITES
from crp.concepts import ChannelConcept
from LCRP.utils.render import vis_opaque_img_border as _vis_opaque_img_border_orig  # ← alias original
from crp.image import imgify

# Add the parent directory to the Python path - bad practice, but it's just for the example
#sys.path.append("/Users/heydari/Desktop/test/FHHI-XAI-PIDNET/")

from src.glocal_analysis import run_analysis
from src.datasets.flood_dataset import FloodDataset
from src.datasets.DLR_dataset import DatasetDLR
from src.plot_crp_explanations import plot_explanations, plot_one_image_explanation
from src.minio_client import MinIOClient
from LCRP.models import get_model

#import plotly.graph_objects as go

# Device handling: use torch.device everywhere
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---- minimal addition: safe wrapper to align heatmaps/masks to image (H,W) ----
def _align_heatmaps_to_imgHW(data_batch, heatmaps):
    """
    Ensure heatmaps' last two dims match image H,W. If they are W,H, transpose.
    Works with torch.Tensor or numpy arrays, and with list/tuple containers.
    """
    # derive image H,W from data_batch
    if isinstance(data_batch, torch.Tensor):
        # expect (B, C, H, W)
        H, W = int(data_batch.shape[-2]), int(data_batch.shape[-1])
    else:
        # list/tuple of PIL images
        sample = data_batch[0]
        H, W = sample.size[1], sample.size[0]  # PIL: (W,H)

    def _align_one(hm):
        is_numpy = isinstance(hm, np.ndarray)
        hm_t = torch.from_numpy(hm) if is_numpy else hm
        if not torch.is_tensor(hm_t):
            return hm  # leave unchanged if unsupported
        if hm_t.ndim >= 2:
            h2, w2 = int(hm_t.shape[-2]), int(hm_t.shape[-1])
            if (h2, w2) != (H, W) and (h2, w2) == (W, H):
                hm_t = hm_t.transpose(-2, -1)  # swap (W,H) -> (H,W)
        return hm_t.numpy() if is_numpy else hm_t

    if isinstance(heatmaps, (list, tuple)):
        return type(heatmaps)(_align_one(hm) for hm in heatmaps)
    else:
        return _align_one(heatmaps)


def vis_opaque_img_border_safe(data_batch, heatmaps, rf, **kwargs):
    """
    Wrapper around original compositor that normalizes heatmap/mask shape
    to image (H,W) to avoid broadcasting errors.
    """
    heatmaps_aligned = _align_heatmaps_to_imgHW(data_batch, heatmaps)
    return _vis_opaque_img_border_orig(data_batch, heatmaps_aligned, rf, **kwargs)
# -------------------------------------------------------------------------------


def _maybe_reset_cuda_max_memory():
    if device.type == "cuda":
        try:
            torch.cuda.reset_max_memory_allocated()
        except Exception:
            # older pytorch may not support or permission issues
            pass


def _maybe_empty_cuda_cache():
    if device.type == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def get_ref_images(fv, topk_ind, layer_name, composite, n_ref=12, ref_imgs_save_path="examples/output/ref_imgs_pidnet/"):
    """
    Get and cache reference images. CPU/PIL based.
    """
    ref_imgs_save_path = os.path.join(ref_imgs_save_path, f"{layer_name}.h5")
    os.makedirs(os.path.dirname(ref_imgs_save_path), exist_ok=True)

    ref_imgs = {}
    missing_keys = list(map(str, topk_ind))

    if os.path.exists(ref_imgs_save_path):
        with h5py.File(ref_imgs_save_path, "a") as f:
            existing_keys = set(f.keys())
            missing_keys = [str(k) for k in topk_ind if str(k) not in existing_keys]

            for k in topk_ind:
                str_k = str(k)
                if str_k in f:
                    group = f[str_k]
                    # read stored PIL images from dataset arrays
                    ref_imgs[int(str_k)] = [Image.fromarray(group[str(idx)][:]) for idx in sorted(group.keys(), key=int)]

            if missing_keys:
                print(f"Calculating and saving missing reference images for keys: {missing_keys}")
                new_refs = fv.get_max_reference([int(k) for k in missing_keys], layer_name, "relevance", (0, n_ref),
                                                composite=composite, rf=True, plot_fn=vis_opaque_img_border_safe)  # ← changed
                for key, images_list in new_refs.items():
                    group = f.create_group(str(key))
                    assert len(images_list) >= n_ref
                    ref_imgs[key] = []
                    for idx, image in enumerate(images_list[:n_ref]):
                        if isinstance(image, Image.Image):
                            arr = np.array(image)
                            group.create_dataset(str(idx), data=arr)
                            ref_imgs[key].append(image)
                        else:
                            print(f"Warning: Item '{idx}' in key '{key}' is not a PIL image and will not be saved.")
    else:
        print("Reference image file does not exist, calculating all.")
        ref_imgs = fv.get_max_reference(topk_ind, layer_name, "relevance", (0, n_ref),
                                        composite=composite, rf=True, plot_fn=vis_opaque_img_border_safe)  # ← changed
        with h5py.File(ref_imgs_save_path, "w") as f:
            for key, images_list in ref_imgs.items():
                group = f.create_group(str(key))
                assert len(images_list) >= n_ref
                for idx, image in enumerate(images_list[:n_ref]):
                    if isinstance(image, Image.Image):
                        arr = np.array(image)
                        group.create_dataset(str(idx), data=arr)
                    else:
                        print(f"Warning: Item '{idx}' in key '{key}' is not a PIL image and will not be saved.")

    return ref_imgs


def plot_pcx_explanations(model_name, model, dataset, sample_id, n_concepts, n_refimgs, num_prototypes,
                          layer_name, ref_imgs_path, output_dir_pcx, output_dir_crp):
    """
    Wrapper that loads the sample, resets memory trackers, calls pidnet version,
    and ensures cleanup.
    """
    try:
        _maybe_reset_cuda_max_memory()

        image_tensor, t = dataset[sample_id]
        image_tensor =image_tensor.to(device)
        fig = plot_pcx_explanations_pidnet(
            model_name, model, dataset, image_tensor,
            n_concepts=n_concepts, n_refimgs=n_refimgs, num_prototypes=num_prototypes,
            layer_name=layer_name, ref_imgs_path=ref_imgs_path,
            output_dir_pcx=output_dir_pcx, output_dir_crp=output_dir_crp
        )

        gc.collect()
        _maybe_empty_cuda_cache()

        return fig

    except Exception as e:
        print(f"Error during explanation: {e}")
        gc.collect()
        _maybe_empty_cuda_cache()
        raise


def plot_pcx_explanations_pidnet(model_name, model, dataset, image_tensor,
                                 n_concepts=5, n_refimgs=12, num_prototypes=2,
                                 layer_name="decoder.center.0.0",
                                 ref_imgs_path="examples/output/ref_imgs_pidnet/",
                                 output_dir_pcx="examples/output/pcx/pidnet_flood/",
                                 output_dir_crp="examples/output/crp/pidnet_flood/"):
    """
    Main function that computes PCX/CRP visualizations.
    This version keeps tensors on GPU when possible and only moves to CPU for
    scikit-learn, plotting and PIL operations.
    """
    # ensure model in eval and on correct device
    model = model.to(device)
    model.eval()

    # get layer names (unchanged)
    layer_names = get_layer_names(model, types=[torch.nn.Conv2d])

    # Setting up CRP / attribution / visualizer objects
    attribution = ATTRIBUTORS[model_name](model)
    composite = COMPOSITES[model_name](canonizers=[CANONIZERS[model_name]()])
    fv = VISUALIZATIONS[model_name](
        attribution,
        dataset,
        layer_names,
        preprocess_fn=lambda x: x,
        path=output_dir_crp,
        max_target="max"
    )
    cc = ChannelConcept()

    # Prepare the image tensor on the computation device
    img = image_tensor[None, ...].to(device)

    # Load attributions file (stored on disk as numpy)
    folder = f"{output_dir_pcx}/{layer_name}/"
    if not os.path.exists(folder + "attributions.npy"):
        raise FileNotFoundError(f"Attributions file not found: {folder + 'attributions.npy'}")
    # load to CPU then to device as needed; keep a CPU copy for sklearn
    attributions_np = np.load(folder + "attributions.npy")  # numpy on CPU
    # torch tensor on device for any tensor ops
    attributions = torch.from_numpy(attributions_np).to(device)

    data = img
    class_id = 1

    # GMM cache paths
    cache_path = f'output/pcx/gmm_cache_{layer_name}.pkl'
    prototype_cache_path = f'output/pcx/prototype_gmms_cache_{layer_name}.pkl'

    # Use CPU numpy arrays for sklearn GaussianMixture
    def _safe_joblib_load(path):
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            try:
                return joblib.load(path)
            except InconsistentVersionWarning:
                return None

    gmm = None
    prototype_gmms = None
    if os.path.exists(cache_path) and os.path.exists(prototype_cache_path):
        gmm = _safe_joblib_load(cache_path)
        prototype_gmms = _safe_joblib_load(prototype_cache_path) if gmm is not None else None

    if gmm is None or prototype_gmms is None:
        # Drop potentially incompatible caches and refit for the current sklearn version
        try:
            if os.path.exists(cache_path):
                os.remove(cache_path)
            if os.path.exists(prototype_cache_path):
                os.remove(prototype_cache_path)
        except OSError:
            pass
        # Fit GMM on CPU copy of attributions
        attributions_for_gmm = attributions_np  # already CPU numpy
        gmm = GaussianMixture(n_components=num_prototypes, reg_covar=1e-5, random_state=0).fit(attributions_for_gmm)
        joblib.dump(gmm, cache_path)

        # Build prototype gmms based on gmm params (keeps sklearn objects on CPU)
        prototype_gmms = [GaussianMixture(n_components=1, covariance_type='full',) for _ in range(num_prototypes)]
        for p, g_ in enumerate(prototype_gmms):
            g_._set_parameters([
                param[p:p + 1] if j > 0 else param[p:p + 1] * 0 + 1
                for j, param in enumerate(gmm._get_parameters())
            ])
        joblib.dump(prototype_gmms, prototype_cache_path)

    # Calculate scores for the dataset (CPU)
    scores = gmm.score_samples(attributions_np)
    # raise ValueError(f"device: {device} data device {data.device} model device {model.device}")
    # Run attribution on the input image (this will happen on device)
    attr = attribution(data.requires_grad_(), [{"y": class_id}], composite, record_layer=[layer_name], init_rel=1)

    # Channel (neuron) relevance on the given layer for this image
    rel_tensor = attr.relevances[layer_name].detach()
    channel_rels = cc.attribute(rel_tensor, abs_norm=True).detach()
    attr_heatmap = attr.heatmap.detach().cpu() if hasattr(attr, "heatmap") else None
    pred_tensor = attr.prediction[0].detach() if hasattr(attr, "prediction") else None

    # Move tensors off GPU as soon as possible to free memory
    channel_rels_cpu = channel_rels.cpu()
    pred_tensor_cpu = pred_tensor.cpu() if pred_tensor is not None else None

    del rel_tensor
    del channel_rels
    del attr
    torch.cuda.empty_cache()

    channel_rels = channel_rels_cpu

    # Score the sample against the GMM (sklearn CPU) - convert to CPU numpy
    channel_rels_np = channel_rels.numpy()
    score_sample = gmm.score_samples(channel_rels_np)
    likelihoods = [g_.score_samples(channel_rels_np) for g_ in prototype_gmms]

    # Compute mean of the most likely prototype, then keep it as tensor on device for comparisons
    mean_np = gmm.means_[np.argmax(likelihoods)]
    mean = torch.from_numpy(mean_np).to(device)

    # Find the closest sample to mean using device tensors (move mean to device)
    # attributions is on device; ensure shapes align
    # attributions shape expected (N, D)
    closest_sample_to_mean = ((attributions - mean[None]).pow(2).sum(dim=1)).argmin().item()
    mean_cpu = mean.detach().cpu()
    del mean
    torch.cuda.empty_cache()

    # Save CPU-safe GMM data for inspection
    try:
        joblib.dump((attributions.detach().cpu().numpy(), gmm, channel_rels.detach().cpu().numpy(), mean_cpu.numpy()), "examples/output/pcx/gmm_data.pkl")
    except Exception:
        # fallback: save only plain things
        try:
            joblib.dump((attributions.detach().cpu(), channel_rels.detach().cpu(), mean_cpu), "examples/output/pcx/gmm_data_fallback.pkl")
        except Exception:
            pass

    # Closest prototype sample from dataset (data_p, target_p usually CPU tensors)
    data_p, target_p = dataset[closest_sample_to_mean]
    # keep a CPU copy of target_p for mask computations; move input data to device for model ops
    target_p_cpu = target_p.detach().cpu() if isinstance(target_p, torch.Tensor) else target_p
    data_p_device = data_p[None].to(device)

    # Getting top concepts/neurons for the given image in the given layer
    topk = torch.topk(channel_rels[0], n_concepts)
    topk_ind = topk.indices.detach().cpu().numpy()

    # Get reference images (CPU / PIL)
    ref_imgs = get_ref_images(fv, topk_ind, layer_name, composite=composite, n_ref=n_refimgs, ref_imgs_save_path=ref_imgs_path)

    # Calculate conditional heatmaps and prototype heatmaps (calls to attribution may return CPU or device tensors)
    conditions = [{"y": class_id, layer_name: int(c)} for c in topk_ind]

    def _to_cpu_container(obj):
        if torch.is_tensor(obj):
            return obj.detach().cpu()
        if isinstance(obj, (list, tuple)):
            return type(obj)(_to_cpu_container(o) for o in obj)
        return obj

    def _extract_heatmap(res):
        if hasattr(res, "heatmap"):
            return res.heatmap
        if isinstance(res, (list, tuple)):
            return res[0]
        return res

    def _collect_conditional_heatmaps(input_tensor, conds):
        heatmaps = []
        for cond in conds:
            # Clone to avoid autograd graph accumulation and keep memory bounded
            inp = input_tensor.detach().clone().requires_grad_()
            result = attribution(inp, [cond], composite)
            heatmaps.append(_to_cpu_container(_extract_heatmap(result)))
            torch.cuda.empty_cache()
        return heatmaps

    cond_heatmap_p = _collect_conditional_heatmaps(data_p_device, conditions)
    cond_heatmap = _collect_conditional_heatmaps(data, conditions)

    # Segmentation mask for plotting (CPU)
    # Use stored prediction to build mask
    if pred_tensor_cpu is not None:
        mask = (pred_tensor_cpu.argmax(dim=0) == class_id).detach().cpu()
    else:
        # In case prediction missing, make empty mask (safe fallback)
        # shape fallback to small size if needed
        mask = torch.zeros((1, 1), dtype=torch.bool)

    # Reverse augmentation expects CPU tensors; move the original input to CPU for reverse_augmentation
    sample_cpu_for_plot = dataset.reverse_augmentation(img.detach().cpu())
    # Resize mask if pidnet-style needs change
    if "pidnet" in model_name:
        # Convert mask to float and add batch + channel dims
        mask_f = mask.float().unsqueeze(0).unsqueeze(0)  # shape (1, 1, H, W)
        # Target spatial size based on reversed sample
        try:
            target_h = sample_cpu_for_plot[:3, :, :][0].shape[1]
            target_w = sample_cpu_for_plot[:3, :, :][0].shape[2]
            resized_mask = torch.nn.functional.interpolate(mask_f, size=(target_h, target_w), mode='nearest')
            mask = resized_mask.bool().squeeze().squeeze()
        except Exception:
            # fallback: keep original mask
            mask = mask.squeeze() if mask.dim() > 0 else mask

    # Draw segmentation overlay using torchvision (CPU tensors)
    try:
        img_with_mask = F.to_pil_image(draw_segmentation_masks(sample_cpu_for_plot[:3, :, :][0], masks=mask, alpha=0.3, colors=["red"]))
    except Exception:
        # fallback: convert sample to PIL without masks
        try:
            img_with_mask = F.to_pil_image(sample_cpu_for_plot[:3, :, :][0])
        except Exception:
            img_with_mask = Image.new("RGB", (150, 150), color=(128, 128, 128))

    # Prototype mask: ensure CPU tensor
    try:
        if isinstance(target_p_cpu, torch.Tensor):
            mask_prototype = (((target_p_cpu - target_p_cpu.min()) / (target_p_cpu.max() - target_p_cpu.min())) > 0.5)[0]
        else:
            # fallback if target not tensor
            mask_prototype = torch.zeros((1, 1), dtype=torch.bool)
    except Exception:
        mask_prototype = torch.zeros((1, 1), dtype=torch.bool)

    # Sample prototype image: reverse augmentation expects CPU input; use data_p_device.cpu()
    try:
        sample_prototype_cpu = dataset.reverse_augmentation(data_p_device.detach().cpu())
        img_prototype = F.to_pil_image(draw_segmentation_masks(sample_prototype_cpu[:3, :, :][0], masks=mask_prototype, alpha=0.3, colors=["red"]))
    except Exception:
        # fallback
        try:
            img_prototype = F.to_pil_image(sample_prototype_cpu[:3, :, :][0])
        except Exception:
            img_prototype = Image.new("RGB", (150, 150), color=(128, 128, 128))

    # --- PLOTTING ---
    # set up figure size depending on n_concepts
    n_rows = n_concepts if n_concepts > 3 else 3
    fig, axs = plt.subplots(n_rows, 6,
                            gridspec_kw={'width_ratios': [1, 1, n_refimgs / 4, 1, 1, 1]},
                            figsize=(4 * n_refimgs / 4, 1.8 * n_rows),
                            dpi=200)
    resize = torchvision.transforms.Resize((150, 150), antialias=True)

    # Ensure axs is 2D array for consistent indexing
    if axs.ndim == 1:
        axs = axs[None, :]

    for r, row_axs in enumerate(axs):
        for c, ax in enumerate(row_axs):
            # Default off for empty cells; we'll enable as necessary
            try:
                if c == 0:
                    if r == 0:
                        ax.set_title("input")
                        input_img = dataset.reverse_augmentation(img.detach().cpu()[0])
                        # input_img is tensor (C,H,W) on CPU
                        try:
                            ax.imshow(input_img.permute(1, 2, 0).cpu().numpy())
                        except Exception:
                            # fallback: convert using torchvision
                            ax.imshow(np.array(F.to_pil_image(input_img)))
                        # overlay the segmentation pil image (already created)
                        ax.imshow(np.asarray(img_with_mask))
                        try:
                            ax.contour(mask.numpy(), colors="black", linewidths=[1])
                        except Exception:
                            pass
                    elif r == 1:
                        ax.set_title("heatmap")
                        try:
                            if attr_heatmap is None:
                                raise ValueError("missing heatmap")
                            heatmap_img = imgify(attr_heatmap, cmap="bwr", symmetric=True)
                            ax.imshow(heatmap_img)
                        except Exception:
                            ax.axis("off")
                    elif r == 2:
                        ax.set_title("class likelihood")
                        a = ax.hist(scores, bins=20, color='k')
                        # score_sample may be array; pick scalar if so
                        s_sample_val = float(score_sample) if np.ndim(score_sample) == 0 else float(score_sample[0])
                        ax.vlines(s_sample_val, 0, a[0].max(), linestyle='--', linewidth=3, label="sample")
                        ax.legend()
                        ax.set_ylabel("density")
                        ax.set_xlabel("log-likelihood")
                        ax.set_yticks([])
                        ax.set_xticks([])

                        # outlier thresholds
                        lower_threshold = np.percentile(scores, 1)
                        upper_threshold = np.percentile(scores, 99)

                        outlier_text = "Outlier" if (s_sample_val < lower_threshold or s_sample_val > upper_threshold) else "Ordinary"
                        bbox_props = dict(boxstyle="round,pad=0.3",
                                          edgecolor="red" if outlier_text == "Outlier" else "green",
                                          facecolor="red" if outlier_text == "Outlier" else "green",
                                          alpha=0.3, linewidth=4)
                        ax.text(0.5, -0.35, outlier_text, transform=ax.transAxes, ha="center", fontsize=10,
                                fontweight='bold', color="red" if outlier_text == "Outlier" else "green", bbox=bbox_props)
                    else:
                        ax.axis("off")

                # Non-first column visualizations
                if c == 1:
                    if r == 0:
                        ax.set_title("Input localization")
                    try:
                        ch = cond_heatmap[r]
                        # ensure it's CPU numpy or tensor
                        ax.imshow(imgify(ch, symmetric=True, cmap="bwr", padding=True))
                    except Exception:
                        ax.axis("off")
                    ax.set_ylabel(f"concept {topk_ind[r]}\n relevance: {(channel_rels[0][topk_ind[r]] * 100):2.1f}%")

                elif c == 2:
                    if r == 0:
                        ax.set_title("concept visualization")
                    # build grid from ref images (PIL)
                    try:
                        grid = make_grid([resize(torch.from_numpy(np.asarray(i).copy()).permute((2, 0, 1))) for i in ref_imgs[topk_ind[r]]],
                                         nrow=int(n_refimgs / 2), padding=0)
                        grid = np.array(zimage.imgify(grid.detach().cpu()))
                        ax.imshow(grid)
                        ax.yaxis.set_label_position("right")
                    except Exception:
                        ax.axis("off")

                elif c == 3:
                    plt.rc('text', usetex=False)
                    plt.rcParams['font.family'] = 'DejaVu Sans'
                    bold_font = FontProperties(weight='bold')

                    if r == 0:
                        ax.set_title("Difference to prot.")
                    ax.imshow(np.zeros((150, 150, 3)), alpha=0.2)
                    try:
                        delta_R = (channel_rels[0][topk_ind[r]].round(decimals=3) - mean_cpu[topk_ind[r]].round(decimals=3)) * 100
                        delta_R = float(delta_R)
                    except Exception:
                        delta_R = 0.0

                    if delta_R > 2:
                        textstr = f"ΔR = {delta_R:+2.1f}%\n⚠ over-used"
                        edge_color = "#ff0000"
                    elif delta_R < -2:
                        textstr = f"ΔR = {delta_R:+2.1f}%\n⚠ under-used"
                        edge_color = "#ff0000"
                    else:
                        textstr = f"ΔR = {delta_R:+2.1f}%\n✓ similar"
                        edge_color = "#00cc00"

                    rect = patches.Rectangle((0, 0), 150, 150, linewidth=3, edgecolor=edge_color, facecolor='white')
                    ax.add_patch(rect)
                    lines = textstr.split('\n')
                    symbol_line = lines[1] if len(lines) > 1 else ""
                    text_line = lines[0] if lines else ""

                    ax.text(75, 60, text_line, fontsize=10, verticalalignment='center', horizontalalignment='center', bbox=dict(facecolor=edge_color, edgecolor='none'))
                    ax.text(75, 90, symbol_line, fontproperties=bold_font, verticalalignment='center', horizontalalignment='center', color=edge_color)

                    ax.set_xlim([0, 150])
                    ax.set_ylim([0, 150])
                    ax.axis("off")

                elif c == 4:
                    if r == 0:
                        ax.set_title("Prot localization")
                    try:
                        ax.imshow(imgify(cond_heatmap_p[r], symmetric=True, cmap="bwr", padding=True))
                        ax.yaxis.set_label_position("right")
                        ax.set_ylabel(f"concept {topk_ind[r]}\n relevance: {(mean_cpu[topk_ind[r]] * 100):2.1f}%")
                    except Exception:
                        ax.axis("off")

                elif c == 5:
                    if r == 0:
                        ax.set_title("prototype")
                        try:
                            fv.dataset = dataset
                            img_sample = imgify(fv.get_data_sample(closest_sample_to_mean, preprocessing=False)[0][0])
                            fv.dataset = dataset
                            ax.imshow(img_sample)
                            ax.imshow(np.asarray(img_prototype))
                            try:
                                ax.contour(mask_prototype, colors="black", linewidths=[1])
                            except Exception:
                                pass
                        except Exception:
                            ax.axis("off")
                    else:
                        ax.axis("off")

            except IndexError:
                axs[r][c].axis("off")
            except Exception:
                # if any plotting step fails, do not crash whole plot
                try:
                    ax.axis("off")
                except Exception:
                    pass

            # remove ticks
            try:
                ax.set_xticks([])
                ax.set_yticks([])
            except Exception:
                pass

    plt.tight_layout()
    return fig


def compute_outlier_scores(model_name, model, dataset, layer_name="decoder.center.0.0", num_prototypes=2, output_dir_pcx="examples/output/pcx/pidnet_flood/"):
    """
    Compute outlier scores using GMM on stored attributions.
    Uses GPU where relevant but keeps sklearn parts on CPU.
    """
    model = model.to(device)
    model.eval()

    layer_names = get_layer_names(model, types=[torch.nn.Conv2d])
    attribution = ATTRIBUTORS[model_name](model)
    composite = COMPOSITES[model_name](canonizers=[CANONIZERS[model_name]()])

    fv = VISUALIZATIONS[model_name](attribution, dataset, layer_names,
                                     preprocess_fn=lambda x: x,
                                     path=output_dir_pcx,
                                     max_target="max")

    folder = f"{output_dir_pcx}/{layer_name}/"
    if not os.path.exists(folder + "attributions.npy"):
        raise FileNotFoundError(f"Attributions file not found: {folder + 'attributions.npy'}")

    attributions_np = np.load(folder + "attributions.npy")
    attributions = torch.from_numpy(attributions_np).to(device)

    cache_path = f'examples/output/pcx/gmm_cache_{layer_name}.pkl'

    if os.path.exists(cache_path):
        gmm = joblib.load(cache_path)
    else:
        # fit on CPU numpy
        gmm = GaussianMixture(n_components=num_prototypes, reg_covar=1e-5, random_state=0).fit(attributions_np)
        joblib.dump(gmm, cache_path)

    # compute log-likelihood scores on CPU
    scores = gmm.score_samples(attributions_np)

    lower_threshold = np.percentile(scores, 1)
    upper_threshold = np.percentile(scores, 99)

    outliers = [i for i, score in enumerate(scores) if score < lower_threshold or score > upper_threshold]

    return outliers, scores, lower_threshold, upper_threshold
