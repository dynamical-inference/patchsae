import argparse

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import torch
from PIL import Image, ImageDraw
from plotly.subplots import make_subplots

from src.demo.utils import load_sae_tester

IMAGE_SIZE = 400
DATASET_LIST = ["imagenet"]
GRID_NUM = 14


def get_grid_loc(evt, image):
    # Get click coordinates
    x, y = evt._data["index"][0], evt._data["index"][1]

    cell_width = image.width // GRID_NUM
    cell_height = image.height // GRID_NUM

    grid_x = x // cell_width
    grid_y = y // cell_height
    return grid_x, grid_y, cell_width, cell_height


def plot_activation(
    evt: gr.EventData,
    current_image,
    activation,
    model_name: str,
    colors: tuple[str, str],
):
    """Plot activation distribution for the full image and optionally a selected tile"""
    mean_activation = activation.mean(0)

    tile_activation = None
    tile_x = None
    tile_y = None

    if evt is not None and evt._data is not None:
        tile_x, tile_y, _, _ = get_grid_loc(evt, current_image)
        token_idx = tile_y * GRID_NUM + tile_x + 1
        tile_activation = activation[token_idx]

    fig = create_activation_plot(
        mean_activation,
        tile_activation,
        tile_x,
        tile_y,
        model_name=model_name,
        colors=colors,
    )

    return fig


def create_activation_plot(
    mean_activation,
    tile_activation=None,
    tile_x=None,
    tile_y=None,
    top_k=5,
    colors=("blue", "cyan"),
    model_name="CLIP",
):
    """Create plotly figure with activation traces and annotations"""
    fig = go.Figure()

    # Add trace for mean activation across full image
    model_label = model_name.split("-")[0]
    add_activation_trace(
        fig, mean_activation, f"{model_label} Image-level", colors[0], top_k
    )

    # Add trace for tile activation if provided
    if tile_activation is not None:
        add_activation_trace(
            fig,
            tile_activation,
            f"{model_label} Tile ({tile_x}, {tile_y})",
            colors[1],
            top_k,
        )

    # Update layout
    fig.update_layout(
        title="Activation Distribution",
        xaxis_title="SAE latent index",
        yaxis_title="Activation Value",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="middle", y=0.5, xanchor="center", x=0.5),
    )

    return fig


def add_activation_trace(fig, activation, label, color, top_k):
    """Add a single activation trace with annotations to the figure"""
    # Add line trace
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(activation)),
            y=activation,
            mode="lines",
            name=label,
            line=dict(color=color, dash="solid"),
            showlegend=True,
        )
    )

    # Add annotations for top activations
    top_indices = np.argsort(activation)[::-1][:top_k]
    for idx in top_indices:
        fig.add_annotation(
            x=idx,
            y=activation[idx],
            text=str(idx),
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-15,
            arrowcolor=color,
            opacity=0.7,
        )


def plot_activation_distribution(
    evt: gr.EventData, current_image, clip_act, maple_act, model_name: str
):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["CLIP Activation", f"{model_name} Activation"],
    )

    fig_clip = plot_activation(
        evt, current_image, clip_act, "CLIP", colors=("#00b4d8", "#90e0ef")
    )
    fig_maple = plot_activation(
        evt, current_image, maple_act, model_name, colors=("#ff5a5f", "#ffcad4")
    )

    def _attach_fig(fig, sub_fig, row, col, yref):
        for trace in sub_fig.data:
            fig.add_trace(trace, row=row, col=col)

        for annotation in sub_fig.layout.annotations:
            annotation.update(yref=yref)
            fig.add_annotation(annotation)
        return fig

    fig = _attach_fig(fig, fig_clip, row=1, col=1, yref="y1")
    fig = _attach_fig(fig, fig_maple, row=2, col=1, yref="y2")

    fig.update_xaxes(title_text="SAE Latent Index", row=2, col=1)
    fig.update_xaxes(title_text="SAE Latent Index", row=1, col=1)
    fig.update_yaxes(title_text="Activation Value", row=1, col=1)
    fig.update_yaxes(title_text="Activation Value", row=2, col=1)
    fig.update_layout(
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig


def get_top_images(model_type, slider_value, toggle_btn):
    out_top_images = sae_tester[model_type].get_top_images(
        slider_value, top_k=5, show_seg_mask=toggle_btn
    )

    out_top_images = [plt_to_pil_direct(img) for img in out_top_images]
    return out_top_images


def get_segmask(image, sae_act, slider_value):
    temp = sae_act[:, slider_value]
    mask = torch.Tensor(temp[1:,].reshape(14, 14)).view(1, 1, 14, 14)
    mask = torch.nn.functional.interpolate(mask, (image.height, image.width))[0][
        0
    ].numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-10)

    base_opacity = 30
    image_array = np.array(image)[..., :3]
    rgba_overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    rgba_overlay[..., :3] = image_array[..., :3]

    darkened_image = (image_array[..., :3] * (base_opacity / 255)).astype(np.uint8)
    rgba_overlay[mask == 0, :3] = darkened_image[mask == 0]
    rgba_overlay[..., 3] = 255  # Fully opaque

    return rgba_overlay


def plt_to_pil_direct(fig):
    # Draw the canvas to render the figure
    fig.canvas.draw()

    # Convert the figure to a NumPy array
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    width, height = fig.canvas.get_width_height()
    image = data.reshape((height, width, 3))

    # Create a PIL Image from the NumPy array
    return Image.fromarray(image)


def show_segmentation_masks(
    selected_image, slider_value, sae_act, model_type, toggle_btn=False
):
    slider_value = int(slider_value.split("-")[-1])
    rgba_overlay = get_segmask(selected_image, sae_act, slider_value)
    top_images = get_top_images(model_type, slider_value, toggle_btn)

    act_values = []
    for dataset in REF_DATASET_LIST:
        act_value = sae_data_dict["mean_act_values"][dataset][slider_value, :5]
        act_value = [str(round(value.item(), 3)) for value in act_value]
        act_value = " | ".join(act_value)
        out = f"#### Activation values: {act_value}"
        act_values.append(out)

    return rgba_overlay, top_images, act_values


def load_results(resized_image, radio_choice, clip_act, maple_act, toggle_btn):
    if clip_act is None:
        return None, None, None, None, None, None, None

    init_seg, init_tops, init_values = show_segmentation_masks(
        resized_image, radio_choice, clip_act, "CLIP", toggle_btn
    )

    slider_value = int(radio_choice.split("-")[-1])
    maple_init_seg = get_segmask(resized_image, maple_act, slider_value)

    out = (init_seg, maple_init_seg)
    out += tuple(init_tops)
    out += tuple(init_values)
    return out


def load_image_and_act(image, clip_act, maple_act, model_name):
    resized_image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    sae_tester["CLIP"].register_image(resized_image)
    clip_act = sae_tester["CLIP"].get_activation_distribution()

    sae_tester[model_name].register_image(resized_image)
    maple_act = sae_tester[model_name].get_activation_distribution()

    neuron_plot = plot_activation_distribution(
        None, resized_image, clip_act, maple_act, model_name
    )

    radio_names = get_init_radio_options(clip_act, maple_act)
    radio_choices = gr.Radio(
        choices=radio_names, label="Top activating SAE latent", value=radio_names[0]
    )
    feautre_idx = radio_names[0].split("-")[-1]
    markdown_display = (
        f"## Segmentation mask for the selected SAE latent - {feautre_idx}"
    )

    return (
        resized_image,
        resized_image,
        neuron_plot,
        clip_act,
        maple_act,
        radio_choices,
        markdown_display,
    )


def highlight_grid(evt: gr.EventData, image, clip_act, maple_act, model_name):
    grid_x, grid_y, cell_width, cell_height = get_grid_loc(evt, image)

    highlighted_image = image.copy()
    draw = ImageDraw.Draw(highlighted_image)
    box = [
        grid_x * cell_width,
        grid_y * cell_height,
        (grid_x + 1) * cell_width,
        (grid_y + 1) * cell_height,
    ]
    draw.rectangle(box, outline="red", width=3)

    neuron_plot = plot_activation_distribution(
        evt, image, clip_act, maple_act, model_name
    )

    radio, choices = update_radio_options(clip_act, maple_act, grid_x, grid_y)
    feautre_idx = choices[0].split("-")[-1]
    markdown_display = (
        f"## Segmentation mask for the selected SAE latent - {feautre_idx}"
    )

    return (highlighted_image, neuron_plot, radio, markdown_display)


def get_init_radio_options(clip_act, maple_act):
    clip_neuron_dict = {}
    maple_neuron_dict = {}

    def _get_top_actvation(activations, neuron_dict, top_k=5):
        activations = activations.mean(0)
        top_neurons = list(np.argsort(activations)[::-1][:top_k])
        for top_neuron in top_neurons:
            neuron_dict[top_neuron] = activations[top_neuron]
        sorted_dict = dict(
            sorted(neuron_dict.items(), key=lambda item: item[1], reverse=True)
        )
        return sorted_dict

    clip_neuron_dict = _get_top_actvation(clip_act, clip_neuron_dict)
    maple_neuron_dict = _get_top_actvation(maple_act, maple_neuron_dict)

    radio_choices = get_radio_names(clip_neuron_dict, maple_neuron_dict)

    return radio_choices


def update_radio_options(clip_act, maple_act, grid_x, grid_y):
    def _sort_and_save_top_k(activations, neuron_dict, top_k=5):
        top_neurons = list(np.argsort(activations)[::-1][:top_k])
        for top_neuron in top_neurons:
            neuron_dict[top_neuron] = activations[top_neuron]

    def _get_top_actvation(activations, neuron_dict, token_idx):
        image_activation = activations.mean(0)
        _sort_and_save_top_k(image_activation, neuron_dict)

        tile_activations = activations[token_idx]
        _sort_and_save_top_k(tile_activations, neuron_dict)

        sorted_dict = dict(
            sorted(neuron_dict.items(), key=lambda item: item[1], reverse=True)
        )
        return sorted_dict

    token_idx = grid_y * GRID_NUM + grid_x + 1
    clip_neuron_dict = {}
    maple_neuron_dict = {}
    clip_neuron_dict = _get_top_actvation(clip_act, clip_neuron_dict, token_idx)
    maple_neuron_dict = _get_top_actvation(maple_act, maple_neuron_dict, token_idx)

    clip_keys = list(clip_neuron_dict.keys())
    maple_keys = list(maple_neuron_dict.keys())

    common_keys = list(set(clip_keys).intersection(set(maple_keys)))
    clip_only_keys = list(set(clip_keys) - (set(maple_keys)))
    maple_only_keys = list(set(maple_keys) - (set(clip_keys)))

    common_keys.sort(
        key=lambda x: max(clip_neuron_dict[x], maple_neuron_dict[x]), reverse=True
    )
    clip_only_keys.sort(reverse=True)
    maple_only_keys.sort(reverse=True)

    out = []
    out.extend([f"common-{i}" for i in common_keys[:5]])
    out.extend([f"CLIP-{i}" for i in clip_only_keys[:5]])
    out.extend([f"MaPLE-{i}" for i in maple_only_keys[:5]])

    radio_choices = gr.Radio(
        choices=out, label="Top activating SAE latent", value=out[0]
    )
    return radio_choices, out


def get_radio_names(clip_neuron_dict, maple_neuron_dict):
    clip_keys = list(clip_neuron_dict.keys())
    maple_keys = list(maple_neuron_dict.keys())

    common_keys = list(set(clip_keys).intersection(set(maple_keys)))
    clip_only_keys = list(set(clip_keys) - (set(maple_keys)))
    maple_only_keys = list(set(maple_keys) - (set(clip_keys)))

    common_keys.sort(
        key=lambda x: max(clip_neuron_dict[x], maple_neuron_dict[x]), reverse=True
    )
    clip_only_keys.sort(reverse=True)
    maple_only_keys.sort(reverse=True)

    out = []
    out.extend([f"common-{i}" for i in common_keys[:5]])
    out.extend([f"CLIP-{i}" for i in clip_only_keys[:5]])
    out.extend([f"MaPLE-{i}" for i in maple_only_keys[:5]])

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-imagenet",
        action="store_true",
        default=False,
        help="Include ImageNet in the Demo",
    )
    args = parser.parse_args()

    sae_tester = load_sae_tester("./data/sae_weight/base/out.pt", args.include_imagenet)
    sae_data_dict = {"mean_act_values": {}}
    if args.include_imagenet:
        REF_DATASET_LIST = ["imagenet", "imagenet-sketch", "caltech101"]
    else:
        REF_DATASET_LIST = ["imagenet-sketch", "caltech101"]
    for dataset in ["imagenet", "imagenet-sketch", "caltech101"]:
        data = torch.load(
            f"./out/feature_data/sae_base/base/{dataset}/max_activating_image_values.pt",
            map_location="cpu",
        )
        sae_data_dict["mean_act_values"][dataset] = data

    with gr.Blocks(
        theme=gr.themes.Citrus(),
        css="""
        .image-row .gr-image { margin: 0 !important; padding: 0 !important; }
        .image-row img { width: auto; height: 50px; } /* Set a uniform height for all images */
    """,
    ) as demo:
        with gr.Row():
            with gr.Column():
                # Left View: Image selection and click handling
                gr.Markdown("## Select input image and patch on the image")

                current_image = gr.State()
                clip_act = gr.State()
                maple_act = gr.State()

                image_display = gr.Image(type="pil", interactive=True)

            with gr.Column():
                gr.Markdown("## SAE latent activations of CLIP and MaPLE")
                model_options = [
                    f"MaPLE-{dataset_name}" for dataset_name in DATASET_LIST
                ]
                model_selector = gr.Dropdown(
                    choices=model_options,
                    value=model_options[0],
                    label="Select adapted model (MaPLe)",
                )

                neuron_plot = gr.Plot(label="Neuron Activation", show_label=False)

        with gr.Row():
            with gr.Column():
                markdown_display = gr.Markdown(
                    "## Segmentation mask for the selected SAE latent - "
                )
                gr.Markdown("### Localize SAE latent activation using CLIP")
                seg_mask_display = gr.Image(type="pil", show_label=False)

                gr.Markdown("### Localize SAE latent activation using MaPLE")
                seg_mask_display_maple = gr.Image(type="pil", show_label=False)

            with gr.Column():
                radio_choices = gr.Radio(
                    choices=[],
                    label="Top activating SAE latent",
                )
                toggle_btn = gr.Checkbox(label="Show segmentation mask", value=False)

                image_display_dict = {}
                activation_dict = {}
                for dataset in REF_DATASET_LIST:
                    image_display_dict[dataset] = gr.Image(
                        type="pil", label=dataset, show_label=False
                    )
                    activation_dict[dataset] = gr.Markdown("")

        image_display.upload(
            fn=load_image_and_act,
            inputs=[image_display, clip_act, maple_act, model_selector],
            outputs=[
                image_display,
                current_image,
                neuron_plot,
                clip_act,
                maple_act,
                radio_choices,
                markdown_display,
            ],
        )

        outputs = [seg_mask_display, seg_mask_display_maple]
        outputs += list(image_display_dict.values())
        outputs += list(activation_dict.values())

        radio_choices.change(
            fn=load_results,
            inputs=[current_image, radio_choices, clip_act, maple_act, toggle_btn],
            outputs=outputs,
        )

        toggle_btn.change(
            fn=load_results,
            inputs=[current_image, radio_choices, clip_act, maple_act, toggle_btn],
            outputs=outputs,
        )

        image_display.select(
            fn=highlight_grid,
            inputs=[current_image, clip_act, maple_act, model_selector],
            outputs=[
                image_display,
                neuron_plot,
                radio_choices,
                markdown_display,
            ],
        )

        radio_choices.change(
            fn=load_results,
            inputs=[current_image, radio_choices, clip_act, maple_act, toggle_btn],
            outputs=outputs,
        )

    demo.launch()
