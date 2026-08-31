import io
import os
import random

import numpy as np
import streamlit as st
import torch
from PIL import Image

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from sam2.sam2_image_predictor import SAM2ImagePredictor

from infer import (
    process_image_with_clipseg,
    apply_sam_and_overlay_masks,
    load_default_colors,
)

st.set_page_config(
    page_title="Text-Guided Image Segmentation",
    page_icon="◈",
    layout="wide",
)
#CUSTOM CSS
st.markdown(
    """
    <style>
        /* ---------- Main application ---------- */

        .stApp {
            background:
                radial-gradient(
                    circle at 10% 0%,
                    rgba(99, 102, 241, 0.14),
                    transparent 28%
                ),
                radial-gradient(
                    circle at 90% 10%,
                    rgba(168, 85, 247, 0.10),
                    transparent 25%
                ),
                #080b12;
            color: #f5f7fb;
        }

        html, body, [class*="css"] {
            font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
        }

        h1, h2, h3, h4 {
            font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif !important;
        }

        .block-container {
            max-width: 1200px;
            padding-top: 2.5rem;
            padding-bottom: 4rem;
        }

        /* ---------- Hero ---------- */

        .eyebrow {
            color: #a5b4fc;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            margin-bottom: 0.8rem;
        }

        .hero-title {
            font-size: clamp(2.3rem, 5vw, 4rem);
            font-weight: 700;
            line-height: 1.05;
            letter-spacing: -0.04em;
            margin: 0;
            background: linear-gradient(
                90deg,
                #ffffff 0%,
                #c7d2fe 50%,
                #d8b4fe 100%
            );
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero-text {
            max-width: 720px;
            color: #a1a9b8;
            font-size: 1rem;
            line-height: 1.7;
            margin-top: 1rem;
        }

        /* Hide Streamlit's top toolbar/header so it does not appear above the app. */
        header[data-testid="stHeader"],
        [data-testid="stHeader"] {
            display: none !important;
        }

        [data-testid="stToolbar"] {
            display: none !important;
        }

        /* Keep the app content close to the top after hiding the header. */
        /* ---------- Model badge ---------- */

        .model-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.45rem 0.8rem;
            border-radius: 999px;
            background: rgba(99, 102, 241, 0.12);
            border: 1px solid rgba(129, 140, 248, 0.25);
            color: #c7d2fe;
            font-size: 0.76rem;
            font-weight: 600;
            margin-bottom: 1.2rem;
        }

        .status-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            display: inline-block;
        }

        /* ---------- Large input container ---------- */

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(17, 24, 39, 0.82);
            border: 1px solid rgba(129, 140, 248, 0.28);
            border-radius: 20px;
            padding: 1.5rem 1.6rem;
            box-shadow: 0 18px 50px rgba(0, 0, 0, 0.25);
        }

        .input-heading {
            color: #f8fafc;
            font-size: 1.35rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .input-description {
            color: #8f99aa;
            font-size: 0.85rem;
            margin-bottom: 1.2rem;
        }

        .field-label {
            color: #e5e7eb;
            font-size: 0.9rem;
            font-weight: 650;
            margin-bottom: 0.45rem;
        }

        /* ---------- File uploader ---------- */

        div[data-testid="stFileUploader"] {
            background: rgba(30, 41, 59, 0.70);
            border: 1px solid rgba(129, 140, 248, 0.28);
            border-radius: 15px;
            padding: 0.75rem;
            margin-bottom: 1.2rem;
        }

        div[data-testid="stFileUploader"] section {
            background: rgba(15, 23, 42, 0.88);
            border: 1px dashed rgba(165, 180, 252, 0.45);
            border-radius: 11px;
        }

        div[data-testid="stFileUploader"] button {
            background: #6366f1;
            color: white;
            border: none;
            border-radius: 9px;
            font-weight: 600;
        }

        div[data-testid="stFileUploader"] button:hover {
            background: #7c3aed;
            color: white;
        }

        /* ---------- Text input ---------- */

        div[data-testid="stTextInput"] > div {
            background: rgba(30, 41, 59, 0.75);
            border: 1px solid rgba(129, 140, 248, 0.35);
            border-radius: 14px;
            padding: 0.15rem;
        }

        div[data-testid="stTextInput"] input {
            background: rgba(15, 23, 42, 0.90);
            color: #f8fafc;
            border: none;
            border-radius: 11px;
            padding: 0.75rem 1rem;
        }

        div[data-testid="stTextInput"] input::placeholder {
            color: #64748b;
        }

        div[data-testid="stTextInput"] input:focus {
            border: 1px solid rgba(129, 140, 248, 0.7);
            box-shadow: 0 0 0 1px rgba(129, 140, 248, 0.25);
        }

        /* ---------- Segment button ---------- */

        div[data-testid="stButton"] > button {
            width: 100%;
            min-height: 52px;
            margin-top: 0.8rem;
            border-radius: 12px;
            border: 1px solid rgba(165, 180, 252, 0.35);
            background: linear-gradient(
                135deg,
                #6366f1,
                #7c3aed
            );
            color: white;
            font-weight: 700;
            font-size: 0.95rem;
            transition: all 0.2s ease;
        }

        div[data-testid="stButton"] > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 30px rgba(99, 102, 241, 0.25);
        }

        div[data-testid="stButton"] > button:disabled {
            opacity: 0.45;
            transform: none;
            box-shadow: none;
        }

        /* ---------- Result containers ---------- */

        .result-label {
            color: #94a3b8;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.7rem;
        }

        /* ---------- Download button ---------- */

        .stDownloadButton > button {
            width: 100%;
            border-radius: 12px;
            background: rgba(30, 41, 59, 0.95);
            border: 1px solid rgba(148, 163, 184, 0.25);
            color: #e2e8f0;
            font-weight: 600;
        }

        .stDownloadButton > button:hover {
            border-color: rgba(129, 140, 248, 0.5);
        }

        /* ---------- Sidebar ---------- */

        [data-testid="stSidebar"] {
            background: #0b0f18;
            border-right: 1px solid rgba(255, 255, 255, 0.07);
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 2rem;
        }

        .sidebar-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #f8fafc;
            margin-bottom: 0.3rem;
        }

        .sidebar-subtitle {
            color: #7f8a9c;
            font-size: 0.78rem;
            line-height: 1.5;
        }

        .concept {
            padding: 0.8rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        }

        .concept:last-child {
            border-bottom: none;
        }

        .concept-title {
            color: #e5e7eb;
            font-size: 0.88rem;
            font-weight: 650;
            margin-bottom: 0.25rem;
        }

        .concept-text {
            color: #8f99aa;
            font-size: 0.76rem;
            line-height: 1.5;
        }

        .footer {
            text-align: center;
            color: #64748b;
            font-size: 0.72rem;
            padding-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner="Loading segmentation models...")
def load_models():
    processor = CLIPSegProcessor.from_pretrained(
        "CIDAS/clipseg-rd64-refined"
    )

    model = CLIPSegForImageSegmentation.from_pretrained(
        "CIDAS/clipseg-rd64-refined"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    sam_predictor = SAM2ImagePredictor.from_pretrained(
        "facebook/sam2.1-hiera-large",
        device=device,
    )

    return processor, model, sam_predictor



def parse_colors(colors_string, count):
    colors = [
        tuple(map(int, color.split(",")))
        for color in colors_string.split(";")
        if color.strip()
    ]

    if not colors:
        colors = [(255, 0, 0)]

    while len(colors) < count:
        colors.extend(colors)

    return colors[:count]

st.markdown(
    '<div class="hero">'
    '<div class="eyebrow">Computer Vision · Text Guided AI</div>'
    '<div class="hero-title">Text-Guided<br>Image Segmentation</div>'
    '<div class="hero-text">Describe an object in an image and generate a '
    'precise segmentation mask using a CLIPSeg and SAM2 pipeline.</div>'
    "</div>",
    unsafe_allow_html=True,
)

device_label = (
    "GPU acceleration available"
    if torch.cuda.is_available()
    else "Running on CPU"
)

device_color = (
    "#34d399"
    if torch.cuda.is_available()
    else "#fbbf24"
)

st.markdown(
    f'<div class="model-badge">'
    f'<span class="status-dot" style="background:{device_color};"></span>'
    f'{device_label}'
    f'</div>',
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown(
        '<div class="sidebar-title">Project Overview</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-subtitle">'
        "A text-guided image segmentation system that combines "
        "semantic localization with precise mask generation."
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    concepts = [
        (
            "Text-Guided Segmentation",
            "The user describes the target object using natural language "
            "instead of manually drawing a region.",
        ),
        (
            "CLIPSeg",
            "Uses the text prompt to identify image regions that are "
            "semantically related to the requested object.",
        ),
        (
            "SAM2",
            "Refines the approximate CLIPSeg locations into detailed "
            "object masks.",
        ),
        (
            "Two-Stage Pipeline",
            "CLIPSeg provides semantic guidance while SAM2 performs "
            "high-quality mask generation.",
        ),
        (
            "Multi-Prompt Support",
            'Multiple objects can be requested using comma-separated '
            'prompts such as "person, car, tree".',
        ),
        (
            "Mask Visualization",
            "The application displays the original image, final masks, "
            "prompt heatmaps, and individual object masks.",
        ),
    ]

    concepts_html = "".join(
        f'<div class="concept">'
        f'<div class="concept-title">{title}</div>'
        f'<div class="concept-text">{text}</div>'
        f'</div>'
        for title, text in concepts
    )

    st.markdown(concepts_html, unsafe_allow_html=True)

    st.divider()

    st.markdown(
        '<div style="color:#f8fafc; font-weight:600;">Pipeline</div>',
        unsafe_allow_html=True,
    )

    st.code(
        "Image\n↓\nCLIPSeg + Text Prompt\n↓\nPoint Selection\n↓\nSAM2\n↓\nSegmentation Mask",
        language="text",
    )


with st.container(border=True):
    st.markdown(
        '<div class="input-heading">Segment an Object</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="input-description">'
        "Upload an image, enter a text prompt, and generate the object mask."
        "</div>",
        unsafe_allow_html=True,
    )

    # ----- Upload image -----

    st.markdown(
        '<div class="field-label">Upload Image</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        label_visibility="collapsed",
    )

    # ----- Prompt -----

    st.markdown(
        '<div class="field-label">Enter Prompt</div>',
        unsafe_allow_html=True,
    )

    prompt_input = st.text_input(
        "Enter Prompt",
        placeholder="Example: tree, car, person",
        label_visibility="collapsed",
    )

    # ----- Segment button -----

    segment_button = st.button(
        "Segment Object",
        type="primary",
        disabled=not uploaded_file or not prompt_input.strip(),
        use_container_width=True,
    )


if segment_button:
    random.seed(24)
    np.random.seed(24)
    torch.manual_seed(24)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(24)

    processor, model, sam_predictor = load_models()

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    prompts = [
        prompt.strip()
        for prompt in prompt_input.split(",")
        if prompt.strip()
    ]

    colors = parse_colors(
        load_default_colors() or "255,0,0",
        len(prompts),
    )

    with st.spinner("Finding objects and generating masks..."):
        sam_predictor.set_image(image)

        all_coords, results = process_image_with_clipseg(
            image,
            prompts,
            processor,
            model,
            40,
            35,
            0.8,
            "gradient",
        )

        final_overlay, final_mask, binary_masks = (
            apply_sam_and_overlay_masks(
                image_np,
                all_coords,
                prompts,
                sam_predictor,
                colors,
                3,
            )
        )


    empty_prompts = [
        result["prompt"]
        for result, coords in zip(results, all_coords)
        if len(coords) == 0
    ]

    if empty_prompts:
        st.warning(
            "No confident region was found for: "
            + ", ".join(empty_prompts)
        )

    with st.container(border=True):
        st.subheader("Segmentation Result")

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown(
                '<div class="result-label">Original</div>',
                unsafe_allow_html=True,
            )

            st.image(
                image,
                use_container_width=True,
            )

        with col2:
            st.markdown(
                '<div class="result-label">Segmented</div>',
                unsafe_allow_html=True,
            )

            st.image(
                final_overlay,
                use_container_width=True,
            )


    with st.container(border=True):
        st.subheader("Prompt Heatmaps")

        heat_cols = st.columns(len(results))

        for col, result in zip(heat_cols, results):
            heatmap = (
                result["heatmap_resized"] * 255
            ).astype(np.uint8)

            with col:
                st.image(
                    heatmap,
                    caption=result["prompt"],
                    use_container_width=True,
                )

    

    if binary_masks:
        with st.container(border=True):
            st.subheader("Individual Masks")

            mask_cols = st.columns(len(binary_masks))

            for col, (prompt, mask_image) in zip(
                mask_cols,
                binary_masks.items(),
            ):
                with col:
                    st.image(
                        mask_image,
                        caption=prompt,
                        use_container_width=True,
                    )

    

    result_image = Image.fromarray(final_overlay)

    buffer = io.BytesIO()
    result_image.save(buffer, format="PNG")

    st.download_button(
        "Download Segmented Image",
        data=buffer.getvalue(),
        file_name="segmented_result.png",
        mime="image/png",
        use_container_width=True,
    )



st.markdown(
    '<div class="footer">'
    "CLIPSeg + SAM2 · Text-Guided Image Segmentation"
    "</div>",
    unsafe_allow_html=True,
)