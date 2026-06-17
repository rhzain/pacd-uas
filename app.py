from __future__ import annotations

import base64
from html import escape
from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image

from homography import (
    auto_detect_quadrilateral,
    correct_perspective,
    draw_polygon,
    order_points,
    project_image,
)

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:  # pragma: no cover - optional UI dependency
    streamlit_image_coordinates = None


CORNER_LABELS = ["top-left", "top-right", "bottom-right", "bottom-left"]
IMAGE_TYPES = ["png", "jpg", "jpeg", "bmp"]
FRAME_WIDTH = 860
FRAME_HEIGHT = 520


st.set_page_config(
    page_title="Homography Perspective Tool",
    layout="wide",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1280px;
            padding-top: 1.25rem;
            padding-bottom: 2rem;
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid #e5e7eb;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.45rem;
        }
        .section-note {
            color: #667085;
            font-size: 0.92rem;
            line-height: 1.4;
        }
        .image-frame {
            align-items: center;
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            display: flex;
            justify-content: center;
            overflow: hidden;
            width: 100%;
        }
        .image-frame img {
            display: block;
            max-height: 100%;
            max-width: 100%;
            object-fit: contain;
        }
        .image-caption {
            color: #667085;
            font-size: 0.86rem;
            margin-top: 0.35rem;
            min-height: 1.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_rgb(uploaded_file) -> np.ndarray:
    if hasattr(uploaded_file, "path"):
        image = Image.open(uploaded_file.path).convert("RGB")
    else:
        image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


def png_bytes(image_rgb: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def show_framed_image(image_rgb: np.ndarray, caption: str = "", height: int = FRAME_HEIGHT) -> None:
    encoded = base64.b64encode(png_bytes(image_rgb)).decode("ascii")
    safe_caption = escape(caption)
    st.markdown(
        f"""
        <div class="image-frame" style="height: min({height}px, 72vh); max-height: 72vh;">
            <img src="data:image/png;base64,{encoded}" alt="{safe_caption}">
        </div>
        <div class="image-caption">{safe_caption}</div>
        """,
        unsafe_allow_html=True,
    )


def resize_rgb(image_rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.array(Image.fromarray(image_rgb).resize(size, Image.Resampling.LANCZOS))


def interactive_canvas(image_rgb: np.ndarray) -> tuple[np.ndarray, float, int, int, int, int]:
    image_h, image_w = image_rgb.shape[:2]
    scale = min(FRAME_WIDTH / image_w, FRAME_HEIGHT / image_h)
    display_w = max(1, int(round(image_w * scale)))
    display_h = max(1, int(round(image_h * scale)))
    offset_x = (FRAME_WIDTH - display_w) // 2
    offset_y = (FRAME_HEIGHT - display_h) // 2

    canvas = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 248, dtype=np.uint8)
    canvas[offset_y : offset_y + display_h, offset_x : offset_x + display_w] = resize_rgb(
        image_rgb,
        (display_w, display_h),
    )

    if st.session_state.points:
        points = np.array(st.session_state.points, dtype=np.float32)
        display_points = points * scale + np.array([offset_x, offset_y], dtype=np.float32)
        canvas = draw_polygon(canvas, display_points)

    return canvas, scale, offset_x, offset_y, display_w, display_h


def ensure_state() -> None:
    st.session_state.setdefault("points", [])
    st.session_state.setdefault("detected_points", None)
    st.session_state.setdefault("last_click", None)
    st.session_state.setdefault("editor_version", 0)
    st.session_state.setdefault("background_signature", None)


def bump_editor_version() -> None:
    st.session_state.editor_version += 1


def reset_points() -> None:
    st.session_state.points = []
    st.session_state.detected_points = None
    st.session_state.last_click = None
    bump_editor_version()


def set_detected_points(points: np.ndarray) -> None:
    st.session_state.points = points.astype(int).tolist()
    st.session_state.detected_points = points.astype(int).tolist()
    st.session_state.last_click = None
    bump_editor_version()


def sync_uploaded_image(uploaded_file) -> None:
    signature = f"{uploaded_file.name}:{getattr(uploaded_file, 'size', 0)}"
    if st.session_state.background_signature != signature:
        reset_points()
        st.session_state.background_signature = signature


def point_count() -> int:
    return min(len(st.session_state.points), 4)


def selected_points() -> np.ndarray | None:
    if point_count() != 4:
        return None
    return np.array(st.session_state.points[:4], dtype=np.float32)


def point_editor(image_shape: tuple[int, int, int]) -> None:
    height, width = image_shape[:2]
    points = st.session_state.points[:4]
    version = st.session_state.editor_version

    cols = st.columns(4)
    edited: list[list[int]] = []
    for index in range(4):
        default_x = int(points[index][0]) if index < len(points) else 0
        default_y = int(points[index][1]) if index < len(points) else 0
        with cols[index]:
            st.markdown(f"**P{index + 1}**")
            x = st.number_input(
                "x",
                min_value=0,
                max_value=max(0, width - 1),
                value=min(default_x, max(0, width - 1)),
                key=f"x_{index}_{version}",
                label_visibility="collapsed",
            )
            y = st.number_input(
                "y",
                min_value=0,
                max_value=max(0, height - 1),
                value=min(default_y, max(0, height - 1)),
                key=f"y_{index}_{version}",
                label_visibility="collapsed",
            )
            edited.append([int(x), int(y)])

    if st.button("Terapkan koordinat", use_container_width=True):
        st.session_state.points = edited
        st.session_state.last_click = None
        bump_editor_version()
        st.rerun()


def clickable_selector(image_rgb: np.ndarray) -> None:
    preview, scale, offset_x, offset_y, display_w, display_h = interactive_canvas(image_rgb)

    if streamlit_image_coordinates is None:
        show_framed_image(preview)
        st.warning("Dependency klik gambar belum tersedia. Pakai editor koordinat.")
        return

    click = streamlit_image_coordinates(
        preview,
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        key="point_selector",
    )
    if click is None or point_count() >= 4:
        return

    raw_x = int(click["x"])
    raw_y = int(click["y"])
    inside_x = offset_x <= raw_x < offset_x + display_w
    inside_y = offset_y <= raw_y < offset_y + display_h
    if not inside_x or not inside_y:
        return

    xy = [
        int(round((raw_x - offset_x) / scale)),
        int(round((raw_y - offset_y) / scale)),
    ]
    if st.session_state.last_click == xy:
        return

    st.session_state.last_click = xy
    st.session_state.points.append(xy)
    bump_editor_version()
    st.rerun()


def show_matrix(matrix: np.ndarray) -> None:
    st.code(np.array2string(matrix, precision=4, suppress_small=True), language="text")
    st.markdown(
        """
        **Representasi Elemen Matriks Homography ($3 \\times 3$):**
        
        $$
        H = \\begin{bmatrix}
        h_{11} & h_{12} & h_{13} \\\\
        h_{21} & h_{22} & h_{23} \\\\
        h_{31} & h_{32} & h_{33}
        \\end{bmatrix}
        $$
        
        * **Transformasi Linear ($h_{11}, h_{12}, h_{21}, h_{22}$)**:
          * $h_{11}, h_{22}$: *Scaling* (skala ukuran objek pada sumbu X dan Y).
          * $h_{12}, h_{21}$: *Rotation* (rotasi) dan *Shearing* (kemiringan geser).
        * **Translasi / Pergeseran ($h_{13}, h_{23}$)**:
          * $h_{13}$: Pergeseran posisi secara horizontal (sumbu X).
          * $h_{23}$: Pergeseran posisi secara vertikal (sumbu Y).
        * **Proyeksi Perspektif ($h_{31}, h_{32}$)**:
          * Efek distorsi perspektif 3D (kemiringan bidang/efek jauh-dekat).
        * **Normalisasi Skala ($h_{33}$)**:
          * Nilai faktor skala normalisasi homogen (biasanya bernilai 1.0).
        """
    )


def render_points_table(points: np.ndarray | None) -> None:
    if points is None:
        return
    ordered = order_points(points).astype(int)
    st.dataframe(
        {
            "corner": CORNER_LABELS,
            "x": ordered[:, 0],
            "y": ordered[:, 1],
        },
        hide_index=True,
        use_container_width=True,
    )


def valid_quad(points: np.ndarray) -> bool:
    ordered = order_points(points)
    area = 0.5 * abs(
        np.dot(ordered[:, 0], np.roll(ordered[:, 1], -1))
        - np.dot(ordered[:, 1], np.roll(ordered[:, 0], -1))
    )
    return area > 50 and len(np.unique(ordered.astype(int), axis=0)) == 4


def sidebar_controls() -> tuple[object, str, str, dict]:
    with st.sidebar:
        st.text("Muhammad Raihan Rizky Zain - 140810230049")

        uploaded_background = st.file_uploader(
            "Gambar utama",
            type=IMAGE_TYPES,
        )

        st.divider()
        selection_mode = st.radio(
            "Seleksi area",
            ["Auto", "Manual"],
            horizontal=True,
        )
        action = st.radio(
            "Aksi",
            ["Koreksi", "Proyeksi"],
            horizontal=True,
        )

        action_options = {
            "overlay_file": None,
            "opacity": 1.0,
        }
        if action == "Koreksi":
            pass
        else:
            action_options["overlay_file"] = st.file_uploader(
                "Gambar overlay",
                type=IMAGE_TYPES,
                key="overlay_upload",
            )
        st.divider()
        with st.expander("ℹ️ Tentang Homography"):
            st.markdown(
                """
                **Homography** adalah matriks transformasi geometri ($3 \\times 3$) yang memetakan titik-titik dari satu bidang datar (2D) ke bidang datar lainnya.
                
                **Konsep Utama:**
                * **Minimal 4 Titik**: Membutuhkan 4 pasangan titik sudut korespondensi ($x, y$) untuk menghitung matriks transformasi.
                * **Persamaan**: $x' = H \\cdot x$ (di mana $H$ adalah matriks homography).
                
                **Penerapan:**
                * **Koreksi**: Meluruskan perspektif gambar miring menjadi tegak lurus dari depan.
                * **Proyeksi**: Menempelkan gambar baru (overlay) ke bidang miring gambar utama.
                """
            )

    return uploaded_background, selection_mode, action, action_options


def render_point_panel(image_shape: tuple[int, int, int]) -> None:
    count = point_count()
    st.metric("Titik", f"{count}/4")
    st.progress(count / 4)

    if st.session_state.points:
        raw_points = np.array(st.session_state.points[:4], dtype=int)
        st.dataframe(
            {
                "point": [f"P{i + 1}" for i in range(len(raw_points))],
                "x": raw_points[:, 0],
                "y": raw_points[:, 1],
            },
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.markdown('<p class="section-note">Belum ada titik aktif.</p>', unsafe_allow_html=True)

    cols = st.columns(2)
    with cols[0]:
        if st.button("Reset", use_container_width=True):
            reset_points()
            st.rerun()
    with cols[1]:
        if st.button("Urutkan", use_container_width=True, disabled=count != 4):
            st.session_state.points = order_points(np.array(st.session_state.points)).astype(int).tolist()
            bump_editor_version()
            st.rerun()

    with st.expander("Koordinat", expanded=False):
        point_editor(image_shape)


def render_auto_detection(image_rgb: np.ndarray) -> None:
    """Auto detection: input and result side-by-side, details below."""
    result = auto_detect_quadrilateral(image_rgb)

    # --- Side-by-side: input image | detected result ---
    col_input, col_result = st.columns(2, gap="medium")
    with col_input:
        st.markdown("**Gambar Input**")
        show_framed_image(image_rgb, f"{image_rgb.shape[1]}×{image_rgb.shape[0]} px")
    with col_result:
        st.markdown("**Hasil Deteksi**")
        show_framed_image(result.debug_image, result.message)

    # --- Details below the images ---
    if result.points is None:
        st.warning(result.message)
        return

    # Show edge map + coordinates in an expander
    with st.expander("Detail deteksi", expanded=True):
        detail_left, detail_right = st.columns([1, 1], gap="medium")
        with detail_left:
            st.markdown("**Edge Map**")
            edge_preview = draw_polygon(result.edge_image, result.points)
            show_framed_image(edge_preview, "Edge map dengan titik koordinat", height=320)
        with detail_right:
            st.markdown("**Koordinat Titik**")
            ordered = order_points(result.points).astype(int)
            st.dataframe(
                {
                    "Corner": CORNER_LABELS,
                    "x": ordered[:, 0],
                    "y": ordered[:, 1],
                },
                hide_index=True,
                use_container_width=True,
            )
            st.metric("Titik terdeteksi", "4/4")
            st.progress(1.0)

    # Auto-apply detected points immediately
    set_detected_points(result.points)


def render_area_workspace(image_rgb: np.ndarray, selection_mode: str) -> None:
    st.subheader("Area")

    if selection_mode == "Auto":
        render_auto_detection(image_rgb)
    else:
        # Manual mode: clickable selector + point panel side-by-side
        left, right = st.columns([1.45, 0.55], gap="large")
        with left:
            clickable_selector(image_rgb)
        with right:
            st.markdown("**Status**")
            render_point_panel(image_rgb.shape)

            points = selected_points()
            if points is None:
                return

            if not valid_quad(points):
                st.error("Area tidak valid.")
                return

            ordered_points = order_points(points)
            st.markdown("**Urutan final**")
            st.dataframe(
                {
                    "corner": CORNER_LABELS,
                    "x": ordered_points[:, 0].astype(int),
                    "y": ordered_points[:, 1].astype(int),
                },
                hide_index=True,
                use_container_width=True,
            )


def render_result(image_rgb: np.ndarray, action: str, action_options: dict) -> None:
    points = selected_points()
    st.divider()
    st.subheader("Hasil")

    if points is None:
        st.info("Pilih 4 titik area.")
        return
    if not valid_quad(points):
        st.error("Empat titik belum membentuk area yang valid.")
        return

    ordered_points = order_points(points)

    if action == "Koreksi":
        corrected, matrix = correct_perspective(image_rgb, ordered_points)
        result_image = corrected

        before_col, after_col = st.columns(2, gap="large")
        with before_col:
            show_framed_image(draw_polygon(image_rgb, ordered_points), "Before", height=420)
        with after_col:
            show_framed_image(result_image, "Corrected", height=420)
            st.download_button(
                "Download hasil koreksi",
                data=png_bytes(result_image),
                file_name="perspective_corrected.png",
                mime="image/png",
                use_container_width=True,
            )

    else:
        overlay_file = action_options["overlay_file"]
        if overlay_file is None:
            st.info("Upload gambar overlay di sidebar.")
            return

        overlay_rgb = load_rgb(overlay_file)
        result_image, matrix = project_image(
            image_rgb,
            overlay_rgb,
            ordered_points,
            opacity=action_options["opacity"],
        )

        overlay_col, result_col = st.columns(2, gap="large")
        with overlay_col:
            show_framed_image(overlay_rgb, "Overlay", height=420)
        with result_col:
            show_framed_image(result_image, "Projection", height=420)
            st.download_button(
                "Download hasil proyeksi",
                data=png_bytes(result_image),
                file_name="image_projection.png",
                mime="image/png",
                use_container_width=True,
            )

    with st.expander("Homography matrix"):
        show_matrix(matrix)


def main() -> None:
    ensure_state()
    inject_styles()

    uploaded_background, selection_mode, action, action_options = sidebar_controls()

    st.title("Implementasi Homography untuk Koreksi Perspektif dan Proyeksi Citra pada Bidang Datar")

    if uploaded_background is None:
        import os
        default_path = os.path.join(os.path.dirname(__file__), "example.jpg")
        if os.path.exists(default_path):
            class DefaultImageWrapper:
                def __init__(self, path):
                    self.path = path
                    self.name = os.path.basename(path)
                    self.size = os.path.getsize(path)
            uploaded_background = DefaultImageWrapper(default_path)
        else:
            st.info("Upload gambar utama dari sidebar.")
            return

    sync_uploaded_image(uploaded_background)
    image_rgb = load_rgb(uploaded_background)

    header_cols = st.columns([0.55, 0.15, 0.15, 0.15])
    with header_cols[0]:
        st.caption(uploaded_background.name)
    with header_cols[1]:
        st.metric("Width", image_rgb.shape[1])
    with header_cols[2]:
        st.metric("Height", image_rgb.shape[0])
    with header_cols[3]:
        st.metric("Channel", image_rgb.shape[2])

    render_area_workspace(image_rgb, selection_mode)
    render_result(image_rgb, action, action_options)


if __name__ == "__main__":
    main()
