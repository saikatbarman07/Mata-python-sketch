"""
================================================================================
Project     : Matadi Python Sketch - Maa Skandamata Artistic Generator
Author      : SAIKAT BARMAN
Repository  : https://github.com/saikatbarman07/Mata-python-sketch
License     : MIT License (Copyright (c) 2026 SAIKAT BARMAN)

Description :
    An advanced computer vision and interactive animation system that replicates
    the real-time artistic process of sketching and painting an illustration
    of Maa Skandamata.

Core Techniques Implemented:
    1. Computer Vision & Edge Processing:
       - Grayscale luminance transformation (cv2.cvtColor)
       - Gaussian Blur filter (3x3 kernel) for high-frequency noise suppression
       - Canny Edge Detection with dual hysteresis thresholds (65, 150)
       - Contour boundary retrieval (cv2.findContours with RETR_LIST)
       - Arc-length noise elimination (filters out small contours < 12 px)
       - Douglas-Peucker polygonal curve approximation (cv2.approxPolyDP, epsilon = 1.0)
       - Arc-length descending priority sorting (primary structural contours first)

    2. Color Quantization & Region Segmentation:
       - 7-bit custom bit-shift color quantization (256 discrete color bins)
         Formula: (R >> 5) * 32 + (G >> 5) * 4 + (B >> 6)
       - 8-way Connected Component Labeling with statistics (cv2.connectedComponentsWithStats)
       - Edge-touching boundary & luminous background heuristics (mean luminance >= 190)
       - Nearest-Neighbor interpolation label matrix mapping to 750x750 display

    3. Real-Time Interactive GUI & Animation Engine:
       - Non-blocking Tkinter scheduler (root.after)
       - In-memory Base64 PNG frame streaming (zero disk I/O latency)
       - Dual-mode simulated cursor (Golden pencil stylus for sketching,
         Saffron round paintbrush for coloring)
       - Interactive speed scaling (0.5x to 8.0x), pause/resume, and replay

Color Palette Specifications:
    - Outline Stroke Ink   : Charcoal / Graphite Black (BGR: 34, 30, 27 | HEX: #1b1e22)
    - Canvas Background    : Pure White (BGR/RGB: 255, 255, 255 | HEX: #ffffff)
    - Pencil Stylus Tip    : Golden Yellow (HEX: #f2c14e), Outline (HEX: #30251b)
    - Paintbrush Tool      : Saffron Terracotta (HEX: #d36b36), Outline (HEX: #4b2b1b)
    - Artwork Palette      : Divine Royal Yellow (Navratri Day 5 sacred color),
                             saffron silks, lotus pink, warm lion ochre, and celestial ivory
================================================================================
"""

import base64
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np


# =============================================================================
# Configuration & Constants
# =============================================================================
PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_PATH = PROJECT_DIR / "madadi picture.png"

# Display & Processing Resolutions
DISPLAY_SIZE = 750           # Active Tkinter animation canvas size (750x750 px)
SEGMENTATION_SIZE = 320      # Downscaled resolution for fast color quantization (320x320 px)
CANVAS_MARGIN = 20           # Outer canvas padding in pixels

# Color Palette Definitions
# Stroke ink: Charcoal / Graphite black in OpenCV BGR format (RGB: 27, 30, 34)
COLOR_OUTLINE_INK = (34, 30, 27)

# Cursor Styling
CURSOR_PENCIL_FILL = "#f2c14e"        # Warm golden yellow pencil body
CURSOR_PENCIL_OUTLINE = "#30251b"     # Deep bronze pencil outline
CURSOR_BRUSH_FILL = "#d36b36"         # Saffron terracotta paintbrush head
CURSOR_BRUSH_OUTLINE = "#4b2b1b"      # Rich mahogany brush outline


def prepare_outline_paths(reference):
    """
    Extracts, simplifies, and sorts continuous stroke paths from the reference image.

    Techniques Used:
        1. Grayscale Conversion: Transforms 3-channel color into luminance intensities.
        2. Gaussian Blur (3x3): Smooths noise without sacrificing critical edge details.
        3. Canny Edge Detection (thresholds: 65, 150): Detects sharp gradient boundaries.
        4. Contour Retrieval (cv2.RETR_LIST, CHAIN_APPROX_NONE): Gathers all edge curves.
        5. Noise Filtering: Removes contours with arc length < 12 pixels.
        6. Douglas-Peucker Simplification: Reduces point redundancy while preserving shape.
        7. Arc-Length Priority Sorting: Sorts paths descending so major contours are drawn first.
    """
    gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 65, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    paths = []

    for contour in contours:
        if cv2.arcLength(contour, False) < 12:
            continue
        path = cv2.approxPolyDP(contour, 1.0, False).reshape(-1, 2)
        if len(path) >= 2:
            paths.append((cv2.arcLength(contour, False), path))

    paths.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in paths]


def prepare_color_regions(reference):
    """
    Segments the image into cohesive color regions using bit-shift quantization
    and 8-way connected component analysis.

    Techniques Used:
        1. Resizing to 320x320 with cv2.INTER_AREA for uniform color clustering.
        2. 7-bit Quantization (256 color bins):
           - Red   : 3 bits (>> 5, weight 32)
           - Green : 3 bits (>> 5, weight 4)
           - Blue  : 2 bits (>> 6, weight 1)
        3. Connected Component Labeling: Groups adjacent pixels of identical quantized color.
        4. Background Heuristic: Detects boundary-touching components with mean intensity >= 190.
        5. Spatio-Semantic Painting Order: Paints foreground subjects first, background last,
           ordered top-to-bottom and left-to-right.
        6. Nearest-Neighbor Matrix Upscaling: Scales label map cleanly to 750x750 display size.
    """
    small = cv2.resize(
        cv2.cvtColor(reference, cv2.COLOR_BGR2RGB),
        (SEGMENTATION_SIZE, SEGMENTATION_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    quantized = (
        (small[:, :, 0] >> 5) * 32
        + (small[:, :, 1] >> 5) * 4
        + (small[:, :, 2] >> 6)
    )
    labels = np.zeros((SEGMENTATION_SIZE, SEGMENTATION_SIZE), dtype=np.int32)
    regions = []
    next_region_id = 1

    for color_id in np.unique(quantized):
        color_mask = (quantized == color_id).astype(np.uint8)
        count, component_labels, stats, _ = cv2.connectedComponentsWithStats(
            color_mask, connectivity=8
        )
        color_pixels = quantized == color_id
        labels[color_pixels] = component_labels[color_pixels] + next_region_id - 1

        for component_id in range(1, count):
            x, y, width, height, area = stats[component_id]
            touches_edge = (
                x == 0
                or y == 0
                or x + width == SEGMENTATION_SIZE
                or y + height == SEGMENTATION_SIZE
            )
            is_background = False
            if touches_edge and area >= SEGMENTATION_SIZE * 4:
                component_mask = (
                    component_labels[y : y + height, x : x + width]
                    == component_id
                ).astype(np.uint8)
                mean_color = cv2.mean(
                    small[y : y + height, x : x + width],
                    mask=component_mask,
                )[:3]
                is_background = min(mean_color) >= 190

            regions.append(
                (
                    next_region_id + component_id - 1,
                    int(x),
                    int(y),
                    int(width),
                    int(height),
                    int(area),
                    is_background,
                )
            )

        next_region_id += count - 1

    labels_at_display_size = cv2.resize(
        labels,
        (DISPLAY_SIZE, DISPLAY_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )
    regions.sort(key=lambda region: (region[6], region[2], region[1]))
    return labels_at_display_size, regions


class DrawingAnimation:
    """
    Manages the desktop GUI, interactive animation lifecycle, and dual-phase rendering.
    """

    def __init__(self, root, preview, outline_paths, color_regions):
        self.root = root
        self.preview = preview
        self.outline_paths = outline_paths
        self.color_regions = color_regions
        self.labels = color_regions[0]
        self.regions = color_regions[1]
        self.speed = tk.DoubleVar(value=5.0)
        self.status = tk.StringVar(value="Preparing the white canvas...")
        self.paused = False
        self.after_id = None
        self.phase = "outline"
        self.path_index = 0
        self.point_index = 0
        self.region_index = 0
        self.last_point = None
        self.outline_frame = np.full_like(preview, 255)
        self.color_frame = None

        root.title("Maa Skandamata - Drawing and Coloring | by SAIKAT BARMAN")
        root.resizable(False, False)

        # Tkinter Canvas with outer padding
        self.canvas = tk.Canvas(
            root,
            width=DISPLAY_SIZE + CANVAS_MARGIN * 2,
            height=DISPLAY_SIZE + CANVAS_MARGIN * 2,
            bg="white",
            highlightthickness=0,
        )
        self.canvas.pack(padx=10, pady=(10, 4))

        self.image = tk.PhotoImage(master=root)
        self.image_item = self.canvas.create_image(
            CANVAS_MARGIN, CANVAS_MARGIN, image=self.image, anchor="nw"
        )

        # Dynamic tool cursor (pencil stylus or paintbrush)
        self.cursor = self.canvas.create_polygon(
            0,
            0,
            0,
            0,
            0,
            0,
            fill=CURSOR_PENCIL_FILL,
            outline=CURSOR_PENCIL_OUTLINE,
            width=2,
            state="hidden",
        )

        # Control Panel UI
        controls = ttk.Frame(root, padding=(12, 4, 12, 10))
        controls.pack(fill="x")
        self.pause_button = ttk.Button(
            controls, text="Pause", command=self.toggle_pause
        )
        self.pause_button.pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Restart", command=self.restart).pack(
            side="left", padx=(0, 16)
        )
        ttk.Label(controls, text="Speed").pack(side="left", padx=(0, 6))
        ttk.Scale(
            controls,
            from_=0.5,
            to=8.0,
            variable=self.speed,
            orient="horizontal",
            length=180,
        ).pack(side="left", padx=(0, 12))
        ttk.Label(controls, textvariable=self.status, width=36).pack(
            side="right"
        )

        # Keyboard shortcuts
        root.bind("<space>", lambda _event: self.toggle_pause())
        root.bind("<KeyPress-r>", lambda _event: self.restart())
        root.bind("<KeyPress-R>", lambda _event: self.restart())

        self._show_frame(self.outline_frame)
        self._set_cursor(0, 0, brush=False)
        self._schedule(50)

    def _show_frame(self, frame):
        """In-memory streaming of OpenCV image frames into Tkinter PhotoImage."""
        success, encoded = cv2.imencode(".png", frame)
        if not success:
            raise OSError("Could not update the animation frame.")
        self.image.configure(data=base64.b64encode(encoded.tobytes()))
        self.canvas.tag_raise(self.cursor)

    def _set_cursor(self, x, y, brush):
        """Renders either a round paintbrush cursor or an angled pencil stylus."""
        x += CANVAS_MARGIN
        y += CANVAS_MARGIN
        if brush:
            radius = 8
            self.canvas.coords(
                self.cursor,
                x - radius,
                y - radius,
                x + radius,
                y - radius,
                x + radius,
                y + radius,
                x - radius,
                y + radius,
            )
            self.canvas.itemconfigure(
                self.cursor,
                fill=CURSOR_BRUSH_FILL,
                outline=CURSOR_BRUSH_OUTLINE,
                width=3,
                state="normal",
            )
        else:
            self.canvas.coords(
                self.cursor,
                x - 8,
                y + 10,
                x - 3,
                y - 2,
                x + 7,
                y - 10,
                x + 3,
                y + 2,
            )
            self.canvas.itemconfigure(
                self.cursor,
                fill=CURSOR_PENCIL_FILL,
                outline=CURSOR_PENCIL_OUTLINE,
                width=2,
                state="normal",
            )

    def _schedule(self, delay):
        if not self.paused and self.after_id is None:
            self.after_id = self.root.after(delay, self._advance)

    def _advance(self):
        self.after_id = None
        if self.paused:
            return
        if self.phase == "outline":
            self._advance_outline()
        elif self.phase == "color":
            self._advance_color()

    def _advance_outline(self):
        """Draws contour strokes incrementally with adaptive stroke thickness."""
        speed = self.speed.get()
        point_budget = max(8, int(100 * speed))
        points_drawn = 0

        while self.path_index < len(self.outline_paths) and points_drawn < point_budget:
            path = self.outline_paths[self.path_index]
            if self.point_index == 0:
                self.last_point = path[0]
                self.point_index = 1

            while self.point_index < len(path) and points_drawn < point_budget:
                point = path[self.point_index]
                # Adaptive line thickness based on total contour length
                thickness = 1 if len(path) < 50 else 2 if len(path) < 250 else 3
                cv2.line(
                    self.outline_frame,
                    tuple(map(int, self.last_point)),
                    tuple(map(int, point)),
                    COLOR_OUTLINE_INK,
                    thickness,
                    cv2.LINE_AA,
                )
                self._set_cursor(int(point[0]), int(point[1]), brush=False)
                self.last_point = point
                self.point_index += 1
                points_drawn += 1

            if self.point_index >= len(path):
                self.path_index += 1
                self.point_index = 0
                self.last_point = None

        self._show_frame(self.outline_frame)
        if self.path_index >= len(self.outline_paths):
            self.phase = "color"
            self.color_frame = self.outline_frame.copy()
            self.status.set("Outline finished - painting color regions")
            self._schedule(100)
            return

        self.status.set(
            f"Drawing outlines: {self.path_index:,} / {len(self.outline_paths):,} strokes"
        )
        self._schedule(max(1, int(12 / speed)))

    def _advance_color(self):
        """Paints segmented color regions batch-by-batch over the outline frame."""
        speed = self.speed.get()
        region_budget = max(8, int(60 * speed))
        stop = min(self.region_index + region_budget, len(self.regions))

        while self.region_index < stop:
            region_id, x, y, width, height, _area, _background = self.regions[
                self.region_index
            ]
            x0 = x * DISPLAY_SIZE // SEGMENTATION_SIZE
            y0 = y * DISPLAY_SIZE // SEGMENTATION_SIZE
            x1 = (x + width) * DISPLAY_SIZE // SEGMENTATION_SIZE + 1
            y1 = (y + height) * DISPLAY_SIZE // SEGMENTATION_SIZE + 1
            x1 = min(x1, DISPLAY_SIZE)
            y1 = min(y1, DISPLAY_SIZE)
            region_mask = self.labels[y0:y1, x0:x1] == region_id
            color_area = self.color_frame[y0:y1, x0:x1]
            reference_area = self.preview[y0:y1, x0:x1]
            color_area[region_mask] = reference_area[region_mask]
            self._set_cursor((x0 + x1) // 2, (y0 + y1) // 2, brush=True)
            self.region_index += 1

        self._show_frame(self.color_frame)
        if self.region_index >= len(self.regions):
            self._show_frame(self.preview)
            self.canvas.itemconfigure(self.cursor, state="hidden")
            self.phase = "done"
            self.status.set("Complete - artwork fully rendered")
            self.pause_button.configure(text="Replay")
            return

        self.status.set(
            f"Painting color regions: {self.region_index:,} / {len(self.regions):,}"
        )
        self._schedule(max(1, int(15 / speed)))

    def toggle_pause(self):
        """Toggles between paused and active drawing animation."""
        if self.phase == "done":
            self.restart()
            return
        self.paused = not self.paused
        if self.paused:
            if self.after_id is not None:
                self.root.after_cancel(self.after_id)
                self.after_id = None
            self.pause_button.configure(text="Resume")
            self.status.set("Paused")
        else:
            self.pause_button.configure(text="Pause")
            self._schedule(1)

    def restart(self):
        """Resets the canvas and restarts the animation from stroke 0."""
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.paused = False
        self.phase = "outline"
        self.path_index = 0
        self.point_index = 0
        self.region_index = 0
        self.last_point = None
        self.outline_frame = np.full_like(self.preview, 255)
        self.color_frame = None
        self.pause_button.configure(text="Pause")
        self.status.set("Drawing outlines...")
        self._show_frame(self.outline_frame)
        self._set_cursor(0, 0, brush=False)
        self._schedule(50)


def main():
    source = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)
    if source is None:
        raise FileNotFoundError(f"Could not load the reference image: {IMAGE_PATH}")

    preview = cv2.resize(
        source,
        (DISPLAY_SIZE, DISPLAY_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    outline_paths = prepare_outline_paths(preview)
    color_regions = prepare_color_regions(preview)

    root = tk.Tk()
    DrawingAnimation(root, preview, outline_paths, color_regions)
    print("=" * 60)
    print("Matadi Python Sketch - Created by SAIKAT BARMAN")
    print(f"Loaded reference picture: {IMAGE_PATH.name} ({source.shape[1]} x {source.shape[0]})")
    print(f"Prepared {len(outline_paths):,} outline strokes.")
    print(f"Prepared {len(color_regions[1]):,} color regions.")
    print("=" * 60)
    root.mainloop()


if __name__ == "__main__":
    main()
