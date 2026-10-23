"""
Maa Durga Python Sketch Generator
================================================================================
Author     : Saikat Barman
GitHub     : @saikatbarman07
Repository : https://github.com/saikatbarman07/Mata-python-sketch
License    : MIT License (Copyright (c) 2026 Saikat Barman)

Description:
    A modular computer vision and vector animation pipeline that transforms
    raster images of Maa Durga into continuous line sketches. The program
    extracts structural edge contours with OpenCV (Gaussian filtering, Canny
    hysteresis, and Douglas-Peucker curve simplification) and renders the
    artwork stroke-by-stroke using Python's standard Turtle graphics engine.
================================================================================
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import turtle
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ==============================================================================
# Default Configuration Constants
# ==============================================================================
BASE_DIR: Path = Path(__file__).resolve().parent
DEFAULT_IMAGE_NAME: str = "maa_durga_reference.png"
DEFAULT_IMAGE_PATH: Path = BASE_DIR / DEFAULT_IMAGE_NAME

DEFAULT_CANVAS_SIZE: int = 750
DEFAULT_BG_COLOR: str = "#FFFFFF"
DEFAULT_PEN_COLOR: str = "#1B1E22"
DEFAULT_PEN_SIZE: int = 1
DEFAULT_SPEED: int = 0
DEFAULT_THRESHOLD1: int = 100
DEFAULT_THRESHOLD2: int = 200
DEFAULT_MIN_LENGTH: int = 15
DEFAULT_EPSILON: float = 1.0


def parse_color_to_bgr(color_value: str) -> Tuple[int, int, int]:
    """
    Converts a color string (hexadecimal or standard name) to an OpenCV BGR tuple.

    Args:
        color_value: Hex string (e.g., '#FFFFFF', '#1B1E22') or color name.

    Returns:
        Tuple of (Blue, Green, Red) integers ranging from 0 to 255.
    """
    clean_value = color_value.strip().lstrip("#")
    if len(clean_value) == 6:
        try:
            r = int(clean_value[0:2], 16)
            g = int(clean_value[2:4], 16)
            b = int(clean_value[4:6], 16)
            return (b, g, r)
        except ValueError:
            pass

    named_colors = {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "gold": (0, 215, 255),
        "yellow": (0, 255, 255),
        "orange": (0, 165, 255),
        "gray": (128, 128, 128),
        "grey": (128, 128, 128),
    }
    return named_colors.get(color_value.strip().lower(), (0, 0, 0))


class SketchGenerator:
    """
    Computer vision and vector animation pipeline for sketching artwork.

    Extracts edge boundaries from an input illustration and recreates the
    sketch incrementally on an interactive Turtle graphics canvas.
    """

    def __init__(
        self,
        image_path: Path | str = DEFAULT_IMAGE_PATH,
        canvas_size: int = DEFAULT_CANVAS_SIZE,
        bg_color: str = DEFAULT_BG_COLOR,
        pen_color: str = DEFAULT_PEN_COLOR,
        pen_size: int = DEFAULT_PEN_SIZE,
        speed: int = DEFAULT_SPEED,
        threshold1: int = DEFAULT_THRESHOLD1,
        threshold2: int = DEFAULT_THRESHOLD2,
        min_contour_length: int = DEFAULT_MIN_LENGTH,
        epsilon: float = DEFAULT_EPSILON,
    ) -> None:
        """
        Initializes the sketch generator configuration.

        Args:
            image_path: Filesystem path to the input reference image.
            canvas_size: Square window dimension in pixels for display.
            bg_color: Canvas background color (hex string or color name).
            pen_color: Stroke ink color (hex string or color name).
            pen_size: Stroke line width in pixels.
            speed: Turtle animation speed (0=fastest batching, 1-10=stepped).
            threshold1: Lower gradient threshold for Canny edge detection.
            threshold2: Upper gradient threshold for Canny edge detection.
            min_contour_length: Minimum arc length to filter out micro-noise.
            epsilon: Tolerance parameter for Douglas-Peucker polygon approximation.
        """
        self.image_path: Path = Path(image_path).resolve()
        self.canvas_size: int = canvas_size
        self.bg_color: str = bg_color
        self.pen_color: str = pen_color
        self.pen_size: int = pen_size
        self.speed: int = speed
        self.threshold1: int = threshold1
        self.threshold2: int = threshold2
        self.min_contour_length: int = min_contour_length
        self.epsilon: float = epsilon

        self.resized_image: Optional[np.ndarray] = None
        self.edge_map: Optional[np.ndarray] = None
        self.contours: List[np.ndarray] = []

    def load_and_preprocess(self) -> np.ndarray:
        """
        Loads the reference image, resizes it, and prepares a smoothed grayscale matrix.

        Returns:
            Preprocessed grayscale image matrix ready for edge detection.

        Raises:
            FileNotFoundError: If the specified image path does not exist.
            ValueError: If OpenCV fails to decode the image file.
        """
        if not self.image_path.is_file():
            raise FileNotFoundError(
                f"Reference image not found at: '{self.image_path}'. "
                f"Please ensure '{DEFAULT_IMAGE_NAME}' exists or specify a valid path with --image."
            )

        source = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if source is None:
            raise ValueError(f"Unable to read or decode image file: {self.image_path}")

        # Standardize resolution to match canvas coordinate space
        self.resized_image = cv2.resize(
            source,
            (self.canvas_size, self.canvas_size),
            interpolation=cv2.INTER_AREA,
        )

        gray = cv2.cvtColor(self.resized_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        return blurred

    def extract_contours(self) -> List[np.ndarray]:
        """
        Performs Canny edge detection, extracts vector contours, and filters noise.

        Contours are simplified with the Douglas-Peucker algorithm and sorted
        in descending order of perimeter so primary structural outlines are
        drawn first.

        Returns:
            List of 2D NumPy arrays representing continuous (x, y) stroke paths.
        """
        preprocessed = self.load_and_preprocess()
        self.edge_map = cv2.Canny(preprocessed, self.threshold1, self.threshold2)

        raw_contours, _ = cv2.findContours(
            self.edge_map,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_NONE,
        )

        paths: List[Tuple[float, np.ndarray]] = []
        for contour in raw_contours:
            arc_length = cv2.arcLength(contour, closed=False)
            if arc_length < self.min_contour_length:
                continue

            simplified = cv2.approxPolyDP(
                contour,
                epsilon=self.epsilon,
                closed=False,
            ).reshape(-1, 2)

            if len(simplified) >= 2:
                paths.append((arc_length, simplified))

        # Sort descending so key structural strokes render before fine details
        paths.sort(key=lambda item: item[0], reverse=True)
        self.contours = [path for _, path in paths]
        return self.contours

    def export_preview(self, output_path: Path | str) -> Path:
        """
        Renders the extracted vector contours to an image file using OpenCV.

        Args:
            output_path: Target filesystem path for the output preview image.

        Returns:
            Resolved Path of the saved image file.
        """
        target = Path(output_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        bg_bgr = parse_color_to_bgr(self.bg_color)
        pen_bgr = parse_color_to_bgr(self.pen_color)

        canvas = np.full(
            (self.canvas_size, self.canvas_size, 3),
            bg_bgr,
            dtype=np.uint8,
        )

        for path in self.contours:
            for index in range(len(path) - 1):
                pt_start = tuple(map(int, path[index]))
                pt_end = tuple(map(int, path[index + 1]))
                cv2.line(
                    canvas,
                    pt_start,
                    pt_end,
                    pen_bgr,
                    self.pen_size,
                    cv2.LINE_AA,
                )

        cv2.imwrite(str(target), canvas)
        return target

    def _to_turtle_coordinates(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """
        Maps top-left origin pixel coordinates to Turtle center-origin coordinates.

        OpenCV Coordinate Space : (0, 0) is top-left, Y increases downward.
        Turtle Coordinate Space : (0, 0) is center,   Y increases upward.
        """
        half_dim = self.canvas_size / 2.0
        turtle_x = pixel_x - half_dim
        turtle_y = half_dim - pixel_y
        return (turtle_x, turtle_y)

    def render(self) -> None:
        """
        Initializes the Turtle graphics screen and draws all extracted contours
        in real time with smooth frame batching.
        """
        if not self.contours:
            self.extract_contours()

        screen = turtle.Screen()
        screen.setup(
            width=self.canvas_size + 40,
            height=self.canvas_size + 40,
        )
        screen.bgcolor(self.bg_color)
        screen.title("Maa Durga Sketch Generator | Saikat Barman (@saikatbarman07)")

        # Configure tracer for smooth real-time animation without freezing
        tracer_batch = 25 if self.speed == 0 else max(1, self.speed * 2)
        screen.tracer(tracer_batch, 0)

        pen = turtle.Turtle()
        pen.color(self.pen_color)
        pen.pensize(self.pen_size)
        pen.speed(self.speed)
        pen.shape("classic")

        print("=" * 65)
        print("Maa Durga Sketch Generator - Saikat Barman (@saikatbarman07)")
        print(f"Loaded reference : {self.image_path.name}")
        print(f"Extracted paths  : {len(self.contours):,} continuous contours")
        print(f"Canvas size      : {self.canvas_size} x {self.canvas_size} px")
        print(f"Palette settings : Pen={self.pen_color} | Background={self.bg_color}")
        print(f"Canny thresholds : Lower={self.threshold1} | Upper={self.threshold2}")
        print("=" * 65)
        print("Rendering sketch strokes... (Close window or press Ctrl+C to cancel)")

        points_drawn = 0
        try:
            for path in self.contours:
                start_x, start_y = self._to_turtle_coordinates(path[0][0], path[0][1])
                pen.penup()
                pen.goto(start_x, start_y)
                pen.pendown()

                for point in path[1:]:
                    tx, ty = self._to_turtle_coordinates(point[0], point[1])
                    pen.goto(tx, ty)
                    points_drawn += 1

                    if points_drawn % tracer_batch == 0:
                        screen.update()

            screen.update()
            pen.hideturtle()
            print(f"Artwork successfully completed! Total vertices rendered: {points_drawn:,}.")
            print("Click on the drawing window to exit.")
            screen.exitonclick()

        except turtle.Terminator:
            print("\nAnimation window closed by user.")
        except KeyboardInterrupt:
            print("\nSketch rendering interrupted by user.")


def build_argument_parser() -> argparse.ArgumentParser:
    """Builds the command-line argument parser for the sketch generator."""
    parser = argparse.ArgumentParser(
        prog="matadi.py",
        description="Maa Durga Sketch Generator - OpenCV Contour Extraction & Turtle Rendering by Saikat Barman.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i", "--image",
        type=Path,
        default=DEFAULT_IMAGE_PATH,
        help="Path to the input reference image.",
    )
    parser.add_argument(
        "--bg-color", "--bg",
        dest="bg_color",
        type=str,
        default=DEFAULT_BG_COLOR,
        help="Canvas background color (hex string or color name).",
    )
    parser.add_argument(
        "--pen-color", "--color",
        dest="pen_color",
        type=str,
        default=DEFAULT_PEN_COLOR,
        help="Sketch pen ink color (hex string or color name).",
    )
    parser.add_argument(
        "--pen-size",
        type=int,
        default=DEFAULT_PEN_SIZE,
        help="Pen stroke line width in pixels.",
    )
    parser.add_argument(
        "-s", "--speed",
        type=int,
        default=DEFAULT_SPEED,
        help="Turtle drawing speed (0 for fastest batching, 1-10 for stepped).",
    )
    parser.add_argument(
        "--threshold1",
        type=int,
        default=DEFAULT_THRESHOLD1,
        help="Lower hysteresis threshold for Canny edge detector.",
    )
    parser.add_argument(
        "--threshold2",
        type=int,
        default=DEFAULT_THRESHOLD2,
        help="Upper hysteresis threshold for Canny edge detector.",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=DEFAULT_MIN_LENGTH,
        help="Minimum contour perimeter to draw (filters noise speckles).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help="Douglas-Peucker simplification tolerance in pixels.",
    )
    parser.add_argument(
        "--canvas-size",
        type=int,
        default=DEFAULT_CANVAS_SIZE,
        help="Square canvas display dimension in pixels.",
    )
    parser.add_argument(
        "--save-preview",
        type=Path,
        default=None,
        metavar="OUTPUT_PATH",
        help="Save an anti-aliased image preview of the sketch to disk without opening GUI.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable interactive Turtle GUI window (useful when paired with --save-preview).",
    )

    return parser


def main() -> None:
    """CLI entry point for running the sketch generator."""
    parser = build_argument_parser()
    args = parser.parse_args()

    generator = SketchGenerator(
        image_path=args.image,
        canvas_size=args.canvas_size,
        bg_color=args.bg_color,
        pen_color=args.pen_color,
        pen_size=args.pen_size,
        speed=args.speed,
        threshold1=args.threshold1,
        threshold2=args.threshold2,
        min_contour_length=args.min_length,
        epsilon=args.epsilon,
    )

    generator.extract_contours()

    if args.save_preview:
        saved_file = generator.export_preview(args.save_preview)
        print(f"Sketch preview exported to: {saved_file}")

    if not args.no_gui:
        generator.render()


if __name__ == "__main__":
    main()
