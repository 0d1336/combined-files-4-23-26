"""Recover obscured content from a PDF by rebuilding pages around dark overlays."""

from __future__ import annotations

import argparse
import importlib
import logging
import re
from pathlib import Path
from typing import Any, Literal


def load_pymupdf() -> Any:
    """Import PyMuPDF with a clear dependency error for local CLI use."""
    try:
        return importlib.import_module("pymupdf")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyMuPDF is required to run recover.py. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc


fitz = load_pymupdf()

LOGGER = logging.getLogger(__name__)

FitzDocument = Any
FitzPage = Any
FitzPixmap = Any
FitzRect = Any
RectCoords = tuple[float, float, float, float]

FITZ_MATRIX = getattr(fitz, "Matrix")
FITZ_OPEN = getattr(fitz, "open")
FITZ_POINT = getattr(fitz, "Point")
FITZ_RECT = getattr(fitz, "Rect")
TEXT_ALIGN_LEFT = getattr(fitz, "TEXT_ALIGN_LEFT")

DEFAULT_INPUT_PDF = Path("Combined-Files-4-23-26.pdf")
DEFAULT_OUTPUT_PDF = Path("Combined-Files-4-23-26.cleaned.pdf")
MIN_DARK_OVERLAY_AVERAGE = 0.35
MIN_DARK_OVERLAY_WIDTH = 20
MIN_DARK_OVERLAY_HEIGHT = 6
REGION_PAD_X = 0.6
REGION_PAD_Y = 0.2
BLANK_NON_WHITE_FRACTION_MAX = 0.015
FAILED_RECOVERY_DARK_FRACTION_MIN = 0.18
FAILED_RECOVERY_DARK_RATIO = 0.75
MIN_TEXT_WORD_INTERSECTION_RATIO = 0.5
PAGE4_EXTRA_PATTERNS = (
    r'^<path transform="matrix\(\.75,0,0,\.75,487\.44,602\.4\)" d="M0 0H67\.84V13\.76H0Z"/>\n?',
    r'^<path transform="matrix\(\.75,0,0,\.75,72\.72,616\.19998\)" d="M0 0H62\.88V13\.76H0Z"/>\n?',
    r'^<path transform="matrix\(\.75,0,0,\.75,150\.36,626\.4\)" d="M0 0H-104\.48-40\.8 0ZM0-13\.6H-40\.8V-16\.32H0V-13\.6Z"/>\n?',
    r'^<path transform="matrix\(\.75,0,0,\.75,119\.759998,626\.4\)" d="M0 0H-63\.68V-16\.32H0V-13\.6H-62\.72V0H0Z"/>\n?',
)
PAGE5_EXTRA_PATTERNS = (
    r'^<path transform="matrix\(\.75,0,0,\.75,156\.36,397\.19999\)" d="M0 0H386\.56V13\.92H0Z"(?: [^>]*)?/>\n?',
    r'^<path transform="matrix\(\.75,0,0,\.75,126\.36,425\.40003\)" d="M0 0H218\.72V13\.76H0Z"(?: [^>]*)?/>\n?',
)

RecoveryTechnique = Literal["svg-clip", "svg-page", "text-words"]
DEFAULT_TECHNIQUE: RecoveryTechnique = "svg-clip"
PAGE_TECHNIQUE_OVERRIDES: dict[int, RecoveryTechnique] = {
    5: "svg-page",
    49: "svg-page",
    50: "svg-page",
    **{page_number: "text-words" for page_number in range(76, 83)},
}
PAGE_EXTRA_REDACTION_RECTS: dict[int, tuple[RectCoords, ...]] = {
    26: ((492.12, 158.64, 507.36, 174.96),),
}
PAGE_IGNORED_REDACTION_RECTS: dict[int, tuple[RectCoords, ...]] = {
    49: ((614.64, 517.32, 686.64, 571.32),),
}
PAGE_ORIGINAL_OVERLAY_RECTS: dict[int, tuple[RectCoords, ...]] = {
    49: ((614.64, 517.32, 686.64, 571.32),),
}


def positive_scale(value: str) -> float:
    scale = float(value)
    if scale <= 0:
        raise argparse.ArgumentTypeError("Scale must be greater than 0.")
    return scale


def normalize_paths(input_pdf: Path, output_pdf: Path) -> tuple[Path, Path]:
    normalized_input = input_pdf.expanduser()
    normalized_output = output_pdf.expanduser()

    if not normalized_input.is_file():
        raise FileNotFoundError(f"Input PDF not found: {normalized_input}")

    if normalized_input.resolve(strict=False) == normalized_output.resolve(strict=False):
        raise ValueError("Output PDF must differ from the input PDF.")

    return normalized_input, normalized_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a full-document cleaned PDF copy by replacing detected redaction overlays "
            "with recovered content when available."
        )
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_PDF,
        help="Input PDF to transform.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PDF,
        help="Output PDF path.",
    )
    parser.add_argument(
        "--scale",
        type=positive_scale,
        default=2.0,
        help="Rasterization scale used when rebuilding each page.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def remove_dark_overlay(svg_text: str) -> str:
    filtered = re.sub(r'^.*fill="#4d4d4f".*\n?', "", svg_text, flags=re.M)
    return re.sub(r'^.*stroke="#4d4d4f".*\n?', "", filtered, flags=re.M)


def remove_page4_default_black_rectangles(svg_text: str) -> str:
    reconstructed = svg_text
    for pattern in PAGE4_EXTRA_PATTERNS:
        reconstructed = re.sub(pattern, "", reconstructed, flags=re.M)
    return reconstructed


def remove_page5_default_black_rectangles(svg_text: str) -> str:
    reconstructed = svg_text
    for pattern in PAGE5_EXTRA_PATTERNS:
        reconstructed = re.sub(pattern, "", reconstructed, flags=re.M)
    return reconstructed


def get_page_technique(page_number: int) -> RecoveryTechnique:
    return PAGE_TECHNIQUE_OVERRIDES.get(page_number, DEFAULT_TECHNIQUE)


def looks_like_redaction(fill: tuple[float, ...] | None, rect: FitzRect | None) -> bool:
    if fill is None or rect is None:
        return False

    return (
        sum(fill) / len(fill) < MIN_DARK_OVERLAY_AVERAGE
        and rect.width > MIN_DARK_OVERLAY_WIDTH
        and rect.height > MIN_DARK_OVERLAY_HEIGHT
    )


def rect_is_ignored(page_number: int, rect: FitzRect) -> bool:
    for ignored_rect_values in PAGE_IGNORED_REDACTION_RECTS.get(page_number, ()):
        ignored_rect = FITZ_RECT(ignored_rect_values)
        intersection = rect & ignored_rect
        if intersection.width <= 0 or intersection.height <= 0:
            continue

        rect_area = rect.get_area()
        if rect_area <= 0:
            continue

        if intersection.get_area() / rect_area >= 0.8:
            return True

    return False


def detect_redaction_rects(page: FitzPage, page_number: int) -> list[FitzRect]:
    redaction_rects: list[FitzRect] = []
    seen_rects: set[tuple[float, float, float, float]] = set()

    for drawing in page.get_drawings():
        rect = drawing.get("rect")
        fill = drawing.get("fill")
        if not looks_like_redaction(fill, rect):
            continue

        candidate_rect = FITZ_RECT(rect)
        if rect_is_ignored(page_number, candidate_rect):
            continue

        key = (
            round(candidate_rect.x0, 2),
            round(candidate_rect.y0, 2),
            round(candidate_rect.x1, 2),
            round(candidate_rect.y1, 2),
        )
        if key in seen_rects:
            continue

        seen_rects.add(key)
        redaction_rects.append(candidate_rect)

    for rect_values in PAGE_EXTRA_REDACTION_RECTS.get(page_number, ()):
        candidate_rect = FITZ_RECT(rect_values)
        key = (
            round(candidate_rect.x0, 2),
            round(candidate_rect.y0, 2),
            round(candidate_rect.x1, 2),
            round(candidate_rect.y1, 2),
        )
        if key in seen_rects:
            continue

        seen_rects.add(key)
        redaction_rects.append(candidate_rect)

    return redaction_rects


def recover_page_svg(page_number: int, page: FitzPage) -> str:
    recovered_svg = remove_dark_overlay(page.get_svg_image())
    if page_number == 4:
        recovered_svg = remove_page4_default_black_rectangles(recovered_svg)
    if page_number == 5:
        recovered_svg = remove_page5_default_black_rectangles(recovered_svg)
    return recovered_svg


def open_recovered_document(page_number: int, page: FitzPage) -> FitzDocument:
    recovered_svg = recover_page_svg(page_number, page)
    return FITZ_OPEN("svg", recovered_svg.encode("utf-8"))


def render_clip_pixmap(page: FitzPage, clip_rect: FitzRect, scale: float) -> FitzPixmap:
    return page.get_pixmap(matrix=FITZ_MATRIX(scale, scale), clip=clip_rect, alpha=False)


def render_page_pixmap(page: FitzPage, scale: float) -> FitzPixmap:
    return page.get_pixmap(matrix=FITZ_MATRIX(scale, scale), alpha=False)


def expand_rect(rect: FitzRect, page_rect: FitzRect) -> FitzRect:
    expanded = FITZ_RECT(
        rect.x0 - REGION_PAD_X,
        rect.y0 - REGION_PAD_Y,
        rect.x1 + REGION_PAD_X,
        rect.y1 + REGION_PAD_Y,
    )
    return expanded & page_rect


def pixmap_fractions(pixmap: FitzPixmap) -> tuple[float, float]:
    channels = pixmap.n
    samples = pixmap.samples
    pixel_count = pixmap.width * pixmap.height
    if pixel_count == 0:
        return 0.0, 0.0

    dark_pixels = 0
    non_white_pixels = 0
    for index in range(0, len(samples), channels):
        average = (samples[index] + samples[index + 1] + samples[index + 2]) / 3.0
        if average < 40:
            dark_pixels += 1
        if average < 245:
            non_white_pixels += 1

    return dark_pixels / pixel_count, non_white_pixels / pixel_count


def recovered_clip_is_usable(original_clip: FitzPixmap, recovered_clip: FitzPixmap) -> bool:
    original_dark_fraction, _ = pixmap_fractions(original_clip)
    recovered_dark_fraction, recovered_non_white_fraction = pixmap_fractions(recovered_clip)

    if recovered_non_white_fraction <= BLANK_NON_WHITE_FRACTION_MAX:
        return False

    dark_ratio_limit = max(
        FAILED_RECOVERY_DARK_FRACTION_MIN,
        original_dark_fraction * FAILED_RECOVERY_DARK_RATIO,
    )
    if recovered_dark_fraction >= dark_ratio_limit:
        return False

    return True


def extract_words_for_rect(page: FitzPage, rect: FitzRect) -> list[tuple[FitzRect, str]]:
    recovered_words: list[tuple[FitzRect, str]] = []
    seen_words: set[tuple[float, float, float, float, str]] = set()

    for word in page.get_text("words"):
        word_rect = FITZ_RECT(word[:4])
        intersection = word_rect & rect
        if intersection.width <= 0 or intersection.height <= 0:
            continue

        word_area = word_rect.get_area()
        if word_area <= 0:
            continue

        if intersection.get_area() / word_area < MIN_TEXT_WORD_INTERSECTION_RATIO:
            continue

        text = word[4]
        key = (
            round(word_rect.x0, 2),
            round(word_rect.y0, 2),
            round(word_rect.x1, 2),
            round(word_rect.y1, 2),
            text,
        )
        if key in seen_words:
            continue

        seen_words.add(key)
        recovered_words.append((word_rect, text))

    return recovered_words


def draw_recovered_word(page: FitzPage, word_rect: FitzRect, text: str) -> None:
    text_rect = FITZ_RECT(
        word_rect.x0,
        word_rect.y0 - 0.5,
        word_rect.x1 + 1.0,
        word_rect.y1 + 0.5,
    )
    font_size = max(6.0, word_rect.height * 0.76)
    result = page.insert_textbox(
        text_rect,
        text,
        fontsize=font_size,
        fontname="helv",
        color=(0.0, 0.0, 0.0),
        align=TEXT_ALIGN_LEFT,
        overlay=True,
    )
    if result < 0:
        page.insert_text(
            FITZ_POINT(text_rect.x0, text_rect.y1 - 1.0),
            text,
            fontsize=max(5.5, word_rect.height * 0.7),
            fontname="helv",
            color=(0.0, 0.0, 0.0),
            overlay=True,
        )


def replace_redaction_region_with_text(
    output_page: FitzPage,
    source_page: FitzPage,
    rect: FitzRect,
) -> None:
    replacement_rect = expand_rect(rect, source_page.rect)
    if replacement_rect.width <= 0 or replacement_rect.height <= 0:
        return

    clear_rect(output_page, replacement_rect)
    for word_rect, text in extract_words_for_rect(source_page, replacement_rect):
        draw_recovered_word(output_page, word_rect, text)


def clear_rect(page: FitzPage, rect: FitzRect) -> None:
    page.draw_rect(
        rect,
        color=(1.0, 1.0, 1.0),
        fill=(1.0, 1.0, 1.0),
        width=0,
        overlay=True,
        stroke_opacity=1.0,
        fill_opacity=1.0,
    )


def overlay_original_regions(
    output_page: FitzPage,
    source_page: FitzPage,
    page_number: int,
    scale: float,
) -> None:
    for rect_values in PAGE_ORIGINAL_OVERLAY_RECTS.get(page_number, ()):
        overlay_rect = FITZ_RECT(rect_values) & source_page.rect
        if overlay_rect.width <= 0 or overlay_rect.height <= 0:
            continue

        original_clip = render_clip_pixmap(source_page, overlay_rect, scale)
        try:
            output_page.insert_image(overlay_rect, pixmap=original_clip)
        finally:
            original_clip = None


def replace_redaction_region(
    output_page: FitzPage,
    source_page: FitzPage,
    recovered_page: FitzPage,
    rect: FitzRect,
    scale: float,
    technique: RecoveryTechnique,
) -> None:
    if technique == "text-words":
        replace_redaction_region_with_text(output_page, source_page, rect)
        return

    replacement_rect = expand_rect(rect, source_page.rect)
    if replacement_rect.width <= 0 or replacement_rect.height <= 0:
        return

    original_clip = render_clip_pixmap(source_page, replacement_rect, scale)
    recovered_clip = render_clip_pixmap(recovered_page, replacement_rect, scale)
    try:
        if recovered_clip_is_usable(original_clip, recovered_clip):
            output_page.insert_image(replacement_rect, pixmap=recovered_clip)
        else:
            clear_rect(output_page, replacement_rect)
    finally:
        original_clip = None
        recovered_clip = None


def build_cleaned_document(input_pdf: Path, output_pdf: Path, scale: float) -> None:
    source_document = FITZ_OPEN(input_pdf)
    output_document = FITZ_OPEN()

    try:
        for page_index in range(source_document.page_count):
            source_page = source_document[page_index]
            page_number = page_index + 1
            technique = get_page_technique(page_number)
            redaction_rects = detect_redaction_rects(source_page, page_number)
            LOGGER.info("Page %s: detected %s redaction region(s)", page_number, len(redaction_rects))

            pixmap = render_page_pixmap(source_page, scale)
            needs_svg_recovery = technique == "svg-page" or (technique == "svg-clip" and bool(redaction_rects))
            recovered_document = open_recovered_document(page_number, source_page) if needs_svg_recovery else None

            try:
                output_page = output_document.new_page(
                    width=source_page.rect.width,
                    height=source_page.rect.height,
                )
                recovered_page = recovered_document[0] if recovered_document is not None else None

                if technique == "svg-page" and recovered_page is not None:
                    recovered_pixmap = render_page_pixmap(recovered_page, scale)
                    try:
                        output_page.insert_image(output_page.rect, pixmap=recovered_pixmap)
                    finally:
                        recovered_pixmap = None
                    overlay_original_regions(output_page, source_page, page_number, scale)
                    continue

                output_page.insert_image(output_page.rect, pixmap=pixmap)

                for rect in redaction_rects:
                    if recovered_page is not None:
                        replace_redaction_region(
                            output_page,
                            source_page,
                            recovered_page,
                            rect,
                            scale,
                            technique,
                        )
                    else:
                        replace_redaction_region_with_text(output_page, source_page, rect)
            finally:
                pixmap = None
                if recovered_document is not None:
                    recovered_document.close()

        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        output_document.save(output_pdf, deflate=True, garbage=3)
        LOGGER.info("Wrote %s", output_pdf)
    finally:
        output_document.close()
        source_document.close()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    input_pdf, output_pdf = normalize_paths(args.pdf, args.output)
    build_cleaned_document(input_pdf, output_pdf, args.scale)


if __name__ == "__main__":
    main()
