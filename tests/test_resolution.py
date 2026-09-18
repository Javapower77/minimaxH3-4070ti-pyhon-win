from PIL import Image

from minimax_h3_fl2v.media import fit_image_without_crop
from minimax_h3_fl2v.resolution import aspect_error, resolve_output_size, size_from_image


def test_official_16_9_ladder():
    assert resolve_output_size(1.0, "16:9") == (1376, 768)
    assert resolve_output_size(0.5, "16:9") == (960, 544)
    assert resolve_output_size(0.5, "9:16") == (544, 960)


def test_multiples_of_32():
    width, height = resolve_output_size(1.0, "1:1")
    assert width % 32 == 0 and height % 32 == 0


def test_image_aspect_snap():
    width, height = size_from_image(1920, 1080, 1.0)
    assert (width, height) == (1376, 768)


def test_arbitrary_image_ratio_is_not_snapped_to_named_preset():
    width, height = size_from_image(1000, 667, 1.0)
    assert width % 32 == 0 and height % 32 == 0
    assert aspect_error(1000, 667, width, height) < 0.02
    assert (width, height) != resolve_output_size(1.0, "4:3")
    assert (width, height) != resolve_output_size(1.0, "16:9")


def test_fit_image_retains_full_source_with_letterbox():
    source = Image.new("RGB", (4, 2), (255, 0, 0))
    fitted = fit_image_without_crop(source, 8, 8)
    assert fitted.size == (8, 8)
    # Entire 2:1 source fits in the central 8x4 area; top/bottom are padding.
    assert fitted.getpixel((0, 0)) == (0, 0, 0)
    assert fitted.getpixel((0, 2)) == (255, 0, 0)
    assert fitted.getpixel((7, 5)) == (255, 0, 0)
    assert fitted.getpixel((7, 7)) == (0, 0, 0)
