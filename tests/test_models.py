import pytest

from seedance_watermark_remover.models import Region


def test_region_validation_accepts_in_bounds_region() -> None:
    Region(10, 20, 100, 50).validate(1920, 1080)


@pytest.mark.parametrize(
    "region",
    [
        Region(-1, 0, 10, 10),
        Region(0, -1, 10, 10),
        Region(0, 0, 0, 10),
        Region(1900, 0, 30, 10),
    ],
)
def test_region_validation_rejects_invalid_region(region: Region) -> None:
    with pytest.raises(ValueError):
        region.validate(1920, 1080)

