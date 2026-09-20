import argparse

import pytest

from cli import parse_region
from models import Region


def test_parse_region() -> None:
    assert parse_region("12, 8, 140, 48") == Region(12, 8, 140, 48)


@pytest.mark.parametrize("value", ["1,2,3", "a,2,3,4", "0,0,-1,10"])
def test_parse_region_rejects_invalid_value(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_region(value)
