from app.domain.asset_metadata import (
    ASSET_CLASS_MAP,
    ASSET_COUNTRY_MAP,
    ASSET_REGISTRY,
    ASSET_SECTOR_MAP,
    ASSET_STYLE_MAP,
    STANDARD_UNIVERSE,
)


def test_standard_universe_contains_100_assets():
    assert len(STANDARD_UNIVERSE) == 100
    assert len(set(STANDARD_UNIVERSE)) == 100


def test_asset_metadata_maps_cover_the_full_universe():
    universe = set(STANDARD_UNIVERSE)
    assert universe == set(ASSET_REGISTRY.keys())
    assert universe == set(ASSET_CLASS_MAP.keys())
    assert universe == set(ASSET_SECTOR_MAP.keys())
    assert universe == set(ASSET_COUNTRY_MAP.keys())
    assert universe == set(ASSET_STYLE_MAP.keys())
