import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("ee_probe", Path(__file__).resolve().parents[1] / "scripts" / "ee_probe.py")
ee_probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ee_probe)


class Node:
    def __init__(self, info):
        self._info = info

    def getInfo(self):  # noqa: N802
        return self._info

    def select(self, _):
        return Node({"crs": "EPSG:4326", "transform": [1, 0, 0, 0, -1, 0]})

    def projection(self):
        return self

    def first(self):
        return Node({"bands": [{"id": "elevation", "data_type": {"type": "PixelType"}, "dimensions": [3, 3]}],
                     "properties": {"b": 1, "a": 2}})

    def size(self):
        return Node(7)


class FakeEE:
    class data:  # noqa: N801
        asset_type = "IMAGE"

        @classmethod
        def getAsset(cls, asset_id):  # noqa: N802
            return {"type": cls.asset_type, "properties": {"title": "T"}, "startTime": "s", "endTime": "e"}

    def Image(self, asset_id):  # noqa: N802
        return Node({"bands": [{"id": "elevation", "data_type": {"type": "PixelType"}, "dimensions": [3, 3]}],
                     "properties": {"b": 1, "a": 2}})

    def ImageCollection(self, asset_id):  # noqa: N802
        return _Coll()


class _Coll(Node):
    def __init__(self):
        super().__init__({})


def test_probe_image():
    out = ee_probe.probe(FakeEE(), "X/Y")
    assert out["type"] == "IMAGE" and out["bands"][0]["id"] == "elevation"
    assert out["projection"]["crs"] == "EPSG:4326"
    assert out["image_properties"] == ["a", "b"]


def test_probe_collection():
    FakeEE.data.asset_type = "IMAGE_COLLECTION"
    try:
        out = ee_probe.probe(FakeEE(), "X/Y")
    finally:
        FakeEE.data.asset_type = "IMAGE"
    assert out["type"] == "IMAGE_COLLECTION" and out["size"] == 7 and out["bands"][0]["id"] == "elevation"
