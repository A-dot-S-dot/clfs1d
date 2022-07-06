from .interval import Interval
from .transformation import LocalCoordinates, WorldCoordinates, AffineTransformation


class Simplex(Interval):
    """Simplex object which contains several mappings."""

    _local_coordinates: LocalCoordinates
    _world_coordinates: WorldCoordinates
    _affine_transformation: AffineTransformation

    def __init__(self, a: float, b: float):
        Interval.__init__(self, a, b)
        self._local_coordinates = LocalCoordinates(self)
        self._world_coordinates = WorldCoordinates(self)
        self._affine_transformation = AffineTransformation(self)

    @property
    def local_coordinates(self) -> LocalCoordinates:
        return self._local_coordinates

    @property
    def world_coordinates(self) -> WorldCoordinates:
        return self._world_coordinates

    @property
    def affine_transformation(self) -> AffineTransformation:
        return self._affine_transformation

    @property
    def Lambda(self):
        return self.local_coordinates.Lambda
