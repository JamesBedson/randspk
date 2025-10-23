import torch
import numpy as np

class SpeakerPosition:
    def __init__(self, azimuth=0.0, elevation=0.0, distance=1.0):
        self.azimuth    = azimuth
        self.elevation  = elevation
        self.distance   = distance

    def set_coords_spherical(self, az: float, el: float, dist: float):
        self.azimuth    = az
        self.elevation  = el
        self.distance   = dist

    def get_coords_spherical(self) -> list[float]:
        return [self.azimuth, self.elevation, self.distance]

    def get_coords_cartesian(self) -> list[float]:
        x = self.distance * np.cos(self.elevation) * np.cos(self.azimuth)
        y = self.distance * np.cos(self.elevation) * np.sin(self.azimuth)
        z = self.distance * np.sin(self.elevation)
        return [x, y, z]

class SpeakerLayout:
    def __init__(
        self,
        num_speakers=8,
        radius=1.0,
        symmetry=False,
        hemisphere=True,
        min_distance=0.2,
        elevation_bias=1.0,
        random_seed=None,
    ):
        self.num_speakers   = num_speakers
        self.radius         = radius
        self.symmetry       = symmetry
        self.hemisphere     = hemisphere
        self.min_distance   = min_distance
        self.elevation_bias = elevation_bias

        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

    # --- Utilities -----------------------------------------------------------

    @staticmethod
    def spherical_to_cartesian(azimuth, elevation, distance):
        x = distance * np.cos(elevation) * np.cos(azimuth)
        y = distance * np.cos(elevation) * np.sin(azimuth)
        z = distance * np.sin(elevation)
        return np.stack([x, y, z], axis=-1)

    def _random_points_on_sphere(self, n, radius):
        az      = 2 * np.pi * np.random.rand(n)
        el      = np.arcsin(2 * np.random.rand(n) - 1)
        dist    = np.full(n, radius)
        return az, el, dist

    def _random_points_on_hemisphere(self, n, radius):
        az = 2 * np.pi * np.random.rand(n)
        u       = np.random.rand(n) ** self.elevation_bias
        el      = np.arcsin(u)  # [0, pi/2]
        dist    = np.full(n, radius)
        return az, el, dist

    @staticmethod
    def _mirror_points(points, axis='x'):
        mirrored = points.copy()
        
        if axis == 'x': mirrored[:, 0] *= -1
        elif axis == 'y': mirrored[:, 1] *= -1
        elif axis == 'z': mirrored[:, 2] *= -1
        else: raise ValueError("Axis must be 'x', 'y', or 'z'")
        
        return mirrored

    @staticmethod
    def _pairwise_distances(points):
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=-1))

    def _generate_special_odd_speaker(self, radius):
        r = np.random.rand()
        
        if r < 0.95: az, el = 0.0, 0.0
        elif r < 0.98: az, el = np.pi, 0.0
        else: az, el = 0.0, np.pi / 2
        
        return self.spherical_to_cartesian(np.array([az]), np.array([el]), np.array([radius]))[0]

    # --- Main ---------------------------------------------------------------

    def generate_random_layout(
        self,
        num_speakers=None,
        radius=None,
        symmetry=None,
        hemisphere=None,
        mirror_axis='x',
        enforce_odd_rule=True,
        return_spherical=False,
        min_distance=None,
    ):
        num_speakers    = num_speakers or self.num_speakers
        radius          = radius or self.radius
        symmetry        = self.symmetry if symmetry is None else symmetry
        hemisphere      = self.hemisphere if hemisphere is None else hemisphere
        min_distance    = min_distance or self.min_distance

        generator = self._random_points_on_hemisphere if hemisphere else self._random_points_on_sphere

        def is_valid(points):
            d = self._pairwise_distances(points)
            np.fill_diagonal(d, np.inf)
            return np.all(d > min_distance)

        for attempt in range(1000):
            if symmetry:
                half            = num_speakers // 2
                az, el, dist    = generator(half, radius)
                cart            = self.spherical_to_cartesian(az, el, dist)
                mirrored        = self._mirror_points(cart, axis=mirror_axis)
                points          = np.concatenate([cart, mirrored], axis=0)
                print(np.shape(points))

                if num_speakers % 2 == 1:
                    if enforce_odd_rule:
                        points = np.vstack([points, self._generate_special_odd_speaker(radius)])
                    else:
                        az, el, dist    = generator(1, radius)
                        rand_point      = self.spherical_to_cartesian(az, el, dist)
                        points          = np.vstack([points, rand_point])
            
            else:
                az, el, dist    = generator(num_speakers, radius)
                points          = self.spherical_to_cartesian(az, el, dist)
                if num_speakers % 2 == 1 and enforce_odd_rule:
                    points[-1] = self._generate_special_odd_speaker(radius)

            if is_valid(points):
                break
        else:
            print("Warning: could not find a valid layout after 1000 attempts.")

        tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0)
        if return_spherical:
            sph = np.stack([az, el, dist], axis=-1)
            return tensor, sph
        return tensor