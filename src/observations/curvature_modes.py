"""Curvature mode extraction via serpenoid fitting.

Extracts compact parameters (amplitude, wave_number, phase) from the snake's
curvature profile by fitting a serpenoid model using FFT analysis.
"""

from typing import Any, Dict

import numpy as np

from .extractors import FeatureExtractor


class CurvatureModeExtractor(FeatureExtractor):
    """Extract serpenoid parameters from curvature profile.

    The serpenoid model assumes curvatures follow:
        kappa(s) = A * sin(k * s + phi)

    where:
        - A: amplitude (peak curvature)
        - k: wave number (spatial frequency)
        - phi: phase offset

    This extractor fits these parameters to the current curvature profile
    using FFT to find the dominant spatial frequency.

    Output features (3 dims):
        [0]: amplitude - peak curvature magnitude
        [1]: wave_number - normalized spatial frequency (0-1 scale)
        [2]: phase - phase offset in radians / (2*pi), normalized to [0, 1]
    """

    def __init__(self, normalize: bool = True):
        """Initialize the curvature mode extractor.

        Args:
            normalize: If True, normalize features to approximately [0, 1] range
        """
        self.normalize = normalize
        # Normalization constants (empirically determined)
        self._amplitude_scale = 5.0  # Max expected amplitude
        self._max_wave_number = 3.0  # Max expected waves along body

    @property
    def feature_dim(self) -> int:
        """Return feature dimension (amplitude, wave_number, phase)."""
        return 3

    def extract(self, state: Dict[str, Any]) -> np.ndarray:
        """Extract serpenoid parameters from curvature profile.

        Args:
            state: State dict containing 'curvatures' key with shape (19,)

        Returns:
            Feature vector of shape (3,): [amplitude, wave_number, phase]
        """
        curvatures = state.get("curvatures", None)
        if curvatures is None:
            return np.zeros(3, dtype=np.float32)

        curvatures = np.asarray(curvatures, dtype=np.float64)
        n = len(curvatures)

        if n == 0:
            return np.zeros(3, dtype=np.float32)

        # Compute FFT to find dominant frequency
        fft = np.fft.rfft(curvatures)
        magnitudes = np.abs(fft)

        # Find dominant frequency (excluding DC component)
        if len(magnitudes) > 1:
            # Ignore DC (index 0), find peak in remaining frequencies
            dominant_idx = np.argmax(magnitudes[1:]) + 1
            amplitude = magnitudes[dominant_idx] * 2.0 / n  # Normalize by length
            phase = np.angle(fft[dominant_idx])  # Phase in radians

            # Wave number: how many complete waves fit along the body
            # dominant_idx corresponds to that many half-periods in the signal
            wave_number = dominant_idx * 2.0 / n * np.pi
        else:
            # Fallback for very short signals
            amplitude = np.std(curvatures)
            wave_number = 1.0
            phase = 0.0

        # Also consider the actual peak-to-peak amplitude
        amplitude = max(amplitude, (np.max(curvatures) - np.min(curvatures)) / 2.0)

        # Normalize phase to [0, 1]
        phase_normalized = (phase + np.pi) / (2 * np.pi)

        features = np.array([amplitude, wave_number, phase_normalized], dtype=np.float32)

        if self.normalize:
            features[0] = np.clip(features[0] / self._amplitude_scale, 0.0, 1.0)
            features[1] = np.clip(features[1] / self._max_wave_number, 0.0, 1.0)
            # phase_normalized is already in [0, 1]

        return features


class ExtendedCurvatureModeExtractor(FeatureExtractor):
    """Extended curvature mode extractor with higher harmonics.

    Extracts the first K harmonic components from the curvature profile,
    providing a richer representation of complex body shapes.

    Output features (3*K dims):
        For each harmonic k in [1, K]:
            - amplitude_k
            - wave_number_k (normalized)
            - phase_k (normalized)
    """

    def __init__(self, num_harmonics: int = 2, normalize: bool = True):
        """Initialize extended curvature mode extractor.

        Args:
            num_harmonics: Number of harmonic components to extract
            normalize: If True, normalize features to approximately [0, 1] range
        """
        self.num_harmonics = num_harmonics
        self.normalize = normalize
        self._amplitude_scale = 5.0
        self._max_wave_number = 3.0

    @property
    def feature_dim(self) -> int:
        """Return feature dimension (3 * num_harmonics)."""
        return 3 * self.num_harmonics

    def extract(self, state: Dict[str, Any]) -> np.ndarray:
        """Extract multiple harmonic components from curvature profile.

        Args:
            state: State dict containing 'curvatures' key

        Returns:
            Feature vector of shape (3 * num_harmonics,)
        """
        curvatures = state.get("curvatures", None)
        if curvatures is None:
            return np.zeros(self.feature_dim, dtype=np.float32)

        curvatures = np.asarray(curvatures, dtype=np.float64)
        n = len(curvatures)

        if n == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)

        # Compute FFT
        fft = np.fft.rfft(curvatures)
        magnitudes = np.abs(fft)

        features = np.zeros(self.feature_dim, dtype=np.float32)

        # Sort frequency indices by magnitude (excluding DC)
        if len(magnitudes) > 1:
            sorted_indices = np.argsort(magnitudes[1:])[::-1] + 1
        else:
            return features

        for k in range(min(self.num_harmonics, len(sorted_indices))):
            idx = sorted_indices[k]
            if idx < len(fft):
                amplitude = magnitudes[idx] * 2.0 / n
                phase = (np.angle(fft[idx]) + np.pi) / (2 * np.pi)
                wave_number = idx * 2.0 / n * np.pi

                if self.normalize:
                    amplitude = np.clip(amplitude / self._amplitude_scale, 0.0, 1.0)
                    wave_number = np.clip(wave_number / self._max_wave_number, 0.0, 1.0)

                base_idx = k * 3
                features[base_idx] = amplitude
                features[base_idx + 1] = wave_number
                features[base_idx + 2] = phase

        return features
