"""Max-Lloyd optimal scalar quantization codebooks for unit-variance Gaussian.

After random orthogonal rotation, coordinates of a high-dimensional unit vector
concentrate to an approximately N(0, 1/D) distribution (Beta(1/2, (D-1)/2) for
finite D; Gaussian in the limit). The Max-Lloyd algorithm finds the quantization
levels that minimise mean-squared error for a given number of levels.

These codebooks are DATA-OBLIVIOUS: they depend only on the distribution shape,
not the specific model. Precomputed once, reused for all TurboQuant runs.

Values sourced from standard references (Lloyd 1982, Max 1960) for unit-variance
normal distribution.
"""


import torch


# Max-Lloyd optimal reconstruction levels for N(0, 1).
# Keyed by bit width; each value is a sorted tensor of 2^b levels.
GAUSSIAN_LLOYD_MAX: dict[int, torch.Tensor] = {
    1: torch.tensor([-0.7979, 0.7979]),
    2: torch.tensor([-1.5104, -0.4528, 0.4528, 1.5104]),
    3: torch.tensor([
        -2.1519, -1.3439, -0.7560, -0.2451,
         0.2451,  0.7560,  1.3439,  2.1519,
    ]),
    4: torch.tensor([
        -2.7326, -2.0690, -1.6181, -1.2562, -0.9424, -0.6568, -0.3881, -0.1284,
         0.1284,  0.3881,  0.6568,  0.9424,  1.2562,  1.6181,  2.0690,  2.7326,
    ]),
}


def lloyd_codebook(bits: int) -> torch.Tensor:
    """Return the Max-Lloyd codebook for N(0,1) at the given bit width."""
    if bits not in GAUSSIAN_LLOYD_MAX:
        raise ValueError(
            f"No precomputed Max-Lloyd codebook for {bits} bits. "
            f"Available: {sorted(GAUSSIAN_LLOYD_MAX.keys())}"
        )
    return GAUSSIAN_LLOYD_MAX[bits].clone()
