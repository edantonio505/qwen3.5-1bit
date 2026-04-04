"""Advanced 1-bit QAT library.

Techniques beyond basic STE:
1. Hadamard rotation — spread outlier energy before binarization
2. Learned scale factors (LSQ-style) — optimize scales via backprop
3. Progressive quantization — anneal from higher bit-width to 1-bit
4. Hessian-aware weight compensation
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

GROUP_SIZE = 128


# ── Hadamard Transform ──

def hadamard_matrix(n):
    """Generate a normalized Hadamard matrix of size n (must be power of 2)."""
    if n == 1:
        return torch.tensor([[1.0]])
    h = hadamard_matrix(n // 2)
    return torch.cat([
        torch.cat([h, h], dim=1),
        torch.cat([h, -h], dim=1),
    ], dim=0) / math.sqrt(2)


def random_hadamard_matrix(n, device="cpu", generator=None):
    """Randomized Hadamard: H * diag(s) where s is random signs."""
    # Pad n to next power of 2
    n_pad = 1 << (n - 1).bit_length()
    H = hadamard_matrix(n_pad).to(device)
    signs = torch.randint(0, 2, (n_pad,), device=device, generator=generator) * 2 - 1
    return H * signs.unsqueeze(0), n_pad


def apply_hadamard_rotation(weight, H_left=None, H_right=None):
    """Rotate weight matrix: W' = H_left @ W @ H_right.T
    This spreads outlier energy, making weight magnitudes more uniform.
    """
    if H_left is not None:
        out_dim = weight.shape[0]
        weight = H_left[:out_dim, :out_dim] @ weight
    if H_right is not None:
        in_dim = weight.shape[1]
        weight = weight @ H_right[:in_dim, :in_dim].T
    return weight


# ── STE with Learned Scale ──

class STEQuantize1Bit(torch.autograd.Function):
    """1-bit quantize with STE. Scale is passed in (can be learned)."""

    @staticmethod
    def forward(ctx, weight, scales):
        """
        weight: (..., GROUP_SIZE) shaped or will be reshaped
        scales: (num_groups, 1) learned or computed scales
        """
        orig_shape = weight.shape
        flat = weight.reshape(-1, GROUP_SIZE)
        signs = flat.sign()
        signs[signs == 0] = 1.0
        quantized = scales * signs
        ctx.save_for_backward(signs)
        return quantized.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output):
        signs, = ctx.saved_tensors
        flat_grad = grad_output.reshape(-1, GROUP_SIZE)
        # Gradient for weights: STE (pass through)
        weight_grad = grad_output
        # Gradient for scales: d(loss)/d(scale) = sum(signs * grad) per group
        scale_grad = (flat_grad * signs).sum(dim=1, keepdim=True)
        return weight_grad, scale_grad


class QuantizedLinear(nn.Module):
    """1-bit quantized linear with learned group scales and optional Hadamard rotation."""

    def __init__(self, original: nn.Linear, use_hadamard=False, use_learned_scales=True):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.use_hadamard = use_hadamard
        self.use_learned_scales = use_learned_scales

        # Master weights (full precision, updated by optimizer)
        self.weight = original.weight
        self.bias = original.bias

        num_groups = self.weight.numel() // GROUP_SIZE

        if use_learned_scales:
            # Initialize learned scales from weight statistics
            with torch.no_grad():
                flat = self.weight.data.reshape(-1, GROUP_SIZE)
                init_scales = flat.abs().mean(dim=1, keepdim=True)
            self.log_scales = nn.Parameter(torch.log(init_scales + 1e-8))
        else:
            self.log_scales = None

        # Hadamard rotation matrices (fixed, not learned)
        if use_hadamard and self.weight.numel() >= GROUP_SIZE:
            self._init_hadamard()
        else:
            self.register_buffer('H_row', None)
            self.register_buffer('H_col', None)

    def _init_hadamard(self):
        """Initialize Hadamard rotation matrices for this layer."""
        out_dim, in_dim = self.weight.shape

        # Only apply to dimensions that are powers of 2 or can be padded cheaply
        # For simplicity, apply column rotation (input dimension) which is most impactful
        if in_dim >= 64 and (in_dim & (in_dim - 1)) == 0:  # power of 2
            H = hadamard_matrix(in_dim).to(self.weight.device, self.weight.dtype)
            self.register_buffer('H_col', H)
        else:
            self.register_buffer('H_col', None)
        self.register_buffer('H_row', None)

    def forward(self, x):
        w = self.weight

        # Apply Hadamard rotation to spread outliers
        if self.use_hadamard and self.H_col is not None:
            w = w @ self.H_col.T

        flat = w.reshape(-1, GROUP_SIZE)

        if self.use_learned_scales:
            scales = torch.exp(self.log_scales)  # ensure positive
        else:
            scales = flat.detach().abs().mean(dim=1, keepdim=True)

        # Quantize with STE
        q_weight = STEQuantize1Bit.apply(w, scales)

        # Undo Hadamard rotation on quantized weights
        if self.use_hadamard and self.H_col is not None:
            q_weight = q_weight @ self.H_col  # H is orthogonal, so H^-1 = H^T, but H is symmetric

        return F.linear(x, q_weight, self.bias)


class ProgressiveQuantizer:
    """Manages progressive bit-width reduction during training.

    Schedule: full_precision -> 4-bit -> 2-bit -> 1-bit
    At each stage, quantization noise is gradually increased.
    """

    def __init__(self, total_steps, warmup_fraction=0.1):
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_fraction)
        # Phase boundaries (as fractions of total steps)
        # 0-10%: warmup (full precision)
        # 10-40%: 4-bit equivalent (add small noise)
        # 40-70%: 2-bit equivalent (more noise)
        # 70-100%: full 1-bit
        self.phases = [
            (0.0, 0.1, "warmup"),
            (0.1, 0.4, "4bit"),
            (0.4, 0.7, "2bit"),
            (0.7, 1.0, "1bit"),
        ]

    def get_noise_scale(self, step):
        """Get quantization noise scale for current step.
        0.0 = no noise (full precision), 1.0 = full 1-bit quantization.
        """
        frac = step / max(self.total_steps, 1)

        if frac < 0.1:
            return 0.0  # warmup: no quantization
        elif frac < 0.4:
            # Linear ramp from 0 to 0.5
            return 0.5 * (frac - 0.1) / 0.3
        elif frac < 0.7:
            # Linear ramp from 0.5 to 0.9
            return 0.5 + 0.4 * (frac - 0.4) / 0.3
        else:
            # Linear ramp from 0.9 to 1.0
            return 0.9 + 0.1 * (frac - 0.7) / 0.3

    def get_phase_name(self, step):
        frac = step / max(self.total_steps, 1)
        for start, end, name in self.phases:
            if start <= frac < end:
                return name
        return "1bit"


class ProgressiveQuantizedLinear(nn.Module):
    """Linear layer with progressive quantization noise injection.

    During training, blends between full-precision and 1-bit based on schedule.
    w_effective = (1 - noise_scale) * w_fp + noise_scale * w_1bit
    """

    def __init__(self, original: nn.Linear, use_learned_scales=True):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.weight = original.weight
        self.bias = original.bias
        self.noise_scale = 1.0  # will be set by training loop

        num_groups = self.weight.numel() // GROUP_SIZE
        if use_learned_scales:
            with torch.no_grad():
                flat = self.weight.data.reshape(-1, GROUP_SIZE)
                init_scales = flat.abs().mean(dim=1, keepdim=True)
            self.log_scales = nn.Parameter(torch.log(init_scales + 1e-8))
        else:
            self.log_scales = None

    def forward(self, x):
        w = self.weight

        if self.log_scales is not None:
            scales = torch.exp(self.log_scales)
        else:
            flat = w.detach().reshape(-1, GROUP_SIZE)
            scales = flat.abs().mean(dim=1, keepdim=True)

        # At full 1-bit (noise_scale=1.0), skip blending entirely
        if self.noise_scale >= 1.0:
            w_eff = STEQuantize1Bit.apply(w, scales)
        elif self.noise_scale <= 0.0:
            w_eff = w
        else:
            # Blend in-place: w_eff = w + noise_scale * (w_1bit - w)
            # This avoids holding w, w_1bit, and w_eff simultaneously
            w_1bit = STEQuantize1Bit.apply(w, scales)
            w_eff = w + self.noise_scale * (w_1bit - w)
            del w_1bit

        return F.linear(x, w_eff, self.bias)


def replace_linears(model, mode="progressive", skip_patterns=None,
                    use_hadamard=False, use_learned_scales=True):
    """Replace nn.Linear with quantized versions.

    mode: "basic" | "progressive" | "hadamard"
    """
    if skip_patterns is None:
        skip_patterns = ["norm", "layernorm", "rmsnorm"]

    replaced = 0
    skipped = 0

    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name

            if isinstance(child, nn.Linear):
                if any(pat in full_name.lower() for pat in skip_patterns):
                    skipped += 1
                    continue
                if child.weight.numel() % GROUP_SIZE != 0:
                    skipped += 1
                    continue

                if mode == "progressive":
                    setattr(module, child_name,
                            ProgressiveQuantizedLinear(child, use_learned_scales))
                elif mode == "hadamard":
                    setattr(module, child_name,
                            QuantizedLinear(child, use_hadamard=True,
                                         use_learned_scales=use_learned_scales))
                else:
                    setattr(module, child_name,
                            QuantizedLinear(child, use_hadamard=False,
                                         use_learned_scales=use_learned_scales))
                replaced += 1

    print(f"Replaced {replaced} linear layers ({mode} mode), skipped {skipped}")
    return model


def set_progressive_noise(model, noise_scale):
    """Set noise_scale on all ProgressiveQuantizedLinear layers."""
    for module in model.modules():
        if isinstance(module, ProgressiveQuantizedLinear):
            module.noise_scale = noise_scale


def extract_1bit_weights(model):
    """Extract final 1-bit weights for export."""
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, (QuantizedLinear, ProgressiveQuantizedLinear)):
            weight = module.weight.data.float()
            flat = weight.reshape(-1, GROUP_SIZE)

            if hasattr(module, 'log_scales') and module.log_scales is not None:
                scales = torch.exp(module.log_scales.data).squeeze().half()
            else:
                scales = flat.abs().mean(dim=1).half()

            signs = (flat.sign() > 0).to(torch.uint8)
            packed = pack_sign_bits(signs)

            result[name + ".weight"] = {
                "type": "q1_0_g128",
                "scales": scales.cpu(),
                "packed_signs": packed.cpu(),
                "shape": weight.shape,
            }
            if module.bias is not None:
                result[name + ".bias"] = module.bias.data.cpu()
        elif isinstance(module, nn.Embedding):
            result[name + ".weight"] = module.weight.data.cpu()
        elif isinstance(module, (nn.LayerNorm,)):
            for pname, param in module.named_parameters(recurse=False):
                result[f"{name}.{pname}"] = param.data.cpu()
    return result


def pack_sign_bits(signs):
    flat = signs.reshape(-1)
    pad = (8 - flat.shape[0] % 8) % 8
    if pad:
        flat = F.pad(flat, (0, pad))
    flat = flat.reshape(-1, 8)
    multipliers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8,
                               device=flat.device)
    return (flat * multipliers).sum(dim=1).to(torch.uint8)
