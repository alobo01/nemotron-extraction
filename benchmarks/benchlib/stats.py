"""Agreement statistics between reference and candidate activations."""

from __future__ import annotations

import json
import math

import torch
from safetensors.torch import save_file


def normalized_mse(error_squared_sum, reference_squared_sum):
    """MSE after scaling both tensors by the reference's site-wide RMS.

    The shared scale preserves magnitude errors. A zero reference has no
    usable scale: return zero for identical zeros and None otherwise.
    """
    if reference_squared_sum:
        return error_squared_sum / reference_squared_sum
    return 0.0 if not error_squared_sum else None


class Agreement:
    """FP32 differences and norms, accumulated in float64; exact cosine quantiles."""

    def __init__(self):
        self.cosines, self.error2, self.reference2, self.max_error = [], 0.0, 0.0, 0.0

    def add(self, reference, candidate):
        if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
            raise ValueError("Activation shapes or dtypes differ")
        for left, right in zip(reference.split(256), candidate.split(256)):
            left, right = left.float(), right.float()
            if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
                raise ValueError("Nonfinite activations")
            diff = right - left
            self.error2 += float(diff.double().square().sum())
            self.reference2 += float(left.double().square().sum())
            self.max_error = max(self.max_error, float(diff.abs().max()))
            a, b = left.norm(dim=-1), right.norm(dim=-1)
            denominator = a * b
            cosine = (left * right).sum(-1) / denominator.clamp_min(torch.finfo(torch.float32).tiny)
            cosine[(a == 0) & (b == 0)] = 1.0
            self.cosines.append(cosine.clamp(-1, 1))

    def merge_result(self, result, cosines):
        """Fold in a saved ``result()`` and its per-row cosines, e.g. from another token range."""
        self.cosines.append(cosines)
        self.error2 += result["error_squared_sum"]
        self.reference2 += result["reference_squared_sum"]
        self.max_error = max(self.max_error, result["maximum_absolute_error"])

    def result(self):
        cosine = torch.cat(self.cosines)
        # A zero reference has relative error zero only when the candidate is also zero.
        nmse = normalized_mse(self.error2, self.reference2)
        relative = math.sqrt(nmse) if nmse is not None else None
        result = {
            "tokens": len(cosine),
            "mean_cosine": float(cosine.mean()),
            "minimum_cosine": float(cosine.min()),
            "first_percentile_cosine": float(cosine.quantile(0.01)),
            "relative_l2": relative,
            "normalized_mse": nmse,
            "mse_normalization": "shared_hf_rms_per_site",
            "maximum_absolute_error": self.max_error,
            "error_squared_sum": self.error2,
            "reference_squared_sum": self.reference2,
        }
        result["passed"] = (
            result["mean_cosine"] >= 0.9999
            and result["minimum_cosine"] >= 0.9990
            and relative is not None
            and relative <= 0.02
        )
        return result


def same_expert_sets(left, right):
    """Per-row equality of selected expert IDs, ignoring top-k order."""
    return (torch.sort(left.long(), dim=-1).values == torch.sort(right.long(), dim=-1).values).all(dim=-1)


class TokenDiagnostics:
    """Exact scalar distributions with at most 40 retained activation pairs.

    Row identity is supplied explicitly; it never assumes a fixed sequence length.
    Shards can be merged without changing ranking or aggregate statistics.
    """

    def __init__(self):
        self.rows = []
        self.samples = {}
        self.width = self.dtype = None

    def add(self, reference, candidate, identities):
        if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
            raise ValueError("Activation shapes or dtypes differ")
        if reference.ndim != 2 or len(identities) != len(reference):
            raise ValueError("Expected one identity per activation row")
        if self.width is not None and (reference.shape[1] != self.width or str(reference.dtype) != self.dtype):
            raise ValueError("Incompatible diagnostic shards")
        self.width, self.dtype = reference.shape[1], str(reference.dtype)
        h, v = reference.detach().cpu().float(), candidate.detach().cpu().float()
        if not bool(torch.isfinite(h).all() and torch.isfinite(v).all()):
            raise ValueError("Nonfinite activations")
        hn, vn = h.norm(dim=-1), v.norm(dim=-1)
        cos = ((h * v).sum(-1) / (hn * vn).clamp_min(torch.finfo(torch.float32).tiny)).clamp(-1, 1)
        cos[(hn == 0) & (vn == 0)] = 1
        err = (v - h).double().square().sum(-1)
        ref = h.double().square().sum(-1)
        if len({r["row_id"] for r in identities}) != len(identities):
            raise ValueError("Duplicate row identity")
        existing = {r["row_id"] for r in self.rows}
        for i, identity in enumerate(identities):
            if identity["row_id"] in existing:
                raise ValueError("Duplicate row identity")
            self.rows.append(
                dict(
                    identity,
                    cosine=float(cos[i]),
                    reference_norm=float(hn[i]),
                    candidate_norm=float(vn[i]),
                    error_squared=float(err[i]),
                    reference_squared=float(ref[i]),
                    maximum_absolute_error=float((v[i] - h[i]).abs().max()),
                    relative_error=float(err[i].sqrt() / hn[i]) if hn[i] else (0.0 if not vn[i] else None),
                    relative_error_undefined=bool(hn[i] == 0 and vn[i] != 0),
                    exact=bool(torch.equal(reference[i], candidate[i])),
                )
            )
        # Only candidates for either global top-20 need their full tensors kept.
        local = set(sorted(range(len(identities)), key=lambda i: (float(cos[i]), identities[i]["row_id"]))[:20])
        local.update(sorted(range(len(identities)), key=lambda i: (-float(err[i]), identities[i]["row_id"]))[:20])
        for i in local:
            self.samples[identities[i]["row_id"]] = (
                reference[i].detach().cpu().clone(),
                candidate[i].detach().cpu().clone(),
            )
        self._prune()

    def _rank(self, key, reverse=False):
        return sorted(self.rows, key=lambda r: ((-r[key] if reverse else r[key]), r["row_id"]))[:20]

    def _prune(self):
        keep = {r["row_id"] for r in self._rank("cosine") + self._rank("error_squared", True)}
        self.samples = {k: v for k, v in self.samples.items() if k in keep}

    def merge(self, other):
        if self.width is not None and (self.width, self.dtype) != (other.width, other.dtype):
            raise ValueError("Incompatible diagnostic shards")
        if {r["row_id"] for r in self.rows} & {r["row_id"] for r in other.rows}:
            raise ValueError("Duplicate row identity")
        self.width, self.dtype = other.width, other.dtype
        self.rows.extend(other.rows)
        self.samples.update(other.samples)
        self._prune()
        return self

    def result(self):
        if not self.rows:
            raise ValueError("No diagnostic rows")
        rows = sorted(self.rows, key=lambda r: r["row_id"])
        cos = torch.tensor([r["cosine"] for r in rows], dtype=torch.float64)
        norms = torch.tensor([r["reference_norm"] for r in rows], dtype=torch.float64)
        error2 = sum(r["error_squared"] for r in rows)
        reference2 = sum(r["reference_squared"] for r in rows)
        nmse = error2 / reference2 if reference2 else (0.0 if not error2 else None)
        relative = nmse**0.5 if nmse is not None else None
        ranked_error = self._rank("error_squared", True)
        count = max(1, int(0.001 * len(rows)))
        top_error = sorted((r["error_squared"] for r in rows), reverse=True)[:count]
        order = torch.argsort(norms, stable=True)
        deciles = []
        for number, indices in enumerate(torch.tensor_split(order, 10)):
            if indices.numel():
                c, n = cos[indices], norms[indices]
                deciles.append(
                    dict(
                        decile=number + 1,
                        rows=len(indices),
                        reference_norm_min=float(n.min()),
                        reference_norm_max=float(n.max()),
                        mean_cosine=float(c.mean()),
                        minimum_cosine=float(c.min()),
                        first_percentile_cosine=float(c.quantile(0.01)),
                    )
                )

        def enriched(items):
            return [dict(r, error_energy_share=r["error_squared"] / error2 if error2 else 0.0) for r in items]

        mean, minimum = float(cos.mean()), float(cos.min())
        one_percent_count = (len(rows) + 99) // 100
        cosine_tail = sorted(rows, key=lambda r: (r["cosine"], r["row_id"]))[:one_percent_count]
        error_tail = sorted(rows, key=lambda r: (-r["error_squared"], r["row_id"]))[:one_percent_count]
        reference_energy_per_row = reference2 / len(rows)
        normalized_row_errors = (
            (torch.tensor([r["error_squared"] for r in rows], dtype=torch.float64) / reference_energy_per_row)
            if reference_energy_per_row
            else None
        )
        return dict(
            tokens=len(rows),
            width=self.width,
            dtype=self.dtype,
            mean_cosine=mean,
            cosine_std=float(cos.std(correction=0)),
            minimum_cosine=minimum,
            maximum_cosine_disagreement=1 - minimum,
            normalized_mse_std=(
                float(normalized_row_errors.std(correction=0))
                if normalized_row_errors is not None
                else (0.0 if not error2 else None)
            ),
            worst_one_percent=dict(
                count=one_percent_count,
                rounding="ceil(0.01 * token_rows)",
                mean_cosine=sum(r["cosine"] for r in cosine_tail) / one_percent_count,
                maximum_cosine_disagreement=1 - minimum,
                error_energy_share=sum(r["error_squared"] for r in error_tail) / error2 if error2 else 0.0,
                mean_normalized_squared_error=(
                    sum(r["error_squared"] for r in error_tail) / one_percent_count / reference_energy_per_row
                    if reference_energy_per_row
                    else (0.0 if not error2 else None)
                ),
                maximum_normalized_squared_error=(
                    max(r["error_squared"] for r in rows) / reference_energy_per_row
                    if reference_energy_per_row
                    else (0.0 if not error2 else None)
                ),
                rows_by_cosine=enriched(cosine_tail),
                rows_by_squared_error=enriched(error_tail),
            ),
            cosine_percentiles={str(p): float(cos.quantile(p / 100)) for p in (0, 0.1, 1, 5, 50, 99, 100)},
            relative_l2=relative,
            normalized_mse=nmse,
            mse_normalization="shared_reference_rms_per_site",
            error_squared_sum=error2,
            reference_squared_sum=reference2,
            maximum_absolute_error=max(r["maximum_absolute_error"] for r in rows),
            exact=all(r["exact"] for r in rows),
            both_zero_rows=sum(r["reference_norm"] == 0 and r["candidate_norm"] == 0 for r in rows),
            one_zero_rows=sum((r["reference_norm"] == 0) != (r["candidate_norm"] == 0) for r in rows),
            high_relative_error_fraction=sum(r["relative_error_undefined"] or r["relative_error"] > 0.02 for r in rows)
            / len(rows),
            worst_point_one_percent_error_share=sum(top_error) / error2 if error2 else 0.0,
            reference_norm_deciles=deciles,
            worst_cosine=enriched(self._rank("cosine")),
            worst_error=enriched(ranked_error),
            passed=mean >= 0.9999 and minimum >= 0.9990 and relative is not None and relative <= 0.02,
        )

    def save_samples(self, path):
        ordered = sorted(self.samples)
        save_file(
            {
                f"{i}_{side}": self.samples[key][j]
                for i, key in enumerate(ordered)
                for j, side in enumerate(("reference", "candidate"))
            },
            str(path),
            metadata={"row_ids": json.dumps(ordered)},
        )
