"""Same-window dual initialization for the frozen Hankel torus observer.

The original optimizer, dictionary, exact Gram and SVD tolerance are reused.
Each window supplies both a raw-Hann seed and an independent free-span seed.
Full measured-window cost selects the solution; truth and clock phases never
enter the API. A finite search is not a global minimization certificate.
"""

import numpy as np

from hankel_torus_projection import HankelTorusProjector


class RobustHankelTorusProjector(HankelTorusProjector):
    def free_linear_projection(self, mass_values, starts):
        cached = getattr(self, "_free_cache", None)
        if cached is not None and cached[0] is mass_values and cached[1] == starts:
            return cached[2]
        result = super().free_linear_projection(mass_values, starts)
        self._free_cache = mass_values, starts, result
        return result

    def project(self, mass_values, seed_phase):
        seed = np.asarray(seed_phase, dtype=float)
        if seed.ndim != 2 or seed.shape[0] != self.rank or not np.isfinite(seed).all():
            raise ValueError("one finite seed phase per inferred dimension and observed window required")
        free = self.free_linear_projection(mass_values, seed.shape[1])
        rows = np.asarray(self.multistart_metadata["strong_independent_character_rows"], dtype=int)
        if rows.shape != (self.rank, self.rank):
            raise ValueError("one independent learned character per inferred phase is required")
        indices = []
        for row in rows:
            matching = np.flatnonzero(np.all(self.labels == row, axis=1))
            if len(matching) != 1:
                raise ValueError("strong independent characters must uniquely index the frozen dictionary")
            indices.append(int(matching[0]))
        indices = np.asarray(indices)
        coefficients = free["free_coefficients"]
        strong_angles = np.arctan2(coefficients[self.count+indices], coefficients[indices])
        second_seed = np.linalg.solve(rows.astype(float), strong_angles)
        # Retain all available primary covering lifts. Allocate the remainder
        # of the predeclared total budget to the secondary seed's same lifts.
        primary_count = len(self.offsets)
        secondary_count = min(primary_count, max(0, self.settings.max_multistarts-primary_count))
        specifications = [("raw_hann", index, offset) for index, offset in enumerate(self.offsets)]
        specifications += [("free_linear", index, offset)
                           for index, offset in enumerate(self.offsets[:secondary_count])]
        if len(specifications) > self.settings.max_multistarts:
            raise ArithmeticError("combined start budget exceeded")
        right, constant = self.sufficient_statistics(mass_values, seed.shape[1])
        phases, diagnostics = [], []
        for column in range(seed.shape[1]):
            attempts = [self._optimize((seed if name == "raw_hann" else second_seed)[:, column]+offset,
                                      right[:, column], constant[column])
                        for name, _, offset in specifications]
            chosen = min(range(len(attempts)), key=lambda which: attempts[which][1]["final_cost"])
            phase, info = attempts[chosen]
            phases.append(np.angle(np.exp(1j*phase)))
            seed_cost = self.cost_gradient(seed[:, column], right[:, column], constant[column])[0]
            diagnostics.append({**info, "selected_start": chosen,
                "selected_seed": specifications[chosen][0], "selected_deck_offset": specifications[chosen][1],
                "seed_cost": seed_cost, "cost_not_above_product_seed": info["final_cost"] <= seed_cost+1e-10})
        phase = np.asarray(phases).T
        angles = np.einsum("ij,jt->it", self.labels, phase, optimize=False)
        design = np.column_stack((np.ones(seed.shape[1]), np.cos(angles.T), np.sin(angles.T)))
        field = np.einsum("nj,jq->qn", design, self.coefficients, optimize=False)
        metadata = {**self.multistart_metadata,
            "raw_hann_starts": primary_count, "free_linear_starts": secondary_count,
            "starts": len(specifications), "combined_start_cap": self.settings.max_multistarts,
            "secondary_cover_enumeration_complete": secondary_count == self.multistart_metadata.get("deck_order"),
            "secondary_cover_truncated_to_preserve_primary_cover_budget": secondary_count < primary_count,
            "secondary_seed": "atan2 of same-window unconstrained cos/sin coefficients at the strongest independent learned labels",
            "same_raw_observation_window_for_both_seeds_objective_and_linear_comparator": True,
            "optimizer_settings_changed": False, "global_optimum_certified": False}
        return {"phase": phase, "mass_field": field, "diagnostics": diagnostics,
            "multistart": metadata, "free_linear_seed_phase": second_seed}
