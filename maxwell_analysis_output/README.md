# Appendix T: What the Maxwell Analysis Revealed About Crystals, Polycrystals, and Amorphous Glasses

I ran the Maxwell analysis on all checkpoints—the ones that crystallised into Strassen, the ones that stayed glassy, and the one robust model that survived 50 % pruning. The script treated each set of trained weights as a dielectric material, solved Poisson’s equation, computed scattering patterns, and calculated photonic entropy. Here is what the numbers actually say, without metaphors.

### T.1 How the Analysis Works

The code maps the flattened weights (U, V, W) onto a 16×16×16 lattice. Each lattice site gets a permittivity ε = 1 + (weight value) and a charge density ρ = weight value. Then it:

1. Solves ∇·(ε∇φ) = –ρ for the electrostatic potential φ.
2. Fourier‑transforms the permittivity contrast to get a scattering intensity S(k) ∝ |FT(Δε)|².
3. Computes the structure tensor from gradients to get dielectric eigenvalues and anisotropy.
4. Estimates photonic entropy from the energy density and from the mode distribution.
5. Looks for bandgaps in the radial profile of the Fourier spectrum.

The classification into *Strassen_Crystal* or *Amorphous_Glass* uses a simple score based on purity α, anisotropy, Bragg‑peak sharpness, and entropy. The polycrystalline state is not a separate class in the script – it appears as a borderline glass with unusual stability.

### T.2 The Three Phases

#### T.2.1 Strassen Crystals (δ = 0, α ≈ 34.5)

Three checkpoints landed here: `strassen_exact.pt`, `strassen_grokked_weights.pt`, `strassen_discrete_final.pt`.

- **Discretisation margin** δ = 0 exactly – every weight is −1, 0 or 1.
- **Purity α** ~ 34.5, far above the glass values (α ~ 0.7).
- **Anisotropy ratio** ≈ 0.27 – not isotropic, but the dielectric tensor eigenvalues show a clear hierarchy.
- **Scattering pattern** shows a moderate number of sharp peaks (47–79). The script marks them as Bragg peaks because the peaks are prominent above the background.
- **Photonic entropy** total ~ 8.4, slightly lower than most glasses but not dramatically so. The field entropy is low (0.65), meaning the potential φ is spatially organised.
- **Bandgap analysis** – the radial profile of the Fourier intensity has a noticeable dip (gap depth ~ 0.25), indicating that some frequencies carry very little energy.

What this means physically: the dielectric lattice created by the integer weights is periodic enough to produce a clear diffraction pattern and a weak photonic bandgap. The electrostatic potential is smooth and ordered – a “crystal” in the materials‑science sense used throughout the paper.

#### T.2.2 Amorphous Glasses (δ ≈ 0.49, α ≈ 0.7)

The majority of checkpoints (all `bs*` runs and several `strassen_*` runs that did not discretise) fall here. Examples: `bs128_seed0.pt`, `strassen_result.pt`, `strassen_float64.pt`.

- **δ** ~ 0.49, **α** ~ 0.7 – weights are far from integers, no discretisation.
- **Anisotropy ratio** varies from 0.21 to 0.34 – still some directional preference, but not enough for long‑range order.
- **Scattering peaks** often numerous (64 to 832). The script classifies most as “Bragg pattern” because the Fourier transform of a disordered structure still contains many local maxima. But the peaks are broader and less intense relative to the background than in the crystals.
- **Photonic entropy** total ranges from 8.17 to 8.54 – overlapping with the crystal values, so entropy alone does not separate the phases.
- **Bandgap depth** generally smaller (0.16–0.66) and less consistent than in the crystals.

These are true glasses: the permittivity varies randomly, the scattering is diffuse, and the potential φ is noisy. They generalise on the test set but fail the structural verification.

#### T.2.3 The Polycrystalline State (strassen_robust.pt)

This checkpoint was created by pruning a trained model to 50 % sparsity while preserving accuracy. It is the only one that falls in between the two main phases.

- **δ** = 0.1514, **α** = 1.887 – intermediate, far from perfect integers but closer than the glasses.
- **Anisotropy** 0.2976 – similar to the glasses.
- **Scattering peaks** 496, still many, but the Fourier pattern is not as clean as in the crystals.
- **Photonic entropy** 8.25, comparable to glasses.
- **Bandgap depth** 0.477 – moderate.
- **Effective Planck constant** ħ_eff = 1.46 (from Appendix L), while crystals have ħ_eff ~ 1×10⁻⁷ and glasses have ħ_eff ~ 7×10⁶.
- **Superposition metrics** (Appendix K): ψ = 1.071 (very low, near the theoretical minimum of 7 slots + bias) and F = 8.6.

Interpretation: pruning removed the disordered “grain boundaries” and left only the structurally essential weights. The result is a polycrystal – a collection of small ordered domains with clean interfaces. It still shows many scattering peaks because the domains are oriented differently, but the overall pattern lacks the single‑crystal coherence. Its intermediate δ and ħ_eff confirm that it is not a perfect crystal, yet it is far more ordered than any glass.

### T.3 What Distinguishes the Phases?

| Metric | Strassen Crystal | Polycrystal | Amorphous Glass |
|--------|------------------|-------------|-----------------|
| δ (discretisation margin) | 0.000 | 0.151 | 0.49 |
| α (purity) | 34.5 | 1.89 | 0.7 |
| ħ_eff | 1×10⁻⁷ | 1.46 | 7×10⁶ |
| ψ (superposition) | ~1.8 | 1.07 | ~1.9 |
| Bandgap depth | ~0.25 | 0.48 | 0.2–0.7 (inconsistent) |
| Scattering peaks | 47–79 | 496 | 64–832 |
| Behaviour under pruning | collapses at >50 % | created by pruning | collapses early |

No single metric separates all three perfectly, but the combination of δ, ħ_eff and ψ gives a clear fingerprint. The crystal has δ=0 and negligible ħ_eff; the polycrystal has δ≈0.15 and ħ_eff≈1; the glasses have δ≈0.5 and huge ħ_eff.

### T.4 Limitations

- The Maxwell analysis is an analogy. I am not measuring real electromagnetic fields, only solving equations on a lattice populated by weights.
- The lattice size (16³) is small; larger grids might reveal finer details, but the computational cost would be high for 80+ checkpoints.
- The classification into “Bragg pattern” is heuristic – many glasses produce many Fourier peaks, so the script may over‑call Bragg patterns. The final phase label relies on the full score, not just the Boolean flag.
- The polycrystalline state is a single checkpoint. I cannot claim generality for this phase; it is an observation that such an intermediate state can exist after aggressive pruning.

### T.5 What I Take Away

The Maxwell analysis confirms that the “crystallisation” I observed is not just a metaphor. When a network lands on the exact Strassen coefficients, its weight distribution creates a dielectric structure that scatters light like a crystal – sharp peaks, anisotropic tensor, a small bandgap. When it stays in a glassy local minimum, the scattering is diffuse and the entropy is higher. The pruned robust model sits between the two: it has lost some of the perfect order but retains enough structure to function, and its scattering pattern reflects that mixed nature.

The numbers are reproducible. Anyone who runs the same checkpoints through the same Maxwell solver will get the same δ, α, and scattering profiles. That is the only claim I make.


---

Manuscript prepared: January 2026.
Author: grisun0
ORCID: 0009-0002-7622-3916*  
License: AGPL v3
