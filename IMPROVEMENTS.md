# Improvement Suggestions — Stanford RNA 3D Folding Part 2

Prioritized, actionable improvements for the solution. Each section is ordered by estimated impact vs. effort.

---

## 1. Algorithm & Template Matching

### 1.1 Use Kabsch superposition after coordinate transfer (high impact, medium effort)

**Current:** Coordinates are transferred by alignment mapping only; no structural superposition.

**Improvement:** After transferring template coords to the target sequence, run Kabsch on the **aligned residue pairs** to get optimal rotation/translation, then apply that transform to the full transferred structure. This corrects for global orientation and scale differences between template and target.

**Where:** `src/template_matcher.py` → inside `transfer_coords_with_alignment` or in `predict_single_chain` after getting `coords`. Use `superposition.kabsch_rmsd` on the aligned subset of template vs. transferred coords, then `apply_superposition` on the full coords.

### 1.2 Increase alignment candidates and tune scoring (medium impact, low effort)

- **max_align:** You cap at 20 candidates for alignment; try 30–50 for targets where the best score is still low (<0.4).
- **Coverage weight:** Combined score is `0.7 * alignment_score + 0.3 * coverage`. For long targets, add a small **length-similarity** term so templates with similar length are slightly favored (reduces bad extrapolation).
- **Prefer global for similar length:** In `compute_alignment_score`, when `length_ratio > 0.85`, force `mode="global"` instead of `"auto"` so you get a full-sequence alignment.

### 1.3 Secondary-structure–aware template selection (high impact, high effort)

Use base-pairing (e.g. from RNAfold or ViennaRNA) to get dot-bracket for test and train sequences. Prefer templates that have similar secondary structure (e.g. same stems/loops). Even a simple stem-count or loop-count similarity can improve template choice for structured RNAs.

### 1.4 BLAST or faster alignment for prefilter (medium impact, medium effort)

Replace or augment k-mer prefilter with **BLAST** (via Biopython or subprocess) for the first pass. BLAST is optimized for this and can use E-value to get a better shortlist before running full NW/SW.

---

## 2. Multi-Chain & Long Targets

### 2.1 Use MSA data for long / ribosomal-like targets (high impact, high effort)

Competition provides **MSA/** FASTA files per target. For long targets (e.g. 9MME, 9ZCC), use MSA-derived conservation or consensus to:

- Refine which template regions to trust.
- Or feed MSA into a model that uses it (e.g. Boltz with MSA).

Download MSA files and add a small module that loads the MSA for a given `target_id` and optionally passes it to the predictor or template scorer.

### 2.2 Smarter extrapolation for target longer than template (medium impact, low effort)

**Current:** When target is longer than template, `get_residue_mapping` and fallback use linear interpolation/extrapolation, which can produce stretched structures.

**Improvement:**

- Cap extrapolation: e.g. only extrapolate up to 20% beyond template length; beyond that, fill with **A-form helix** segments parameterized to connect smoothly to the last transferred residue (use your existing `generate_aform_helix` with offset/orientation).
- Or use the **last few template residues** to define a local helix axis and extend along that direction with A-form geometry.

### 2.3 Hetero-oligomer chain order (medium impact, low effort)

For hetero-oligomers (e.g. 9G4P B:1;A:1), the submission row order may follow a specific chain order. Verify that `assemble_multimer` and the way you iterate `chains` match the submission’s residue order (e.g. by checking that `submission_targets` residue IDs match your concatenation order). If not, reorder chain predictions to match.

---

## 3. Diversity & Best-of-5

### 3.1 Order the five predictions by estimated quality (medium impact, low effort)

TM-score is **best-of-5**. So the first prediction that is “good” matters more than the last. Order your five predictions by **decreasing template score** (and optionally by structural diversity so the first few are not nearly identical). Put the best template first, then next best, etc., so the scorer is more likely to hit a good one early.

### 3.2 Explicit structural diversity for the five (medium impact, medium effort)

Besides “different templates” and “different copies,” add a simple diversity check: e.g. RMSD between pairs of the five. If two predictions have very low RMSD, replace one with another template or a slightly perturbed version so the five are more spread out in structure space and best-of-5 has a better chance.

### 3.3 Ensemble with A-form baseline (low impact, low effort)

For targets with weak template scores (e.g. <0.25), set one of the five predictions to an **A-form helix** (your `generate_aform_helix`). For some targets, a regular helix may score better than a bad template.

---

## 4. Code Quality & Robustness

### 4.1 Temporal cutoff / data leakage (high impact for fairness, low effort)

Competition rules often restrict use of structures **after** a certain date. `train_sequences` has `temporal_cutoff`. Filter training structures so you only use templates whose **release date** (or a date you infer from `temporal_cutoff` of the test target) is before the test target’s cutoff. Otherwise you may be inadvertently using “future” data.

### 4.2 Unit tests for alignment and transfer (medium impact, medium effort)

Add tests (e.g. in `tests/`) for:

- `alignment.needleman_wunsch` / `smith_waterman`: known pairs of sequences and expected score/mapping.
- `get_residue_mapping`: aligned_pairs + lengths → expected mapping array.
- `transfer_coords_with_alignment`: small template coords + alignment → expected shape and no NaNs.

This will make tuning alignment and transfer safer.

### 4.3 Config file for hyperparameters (low impact, low effort)

Move magic numbers to a small config (e.g. `config.yaml` or `src/config.py`): k-mer size, top_n prefilter, max_align, gap open/extend, score weights, length_ratio threshold, etc. Makes ablation and submission variants easier.

### 4.4 Logging and reproducibility (low impact, low effort)

- Use a fixed **random seed** everywhere (you already use hashes for some RNG; ensure one global seed at the start of `run.py`).
- Log for each target: chosen template IDs, alignment score, and which of the five came from which source (template vs. copy vs. perturbation). Saves to a small JSON/CSV next to the submission for debugging and ablation.

---

## 5. Evaluation & Validation

### 5.1 Local TM-score on training (high impact, high effort)

Implement or call **TM-score** (or use a Python wrapper if available) and run it on a **validation split** of training targets: hide some training structures, predict them from sequence using the rest as templates, and compute TM-score. This gives a local proxy for the leaderboard and lets you tune template selection and diversity without submitting.

### 5.2 Sanity checks on submission (medium impact, low effort)

Before writing `submission.csv`:

- No NaN/Inf in coordinates.
- Residue count per target matches `sample_submission`.
- Optional: check that C1'–C1' distances along the chain are in a plausible range (e.g. 3–8 Å) for a random sample of rows; flag large outliers.

You already validate format; add these as optional checks in `submission.py`.

---

## 6. Deep Learning & External Tools

### 6.1 Boltz integration (already started)

- Make Boltz output parsing robust to different CIF/PDB layouts (column order, atom naming).
- For long sequences, consider running Boltz on **chunks** (e.g. 200–300 nt) and then stitching with overlap and Kabsch, or only use Boltz for short targets and keep template-only for long ones.

### 6.2 RhoFold2 or other RNA-specific models (high impact, high effort)

If you have GPU, add RhoFold2 (or similar) as an alternative to Boltz for RNA. Use it for targets with no good template (e.g. alignment score < 0.3) and optionally ensemble with template predictions.

---

## 7. Competition Tactics

### 7.1 Multiple submission variants

- **Variant A:** Current pipeline.
- **Variant B:** Same but order the five by template score (best first).
- **Variant C:** Replace the 5th prediction with A-form for low-score targets.
- **Variant D:** Increase max_align to 40 and add length-similarity term.

Submit A/B/C/D and keep the best.

### 7.2 Per-target strategy

For a few high-weight or uncertain targets, maintain a small “override” list (e.g. target_id → list of template IDs to try first) based on manual inspection or previous submission feedback, and use that in template selection when available.

---

## Quick wins (implement first)

1. **Order the five predictions by template score** (best first).
2. **Use Kabsch** on aligned residues after transfer and apply the transform to the full structure.
3. **Filter templates by temporal_cutoff** to avoid data leakage.
4. **Add length-similarity** to the combined alignment score and **force global alignment** when length ratio > 0.85.
5. **Config + logging**: one seed, one config, log template IDs and scores per target.

After that, focus on local TM-score validation and MSA usage for long targets.
