# Handoff: grow the stone-benchmark reference on the lab PC

Paste this whole file into Claude Code on the lab Windows PC. That session has no other context.

## 1. Goal

Make the patient-stone reference bigger, then run the benchmark again.

The reference now has only 30 calcium oxalate stones (20 patients). The envelope is too wide: all 15 uric acid control stones fall inside it. The lab PC has micro-CT results for about 90 stones, but most aren't in the master spreadsheet. Merge them in.

## 2. Background (read before you start)

- Repo: `https://github.com/Reaper8202/stone-benchmark` (private). Read `README.md` and `stone_benchmark.py` first.
- The benchmark needs these values for each stone: patient, composition percentages, total volume (mm³), porosity (%), and surface area (mm²).
- Sphericity = π^(1/3) · (6V)^(2/3) / SA. The reference and the farm stones **must** use the same surface-area estimator. The primary estimator is Lindblad; Lorensen is the sensitivity check.
- The master spreadsheet's "Porosity %" means **closed porosity only**. Any porosity you merge in must use the same definition.

### Baseline to beat (Mac run, Oct 16 spreadsheet)

| Metric | Value |
|---|---|
| Usable stones / patients | 46 / 27 |
| Calcium oxalate reference | 30 stones, 20 patients |
| Uric acid inside the 90% envelope | 15 / 15 |
| Envelope cut (LOPO D², 90th pct) | 11.15 (χ²₂ would be 4.61) |
| AUC, both features / sphericity / porosity | 0.69 / 0.78 / 0.65 |
| Stones stretching the envelope | 9, 21a, 21b, 21c |

## 3. Data on this PC

Find the project root first; paths below are relative to it. Some names may differ, so confirm each one.

- `04_analysis/KS-*/`: per-stone pipeline outputs.
- `microct_results.csv`: about 90 stones with porosity, Feret diameter and sphericity.
- `microct_hu.csv`: approximate HU. Not needed here.
- `MASTER STUDY DATA - ALL.csv` (or its .xlsx): composition labels and the patient number.

### Known problems with the data

1. **Two ID systems.** The pipeline uses `KS-###` (fragments look like `KS-024.1-B`). The spreadsheet uses patient numbers with letters (`21a`). `KS-###` IDs are **not** patient IDs. No checked crosswalk exists, and the fragment-to-assay links are unverified.
2. **Two scanners.**
   - RX Solutions EasyTom at about 26.6 µm: a pilot batch of 27 and a general batch of 26.
   - Zeiss Xradia SickKids cohort, KS-035 to KS-053, at 11 to 12.7 µm.
   - KS-063 is stitched at 36.7 µm.

   Resolution changes the measured porosity and surface area, so record the scanner for every stone.
3. **Bad segmentations.** Low-density stones segment poorly: KS-004, KS-010, KS-024.1. SickKids exports KS-050, KS-051 and all of KS-053 must be redone. Flag these stones; don't silently drop them.
4. **Pipeline sphericity may not match.** Find out how `microct_results.csv` computes sphericity and which surface-area method it uses. If you can get Lindblad SA, compute sphericity with the formula above. If not, recompute SA the same way for every stone and note which estimator you used.
5. **Box order.** The pilot batch box order is reversed. This is already fixed in the pipeline code, so check you're reading the fixed outputs.

## 4. Steps

1. Clone the repo and install: `pip install -r requirements.txt`.
2. Make a `data/` folder in the repo. It's gitignored. Put all patient data there and nowhere else in the repo.
3. **Inventory.** List every `04_analysis/KS-*` folder and the columns of `microct_results.csv`. Note the scanner and voxel size for each stone. Write it to `data/inventory.csv`.
4. **Crosswalk.** Build `data/crosswalk.csv` with these columns: `ks_id, master_id, patient, scanner, voxel_um, link_confidence (verified|probable|unknown), note`. Use any ID mapping the lab already has. **If a link is ambiguous, stop and ask Derek. Don't guess.**
5. **Tidy table.** Build `data/reference_stones.csv`: one row per stone. Include composition columns `COM, COD, UA, UAD, Apatite, Brushite, Struvite, Cystine`, plus `volume_mm3, porosity_pct, sa_lindblad, sa_lorensen, scanner, voxel_um, flag`.
6. **Loader.** Add a `load_table(path)` function to `stone_benchmark.py`. It reads the tidy table and returns the same columns as `load()`. Also add a CLI switch, for example `--table data/reference_stones.csv`. Keep `load()` working for the master CSV.
7. **Runs.** Each run writes to its own output folder.
   - A: all usable stones.
   - B: A minus flagged stones (step 3 problems).
   - C: RX scanner only.
   - D: SickKids scanner only, if there are enough calcium oxalate stones.
   - E: verified crosswalk links only.
8. **Check scanner bias.** Compare the feature distributions between scanners within calcium oxalate stones (patient-cluster bootstrap or Mann-Whitney). If they differ, don't pool them. Report this.
9. **Commit code only.** Never commit `data/`, any CSV or xlsx, or `benchmark_outputs/`. Run `git status` before each commit.

## 5. What to send back to the Mac

Write `RESULTS_PC.md` (it's fine to commit it, since it has aggregate numbers only) with:

- Stone and patient counts for each run, by composition class.
- For each run: uric acid inside the envelope (n/N), the envelope cut, AUCs, and stones with LOPO D² > 5.
- The scanner-bias result.
- How many crosswalk links are verified, probable and unknown.
- Every assumption you made, and every open question for Derek.

Copy the run figures into `docs/figures/pc/`. They have no identifiers, so they can be committed.

## 6. Rules

- Patient data stays on this PC. The repo is private, but still: no CSV, xlsx or DICOM in git.
- Use the GitHub alias Reaper8202 for commits. Don't use a real name.
- Keep fragments from one patient together in every resample. The script already does this, so don't break it.
- Report numbers only from the runs you did. If a run has too few stones to mean anything, say so.
