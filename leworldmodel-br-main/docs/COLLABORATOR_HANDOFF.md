# Collaborator Handoff: Brain-Regularized LeWorldModel

## 1. What this project is about

This project asks a narrow but interesting question:

Can a compact predictive world model learn *better latent states* if we lightly regularize it to also predict human brain responses to naturalistic video?

The intended contribution is **not** "we built a brain-like agent" and **not** "we fused two giant foundation models."

The actual claim we are trying to test is:

**Cortical supervision may act as an inductive bias on predictive latent representations.**

In plainer terms: if a latent state is useful both for predicting future visual experience *and* for explaining measured brain responses, maybe it preserves more abstract or human-salient structure than a purely predictive latent alone.

The project is currently in a **validation-first** phase. We are not yet trying to maximize benchmark scores or make strong planning claims. We are trying to determine whether the idea has real signal and what interface between world-model latents and brain supervision actually works.

## 2. Background: what LeWorldModel does

LeWorldModel is the backbone idea we started from.

At a high level:

- It takes raw visual observations.
- An encoder maps each frame to a latent representation.
- A predictor tries to predict the next latent from previous latents.
- Training is end-to-end.
- It uses a latent regularizer to prevent collapse and keep the representation well-behaved.

You can think of it as a compact JEPA-style predictive model:

- frame `x_t`
- latent `z_t = E(x_t)`
- predicted next latent `\hat{z}_{t+1} = P(z_{\le t})`

The core world-model objective is roughly:

`L_world = L_pred + lambda_sig * L_sigreg`

where:

- `L_pred` encourages the predictor to match the next latent
- `L_sigreg` keeps the latent distribution from collapsing and promotes a healthier geometry

The reason this backbone is attractive here is that it is:

- compact
- stable
- already designed to produce predictive latents from pixels

So it is a good substrate for testing representation regularization.

## 3. Background: what TRIBE does

TRIBE is the other conceptual source.

At a high level:

- It takes rich naturalistic stimuli such as video, audio, and text
- It builds representations over time
- It predicts fMRI responses across the cortex

The important thing for us is not "TRIBE as a whole multimodal system." The important thing is the **brain-encoding idea**:

- a learned representation can be evaluated or shaped by how well it predicts measured cortical responses

TRIBE also matters because it explicitly deals with the fact that fMRI is a **slow** signal. Brain measurements are not aligned to individual video frames. They are temporally delayed and temporally blurred.

That temporal mismatch is one of the central scientific and engineering issues in our project.

## 4. Our actual idea

We are **not** symmetrically merging LeWorldModel and TRIBE.

The design is asymmetric:

- Keep a LeWorldModel-style predictive latent backbone
- Add a TRIBE-style auxiliary brain head
- Use the brain loss as a regularizer, not as the main objective

The original intended architecture was:

- `z_t = E(x_t)`
- `\hat{z}_{t+1} = P(z_{\le t})`
- `h_tau = G(z_{1:T})`
- `\hat{y}_tau = B(h_tau)`

where:

- `E` is the frame encoder
- `P` is the predictive latent dynamics model
- `G` is a temporal aligner from fast video latents to slow fMRI timescale
- `B` is a parcel-level brain readout head
- `y_tau` is the measured fMRI target at TR index `tau`

The original intended loss was:

`L = L_pred + lambda_sig * L_sigreg + lambda_brain * L_brain`

with:

- `L_pred`: predictive world-model loss
- `L_sigreg`: latent anti-collapse / geometry regularizer
- `L_brain`: parcel-level brain prediction loss

The original conceptual bet was:

- predictive training gives you a meaningful latent state
- cortical supervision may refine that latent toward more abstract structure

## 5. The key math intuition

### 5.1 Predictive world-model path

For each frame sequence:

- `x_1, ..., x_T`
- `z_t = E(x_t)`
- `\hat{z}_{t+1} = P(z_{\le t})`

and the predictive loss is conceptually:

`L_pred = sum_t || \hat{z}_{t+1} - z_{t+1} ||^2`

This is what makes the model a predictive world model rather than a pure encoding model.

### 5.2 Brain-encoding path

The brain head does not really want a single frame latent. It wants a slower representation that matches the timescale of BOLD fMRI:

`h_tau = G(z_{1:T})`

`y_hat_tau = B(h_tau)`

and the brain loss is some combination of:

- MSE on parcel predictions
- correlation-aware objective

conceptually:

`L_brain = alpha * MSE(y_hat, y) + beta * (1 - corr(y_hat, y))`

### 5.3 Why timescale mismatch matters

This is the most important math issue in the project.

fMRI is not a direct readout of an instantaneous latent `z_t`. A better approximation is:

`y_tau ≈ sum_{k=0}^K h_k * B(c_{tau-k}) + eta_tau`

where:

- `c_tau` is some latent neural-style representation at slower timescale
- `h_k` is an HRF-like temporal kernel
- `eta_tau` is noise

So if we try to supervise `z_t` directly against `y_tau`, the useful signal can get badly attenuated.

We wrote this informally as:

`delta_eff = rho_align * delta`

and therefore the benefit of brain supervision scales like:

`alpha^2 + lambda * rho_align^2 * delta^2 > beta^2`

Interpretation:

- if temporal alignment is poor, `rho_align` is small
- usefulness falls quadratically in `rho_align`
- so a good idea can look bad simply because the temporal interface is wrong

That is a genuine mathematical risk in the idea, not just an engineering nuisance.

## 6. What we built in code

We built a compact research scaffold with:

- LeWorldModel-style backbone
- slow-state / HRF-aware brain path
- synthetic smoke-test dataset
- Algonauts preprocessing pipeline
- staged Hyak Slurm jobs
- run reporting utilities

Key code locations:

- model backbone: [`/Users/divijchawla/Documents/important docs/leWorldModel-br?/src/slow_state_wm/models/backbone.py`](/Users/divijchawla/Documents/important docs/leWorldModel-br?/src/slow_state_wm/models/backbone.py)
- brain path / temporal alignment: [`/Users/divijchawla/Documents/important docs/leWorldModel-br?/src/slow_state_wm/models/brain.py`](/Users/divijchawla/Documents/important docs/leWorldModel-br?/src/slow_state_wm/models/brain.py)
- training loop: [`/Users/divijchawla/Documents/important docs/leWorldModel-br?/src/slow_state_wm/trainer.py`](/Users/divijchawla/Documents/important docs/leWorldModel-br?/src/slow_state_wm/trainer.py)
- preprocessing: [`/Users/divijchawla/Documents/important docs/leWorldModel-br?/src/slow_state_wm/preprocess_algonauts.py`](/Users/divijchawla/Documents/important docs/leWorldModel-br?/src/slow_state_wm/preprocess_algonauts.py)
- config preparation: [`/Users/divijchawla/Documents/important docs/leWorldModel-br?/src/slow_state_wm/prepare_algonauts.py`](/Users/divijchawla/Documents/important docs/leWorldModel-br?/src/slow_state_wm/prepare_algonauts.py)
- result matrix: [`/Users/divijchawla/Documents/important docs/leWorldModel-br?/src/slow_state_wm/report_matrix.py`](/Users/divijchawla/Documents/important docs/leWorldModel-br?/src/slow_state_wm/report_matrix.py)
- staged Hyak jobs: [`/Users/divijchawla/Documents/important docs/leWorldModel-br?/slurm`](/Users/divijchawla/Documents/important docs/leWorldModel-br?/slurm)

## 7. Experimental design so far

We used a staged set of experiments.

### V0

World model only.

- Train predictive latent backbone
- No brain supervision

Question:

Can the model learn stable predictive latents on these movie slices?

### V1

Frozen-latent brain readout.

- Freeze backbone
- Train temporal aligner + brain head only

Question:

Are the learned predictive latents already brain-readable?

### V2

Joint regularization.

- Keep predictive training on
- Unfreeze part of the predictive stack
- Add brain loss as auxiliary supervision

Question:

Does brain supervision improve the latent representation itself?

### Additional ablations

#### `fast_latent`

Apply brain supervision to a faster latent path rather than the slow pooled state.

Question:

Is the full slow-state alignment helping, or is it over-constraining the signal?

#### `no_temporal`

Remove the temporal alignment module.

Question:

Is the explicit temporal aligner helping on these slices?

#### `no_pred`

Remove predictive loss during the joint phase.

Question:

Does the method only work when it stays world-model-first?

This ablation is particularly important because it checks whether the whole project really needs predictive latent learning or whether it is effectively becoming "just a brain regressor."

## 8. Results so far

### `life` slice

Best validation parcel correlations:

- `V0`: `-0.000609`
- `V1`: `0.003577`
- `V2`: `0.002175`
- `fast_latent`: `0.010782`
- `no_temporal`: `0.010457`
- `no_pred`: `0.009877`

Interpretation:

- baseline `V1` and `V2` were at least mildly positive
- `fast_latent` and `no_temporal` were both much better than baseline `V2`
- the idea showed signal, but not in the exact form we initially expected

### `figures` slice

Best validation parcel correlations:

- `V0`: `0.000148`
- `V1`: `-0.000057`
- `V2`: `0.000311`
- `fast_latent`: `0.019998`
- `no_temporal`: `0.001224`
- `no_pred`: `0.023837`

Interpretation:

- baseline slow-state `V2` again did not win
- `fast_latent` was very strong
- `no_pred` was even stronger on this slice
- `V1` was essentially flat

### Current scientific reading

The evidence currently suggests:

- there is real slice-level signal in the setup
- the exact slow-state implementation we started with is **not** clearly the best interface
- `fast_latent` looks promising across completed slices
- the role of predictive loss is now more ambiguous than we initially expected

At this point, the data do **not** support a strong claim that our original slow-state aligned `V2` is the best version.

They *do* support the weaker but still interesting claim that:

**brain supervision changes the representation in meaningful ways, but the correct interface is still an open question.**

## 9. What failed, and what kind of failure it was

This is important.

### 9.1 Pure engineering failures

We had several failures that were engineering failures, not theory failures.

#### Hyak job structure

Originally we ran:

- fetch
- preprocess
- `V0`
- `V1`
- `V2`
- ablations

inside one long dependent chain / large job structure.

That timed out.

This was an **engineering failure**, not evidence against the idea.

Why:

- stages that should have been separate were bundled together
- timeouts were destroying too much progress at once
- environment setup and data fetch were mixed with actual experiments

We fixed this by introducing staged jobs:

- fetch only
- preprocess only
- one phase per job

This is the correct design, and it worked.

#### Video decoding / environment issues

At one point, `torchvision` on Hyak lacked the expected video decoder path, and preprocessing failed until we added a PyAV fallback.

This was also an **engineering failure**, not a conceptual failure.

#### Repeated environment bootstrap overhead

Some jobs spent too much time recreating or refreshing package state.

Again: **engineering failure**, not theory failure.

### 9.2 Where failure could genuinely come from the math / idea

There are also ways the idea itself could fail, and those are different.

#### Temporal alignment may genuinely be the wrong interface

The current data are not favoring our original slow-state aligned `V2`.

This *could* be because:

- our implementation is still imperfect

but it also *could* reflect a genuine modeling fact:

- the useful brain-supervision signal on these data may attach more directly to a faster latent path than to our current pooled slow state

This is a place where failure may come from the **actual idea/interface**, not just engineering.

#### Predictive loss may not help as much as expected

Originally we thought:

- `no_pred` should clearly be worse

That is not what we are currently seeing on `figures`.

This means one of two things:

1. our current training setup is still under-tuned, or
2. the mathematical story "brain regularization must stay world-model-first" is weaker than we assumed

That is a real conceptual pressure point.

#### Slice instability / low-signal regime

Another genuine possibility is that we are operating in a regime where:

- datasets are too small
- slices are too heterogeneous
- correlations are tiny and noisy

In that regime, model ranking itself may be unstable.

That would not mean the idea is wrong, but it would mean our current experimental unit is too weak to adjudicate it cleanly.

## 10. Where the project stands right now

### What is solid

- the overall research direction is still alive
- the infrastructure now basically works
- we have real results on multiple slices
- the model family is sensitive enough to show differences across ablations

### What is not settled

- whether slow-state alignment is actually the right interface
- whether predictive loss is essential in the final best version
- whether the best-performing variant is robust across more slices and subjects

### Immediate next step

Continue the staged cross-movie rigor pass:

- finish `wolf`
- finish `bourne`
- compare all four completed slices:
  - `life`
  - `figures`
  - `wolf`
  - `bourne`

Then decide whether the lead interface is:

- baseline `V2`
- `fast_latent`
- `no_temporal`
- or a revised formulation

## 11. Practical orientation for a new collaborator

If you are joining this project fresh, the fastest useful mental model is:

1. Think of the project as **representation learning with auxiliary brain supervision**, not as neuroscience grand theory.
2. The key scientific issue is **where** to attach the brain loss.
3. The key systems issue is **how** to run the experiments reliably on Hyak without recomputing everything.

If you want to orient in code first:

1. Read the README.
2. Read `models/backbone.py`.
3. Read `models/brain.py`.
4. Read `trainer.py`.
5. Read `prepare_algonauts.py`.
6. Read the staged Slurm scripts.

If you want to orient in experiment logic first:

1. Understand `V0`, `V1`, `V2`
2. Understand the three ablations
3. Compare `life` vs `figures`
4. Notice that the lead hypothesis is already being challenged by the data

## 12. Is this on GitHub or otherwise shareable?

Right now, **there is no Git repo in the local workspace**. Running `git` in the project root reports that it is not a repository, so there is no local GitHub remote or branch to hand off today.

So:

- the code exists locally in this workspace
- the working research copy also exists on Hyak under:
  - `/gscratch/stf/dc245/leworldmodel-br/repo`
- but there is not yet a clean GitHub repository wired up from this workspace

That means the project is shareable in practice via:

- this workspace directory
- the Hyak repo directory
- this handoff document

but **not yet via an existing GitHub link**.

If we want a clean collaborator handoff, the next administrative step should be:

1. initialize a proper Git repository here if needed
2. connect it to GitHub
3. push the current code and docs
4. optionally add a short experiment status note summarizing which Hyak jobs completed and where the result files live

## 13. Bottom line

The project is in a good exploratory state.

The original high-level idea remains worth pursuing.

Several of the setbacks so far were **engineering failures** and have been or can be fixed without changing the scientific direction.

At the same time, the data are already telling us something important:

the **specific slow-state aligned formulation we started from is not yet the empirical winner**.

That is not bad news. It is the core thing we are learning.

The project is no longer "does this idea have any signal at all?"

It is now:

**which brain-regularization interface actually improves predictive visual representations on real movie-fMRI data?**
