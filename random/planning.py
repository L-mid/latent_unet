
"""
What I actually do.

1). Rewriting repo parts from scratch -> Software engineering intuition. 

2). Math & Paper recreation -> Research layer. 

3). Debugging -> Debugging.

4). Training models -> Applided outcome.



Weighting:
70% on building/debugging codebase -> (what I'm doing now almost exactly).
20% on math/paper recreation -> conceptual mucsle matters, after repo stable.
10% on training runs -> treat as demonstrations, not main practice until base is reliable.

"""

# good sanity check
"""
Signs I'm doing it!

You can refactor something ugly into a clean, reusable abstraction.
You can debug errors by narrowing the cause systemically (not just random trial).
You can implement math from a paper without copying code.
You notice you're reading papers differently: less "this is alien" and more "ok, so they changes X in the loss, added Y in the block".
You can explain design tradeoffs in your repo, not just code them.

"""


# schedule allocation cal (only messy cal here)
"""
Time Split:

Base (repo/debugging): 28h (50%) (20 if day off)
Research (math/papers): 16h (30%)
Demonstrations (training): 8h (15%)
Stretch (exploration): 4h (5%)                               # idk if these %ages are correct


I think it's fair to say "do repo everyday for 4 hrs" so thats that. If not doing 8 hrs per week, use this as baseline. 


Rest of 4 hrs:
And then it's fair to say Reserch is done 50% of the remaining time.

2hs left:

Only 4 hours goes to stretch (exploration) a week. 1X per week. 
8 goes to training and vizusalizations (one day, or split over 2). 2X per week.


So that leaves: 4hrs * 7 = 28 taken (repo/debugging). 

28 left:
28 - 8 (training/viz)
20 - 4 (strech)

16 left: (goes to research) (= 8 * 2) = two days OR four sessions (4hrs)

"""


# Actcual schedule:
"""
So: 1x stretch, 2x training/viz, 4x research/math, 7x repo


Days (started on a wednesday): 

1 - research/math (4 hrs)   - repo (4 hrs) 🔹    (day off)
2 - research/math (4 hrs)   - repo (4 hrs) 
3 - training/viz (4 hrs)    - repo (4 hrs)
4 - research/math (4 hrs)   - repo (4 hrs)
5 - stretch (4 hrs)          - repo (4 hrs)
6 - research/math (4 hrs)   - repo (4 hrs)
7 - training/viz (4 hrs)    - repo (4 hrs)

"""

# On "look at other repos" vs "look at others":
"""
On "look at other repos" vs "look at others":

Potental allocation split:
Your-repo: 70% (work on your own)
External reading: 30% (hey look what are other people doing?) / Maybe toturials 

(More on this once i see schedule and am like damn wtf do i do i have here)

|
V
"""
import random

print(random.choices(["Look at other Repo.", "Do your Repo."], weights=[30, 70]))   #yay


# For when "['Look at other Repo']":
"""

There's a ramp and order to this:
1) Learn-by-running (tutorial level)
    Hugging Face's Diffusion Course: unit 1. (https://huggingface.co/learn/diffusion-course/en/unit1/1). 

2) Minimal PyTorch implementation
lucidrains/denoising-diffusion-pytorch + "The 'First Repo' plan (2nd down in "Learning from Repos", do it to completion, then move on).

3) Canonical/production-ish reference
    OpenAI inproved-diffusion. This repo. 
    Treat as "spec" once comfortable with above. 


Also consider finsishing that huggingface tutorial looked fun.
Maybe as stretch.


-- main
The idea is to learn parts, not the whole.

1) Set your target.
    One sentence goal: "From repo X I want to learn Y (e.g, EMA + training loop design)
    and reproduce Z (e.g., DDIM sampling) on a toy dataset".

    Scope guardrails: Decide what you will not learn this round (e.g., multi-node DDP, exotic schedulers).

    
2) Do three passes (increasing depth)

Pass A -- Map:
    Skim README, examples, configs/, train.py/CLI
    Collect nouns (entities: Dataset, UNet, Sampler) and verbs (fit, sample, save_ckpt).
    Draw a 5-box dataflow: config -> builder -> model -> loop -> outputs.

Pass B - Run:
    Create a minimal env; run the smallest example end-to-end with fixed seed.
    Save: exact CLI, config snapshot, stdout, first checkpoint, first samples.
    Note all side-effects (folders, logs, metrics). This becomes your ground truth.

Pass C - Trace:
    Set breakpoints or add temporary logs through the hot path:
    entrypoint -> config parse -> model build -> forward -> loss -> backward -> optimizer -> checkpoint.
    Write a short "walkthrough notebook" that prints tensor shapes and key values for one batch.

    
3) Build the skeleton map
    Entrypoints: scripts/CLI, main functions.
    Configs: where defaults come from; override order; how objects are built.
    Builders/factories: build_unet, build_sampler, etc.
    Training loop: where step happens; gradient/AMP/EMA; logging cadence.
    I/O: dataset pipeline -> transforms -> batching -> device.
    State: checkpoint format; resume logic; RNG seeding.
    Tests: what's covered vs missing.

4) Thin-slice reimplementation (parity first)
    Pick one slice and match it numerically on a toy problem:
        Example slices: beta schedule, noise add/denoise step, UNet forward, EMA update, DDIM step.
        Write a tiny pytest that:
            1: Imports their function to produce a value,
            2. Runs your function on the same inputs,
            3. Asserts closeness (e.g., atol=1e-6).
        Only when this passes, expand the slice (e.g., from one sampling step -> full sampler.)

5) Life patterns, not files
Extract design ideas, not just code:
    Config -> Builder -> Component pattern (clean seams, swappable blocks).
    Logging/metrics contract (step, epoch, sample hooks).
    Checkpoint contract (what keys live where; forward/backward compatibility).
    Error-handling + invarients (assert shapes/dtypes/devices early).

6) Write "explainer artifacts" as you go
Keep these minimal but durable:
    Call graph for one training step (who calls whom).
    Shape table for a single batch (N, C, H, W at each block).
    Glossary of repo-specific terms.
    Design tradeoffs you notices (why they chose X over Y).

7) Critique with prompts that reveal depth
    What are the three strongest engineering choices here?
    Where does the technical debt likely live?
    If you had to port this to a different data modality, what breask first?
    What's the smallest change that would cause silent wrongness? (add a test for it)

8) Reproduce a result (scaled-down)
    Fix all seeds; log library versions.
    Use a tiny dataset (e.g., 1k images) and a reduced model to reproduce a curve or sample grid.
    Capture a "before/after" plot or grid; record exact CLI/config to make it repeatable.

9) Rebuild the piece your way
Now implement the same subsystem in your own style (clean API, tests, docstrings). Keep:
    Unit tests (numerical parity),
    Golden samples (known outputs for fixed seed),
    Load/save parity (their ckpt <-> your ckpt if feasible).

10) Publish a learning note
Short write-up: goal, slice, parity proof, what you'd reuse, what you'd change. 
May become portfolio-quality evidence of understanding.


Concrete checklists
    Fast tools:
        Create a clean env; install the repo
        rg / ripgrep , to find entrypoints: rg "if __name__|argparse|Trainer|fit\(|build_|main\("
        pytest -k <keyword> to run a narrow test
        torch.manual_seed(0) (and CUDA/cudnn seeds) for reproducibility. 
        line_profiler or simple timers around forward/backward
        assert shape/device/dtype at module boundaries
        Open repo tree:
        Windows PowerShell: Get-ChildItem -Recurse - Depth 2 | Where-Object{$_.PSIsContainer} | Select-Object FullName


    Diffusion-repo specifics to map:
        Beta/noise schedule & parameterization (ε-prediction vs x₀-prediction)
        Timestep embedding path and where it enters blocks
        Attention variants (vanilla/window/flash) and where used
        EMA: update rate schedule and when applied (before/after optimizer.step)
        Mixed precision + GradScaler boundaries
        Checkpoint contents: model/optimizer/EMA/step and resume edge-cases
        Sample hooks (how images are denormed/saved)

    




"""



# This is for Legend marking. You'll find it in planning for some reason.
"""
For this Legend: 

Legend:
🔹: Implimented once - (copied direct)                      - You typed it in, but it's "dead muscle memory".
🔺: Known            - (wrote with some copying)            - Can explain dataflow, run small tests, still peaks for APIs.
🔸: Level 1          - (can write without help)             - Can re-impliment with <3 peaks per function, tests pass.
🔸🔸: Level 2        - (can write and change it a bit)     - Can re-impliment model+train loop cold; contract tests all pass.       
🔸🔸🔸: Level 3     - (can write and refactor)             - Can inject bugs and your asserts/localization catch them quickly; can swap varients easily without breaking.
🌟: Mastered         - (automatic)                          - Can design new abstractions, refactor repo-scale systems, anticipate faliure modes easily.


HOW TO:

=== The progression:

0) First contact (you can't code it yet)

Goal: Build understanding before code.
- One-paragraph summary: What is it? Inputs -> outputs -> invariants.
- Box-and-arrow sketch: tensors + shapes + key ops (e.g., x_t, eps_theta, alpha_t)
- Tiny glossary: 5-10 symbols with plain-English meaning (like ε).
- Golden tests you'd want: What must always hold true? (e.g., shape invarients, monotonic schedules, energy bounds).
Outcomes: a 1-pager you can explain out loud in 90 secs.


1) Guiden scaffold (copying allowed, but only structure)

Goal: understand the skeleton and seams.
- Write a minimal interface: forward(...), expected shapes, return type.
- Create failing tests first (red): shapes, dtype, device, determinism with a fixed seed.
- Paste only function signatures + TODOs, not solution code.
- Add debug prints/asserts for invariants (norms, ranges, NaNs).
Outcomes: a red test suite and a full file full of TODOs with a clear API.


2) Peeking protocol (disciplined looking)

Goal: learn, not memorize blindly. 
- Code attempt -> if stuck, peak at exactly one line/idea -> close it -> continue.
- When you peek, write why you were stuck (missing identity, shape, convention).
- After each peek, generalize the rule (e.g., "DDIM update is determinitic Euler step under reparam with σ=0")
- Keep a Stuck Log: symptom, root cause, the rule you learned.
Outcome: working code with 3-6 short "rules I learned" bullets.


3) Explain-back & invarients

Goal: prove understanding in your own words. 
- Write a comment block at the top, e.g: "We compute ... because...; if β_t->0; limits at t=0 look like ..."
- Encode invariants as asserts (e.g., x0_clip ∈ [-1, 1], schedule len == T).
- Add property tests: sample random shapes/dtypes/devices and ensure your function is total (no errors/NaNs).
Outcome: code that teaches how it workds, plus invarients that catch regressions. 


4) Blind-rebuild drill (no peeking)

Goal: convert understanding into recallable skill. 
- Close everything. From blank file, impliment module using only 1-pager, glossary, and tests.
- If you fail a test, don't open references yet. Use the Stuck Log and derive. 
- Only if blocked > 15 min, do a 60s peek, then immediately re-close and continue. 
Do this 3 times across a few days (spaced repetition). 
Outcome: Faster, fewer peeks.


5) Black-box parity & adversarial tests

Goal: hard validation that impl is correct.
- Compare against a reference impl with fixed seeds; assert max|y_ref-y| < ε .
- Adversarial inputs: extreme timesteps, tiny betas, random masks, mixed precision on/off.
- Gradient checks: finite-difference vs autograd on small tensors (catch sign mistakes).
Outcome: parity report + adversarial test file.


6) Transfer & variation

Goal: prove it's not rote. 
- Implement the variant (e.g., switch σ schedule; change time embedding; swap norm layer).
- Port to a diff framework or different tensor layout (NumPy, JAX, TensorFlow, (B, H, W, C)!). 
- Integrate into a new pipeline location (e.g., use in a different sampler loop).
Outcome: same tests pass with varients; notes on what changed and why.


"""






