# Generative models as transport -- IAIFI Summer School 2026

This repository holds the hands-on material for the generative-models tutorial at the school, accompanying the lectures. See the companion post, [*Tutorial: Generative models as transport*](https://notes.mathisgerdes.com/posts/generative-models-as-transport/), as a starting point and for deeper explanations.

## Setup

We recommend you create a fresh environment on Python 3.11 or newer (`python -m venv` and conda both work), then install from this folder:

```bash
pip install -e .
python -c "import iaifi_gm as gm; gm.install_check()"
```

The second line is the install check. Ideally run it **in advance of the
session** so that environment problems surface early. If you use
[uv](https://docs.astral.sh/uv/), `uv sync` works too.

The notebooks run locally or on any Jupyter host. If you want to use Colab, open a notebook and run this in a cell first (the notebooks load checkpoints from the cloned repo):

```python
!git clone https://github.com/mathisgerdes/iaifi-generative-models.git
%cd iaifi-generative-models
!pip install -e .
```

Indicated runtimes, where reported, come from a relatively recent MacBook
running on CPU; expect different numbers on other hardware. The `(~N min)`
totals on notebook titles are a rough budget for the session.

## The notebooks

Exercise notebooks are `notebooks/<name>.ipynb`; complete versions are
`notebooks/<name>_solution.ipynb`.

- **nb0_jax**: a JAX/nnx warm-up (optional, ~30 min). Work through it in advance if JAX is new to you.
- **nb1_core**: the core notebook, from transport to diffusion.
- **nb2a_flow_matching**: the flow-matching track -- simulation-free losses and straighter paths.
- **nb2b_phi4**: the lattice-field-theory track -- a φ⁴ CNF built with bijx, plus an SU(3) flow demo.
- **nb2c_distillation**: the distillation track -- few-step inference from a trained diffusion teacher.

## Prerequisites

You should be comfortable with Python and NumPy, and with probability at the
level of a science graduate course (Gaussians, change of variables); some
neural-network training experience helps. JAX experience is not necessary --
Notebook 0 and the JAX links below cover the patterns the notebooks use.

## Advance resources

A few pointers.

1. [The Principles of Diffusion Models](https://arxiv.org/abs/2510.21890).
2. [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264).
3. [TinyML / Efficient Deep Learning course](https://hanlab.mit.edu) -- the training-free efficiency axis referenced in track 2c.
4. New to JAX: the [official JAX quickstart](https://docs.jax.dev/en/latest/quickstart.html) and the [UvA DL Course JAX/Flax notebooks](https://uvadlc-notebooks.readthedocs.io/) (explicitly PyTorch-comparative), alongside our Notebook 0.
5. [MIT "Intro to Flow Matching and Diffusion" course notes](https://diffusion.csail.mit.edu/) (Holderrieth & Erives, 6.S184) -- the best student-level treatment of exactly this material.

## What else is here

| Path | Contents |
| --- | --- |
| `src/iaifi_gm/` | helper package (`import iaifi_gm as gm`): targets, phi4, data, models, plotting, checkpoints, metrics |
| `checkpoints/` | pretrained model checkpoints used by the notebooks + diagnostic figures |
| `scripts/` | one training script per checkpoint (hyperparameters at the top) + an HMC reference + the SU(3) demo |

Every checkpoint the notebooks load can be retrained from its script in `scripts/`; the small 2D and φ⁴ models take seconds to a few minutes on a laptop CPU, while the fashion-MNIST ones want a GPU.
