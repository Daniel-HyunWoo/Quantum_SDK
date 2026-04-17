# Quantum SDK — Gemini Guide

## Defaults

- Default to Korean unless the user explicitly asks otherwise.
- Be concise and direct.
- Do not repeat prior context unless it is necessary for the current task.
- When giving shell commands, prefer minimal commands that can be run as-is.
- Before large or potentially disruptive edits, briefly state the plan.
- When uncertain, state the uncertainty explicitly instead of guessing.

- Default environment: `quantum_env` (Conda or venv, Python 3.11)
- Default runtime context: remote SSH Ubuntu in VS Code (A100 GPU)
- Primary workspace: `/home/ubuntu/Quantum_SDK`

## Gemini-specific notes

- Treat this file as the repository-level operating guide for Gemini in this workspace.
- Inspect real repository files before making claims; do not rely on generic prior assumptions.
- Prefer actionable outputs over long explanations: exact file paths, exact commands, and artifact-grounded statements.

## Scope and task routing

- The primary task family is developing and testing quantum algorithms across multiple SDKs.
- Identify the target SDK before proposing edits or interpretations:
  - Qiskit (IBM Quantum)
  - PennyLane (Xanadu)
  - CUDA-Q (NVIDIA)
  - cuQuantum (NVIDIA)

## Source of truth

- The project relies on the following primary frameworks:
  - **CUDA-Q**: `cudaq` python package (Version 0.14.0+)
  - **cuQuantum**: `cuquantum-python` (includes `cuPauliProp` 0.3.0+ via `nvmath-python`)
  - **PennyLane**: `pennylane` (requires `pennylane-lightning-gpu` for A100 execution)
  - **Qiskit**: `qiskit` 1.0+ and `qiskit-aer`

- Preferred execution flow (GPU Optimization):
  - Always default to GPU-accelerated backends when available.
  - For PennyLane, prefer `lightning.gpu` over `default.qubit`.
  - For Qiskit, utilize `AerSimulator` with `device='GPU'`.
  - For cuQuantum, utilize explicit GPU memory management when working with large state vectors or tensor networks.

## Edit boundaries

- Prefer minimal diffs.
- Do not change public interfaces, file layout, or path conventions unless needed for the requested task.
- Maintain the established directory structure:
  - `CUDA-Q/`
  - `PennyLane/`
  - `Qiskit/`
  - `cuQuantum/`
- When integrating user custom modules (like `easy-module`), isolate them in a specific sub-folder (e.g., `cuQuantum/easy-module/`) and organize general scripts into standard categories (`Basics/`, `Examples/`, `Applications/`).

## Debugging, implementation, and review behavior

- When debugging GPU issues, first verify CUDA compatibility and package versions (e.g., `cupy-cuda12x` vs `cupy`).
- When implementing, prioritize established tutorials and documentation conventions over ad-hoc scripts.

## Interaction and output style

- No unnecessary verbosity.
- Answer directly and immediately.
- Maintain a professional and concise tone.
- Avoid emotional or overly friendly language.
- Minimize unnecessary formatting.

- Prefer:
  - short diagnosis
  - concrete next action
  - exact file paths
  - exact commands

- Avoid:
  - generic high-level advice without repo grounding
  - repeated obvious context

## Safety and scope

- Do not execute destructive operations without explicit user approval.
- Do not modify unrelated files.
- Do not assume missing context when the answer materially depends on it.

