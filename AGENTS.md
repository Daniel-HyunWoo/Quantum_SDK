# Quantum SDK — Agent Guide

## Defaults

- Default to Korean unless the user explicitly asks otherwise.
- Be concise and direct.
- Do not repeat prior context unless it is necessary for the current task.
- When giving shell commands, prefer minimal commands that can be run as-is.
- Before large or potentially disruptive edits, briefly state the plan.
- When uncertain, state the uncertainty explicitly instead of guessing.

- Default environment: `quantum_env`
- Default runtime context: remote SSH Ubuntu in VS Code (A100)
- Primary workspace: `/home/ubuntu/Quantum_SDK`

## Context & Routing

This file defines the sub-agents and specialized workflows available in the `Quantum_SDK` workspace.
Before executing tasks involving multi-file refactoring, deep quantum API research, or large-scale artifact analysis, check if a sub-agent is a better fit.

## Available Sub-Agents

1. **`codebase_investigator`**
   - Role: System-wide search, architectural analysis, and deep API mapping.
   - Triggers:
     - "How does PennyLane integrate with CUDA-Q in this repo?"
     - "Find all instances where we manage GPU memory manually in Qiskit."
     - Unclear or vague bugs requiring deep context retrieval.

2. **`generalist`**
   - Role: Broad execution, batch tasks, and repetitive cross-file edits.
   - Triggers:
     - "Update all `import cupy` to `import cupy_cuda12x` across the `cuQuantum/` folder."
     - "Format all `.py` files in `CUDA-Q/` according to standard PEP8."

## Sub-agent rules
- Never run multiple sub-agents concurrently if they might mutate the same files.
- Always provide a clear, highly detailed `objective` or `request` when delegating to a sub-agent.
- Rely on sub-agents to "compress" context when doing large-scale file reads or complex searches across the four primary SDKs (`Qiskit`, `PennyLane`, `CUDA-Q`, `cuQuantum`).