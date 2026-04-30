---
created: "2026-03-10T03:35:18.517Z"
title: Update Dockerfile for advisor macOS deployment
area: tooling
files:
  - Dockerfile
  - docker-compose.yml
---

## Problem

Advisor needs to run this project on his macOS machine. The existing Dockerfile and docker-compose.yml exist but may not be tested/optimized for macOS (ARM/Apple Silicon). Need to ensure the full pipeline (data collection, surrogate training, RL training) works out of the box via Docker on a Mac.

## Solution

- Review and update existing Dockerfile for multi-arch support (amd64 + arm64)
- Test docker build and docker run on macOS/Apple Silicon
- Ensure all Python deps (PyElastica, TorchRL, numba, wandb) install cleanly in container
- Handle GPU vs CPU gracefully (advisor's Mac likely has no CUDA)
- Add clear usage instructions (README section or docker-compose comments)
- Consider pre-built image on Docker Hub (albatross679 account already configured)
