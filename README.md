<picture>
  <source media="(prefers-reduced-motion: reduce)" srcset="assets/logo.svg">
  <img src="assets/logo.gif" alt="Pixel-art brain with a terminal inside" width="128" height="128">
</picture>

# Latent Machines

Neural models as programmable machines.

Research into programs compiled into weights, persistent memory, and learned control over exact execution.

> Can a model use a machine through an internal interface instead of textual tool calls?

> When does this improve correctness, generalization, or cost?

## Scope

- Learned controllers with explicit memory and machine operations.
- Programs compiled into neural weights.
- Native and tensor implementations of the same state transitions.
- Experiments that separate interface benefits from implementation benefits.

"Exact" means following specified computational semantics within stated numerical and resource limits. It does not exclude weight-space programming. "Persistent" means state survives computation steps; persistence across runs requires saving and restoring it.

## Research

See [RESEARCH.md](RESEARCH.md) for the source-backed landscape, programmable layer families, and alternative computation models.

See [EXPERIMENTS.md](EXPERIMENTS.md) for the proposed repository layout, experiment dependencies, correctness gates, and reproducibility contract.

The program separates four questions:

1. **Machine:** what state transitions should execute?
2. **Backend:** which layer family or conventional implementation executes them?
3. **Interface:** how does a controller observe and invoke the machine?
4. **Task:** does the combination improve correctness, generalization, or cost?

Start with executor conformance on a small memory and streaming machine. Then run the four-way controller comparison: no machine, textual tools, internal native execution, and internal compiled tensor execution. Broaden to latent ticks, dataflow, graphs, and spatial machines only after those foundations work.

This repository currently contains documentation only. The implementation layout and experiments are proposals, not existing packages or reproduced results.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for code style and research standards: minimal code and dependencies, uv, pre-commit, small frequent commits, commit messages, reproducible experiments, and mandatory result verification.

## License

[Apache License 2.0](LICENSE).
