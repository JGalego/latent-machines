<picture>
  <source media="(prefers-reduced-motion: reduce)" srcset="assets/logo.svg">
  <img src="assets/logo.gif" alt="Pixel-art brain with a terminal inside" width="128" height="128">
</picture>

# Latent Machines

Neural models as programmable machines. **[Project page](https://jgalego.github.io/latent-machines/)**

Experiments with programs compiled into neural weights and learned control over persistent machine state.

E00 and E01 have CPU conformance results: delta-rule memory, compiled state machines, and a delta-rule stack matched their references on the tested traces, up to 10^6 actions per variant ([E00](results/E00/README.md), [E01](results/E01/README.md)). Learned controllers and interface comparisons are not implemented yet.

> Can a model use a machine through an internal interface instead of textual tool calls?

> When does this improve correctness, generalization, or cost?

## Scope

- Learned controllers with explicit memory and machine operations.
- Programs compiled into neural weights.
- Native and tensor implementations of the same state transitions.
- Experiments that separate interface benefits from implementation benefits.

"Exact" means following specified computational semantics within stated numerical and resource limits. It does not exclude weight-space programming. "Persistent" means state survives computation steps; persistence across runs requires saving and restoring it.

## Research

See [RESEARCH.md](RESEARCH.md) for related systems, their evidence limits, and candidate layer families.

See [EXPERIMENTS.md](EXPERIMENTS.md) for the proposed repository layout, experiment dependencies, correctness gates, and reproducibility contract.

The experiments distinguish:

- **Machine:** what state transitions should execute?
- **Backend:** which layer family or conventional implementation executes them?
- **Interface:** how does a controller observe and invoke the machine?
- **Task:** does the combination improve correctness, generalization, or cost?

The next study compares a model without a machine, textual tools, internal native execution, and internal compiled tensor execution. Later experiments cover latent feedback, dataflow, graphs, and spatial machines, subject to the [scope gates](EXPERIMENTS.md#gates-for-expanding-scope).

Implementation follows the [first implementation slice](EXPERIMENTS.md#first-implementation-slice). E02-E08 remain proposals.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for code, [writing](CONTRIBUTING.md#14-write-for-the-reader), and research standards.

## Citing

See [CITATION.cff](CITATION.cff), or use:

```bibtex
@software{galego2026latentmachines,
  author  = {Galego, João},
  title   = {Latent Machines: neural models as programmable machines},
  year    = {2026},
  url     = {https://github.com/JGalego/latent-machines},
  license = {Apache-2.0}
}
```

## License

[Apache License 2.0](LICENSE).
