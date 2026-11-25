<picture>
  <source media="(prefers-reduced-motion: reduce)" srcset="assets/logo.svg">
  <img src="assets/logo.gif" alt="Pixel-art brain with a terminal inside" width="128" height="128">
</picture>

# Latent Machines

Neural models as programmable machines. **[Project page](https://jgalego.github.io/latent-machines/)**

Experiments with programs compiled into neural weights and learned control over persistent machine state.

Every experiment so far has a first recorded run. Results:

- [E00: memory](results/E00/README.md) and [E01: streaming machines](results/E01/README.md). A delta-rule memory, compiled state machines, and a delta-rule stack matched their integer oracles on every tested trace, up to 10^6 actions per variant.
- [E02: learned interfaces](results/E02/README.md). A learned controller on a frozen SmolLM2-135M answered more record-update questions correctly through a typed interface than through a text tool on programs longer than those seen in training. The delta-rule executor gave the same answers as the native one.
- [E03: latent feedback](results/E03/README.md). Feeding each machine step back into the model as a trained vector raised accuracy on long multi-step programs. A control with no feedback in the vector did nearly as well, so the gain is mostly trained steering.
- [E04: dataflow](results/E04/README.md) to [E08: search with a verifier](results/E08/README.md). Dataflow graphs, graph and Datalog fixed points, cellular automata, optimizer-written memory, and a SAT verifier matched their references in float32 and float64 within declared limits. They were rarely faster than conventional algorithms and failed in bfloat16 where precision ran out.
- [E09: state in the machine](results/E09/README.md). A controller that reads one statement at a time, with the machine holding all state, made no errors up to 256 statements, where full-context controllers fell to about 0.13. On 8 statements it never trained on it was right on 7, against 2 for a random untrained model. One prediction held vacuously, one failed, and one follow-up prediction was inconclusive.

Next are [E10: machine layer](EXPERIMENTS.md#e10---a-machine-layer-inside-the-model), which moves the memory inside a model layer, and [E11: learning from outcomes](EXPERIMENTS.md#e11---learning-to-use-the-machine-from-outcomes), which trains a controller from the answer loss alone. Both are proposals.

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

See [EXPERIMENTS.md](EXPERIMENTS.md) for the experiment program, repository layout, correctness gates, and reproducibility contract.

The experiments distinguish:

- **Machine:** what state transitions should execute?
- **Backend:** which layer family or conventional implementation executes them?
- **Interface:** how does a controller observe and invoke the machine?
- **Task:** does the combination improve correctness, generalization, or cost?

Results so far are initial runs with one or a few seeds. Larger machines, other models, and a shared compiler wait on the [scope gates](EXPERIMENTS.md#gates-for-expanding-scope).

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
