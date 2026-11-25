<picture>
  <source media="(prefers-reduced-motion: reduce)" srcset="assets/logo.svg">
  <img src="assets/logo.gif" alt="Pixel-art brain with a terminal inside" width="128" height="128">
</picture>

# Latent Machines

Neural models as programmable machines. **[Project page](https://jgalego.github.io/latent-machines/)**

Experiments with programs compiled into neural weights and learned control over persistent machine state.

Every planned experiment, E00 to E08, has a first recorded run. E00 and E01 have CPU conformance results: delta-rule memory, compiled state machines, and a delta-rule stack matched their references on the tested traces, up to 10^6 actions per variant ([E00](results/E00/README.md), [E01](results/E01/README.md)). In E02, a learned controller on a frozen SmolLM2-135M answered more record-update questions correctly through a typed internal interface than through a text tool on programs longer than those seen in training, and the delta-rule executor gave the same answers as the native one ([E02](results/E02/README.md)). In E03, feeding each machine step back into the model as a trained vector raised accuracy on long multi-step programs, but a control whose vectors carried no feedback did nearly as well, so the gain is mostly trained steering ([E03](results/E03/README.md)). Dataflow graphs, graph and Datalog fixed points, cellular automata, optimizer-written memory, and a SAT verifier matched their references in float32 and float64 within declared limits, were rarely faster than conventional algorithms, and failed in bfloat16 where precision ran out ([E04](results/E04/README.md)-[E08](results/E08/README.md)).

Next is [E09](EXPERIMENTS.md#e09---state-in-the-machine-not-the-context), pre-registered before any run: a controller that reads one statement at a time, with the machine holding all state, against full-context controllers on long programs. The plan then moves the memory inside a model layer ([E10](EXPERIMENTS.md#e10---a-machine-layer-inside-the-model)) and tests learning to use it from the answer loss alone ([E11](EXPERIMENTS.md#e11---learning-to-use-the-machine-from-outcomes)); both are proposals.

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
