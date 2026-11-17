"""A frozen language model that turns text chunks into feature vectors.

Text is processed as a list of chunks, such as a preamble and one chunk per
statement. Each chunk is tokenized on its own and the token ids are
concatenated, so token boundaries are identical whether the chunks are fed
at once (``chunk_features``) or one at a time with a key-value cache
(``Session``). A chunk's feature is the last layer's hidden state at its last
token. No parameter of the model is trained.

``load("tiny-random")`` builds a 2-layer random Llama with a byte tokenizer
for tests; it downloads nothing. With ``device="cuda"`` the model runs on the
GPU and features are returned on the CPU, where the heads and executors run.
"""

from collections.abc import Callable

import torch

TINY_RANDOM = "tiny-random"


class FrozenLM:
    def __init__(
        self,
        model: torch.nn.Module,
        encode: Callable[[str], list[int]],
        name: str,
        revision: str | None,
        device: str = "cpu",
    ) -> None:
        self.model, self.encode, self.name, self.revision = (
            model,
            encode,
            name,
            revision,
        )
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.hidden_size: int = model.config.hidden_size
        self.parameters = sum(p.numel() for p in model.parameters())

    @classmethod
    def load(
        cls, name: str, revision: str | None = None, device: str = "cpu"
    ) -> "FrozenLM":
        from transformers import AutoModel, AutoTokenizer, LlamaConfig, LlamaModel

        if name == TINY_RANDOM:
            torch.manual_seed(0)
            config = LlamaConfig(
                vocab_size=256,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                max_position_embeddings=8192,
            )
            model = LlamaModel(config)
            return cls(model, lambda text: list(text.encode()), name, None, device)
        tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        model = AutoModel.from_pretrained(name, revision=revision, dtype=torch.float32)

        def encode(text: str) -> list[int]:
            return tokenizer(text, add_special_tokens=False)["input_ids"]

        return cls(model, encode, name, revision, device)

    def tokens(self, chunks: list[str]) -> list[list[int]]:
        ids = [self.encode(chunk) for chunk in chunks]
        if any(not chunk_ids for chunk_ids in ids):
            raise ValueError("every chunk must encode to at least one token")
        return ids

    @torch.no_grad()
    def chunk_features(self, chunks: list[str]) -> torch.Tensor:
        """Features ``(len(chunks), hidden_size)`` from one forward pass."""
        ids = self.tokens(chunks)
        flat = torch.tensor([token for chunk_ids in ids for token in chunk_ids])
        hidden = self.model(input_ids=flat[None].to(self.device)).last_hidden_state[0]
        ends = torch.tensor(ids_ends(ids), device=self.device)
        return hidden[ends].cpu()

    def session(self) -> "Session":
        return Session(self)


def ids_ends(ids: list[list[int]]) -> list[int]:
    """Index of the last token of each chunk in the concatenated sequence."""
    ends, position = [], -1
    for chunk_ids in ids:
        position += len(chunk_ids)
        ends.append(position)
    return ends


class Session:
    """Feeds chunks one at a time, reusing the key-value cache."""

    def __init__(self, lm: FrozenLM) -> None:
        self.lm, self.cache, self.tokens = lm, None, 0

    @torch.no_grad()
    def feed(self, chunk: str) -> torch.Tensor:
        (ids,) = self.lm.tokens([chunk])
        output = self.lm.model(
            input_ids=torch.tensor(ids, device=self.lm.device)[None],
            past_key_values=self.cache,
            use_cache=True,
        )
        self.cache = output.past_key_values
        self.tokens += len(ids)
        return output.last_hidden_state[0, -1].cpu()
