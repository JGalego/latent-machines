"""The frozen language model wrapper, on a tiny random model."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from latent_machines.controllers.frozen_lm import FrozenLM, ids_ends  # noqa: E402
from latent_machines.controllers.heads import (  # noqa: E402
    ActionHead,
    AnswerHead,
    parameter_count,
)

CHUNKS = ["Records a to h start at 0.", " Set c to 7.", " What is c?"]


@pytest.fixture(scope="module")
def lm() -> FrozenLM:
    return FrozenLM.load("tiny-random")


def test_model_is_frozen(lm: FrozenLM) -> None:
    assert not any(p.requires_grad for p in lm.model.parameters())


def test_incremental_features_match_one_pass(lm: FrozenLM) -> None:
    batch = lm.chunk_features(CHUNKS)
    session = lm.session()
    incremental = torch.stack([session.feed(chunk) for chunk in CHUNKS])
    assert batch.shape == (3, lm.hidden_size)
    assert torch.allclose(batch, incremental, atol=1e-5)
    assert session.tokens == sum(len(chunk.encode()) for chunk in CHUNKS)


def test_chunk_ends() -> None:
    assert ids_ends([[1, 2], [3], [4, 5, 6]]) == [1, 2, 5]


def test_empty_chunk_is_rejected(lm: FrozenLM) -> None:
    with pytest.raises(ValueError):
        lm.chunk_features(["a", ""])


def test_head_shapes_and_counts() -> None:
    action = ActionHead(32, 16, 2, 8, 16)
    op, record, value = action(torch.zeros(5, 32))
    assert (op.shape, record.shape, value.shape) == ((5, 2), (5, 8), (5, 16))
    answer = AnswerHead(32, 16, 16, observations=16)
    assert answer(torch.zeros(5, 32), torch.zeros(5, dtype=torch.long)).shape == (5, 16)
    with pytest.raises(ValueError):
        answer(torch.zeros(5, 32))
    assert parameter_count(action, None) == sum(p.numel() for p in action.parameters())


def test_feeding_token_embeddings_as_vectors_matches_tokens(lm: FrozenLM) -> None:
    session = lm.session()
    session.feed(CHUNKS[0])
    by_tokens = session.feed(" c")
    vectors = lm.session()
    vectors.feed(CHUNKS[0])
    (ids,) = lm.tokens([" c"])
    for vector in lm.token_embeddings(ids):
        by_vector = vectors.feed_vector(vector)
    assert torch.allclose(by_tokens, by_vector, atol=1e-5)
    assert (vectors.tokens, vectors.vectors) == (len(CHUNKS[0]), len(ids))


def test_tick_head_reads_feedback_codes() -> None:
    from latent_machines.controllers.heads import FeedbackEmbedding, TickHead

    head = TickHead(32, 16, (3, 8), codes=(3, 9))
    codes = torch.tensor([[0, 1], [2, 8]])
    op, record = head(torch.zeros(2, 32), codes)
    assert (op.shape, record.shape) == ((2, 3), (2, 8))
    with pytest.raises(ValueError):
        head(torch.zeros(2, 32))
    embed = FeedbackEmbedding((3, 9), 4)
    assert torch.equal(
        embed(codes[0]), embed.tables[0].weight[0] + embed.tables[1].weight[1]
    )
