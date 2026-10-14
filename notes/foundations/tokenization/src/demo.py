"""Small executable teaching examples, not a production tokenizer.

WordPiece training uses the common normalized-pair-score teaching approximation.
The Unigram example decodes fixed probabilities; it does not train SentencePiece.
"""
from collections import Counter
from fractions import Fraction
import json
import math

CORPUS = {"hug": 10, "pug": 5, "pun": 12, "bun": 4, "hugs": 5}
UNIGRAM = {"ab": 0.40, "a": 0.15, "b": 0.15, "aba": 0.10, "bab": 0.20}


def pair_counts(splits, corpus):
    counts = Counter()
    for word, units in splits.items():
        for pair in zip(units, units[1:]):
            counts[pair] += corpus[word]
    return counts


def merge_units(units, pair, wordpiece=False):
    result = []
    i = 0
    joined = pair[0] + (pair[1].removeprefix("##") if wordpiece else pair[1])
    while i < len(units):
        if i + 1 < len(units) and tuple(units[i:i + 2]) == pair:
            result.append(joined)
            i += 2
        else:
            result.append(units[i])
            i += 1
    return result


def train_bpe(corpus=CORPUS, steps=7):
    splits = {word: list(word) for word in corpus}
    vocab = {char for word in corpus for char in word}
    history = [{"step": 0, "vocab_size": len(vocab), "splits": splits,
                "token_count": sum(len(splits[w]) * corpus[w] for w in corpus)}]
    merges = []
    for step in range(1, steps + 1):
        counts = pair_counts(splits, corpus)
        if not counts:
            break
        # Deterministic tie-breaking: lexicographic pair order.
        pair = min(counts, key=lambda p: (-counts[p], p))
        merges.append(pair)
        vocab.add("".join(pair))
        splits = {w: merge_units(units, pair) for w, units in splits.items()}
        history.append({"step": step, "pair": list(pair), "count": counts[pair],
                        "vocab_size": len(vocab), "splits": splits,
                        "token_count": sum(len(splits[w]) * corpus[w] for w in corpus)})
    return merges, vocab, history


def encode_bpe(word, merges):
    units = list(word)
    ranks = {tuple(pair): rank for rank, pair in enumerate(merges)}
    while len(units) > 1:
        available = [pair for pair in zip(units, units[1:]) if pair in ranks]
        if not available:
            break
        pair = min(available, key=ranks.__getitem__)
        units = merge_units(units, pair)
    return units


def wordpiece_scores(splits, corpus):
    counts = Counter()
    for word, units in splits.items():
        for unit in units:
            counts[unit] += corpus[word]
    pairs = pair_counts(splits, corpus)
    return [{"pair": list(pair), "pair_count": count,
             "left_count": counts[pair[0]], "right_count": counts[pair[1]],
             "score": float(Fraction(count, counts[pair[0]] * counts[pair[1]]))}
            for pair, count in sorted(pairs.items())]


def train_wordpiece_teaching(corpus=CORPUS, steps=3):
    splits = {word: [c if i == 0 else "##" + c for i, c in enumerate(word)]
              for word in corpus}
    vocab = {unit for units in splits.values() for unit in units}
    history = []
    for step in range(1, steps + 1):
        scores = wordpiece_scores(splits, corpus)
        if not scores:
            break
        best = min(scores, key=lambda item: (
            -Fraction(item["pair_count"], item["left_count"] * item["right_count"]),
            tuple(item["pair"])))
        pair = tuple(best["pair"])
        vocab.add(pair[0] + pair[1].removeprefix("##"))
        splits = {w: merge_units(units, pair, wordpiece=True) for w, units in splits.items()}
        history.append({"step": step, "selected": best, "candidates": scores, "splits": splits})
    return vocab, history


def encode_wordpiece(word, vocab, unk="[UNK]"):
    start, tokens = 0, []
    while start < len(word):
        found = None
        for end in range(len(word), start, -1):
            piece = ("##" if start else "") + word[start:end]
            if piece in vocab:
                found = piece, end
                break
        if found is None:
            return [unk]
        piece, start = found
        tokens.append(piece)
    return tokens


def all_unigram_paths(text, probabilities=UNIGRAM):
    # Explicit enumeration is only appropriate for this tiny teaching example.
    def visit(offset, pieces, probability):
        if offset == len(text):
            yield {"pieces": pieces, "probability": probability}
        for piece, p in probabilities.items():
            if text.startswith(piece, offset):
                yield from visit(offset + len(piece), pieces + [piece], probability * p)
    return sorted(visit(0, [], 1.0), key=lambda item: -item["probability"])


def unigram_best_path(text, probabilities=UNIGRAM):
    costs = [math.inf] * (len(text) + 1)
    previous = [None] * (len(text) + 1)
    costs[0] = 0.0
    for start in range(len(text)):
        if math.isinf(costs[start]):
            continue
        for piece, p in probabilities.items():
            if not text.startswith(piece, start):
                continue
            end = start + len(piece)
            candidate = costs[start] - math.log(p)
            if candidate < costs[end]:
                costs[end], previous[end] = candidate, (start, piece)
    if previous[-1] is None:
        raise ValueError("Input cannot be segmented using the supplied vocabulary")
    pieces, offset = [], len(text)
    while offset:
        offset, piece = previous[offset]
        pieces.append(piece)
    return list(reversed(pieces)), math.exp(-costs[-1])


def examples():
    merges, vocab, bpe = train_bpe()
    _, wp = train_wordpiece_teaching()
    fixed_vocab = {"play", "##ing", "##er", "##s", "[UNK]"}
    paths = all_unigram_paths("abab")
    best, probability = unigram_best_path("abab")
    # Check the mechanisms illustrated in the prose, including failure behavior.
    assert merges[:3] == [("u", "g"), ("u", "n"), ("h", "ug")]
    assert encode_bpe("hugs", merges[:3]) == ["hug", "s"]
    assert encode_bpe("abc", [("b", "c"), ("a", "b")]) == ["a", "bc"]
    assert wp[0]["selected"]["pair"] == ["##g", "##s"]
    assert encode_wordpiece("playing", fixed_vocab) == ["play", "##ing"]
    assert encode_wordpiece("playz", fixed_vocab) == ["[UNK]"]
    assert best == ["ab", "ab"] and math.isclose(probability, 0.16)
    assert math.isclose(paths[0]["probability"], probability)
    assert math.isclose(sum(UNIGRAM.values()), 1)
    return {"corpus": CORPUS, "bpe_history": bpe,
            "bpe_hugs_after_three_merges": encode_bpe("hugs", merges[:3]),
            "bpe_rank_counterexample": encode_bpe("abc", [("b", "c"), ("a", "b")]),
            "wordpiece_teaching_history": wp,
            "wordpiece_playing": encode_wordpiece("playing", fixed_vocab),
            "wordpiece_playz": encode_wordpiece("playz", fixed_vocab),
            "unigram_probabilities": UNIGRAM, "unigram_paths": paths,
            "unigram_best": best, "unigram_best_probability": probability,
            "unigram_string_probability": sum(path["probability"] for path in paths)}


if __name__ == "__main__":
    # ASCII JSON works with legacy Windows terminal encodings as well.
    print(json.dumps(examples(), indent=2, ensure_ascii=True))
