#!/usr/bin/env python3
"""P1 Step 6: rerank with the repository's own inference code.

Uses `src/implicit/run_beir.py::run_topics` unchanged. That file is gzipped in
the working tree, so it is decompressed to a scratch module and imported; the
logic is byte-identical to the committed source.

Three adaptations are applied. None touches scoring, candidate depth, ordering
or output format:

1. The bi-encoder loader calls `.cuda()`, unavailable on Apple silicon, so the
   model is moved to the selected torch device instead.
2. `rankers` 0.1.0's CatTransformer.from_pretrained assigns a bare HF model
   while transform() expects the `Cat` wrapper; cls_architecture is pointed at
   `Cat`, the class train_cat.py uses. Scoring stays
   log_softmax(logits, -1)[:, 1].
3. Document texts are resolved by streaming `docs_iter()` and keeping only the
   doc_ids present in the candidate run, instead of `docs_store()`. On
   BEIR hotpotqa (5.2M docs) docs_store() allocated 43GB on a 24GB machine and
   wedged in swap. The texts handed to the model are identical; only the
   lookup mechanism differs. Verified by the determinism check, which compares
   regenerated runs against the originals.

    python recompute/runs_p1/rerank.py \
        --candidates recompute/runs_p1/candidates/run.beir.bm25-flat.nfcorpus.txt.gz \
        --checkpoint recompute/runs_p1/checkpoints/cat-LCE-ensemble.16-16 \
        --out recompute/runs_p1/out --cat
"""

from __future__ import annotations

import argparse
import gzip
import logging
import importlib.util
import os
import shutil
import sys
import tempfile

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_GZ = os.path.join(REPO, "src", "implicit", "run_beir.py.gz")


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_repo_module():
    """Import src/implicit/run_beir.py verbatim from its gzipped form."""
    tmpdir = tempfile.mkdtemp(prefix="implicit_runbeir_")
    dest = os.path.join(tmpdir, "run_beir_repo.py")
    with gzip.open(SRC_GZ, "rt", encoding="utf-8") as fin:
        source = fin.read()
    with open(dest, "w", encoding="utf-8") as fout:
        fout.write(source)

    spec = importlib.util.spec_from_file_location("run_beir_repo", dest)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_beir_repo"] = mod
    spec.loader.exec_module(mod)
    return mod, tmpdir


def patch_bi_encoder(mod, device: str) -> None:
    """Replace only the `.cuda()` call in the bi-encoder loader."""

    def load_bi_encoder(checkpoint: str, batch_size: int = 64, **kwargs):
        from pyterrier_dr import BiScorer, HgfBiEncoder
        from transformers import AutoModel, AutoTokenizer

        model = AutoModel.from_pretrained(checkpoint).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        backbone = HgfBiEncoder(model, tokenizer, {}, device=model.device)
        return BiScorer(backbone, batch_size=batch_size, verbose=True)

    mod.load_bi_encoder = load_bi_encoder


def patch_cross_encoder(mod, device: str) -> None:
    """Pass the device through, and work around a bug in rankers 0.1.0.

    CatTransformer.transform() scores with `self.model.model(**inps).logits`,
    i.e. it expects `self.model` to be the `Cat` wrapper that train_cat.py
    builds. But CatTransformer.cls_architecture is
    AutoModelForSequenceClassification, so from_pretrained() assigns a bare
    ElectraForSequenceClassification and `.model` raises AttributeError.

    Pointing cls_architecture at `Cat` restores the path the training code
    used. Scoring is untouched: still log_softmax(logits, -1)[:, 1].
    """

    def load_cross_encoder(checkpoint: str, batch_size: int = 64, **kwargs):
        from rankers import CatTransformer
        from rankers.modelling.cat.cat import Cat

        CatTransformer.cls_architecture = Cat
        return CatTransformer.from_pretrained(
            checkpoint, batch_size=batch_size, verbose=True, device=device
        )

    mod.load_cross_encoder = load_cross_encoder


def wanted_doc_ids(candidates: str, depth: int) -> set:
    """Doc ids that run_topics will actually score.

    Replicates its filtering exactly:
        run = pt.io.read_results(file)
        run = run.sort_values(["qid","score"], ascending=[True,False])
                 .groupby("qid").head(depth)
    """
    import pyterrier as pt

    run = pt.io.read_results(candidates)
    run = (
        run.sort_values(["qid", "score"], ascending=[True, False])
        .groupby("qid")
        .head(depth)
    )
    return set(run["docno"].astype(str))


def patch_text_lookup(mod, wanted: set) -> None:
    """Resolve texts by streaming docs_iter over only the ids we need.

    The repository builds a full docstore:

        docstore = dataset.docs_store()
        run['text'] = run['docno'].map(lambda docno: docstore.get(docno).text)

    `docs_store()` does not fit in memory for the larger BEIR collections: on
    hotpotqa (5.2M docs) it allocated 43GB on a 24GB machine and wedged in
    swap. This keeps only the candidate documents' texts.
    """
    import ir_datasets as irds

    class _Doc:
        __slots__ = ("text",)

        def __init__(self, text: str) -> None:
            self.text = text

    class StreamingDocstore:
        def __init__(self, dataset, wanted: set) -> None:
            self._texts = {}
            for doc in dataset.docs_iter():
                if doc.doc_id in wanted:
                    self._texts[doc.doc_id] = doc.text
                    if len(self._texts) == len(wanted):
                        break
            missing = len(wanted) - len(self._texts)
            if missing:
                logging.warning(
                    "%d of %d candidate doc_ids not found in corpus",
                    missing, len(wanted),
                )
            else:
                logging.info("resolved all %d candidate texts", len(wanted))

        def get(self, doc_id):
            return _Doc(self._texts.get(doc_id, ""))

    original_load = irds.load

    class _Wrapper:
        def __init__(self, inner) -> None:
            self._inner = inner

        def __getattr__(self, item):
            return getattr(self._inner, item)

        def docs_store(self):
            return StreamingDocstore(self._inner, wanted)

    class _Shim:
        @staticmethod
        def load(name):
            return _Wrapper(original_load(name))

    mod.irds = _Shim


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cat", action="store_true", help="cross-encoder")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    # run_beir.py only calls basicConfig under __main__, which importing it
    # skips, so its logging.info diagnostics would otherwise be invisible.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    device = pick_device(args.device)
    torch.manual_seed(0)  # plan rule 6
    print(f"device={device}  torch={torch.__version__}")

    os.makedirs(args.out, exist_ok=True)
    mod, tmpdir = load_repo_module()
    try:
        patch_bi_encoder(mod, device)
        patch_cross_encoder(mod, device)

        wanted = wanted_doc_ids(args.candidates, args.depth)
        print(f"candidate documents to resolve: {len(wanted)}")
        patch_text_lookup(mod, wanted)
        result = mod.run_topics(
            file=args.candidates,
            model_name_or_path=args.checkpoint,
            out_path=args.out,
            batch_size=args.batch_size,
            cat=args.cat,
            depth=args.depth,
        )
        print(result)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
