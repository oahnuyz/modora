from __future__ import annotations

import argparse
import asyncio
import collections
import json
import logging
import random
import re
import shutil
import string
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from tqdm import tqdm

from modora.core.domain.cctree import CCTree, CCTreeNode
from modora.core.domain.component import Location
from modora.core.infra.llm import (
    AsyncLLMFactory,
    ensure_llm_local_loaded,
    shutdown_llm_local,
)
from modora.core.services.retrieve import SemanticRetriever
from modora.core.settings import Settings
from modora.core.utils.config import (
    load_ui_settings_from_config,
    settings_from_ui_payload,
)


EVIDENCE_BASED_ASSESSMENT_INSTRUCTION = """IMPORTANT: Answer strictly based on the provided context above. Do NOT use external knowledge or information not present in the context.

Before answering, audit the provided context against the question. Provide a concise evidence analysis that can be checked:
- Point 1: Quote the exact sentence(s) from the context that directly answer the question, focusing on content that matches the question's key terms.
- Point 2: Identify any additional evidence, constraints, dates, entities, numbers, or multi-hop links needed for the answer.
- Point 3: State whether any key information is missing or conflicting.

If the context is INSUFFICIENT (missing key facts, conflicting information, or would require guessing), set "sufficient" to false, list the missing information, and set "answer" to "Not mentioned". Do NOT guess or fabricate.

If the context is SUFFICIENT, set "sufficient" to true and provide a complete answer in the "answer" field. Include all relevant details (dates, ranges, names) rather than oversimplifying.

Respond ONLY as a JSON object in the following format:
{
  "sufficient": true/false,
  "evidence_analysis": [
    "Point 1: [Quote] ...",
    "Point 2: ...",
    "Point 3: ..."
  ],
  "missing_info": [],
  "answer": "<final answer or Not mentioned>",
  "reasoning": "<one short sentence summarizing why the answer is supported or why it is insufficient>"
}"""


FINANCEBENCH_QA_PROMPT = """Based on the financial document excerpts above, answer the following question accurately and concisely.
If the answer involves a numerical value, include the unit (e.g., USD millions, %, etc.).

Question: {}"""


JUDGE_SYSTEM_PROMPT = """
You are an expert evaluator scoring how well an AI-generated answer matches a gold standard (ground truth).
"""


JUDGE_PROMPT_TEMPLATE = """
Please score the Generated Answer against the Gold Answers on a scale of 0 to 4.

[Evaluation Rubric]
- Score 4 (Perfect): Fully and accurately captures the core meaning and key facts of any of the Gold Answers. Additional relevant explanation or context is acceptable and does NOT reduce the score, as long as it is consistent with and does not contradict the Gold Answers. Minor differences in wording, capitalization, punctuation, or phrasing are acceptable if the core meaning is preserved.
- Score 3 (Good): Correctly captures the main answer and most key facts, but has minor issues such as slight imprecision, small omissions of non-critical details, or wording that is somewhat vague or ambiguous. The overall answer is still clearly correct.
- Score 2 (Partial): Partially correct, but missing at least one important fact, condition, or detail needed for a fully correct answer. The answer is related to the correct topic, but is incomplete or insufficient.
- Score 1 (Poor): Mostly incorrect, seriously incomplete, or only weakly related to the Gold Answers.
- Score 0 (Wrong): Incorrect, contradictory to the Gold Answers, or contains fabricated / hallucinated core content.

Important Notes:
- Gold answers are multiple possible correct answers separated by " | ". The generated answer only needs to match any one of them.
- The gold answers may be concise, but the generated answer can be longer and include additional explanations - this is acceptable for Score 4 as long as the core information is correct.
- Do NOT penalize for additional relevant information that doesn't contradict the gold answers. Examples of acceptable extra information: titles ("King Padella" vs "Padella"), locations ("Paflagonia" vs "the capital of Paflagonia"), or additional context that supports the answer.
- Only penalize for actual incorrect information, missing key facts, or contradictions.
- Ignore minor differences in capitalization (e.g., "CRIM TARTARY" vs "Crim Tartary") or punctuation (e.g., with or without a period at the end).

Question: {question}
Gold Answers: {gold_answers}
Generated Answer: {response}

First, briefly explain the rating in 1 sentence. Then output the integer score.
Respond ONLY with a JSON object: {{"score": 0 to 4, "reasoning": "string"}}
"""


@dataclass
class ParsedAnswer:
    sufficient: bool
    answer: str
    reasoning: str
    evidence_analysis: list[str]
    missing_info: list[str]
    raw: str


def register(sub: argparse._SubParsersAction) -> None:
    prepare = sub.add_parser(
        "financebench-prepare",
        help="Prepare a sampled FinanceBench dataset for MoDora experiments",
    )
    prepare.add_argument(
        "--source",
        required=True,
        help="FinanceBench root directory or financebench_open_source.jsonl path",
    )
    prepare.add_argument(
        "--pdf-dir",
        default=None,
        help="PDF directory. Defaults to <source>/pdfs or <source>/../pdfs",
    )
    prepare.add_argument(
        "--output",
        required=True,
        help="Output directory for MoDora-ready PDFs and test.json",
    )
    prepare.add_argument(
        "--sample-size",
        type=int,
        default=12,
        help="Number of QA rows to sample. Use 0 with --num-docs to keep all QAs in selected docs.",
    )
    prepare.add_argument(
        "--num-docs",
        type=int,
        default=3,
        help="Number of documents to sample. Use 0 to sample from all documents.",
    )
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument(
        "--sample-mode",
        choices=["stratified", "random"],
        default="stratified",
    )
    prepare.add_argument(
        "--full",
        action="store_true",
        help="Use all FinanceBench QA rows and all referenced documents",
    )
    prepare.add_argument("--start-question-id", type=int, default=1)
    prepare.add_argument(
        "--max-pages-per-pdf",
        type=int,
        default=0,
        help=(
            "If >0, write only the first N pages of each selected PDF to the "
            "prepared dataset. Intended for small smoke tests."
        ),
    )
    prepare.set_defaults(_handler=_handle_financebench_prepare)

    qa = sub.add_parser(
        "financebench-qa",
        help="Run FinanceBench QA with MoDora semantic retrieval and OpenViking prompts",
    )
    qa.add_argument("--dataset", required=True, help="MoDora FinanceBench test.json")
    qa.add_argument("--cache", required=True, help="Cache directory containing tree.json")
    qa.add_argument("--output", required=True, help="Output directory")
    qa.add_argument("--concurrency", type=int, default=4)
    qa.add_argument("--limit", type=int, default=0)
    qa.add_argument("--tag", default=None)
    qa.add_argument("--resume", action="store_true")
    qa.add_argument("--debug", action="store_true", help="Store generated prompts")
    qa.set_defaults(_handler=_handle_financebench_qa)

    evaluate = sub.add_parser(
        "financebench-evaluate",
        help="Evaluate FinanceBench results with OpenViking-compatible metrics",
    )
    evaluate.add_argument("--dataset", required=True, help="MoDora FinanceBench test.json")
    evaluate.add_argument("--result", required=True, help="Result JSON from financebench-qa or batch-qa")
    evaluate.add_argument("--output-dir", default=None)
    evaluate.add_argument("--concurrency", type=int, default=4)
    evaluate.add_argument("--judge-instance", default=None)
    evaluate.add_argument(
        "--skip-llm-judge",
        action="store_true",
        help="Only compute F1/Recall; Accuracy is set to 0",
    )
    evaluate.set_defaults(_handler=_handle_financebench_evaluate)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resolve_financebench_paths(source: Path, pdf_dir: str | None) -> tuple[Path, Path]:
    if source.is_file():
        jsonl_path = source
        base_dir = source.parent
    else:
        candidate = source / "financebench_open_source.jsonl"
        if candidate.exists():
            jsonl_path = candidate
            base_dir = source
        else:
            candidate = source / "data" / "financebench_open_source.jsonl"
            if not candidate.exists():
                raise FileNotFoundError(
                    f"financebench_open_source.jsonl not found under {source}"
                )
            jsonl_path = candidate
            base_dir = source

    if pdf_dir:
        resolved_pdf_dir = Path(pdf_dir).expanduser().resolve()
    else:
        candidates = [base_dir / "pdfs", jsonl_path.parent / "pdfs", jsonl_path.parent.parent / "pdfs"]
        resolved_pdf_dir = next((p for p in candidates if p.exists()), candidates[0])

    if not resolved_pdf_dir.exists():
        raise FileNotFoundError(f"FinanceBench PDF directory not found: {resolved_pdf_dir}")
    return jsonl_path.resolve(), resolved_pdf_dir.resolve()


def _group_by_doc(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        doc_name = str(row.get("doc_name") or "").strip()
        if doc_name:
            groups[doc_name].append(row)
    return dict(groups)


def _sample_rows(
    rows: list[dict[str, Any]],
    *,
    sample_size: int | None,
    num_docs: int | None,
    seed: int,
    sample_mode: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    doc_groups = _group_by_doc(rows)
    all_doc_names = list(doc_groups.keys())
    selected_docs = all_doc_names

    if num_docs is not None and num_docs < len(all_doc_names):
        rng = random.Random(seed)
        sorted_docs = sorted(all_doc_names, key=lambda d: len(doc_groups[d]), reverse=True)
        rng.shuffle(sorted_docs)
        selected_docs = sorted(sorted_docs, key=lambda d: len(doc_groups[d]), reverse=True)[:num_docs]

    candidate_rows: list[dict[str, Any]] = []
    for doc_name in selected_docs:
        candidate_rows.extend(doc_groups[doc_name])

    if sample_size is None or sample_size >= len(candidate_rows):
        selected_rows = candidate_rows
    elif sample_mode == "random":
        selected_rows = random.Random(seed).sample(candidate_rows, sample_size)
    else:
        selected_rows = _stratified_sample(candidate_rows, sample_size, seed)

    selected_doc_set = {str(row.get("doc_name")) for row in selected_rows}
    ordered_docs = [doc for doc in selected_docs if doc in selected_doc_set]
    if not ordered_docs:
        ordered_docs = sorted(selected_doc_set)

    grouped_selected = _group_by_doc(selected_rows)
    ordered_rows: list[dict[str, Any]] = []
    for doc_name in ordered_docs:
        ordered_rows.extend(grouped_selected.get(doc_name, []))
    return ordered_rows, ordered_docs


def _stratified_sample(
    rows: list[dict[str, Any]], sample_size: int, seed: int
) -> list[dict[str, Any]]:
    by_category: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_category[str(row.get("question_type") or "domain-relevant")].append(row)

    categories = sorted(by_category.keys())
    if not categories or sample_size < len(categories):
        return random.Random(seed).sample(rows, min(sample_size, len(rows)))

    base = sample_size // len(categories)
    remainder = sample_size % len(categories)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []

    for idx, category in enumerate(categories):
        target = base + (1 if idx < remainder else 0)
        items = list(by_category[category])
        rng.shuffle(items)
        selected.extend(items[: min(target, len(items))])

    if len(selected) < sample_size:
        remaining = [row for row in rows if row not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: sample_size - len(selected)])

    return selected[:sample_size]


def _evidence_texts(row: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    evidence = row.get("evidence")
    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, dict):
                text = str(item.get("evidence_text") or "").strip()
                if text:
                    texts.append(text)
    return texts


def _copy_or_truncate_pdf(src: Path, dst: Path, max_pages: int) -> dict[str, Any]:
    """Copy a PDF, optionally keeping only its first N pages."""
    if max_pages <= 0:
        shutil.copy2(src, dst)
        return {
            "source": str(src),
            "output": str(dst),
            "source_pages": None,
            "output_pages": None,
            "truncated": False,
        }

    import fitz

    tmp = dst.with_name(f"{dst.name}.tmp")
    if tmp.exists():
        tmp.unlink()

    with fitz.open(src) as in_doc:
        source_pages = int(in_doc.page_count)
        output_pages = min(max_pages, source_pages)
        with fitz.open() as out_doc:
            if output_pages > 0:
                out_doc.insert_pdf(in_doc, from_page=0, to_page=output_pages - 1)
            out_doc.save(tmp)

    tmp.replace(dst)
    shutil.copystat(src, dst)
    return {
        "source": str(src),
        "output": str(dst),
        "source_pages": source_pages,
        "output_pages": output_pages,
        "truncated": output_pages < source_pages,
    }


def _handle_financebench_prepare(args: argparse.Namespace, logger: logging.Logger) -> int:
    try:
        source = Path(args.source).expanduser().resolve()
        jsonl_path, pdf_dir = _resolve_financebench_paths(source, args.pdf_dir)
        output_dir = Path(args.output).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        rows = _load_jsonl(jsonl_path)
        sample_size_arg = int(args.sample_size)
        num_docs_arg = int(args.num_docs)
        sample_size = None if args.full or sample_size_arg <= 0 else sample_size_arg
        num_docs = None if args.full or num_docs_arg <= 0 else num_docs_arg
        selected_rows, selected_docs = _sample_rows(
            rows,
            sample_size=sample_size,
            num_docs=num_docs,
            seed=int(args.seed),
            sample_mode=str(args.sample_mode),
        )
        max_pages_per_pdf = int(getattr(args, "max_pages_per_pdf", 0) or 0)
        if max_pages_per_pdf < 0:
            raise ValueError("--max-pages-per-pdf must be >= 0")

        doc_to_pdf_id = {
            doc_name: f"{idx}.pdf" for idx, doc_name in enumerate(selected_docs, start=1)
        }
        missing_pdfs: list[str] = []
        pdf_page_info: dict[str, Any] = {}
        for doc_name, pdf_id in doc_to_pdf_id.items():
            src = pdf_dir / f"{doc_name}.pdf"
            if not src.exists():
                missing_pdfs.append(str(src))
                continue
            pdf_page_info[pdf_id] = {
                "doc_name": doc_name,
                **_copy_or_truncate_pdf(src, output_dir / pdf_id, max_pages_per_pdf),
            }

        if missing_pdfs:
            logger.error("missing FinanceBench PDFs", extra={"missing": missing_pdfs})
            return 2

        start_qid = int(args.start_question_id)
        modora_rows: list[dict[str, Any]] = []
        for idx, row in enumerate(selected_rows):
            doc_name = str(row.get("doc_name") or "")
            item = {
                "questionId": start_qid + idx,
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "pdf_id": doc_to_pdf_id[doc_name],
                "tag": row.get("question_type"),
                "financebench_id": row.get("financebench_id"),
                "doc_name": doc_name,
                "company": row.get("company"),
                "question_type": row.get("question_type"),
                "evidence_texts": _evidence_texts(row),
                "question_reasoning": row.get("question_reasoning"),
                "justification": row.get("justification", ""),
            }
            modora_rows.append(item)

        (output_dir / "test.json").write_text(
            json.dumps(modora_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _write_jsonl(output_dir / "financebench_open_source.jsonl", selected_rows)

        mapping = {
            "doc_to_pdf_id": doc_to_pdf_id,
            "pdf_id_to_doc": {v: k for k, v in doc_to_pdf_id.items()},
            "pdf_page_info": pdf_page_info,
        }
        (output_dir / "doc_mapping.json").write_text(
            json.dumps(mapping, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        metadata = {
            "dataset": "FinanceBench",
            "source_jsonl": str(jsonl_path),
            "source_pdf_dir": str(pdf_dir),
            "output_dir": str(output_dir),
            "original_total_qas": len(rows),
            "original_num_docs": len(_group_by_doc(rows)),
            "sampled_total_qas": len(selected_rows),
            "sampled_num_docs": len(selected_docs),
            "sample_size": sample_size,
            "num_docs": num_docs,
            "seed": int(args.seed),
            "sample_mode": str(args.sample_mode),
            "is_full": bool(args.full),
            "max_pages_per_pdf": max_pages_per_pdf or None,
        }
        (output_dir / "sampling_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"Prepared FinanceBench dataset -> {output_dir}")
        print(f"QAs: {len(selected_rows)}, PDFs: {len(selected_docs)}")
        if max_pages_per_pdf > 0:
            print(f"PDF page limit: first {max_pages_per_pdf} pages per PDF")
        return 0
    except Exception as e:
        logger.exception("financebench prepare failed", extra={"error": str(e)})
        return 2


def _dict_to_node(data: dict[str, Any]) -> CCTreeNode:
    node = CCTreeNode(
        type=data.get("type", "unknown"),
        metadata=data.get("metadata"),
        data=data.get("data", ""),
        location=[Location.from_dict(loc) for loc in data.get("location", [])],
        children={},
    )
    for key, value in data.get("children", {}).items():
        if isinstance(value, dict):
            node.children[key] = _dict_to_node(value)
    return node


def _load_tree(tree_path: Path) -> CCTree:
    data = json.loads(tree_path.read_text(encoding="utf-8"))
    root_data = data.get("root", data) if isinstance(data, dict) else {}
    return CCTree(root=_dict_to_node(root_data))


def _parse_answer(raw: str) -> ParsedAnswer:
    text = (raw or "").strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    def coerce_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    try:
        obj = json.loads(text)
        sufficient = obj.get("sufficient", True)
        if isinstance(sufficient, str):
            sufficient = sufficient.lower() in {"true", "1", "yes"}
        return ParsedAnswer(
            sufficient=bool(sufficient),
            answer=str(obj.get("answer", "")).strip(),
            reasoning=str(obj.get("reasoning", "")).strip(),
            evidence_analysis=coerce_list(obj.get("evidence_analysis")),
            missing_info=coerce_list(obj.get("missing_info")),
            raw=raw,
        )
    except Exception:
        return ParsedAnswer(
            sufficient=True,
            answer=(raw or "").strip(),
            reasoning="",
            evidence_analysis=[],
            missing_info=[],
            raw=raw,
        )


def _build_financebench_prompt(question: str, context_blocks: list[str]) -> str:
    context_text = "\n\n".join(context_blocks)
    qa_prompt = FINANCEBENCH_QA_PROMPT.format(question)
    return f"{context_text}\n\n{EVIDENCE_BASED_ASSESSMENT_INSTRUCTION}\n\n{qa_prompt}"


def _retrieved_documents(result: Any) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    locations_by_path = getattr(result, "locations_by_path", {}) or {}
    if locations_by_path:
        for path, locs in locations_by_path.items():
            content = result.text_map.get(path, "")
            docs.append(
                {
                    "path": path,
                    "content": content,
                    "locations": [loc.to_dict() for loc in locs],
                    "retrievers": ["semantic"],
                }
            )
    else:
        for path, content in getattr(result, "text_map", {}).items():
            docs.append(
                {
                    "path": path,
                    "content": content,
                    "locations": [],
                    "retrievers": ["semantic"],
                }
            )
    return docs


def _resolve_financebench_job(
    item: dict[str, Any], dataset_path: Path, cache_dir: Path, output_dir: Path
) -> dict[str, Any] | None:
    pdf_id = str(item.get("pdf_id") or "")
    if not pdf_id:
        return None
    pdf_path = dataset_path.parent / pdf_id
    tree_path = cache_dir / pdf_id.replace(".pdf", "") / "tree.json"
    if not pdf_path.exists() or not tree_path.exists():
        return None
    return {
        "item": item,
        "pdf_path": pdf_path,
        "tree_path": tree_path,
        "output_path": output_dir / f"qa_{item['questionId']}_result.json",
    }


async def _run_financebench_qa_job(
    job: dict[str, Any],
    retriever: SemanticRetriever,
    llm: Any,
    sem: asyncio.Semaphore,
    debug: bool,
) -> dict[str, Any]:
    async with sem:
        item = job["item"]
        question = str(item.get("question") or "")
        t0 = time.monotonic()
        tree = await asyncio.to_thread(_load_tree, job["tree_path"])
        result = await retriever.retrieve(tree, question, str(job["pdf_path"]))
        latency = time.monotonic() - t0

        context_blocks = list(result.text_map.values())
        prompt = _build_financebench_prompt(question, context_blocks)
        raw_answer = await llm.generate_text(prompt)
        parsed = _parse_answer(raw_answer)
        final_answer = parsed.answer.strip() or raw_answer.strip()

        retrieved_docs = _retrieved_documents(result)
        recall = _check_recall(
            [doc.get("content", "") for doc in retrieved_docs],
            _coerce_string_list(item.get("evidence_texts")),
        )
        output = {
            "questionId": item.get("questionId"),
            "pdf_id": item.get("pdf_id"),
            "doc_name": item.get("doc_name"),
            "financebench_id": item.get("financebench_id"),
            "question": question,
            "ground_truth": item.get("answer"),
            "answer": item.get("answer"),
            "tag": item.get("tag"),
            "prediction": final_answer,
            "evidence": retrieved_docs,
            "retrieval": {
                "latency_sec": latency,
                "uris": list(result.text_map.keys()),
            },
            "llm": {
                "final_answer": final_answer,
                "raw_answer": raw_answer,
                "sufficient": parsed.sufficient,
                "reasoning": parsed.reasoning,
                "evidence_analysis": parsed.evidence_analysis,
                "missing_info": parsed.missing_info,
            },
            "metrics": {"Recall": recall},
            "status": "success",
        }
        if debug:
            output["debug_prompt"] = prompt

        await asyncio.to_thread(
            job["output_path"].write_text,
            json.dumps(output, ensure_ascii=False, indent=2),
            "utf-8",
        )
        return output


async def _run_financebench_qa(args: argparse.Namespace, logger: logging.Logger) -> int:
    config_path = (getattr(args, "config", None) or "").strip() or None
    settings = Settings.load(config_path)
    ensure_llm_local_loaded(settings, logger, config_path=config_path)

    ui_settings = load_ui_settings_from_config(config_path)
    qa_settings, _, qa_instance_id, cfg = settings_from_ui_payload(
        settings, ui_settings, module_key="qaService"
    )
    retriever_settings, _, retriever_instance_id, _ = settings_from_ui_payload(
        settings, cfg, module_key="retriever"
    )
    retriever_settings = replace(retriever_settings, enable_vector_search=False)

    retriever = SemanticRetriever(retriever_settings, instance_id=retriever_instance_id)
    llm = AsyncLLMFactory.create(qa_settings, instance_id=qa_instance_id)

    dataset_path = Path(args.dataset).expanduser().resolve()
    cache_dir = Path(args.cache).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(dataset, list):
        raise ValueError("FinanceBench dataset must be a JSON array")
    if args.tag:
        tags = {x.strip() for x in str(args.tag).split(",") if x.strip()}
        dataset = [item for item in dataset if str(item.get("tag")) in tags]
    if int(args.limit or 0) > 0:
        dataset = dataset[: int(args.limit)]

    completed: dict[Any, dict[str, Any]] = {}
    result_path = output_dir / "result.json"
    if args.resume and result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        existing_rows = existing.get("results", existing) if isinstance(existing, dict) else existing
        if isinstance(existing_rows, list):
            completed = {
                row.get("questionId"): row
                for row in existing_rows
                if isinstance(row, dict) and row.get("status") == "success"
            }

    jobs = []
    skipped_missing = 0
    for item in dataset:
        if item.get("questionId") in completed:
            continue
        job = _resolve_financebench_job(item, dataset_path, cache_dir, output_dir)
        if job is None:
            skipped_missing += 1
            logger.warning("missing pdf or tree for FinanceBench item", extra={"item": item})
            continue
        jobs.append(job)

    sem = asyncio.Semaphore(max(1, int(args.concurrency or 1)))
    results = list(completed.values())
    tasks = [
        _run_financebench_qa_job(job, retriever, llm, sem, bool(args.debug))
        for job in jobs
    ]
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="FinanceBench QA"):
        try:
            results.append(await task)
        except Exception as e:
            logger.exception("FinanceBench QA failed", extra={"error": str(e)})

    results.sort(key=lambda x: int(x.get("questionId", 0) or 0))
    summary = {
        "total": len(dataset),
        "success": sum(1 for row in results if row.get("status") == "success"),
        "skipped_missing_inputs": skipped_missing,
        "semantic_only": True,
        "prompt": "OpenViking FinanceBench evidence-audit JSON prompt",
    }
    result_path.write_text(
        json.dumps({"metrics": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"FinanceBench QA results saved to {result_path}")
    return 0 if summary["success"] == len(dataset) - skipped_missing else 2


def _handle_financebench_qa(args: argparse.Namespace, logger: logging.Logger) -> int:
    try:
        return asyncio.run(_run_financebench_qa(args, logger))
    finally:
        shutdown_llm_local()


def _normalize_answer(value: Any) -> str:
    text = str(value).replace(",", "").lower()
    text = re.sub(r"\b(a|an|the|and)\b", " ", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(text.split())


def _calculate_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    truth_tokens = _normalize_answer(ground_truth).split()
    if not pred_tokens or not truth_tokens:
        return 0.0
    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def _check_refusal(text: str) -> bool:
    refusals = [
        "not mentioned",
        "no information",
        "cannot be answered",
        "none",
        "unknown",
        "don't know",
    ]
    return any(refusal in str(text).lower() for refusal in refusals)


def _check_recall(
    retrieved_texts: list[str],
    evidence_list: list[str],
    soft_threshold: float = 0.8,
    min_soft_match_tokens: int = 4,
) -> float:
    if not evidence_list:
        return 0.0
    combined_retrieved = " ".join(str(x) for x in retrieved_texts)
    normalized_retrieved = _normalize_answer(combined_retrieved)
    ret_tokens = set(normalized_retrieved.split())
    hit_count = 0

    for evidence in evidence_list:
        if evidence in combined_retrieved:
            hit_count += 1
            continue
        normalized_ev = _normalize_answer(evidence)
        ev_tokens = set(normalized_ev.split())
        if len(ev_tokens) < min_soft_match_tokens:
            continue
        coverage = len(ev_tokens & ret_tokens) / len(ev_tokens)
        if coverage >= soft_threshold:
            hit_count += 1
    return hit_count / len(evidence_list)


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(value).strip()] if str(value).strip() else []


def _load_result_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("results", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError("Result file must be a list or an object with results")
    return [row for row in rows if isinstance(row, dict)]


def _retrieved_texts_from_prediction(row: dict[str, Any]) -> list[str]:
    docs = row.get("evidence") or row.get("retrieved_documents") or []
    texts: list[str] = []
    if isinstance(docs, list):
        for doc in docs:
            if isinstance(doc, dict):
                text = str(doc.get("content") or "").strip()
                if text:
                    texts.append(text)
            elif isinstance(doc, str) and doc.strip():
                texts.append(doc.strip())
    return texts


async def _judge_answer(
    llm: Any,
    question: str,
    gold_answers: list[str],
    response: str,
    max_retries: int = 10,
) -> dict[str, Any]:
    gold_answer_str = " | ".join(gold_answers)
    prompt = (
        JUDGE_SYSTEM_PROMPT.strip()
        + "\n\n"
        + JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            gold_answers=gold_answer_str,
            response=response,
        )
    )
    content = ""
    for attempt in range(max_retries):
        try:
            content = await llm.generate_text(prompt)
            text = content.strip()
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()
            result = json.loads(text)
            score = int(result.get("score", 0))
            return {
                "score": max(0, min(4, score)),
                "reasoning": str(result.get("reasoning", "")),
                "prompt_type": "Generic_0-4",
                "raw": content,
            }
        except Exception as e:
            err = str(e)
            if any(x in err for x in ["429", "RateLimit", "TooManyRequests", "TPM"]):
                await asyncio.sleep(5.0 * (2 ** min(attempt, 6)))
                continue
            score_match = re.search(r'"score"\s*:\s*([0-4])', content)
            if not score_match:
                score_match = re.search(r"\b([0-4])\b", content)
            score = int(score_match.group(1)) if score_match else 0
            return {
                "score": max(0, min(4, score)),
                "reasoning": f"Parse fallback from raw output: {content.strip()}",
                "prompt_type": "Generic_0-4",
                "raw": content,
            }

    return {
        "score": 0,
        "reasoning": "Parse failed or model invocation failed. Defaulted to 0.",
        "prompt_type": "Generic_0-4",
        "raw": content,
    }


async def _evaluate_item(
    item: dict[str, Any],
    prediction: dict[str, Any] | None,
    llm: Any | None,
    sem: asyncio.Semaphore,
    skip_llm_judge: bool,
) -> dict[str, Any]:
    async with sem:
        gold_answers = _coerce_string_list(item.get("answer"))
        answer = "" if prediction is None else str(prediction.get("prediction") or "")
        f1 = max((_calculate_f1(answer, gt) for gt in gold_answers), default=0.0)
        recall = _check_recall(
            [] if prediction is None else _retrieved_texts_from_prediction(prediction),
            _coerce_string_list(item.get("evidence_texts")),
        )

        if skip_llm_judge:
            judge = {
                "score": 0,
                "reasoning": "LLM judge skipped.",
                "prompt_type": "Skipped",
                "raw": "",
            }
        else:
            judge = await _judge_answer(
                llm,
                str(item.get("question") or ""),
                gold_answers,
                answer,
            )

        if _check_refusal(answer) and any(_check_refusal(gt) for gt in gold_answers):
            f1 = 1.0
            judge["score"] = 4
            judge["reasoning"] = "System successfully identified Unanswerable/Refusal condition."
            judge["prompt_type"] = "Heuristic_Refusal_Check"

        return {
            "questionId": item.get("questionId"),
            "pdf_id": item.get("pdf_id"),
            "doc_name": item.get("doc_name"),
            "financebench_id": item.get("financebench_id"),
            "category": item.get("tag") or item.get("question_type"),
            "question": item.get("question"),
            "gold_answers": gold_answers,
            "prediction": answer,
            "evidence": _coerce_string_list(item.get("evidence_texts")),
            "retrieved_texts": [] if prediction is None else _retrieved_texts_from_prediction(prediction),
            "metrics": {
                "Recall": recall,
                "F1": f1,
                "Accuracy": judge["score"],
            },
            "llm_evaluation": {
                "prompt_used": judge["prompt_type"],
                "reasoning": judge["reasoning"],
                "normalized_score": judge["score"],
                "raw": judge.get("raw", ""),
            },
        }


async def _run_financebench_evaluate(
    args: argparse.Namespace, logger: logging.Logger
) -> int:
    dataset_path = Path(args.dataset).expanduser().resolve()
    result_path = Path(args.result).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else result_path.parent / "financebench_eval"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(dataset, list):
        raise ValueError("FinanceBench dataset must be a JSON array")
    results = _load_result_rows(result_path)
    prediction_map = {row.get("questionId"): row for row in results}

    llm = None
    if not args.skip_llm_judge:
        config_path = (getattr(args, "config", None) or "").strip() or None
        settings = Settings.load(config_path)
        ensure_llm_local_loaded(settings, logger, config_path=config_path)
        ui_settings = load_ui_settings_from_config(config_path)
        judge_settings, _, judge_instance_id, _ = settings_from_ui_payload(
            settings, ui_settings, module_key="qaService"
        )
        llm = AsyncLLMFactory.create(
            judge_settings,
            instance_id=args.judge_instance or judge_instance_id,
        )

    sem = asyncio.Semaphore(max(1, int(args.concurrency or 1)))
    tasks = [
        _evaluate_item(
            item,
            prediction_map.get(item.get("questionId")),
            llm,
            sem,
            bool(args.skip_llm_judge),
        )
        for item in dataset
    ]

    detailed: list[dict[str, Any]] = []
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="FinanceBench Eval"):
        detailed.append(await task)
    detailed.sort(key=lambda x: int(x.get("questionId", 0) or 0))

    total = len(detailed)
    avg_f1 = sum(row["metrics"]["F1"] for row in detailed) / total if total else 0.0
    avg_recall = sum(row["metrics"]["Recall"] for row in detailed) / total if total else 0.0
    avg_acc = sum(row["metrics"]["Accuracy"] for row in detailed) / total if total else 0.0
    report = {
        "Dataset": "FinanceBench",
        "Total Queries Evaluated": total,
        "Performance Metrics": {
            "Average F1 Score": avg_f1,
            "Average Recall": avg_recall,
            "Average Accuracy (Hit 0-4)": avg_acc,
            "Average Accuracy normalized": avg_acc / 4 if total else 0.0,
        },
    }

    (output_dir / "qa_eval_detailed_results.json").write_text(
        json.dumps(detailed, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "benchmark_metrics_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"FinanceBench evaluation saved to {output_dir}")
    return 0


def _handle_financebench_evaluate(args: argparse.Namespace, logger: logging.Logger) -> int:
    try:
        return asyncio.run(_run_financebench_evaluate(args, logger))
    finally:
        shutdown_llm_local()
