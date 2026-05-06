"""
MULTI-STEP RETRIEVAL v1 — Inspired by MSA Memory Interleave

Implements iterative retrieval for multi-hop reasoning:
  1. Initial search with the user's original query
  2. Fast LLM evaluates if the retrieved context is sufficient
  3. If not, the LLM generates a refined follow-up query
  4. Second search with the follow-up query
  5. Contexts are merged and deduplicated

Design principles:
  - Zero-cost when disabled (MULTI_STEP_ENABLED=False)
  - Minimal overhead when context is sufficient on first try
    (only one fast-model call to evaluate, ~1-2s)
  - Agent-agnostic: works with any agent that exposes an async search(query)->str
  - Does NOT touch server.py — all logic is inside the agents
  - Max 2 retrieval rounds (configurable) to bound latency

Inspired by: "MSA: Memory Sparse Attention" (Chen et al., 2026)
  → Section 3.5 Memory Interleave: iterative retrieval where retrieved
    documents become part of the query for the next round.
"""

import sys
import os
import logging
import json
import hashlib
from typing import Callable, Awaitable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    MULTI_STEP_ENABLED,
    MULTI_STEP_MAX_ROUNDS,
    MULTI_STEP_MIN_CONTEXT_LEN,
    LLM_MODEL_FAST,
)
from llm_client import chat_complete_json

logger = logging.getLogger("multi_step_retrieval")

# Type alias for agent search functions: async (query: str) -> str
SearchFn = Callable[[str], Awaitable[str]]


# ── Evaluation prompt — kept minimal for speed ────────────────────────────────

_EVAL_SYSTEM = (
    "You are a retrieval quality evaluator. "
    "Analyze whether the retrieved context contains enough information "
    "to fully answer the user's question. "
    "Respond ONLY with a JSON object, no other text."
)

_EVAL_USER_TEMPLATE = """QUESTION: {query}

RETRIEVED CONTEXT (first 2000 chars):
{context_preview}

Evaluate the retrieved context and respond with this exact JSON format:
{{
  "sufficient": true/false,
  "reason": "brief explanation in 10 words max",
  "follow_up_query": "refined search query if sufficient=false, empty string if true"
}}

Rules for follow_up_query:
- Must be a DIFFERENT query from the original, targeting missing information
- Should be specific: names, dates, codes, technical terms
- Max 10 words
- If the question asks about relationships between topics (e.g. "compare X and Y"),
  and the context only covers one of them, query for the missing one
- If the context has partial info, query for the specific missing detail"""


# ── Core logic ────────────────────────────────────────────────────────────────

async def _evaluate_context(query: str, context: str) -> dict:
    """
    Ask the fast LLM whether the context is sufficient to answer the query.

    Returns dict with keys:
      - sufficient (bool): True if no further retrieval is needed
      - reason (str): brief explanation
      - follow_up_query (str): refined query for next round (empty if sufficient)
    """
    # Truncate context preview to keep the evaluation fast
    context_preview = context[:2000] if context else "(empty)"

    prompt = _EVAL_USER_TEMPLATE.format(
        query=query,
        context_preview=context_preview,
    )

    try:
        raw = await chat_complete_json(
            model=LLM_MODEL_FAST,
            system=_EVAL_SYSTEM,
            user=prompt,
            temperature=0.0,
        )

        # Parse JSON — handle both clean JSON and markdown-wrapped JSON
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        result = json.loads(text)

        return {
            "sufficient": bool(result.get("sufficient", True)),
            "reason": str(result.get("reason", "")),
            "follow_up_query": str(result.get("follow_up_query", "")),
        }

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"[multi-step] Failed to parse evaluation response: {e}")
        # On parse failure, assume context is sufficient (safe default)
        return {"sufficient": True, "reason": "parse_error", "follow_up_query": ""}

    except Exception as e:
        logger.warning(f"[multi-step] Evaluation call failed: {e}")
        return {"sufficient": True, "reason": "llm_error", "follow_up_query": ""}


def _merge_contexts(context_a: str, context_b: str) -> str:
    """
    Merge two context strings, removing duplicate chunks.

    Strategy: split by double newlines (chunk boundaries), deduplicate
    by content hash, then reassemble. Preserves order (context_a first).
    """
    if not context_b or not context_b.strip():
        return context_a
    if not context_a or not context_a.strip():
        return context_b

    # Split into blocks (chunks are typically separated by blank lines)
    blocks_a = [b.strip() for b in context_a.split("\n\n") if b.strip()]
    blocks_b = [b.strip() for b in context_b.split("\n\n") if b.strip()]

    # Track seen blocks by hash to detect duplicates
    seen_hashes = set()
    merged = []

    for block in blocks_a:
        h = hashlib.md5(block.encode()).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            merged.append(block)

    new_blocks = 0
    for block in blocks_b:
        h = hashlib.md5(block.encode()).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            merged.append(block)
            new_blocks += 1

    if new_blocks > 0:
        logger.info(f"[multi-step] Merged {new_blocks} new blocks from follow-up search")

    return "\n\n".join(merged)


# ── Public API ────────────────────────────────────────────────────────────────

async def multi_step_search(
    query: str,
    search_fn: SearchFn,
    agent_name: str,
    max_rounds: Optional[int] = None,
) -> str:
    """
    Perform iterative retrieval with context evaluation.

    Args:
        query:       User's original question
        search_fn:   Agent's async search function (query -> context string)
        agent_name:  Agent identifier for logging (e.g. "documents", "financial")
        max_rounds:  Override max retrieval rounds (default from config)

    Returns:
        Merged context string from all retrieval rounds
    """
    # Fast path: feature disabled
    if not MULTI_STEP_ENABLED:
        return await search_fn(query)

    rounds = max_rounds or MULTI_STEP_MAX_ROUNDS

    # Step 1: Initial search
    context = await search_fn(query)

    # Sanity check: if context is too short, no point evaluating
    if not context or len(context.strip()) < MULTI_STEP_MIN_CONTEXT_LEN:
        logger.info(
            f"[multi-step] [{agent_name}] Context too short "
            f"({len(context.strip()) if context else 0} chars), "
            f"skipping evaluation"
        )
        return context

    # Iterative refinement loop
    all_queries = [query]

    for round_num in range(1, rounds + 1):
        # Step 2: Evaluate current context
        evaluation = await _evaluate_context(query, context)

        if evaluation["sufficient"]:
            logger.info(
                f"[multi-step] [{agent_name}] Context sufficient after "
                f"{round_num} round(s) — {evaluation['reason']}"
            )
            return context

        follow_up = evaluation["follow_up_query"]

        # Guard: empty or duplicate follow-up query
        if not follow_up or follow_up.strip() in all_queries:
            logger.info(
                f"[multi-step] [{agent_name}] No useful follow-up query, "
                f"proceeding with current context"
            )
            return context

        all_queries.append(follow_up.strip())

        logger.info(
            f"[multi-step] [{agent_name}] Round {round_num}: "
            f"follow-up query = '{follow_up}' — reason: {evaluation['reason']}"
        )

        # Step 3: Follow-up search
        additional_context = await search_fn(follow_up)

        # Step 4: Merge contexts
        context = _merge_contexts(context, additional_context)

    logger.info(
        f"[multi-step] [{agent_name}] Reached max rounds ({rounds}), "
        f"returning merged context"
    )
    return context