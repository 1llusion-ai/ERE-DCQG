"""FairytaleQA Evidence Audit.

For each QA pair, assess evidence difficulty using the story section as context.
Determines whether the answer can be found from a single sentence or requires
multi-sentence evidence chains.

This is the narrative-QA counterpart to dcqg.path.evidence_necessity
(which audits MAVEN-ERE event-based evidence).

Phase 1: audit only — no question generation.
"""
import json
import re

from dcqg.utils.api_client import call_openai_compatible
from dcqg.utils.config import get_api_config


# Speech verbs that can introduce or follow quoted dialog in FairytaleQA text.
_SPEECH_VERBS = {
    'said', 'asked', 'replied', 'cried', 'answered', 'shouted',
    'say', 'says', 'answer', 'repeat', 'repeated', 'thought',
    'whispered', 'told', 'called', 'exclaimed', 'murmured', 'muttered',
    'screamed', 'yelled', 'begged', 'began', 'continued', 'added',
    'declared', 'announced', 'responded', 'retorted', 'demanded',
    'inquired', 'sighed', 'sobbed', 'laughed', 'smiled',
    'pleaded', 'groaned', 'gasped', 'snapped', 'growled', 'roared',
    'bellowed', 'hissed', 'rasped', 'breathed', 'wondered',
    'questioned', 'rejoined', 'barked', 'resumed', 'insisted',
    'suggested', 'explained', 'remarked', 'warned', 'ordered',
    'commanded', 'urged', 'confessed', 'admitted', 'protested',
    'threatened', 'blurted', 'interrupted', 'interjected', 'chimed',
    'piped', 'quipped', 'observed',
}

# Titles / honorifics whose trailing period must not trigger a sentence
# boundary (Mr., Mrs., Ms., Dr., St., Prof., Capt., Rev.).
_ABBREVIATIONS = frozenset({
    'mr', 'mrs', 'ms', 'dr', 'st', 'prof', 'capt', 'rev',
})

# Tokens that are *not* quote characters even though they contain a
# single-quote glyph.  These are FairytaleQA contractions.
_CONTRACTION_TOKENS = frozenset({
    "'s", "n't", "'ll", "'m", "'re", "'ve", "'d", "'t",
})

# Opening-quote tokens (standalone).
_OPEN_QUOTE_TOKENS = frozenset({'"', '“', "'", '‘'})
# Closing-quote tokens (standalone).
_CLOSE_QUOTE_TOKENS = frozenset({'"', '”', "'", '’'})


def _is_standalone_quote(token):
    """True when *token* is a quote mark, not a contraction apostrophe."""
    if token in _CONTRACTION_TOKENS:
        return False
    return token in _OPEN_QUOTE_TOKENS or token in _CLOSE_QUOTE_TOKENS


def _quote_kind(token):
    """Return the quote kind for a standalone quote token."""
    if token in _CONTRACTION_TOKENS:
        return None
    if token == '"':
        return 'double'
    if token == "'":
        return 'single'
    if token in _OPEN_QUOTE_TOKENS or token in _CLOSE_QUOTE_TOKENS:
        return token
    return None


def _looks_like_word(token):
    """True for ordinary word tokens around dialect apostrophes."""
    token = token.strip('",.;:!?')
    return bool(token) and any(ch.isalpha() for ch in token)


def _is_sentence_end_token(token):
    """True for tokenized sentence-ending punctuation."""
    return token in ('.', '!', '?') or bool(re.fullmatch(r'[.!?]+', token))


def _quote_kind_at(tokens, idx):
    """Context-aware quote kind.

    FairytaleQA sometimes tokenizes dialect elisions as a standalone
    apostrophe, as in ``lickin ' her`` or ``an ' how``. Those should not
    be paired as single quotation marks.
    """
    kind = _quote_kind(tokens[idx])
    if kind != 'single':
        return kind
    prev_token = tokens[idx - 1] if idx > 0 else ''
    next_token = tokens[idx + 1] if idx + 1 < len(tokens) else ''
    if _looks_like_word(prev_token) and _looks_like_word(next_token):
        return None
    return kind


def _is_speech_verb(tokens, idx):
    """Check if token at *idx* is a speech verb (case-insensitive)."""
    if idx < 0 or idx >= len(tokens):
        return False
    word = tokens[idx].strip('",.;:!?\'').lower()
    return word in _SPEECH_VERBS


def _find_quote_spans(tokens):
    """Return local (open_idx, close_idx) pairs for matched quotes.

    Each opening quote is paired with the next standalone quote of the
    same kind.  This local pairing avoids the previous greedy behavior
    where an unmatched quote could consume several ordinary narrative
    sentences.  Contraction tokens such as ``'s``, ``n't``, ``'ll`` are
    never treated as quote markers.
    """
    spans = []
    open_idx = None
    open_kind = None

    for i, token in enumerate(tokens):
        kind = _quote_kind_at(tokens, i)
        if kind is None:
            continue
        if open_idx is None:
            open_idx = i
            open_kind = kind
            continue
        if kind == open_kind:
            spans.append((open_idx, i))
            open_idx = None
            open_kind = None

    return spans


def _next_token_after(tokens, idx):
    """Return the first non-punctuation token after *idx*, or ``None``."""
    for k in range(idx + 1, len(tokens)):
        t = tokens[k]
        if t in (',', ';', ':') or _is_sentence_end_token(t):
            continue
        return t
    return None


def _has_speech_attribution_after_quote(tokens, close_idx, max_scan=8):
    """True if a closing quote is followed by a local speech attribution."""
    first_idx = None
    for k in range(close_idx + 1, min(len(tokens), close_idx + max_scan + 1)):
        if tokens[k] in (',', ';', ':') or _is_sentence_end_token(tokens[k]):
            continue
        first_idx = k
        break
    if first_idx is None:
        return False

    first = tokens[first_idx].lower()
    if first in {
        'and', 'but', 'then', 'so', 'with', 'when', 'after', 'before', 'as',
        'thereupon', 'upon', 'now',
    }:
        return False

    # Verb-first attribution: "..." said the boy .
    if _is_speech_verb(tokens, first_idx):
        return True

    # Copula attribution: "..." was the answer / was again the reply .
    # After a closing quote, a copula without its own subject is a
    # fragment that belongs to the quote.
    if first in {'was', 'were', 'is', 'are', 'be', 'being', 'been',
                 'seemed', 'appeared', 'became', 'remained'}:
        return True

    # Pronoun-first attribution: "..." he said .
    # Named subjects after a quote are often the next speaker/action in these
    # stories ("..." the maiden said that...), so keep this intentionally narrow.
    if first in {'he', 'she', 'they', 'it', 'i', 'we', 'you'}:
        for k in range(first_idx + 1, min(len(tokens), first_idx + 4)):
            token = tokens[k]
            if token in (';', ':') or _is_sentence_end_token(token):
                return False
            if token == ',':
                continue
            if not _looks_like_word(token):
                continue
            if _is_speech_verb(tokens, k):
                next_word = _next_token_after(tokens, k)
                if next_word and next_word.lower() in {
                    'that', 'whether', 'if', 'what', 'why', 'how', 'where',
                    'when', 'who', 'which',
                }:
                    return False
                return True
            return False

    return False


def _span_is_nested(span, spans):
    """True if *span* is fully contained in another quote span."""
    start, end = span
    return any(other_start < start and end < other_end
               for other_start, other_end in spans)


def _indices_inside_spans(spans):
    """Return token indices strictly inside any span."""
    inside = set()
    for start, end in spans:
        inside.update(range(start + 1, end))
    return inside


def _sentence_ends_in_quote_span(tokens, span, nested_spans):
    """Sentence ends in a quote span, ignoring nested quoted names/fragments."""
    start, end = span
    nested_inside = _indices_inside_spans(nested_spans)
    ends = []
    inner_quote_kind = None
    for k in range(start + 1, end):
        if k in nested_inside:
            continue
        kind = _quote_kind_at(tokens, k)
        if kind is not None:
            if inner_quote_kind is None:
                inner_quote_kind = kind
            elif kind == inner_quote_kind:
                inner_quote_kind = None
            continue
        if inner_quote_kind is not None:
            continue
        token = tokens[k]
        if _is_sentence_end_token(token):
            ends.append(k)
    return ends


def _split_story_sentences(text):
    """Split FairytaleQA story text into complete semantic sentences.

    Keeps direct speech together with its speech-verb attribution so that
    evidence counting reflects the number of *semantic* sentences rather
    than the number of typographic fragments.

    For example::

        " I will go , " said the boy .           -> 1 sentence
        the boy said , " I will go . "           -> 1 sentence
        The sun set . " Let us go , " she said . -> 2 sentences

    Sentence identifiers use the [S0], [S1], ... numbering used across the
    evidence-audit pipeline.
    """
    if not text:
        return []

    text = text.strip()
    if not text:
        return []

    text = re.sub(r'(["“”])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    n = len(tokens)

    # ---- conservative sentence-ending positions -----------------------
    # Ordinary narrative punctuation remains a sentence boundary. Matched
    # quote spans are treated as local speech units: a one-sentence quote
    # stays with its attribution, while a multi-sentence quote is split at
    # internal complete-sentence boundaries. Nested quoted names/fragments
    # such as "' take hold ! '" are ignored when deciding whether the outer
    # quote contains multiple complete sentences.
    quote_spans = _find_quote_spans(tokens)
    open_to_close = {q_open: q_close for q_open, q_close in quote_spans}
    close_to_open = {q_close: q_open for q_open, q_close in quote_spans}
    non_nested_spans = [
        span for span in quote_spans
        if not _span_is_nested(span, quote_spans)
    ]
    nested_spans_by_outer = {
        span: [
            other for other in quote_spans
            if span[0] < other[0] and other[1] < span[1]
        ]
        for span in non_nested_spans
    }
    non_nested_closes = {end for _, end in non_nested_spans}
    inside_non_nested_quotes = _indices_inside_spans(non_nested_spans)

    inner_quote_breaks = set()
    inner_quote_break_quotes = {}
    quote_sentence_ends_by_close = {}
    for span in non_nested_spans:
        ends = _sentence_ends_in_quote_span(
            tokens, span, nested_spans_by_outer[span]
        )
        quote_sentence_ends_by_close[span[1]] = ends
        if len(ends) > 1:
            for break_idx in ends[:-1]:
                inner_quote_breaks.add(break_idx)
                inner_quote_break_quotes[break_idx] = (tokens[span[0]], tokens[span[1]])

    final_breaks = set()

    for i, token in enumerate(tokens):
        if i in open_to_close:
            continue

        if i in close_to_open:
            if i not in non_nested_closes:
                continue
            if i == n - 1:
                final_breaks.add(i)
                continue
            sentence_ends = quote_sentence_ends_by_close.get(i, [])
            if not sentence_ends:
                continue
            nxt = tokens[i + 1]
            if nxt in (';', ':', ','):
                continue
            if _has_speech_attribution_after_quote(tokens, i):
                continue
            final_breaks.add(i)
            continue

        if _is_sentence_end_token(token):
            if i in inner_quote_breaks:
                final_breaks.add(i)
                continue
            if i in inside_non_nested_quotes:
                continue
            # Abbreviation period (Mr., Mrs., Dr., etc.) — not a sentence end.
            if token == '.' and i > 0:
                prev_word = tokens[i - 1].strip('",;:!?\'').lower()
                if prev_word in _ABBREVIATIONS:
                    continue
            final_breaks.add(i)

    if not final_breaks:
        return [text]

    boundaries = sorted(final_breaks)
    sentences = []
    prev = 0
    prefix_quote = None
    for b in boundaries:
        sent_tokens = tokens[prev:b + 1]
        if prefix_quote:
            sent_tokens = [prefix_quote] + sent_tokens
            prefix_quote = None
        if b in inner_quote_break_quotes:
            open_quote, close_quote = inner_quote_break_quotes[b]
            if not sent_tokens or sent_tokens[-1] != close_quote:
                sent_tokens = sent_tokens + [close_quote]
            prefix_quote = open_quote
        sent_text = ' '.join(sent_tokens).strip()
        if sent_text:
            sentences.append(sent_text)
        prev = b + 1

    if prev < n:
        remaining_tokens = tokens[prev:]
        if prefix_quote:
            remaining_tokens = [prefix_quote] + remaining_tokens
        remaining = ' '.join(remaining_tokens).strip()
        if remaining:
            sentences.append(remaining)

    sentences = _cleanup_sentence_starts(sentences)
    sentences = _merge_orphan_attributions(sentences)
    return sentences if sentences else [text]


def _merge_orphan_attributions(sentences):
    """Post-process: merge pure-quote sentences with orphan attribution fragments.

    When a sentence is entirely quoted speech and the next sentence is a short
    attribution fragment (speech verb or copula), merge them.  This catches
    cases that the token-level attribution check may miss (e.g. rare speech
    verbs, inverted subjects).
    """
    if len(sentences) <= 1:
        return sentences

    _ALL_QUOTES = _OPEN_QUOTE_TOKENS | _CLOSE_QUOTE_TOKENS

    def _is_pure_quoted_speech(sent):
        tokens = sent.strip().split()
        if len(tokens) < 2:
            return False
        if tokens[0] not in _ALL_QUOTES or tokens[-1] not in _ALL_QUOTES:
            return False
        quote_count = sum(
            1 for t in tokens
            if t in _ALL_QUOTES and t not in _CONTRACTION_TOKENS
        )
        return quote_count >= 2 and quote_count % 2 == 0

    def _is_attribution_fragment(sent):
        tokens = sent.strip().split()
        if not tokens or len(tokens) > 6:
            return False
        first = tokens[0].strip('",.;:!?\'').lower()
        if first in _SPEECH_VERBS:
            return True
        if first in {'was', 'were', 'is', 'are', 'be', 'being', 'been',
                     'seemed', 'appeared', 'became', 'remained'}:
            return True
        return False

    merged = []
    i = 0
    while i < len(sentences):
        sent = sentences[i].strip()
        if _is_pure_quoted_speech(sent) and i + 1 < len(sentences):
            nxt = sentences[i + 1].strip()
            if _is_attribution_fragment(nxt):
                merged.append(sent + ' ' + nxt)
                i += 2
                continue
        merged.append(sent)
        i += 1

    return merged


def _check_split_anomalies(sentences):
    """Return anomaly descriptions for a split-sentence list.

    Detects orphan quote-characters, stray abbreviation fragments, and
    consecutive quote-only sentences — signals that the split may need
    manual review.  An empty list means the split looks clean.
    """
    if not sentences:
        return []

    _ALL_QUOTES = _OPEN_QUOTE_TOKENS | _CLOSE_QUOTE_TOKENS
    issues = []

    for j, sent in enumerate(sentences):
        stripped = sent.strip()
        tokens = stripped.split()
        if not tokens:
            continue

        # Sentence consisting solely of quote characters.
        if all(t in _ALL_QUOTES for t in tokens):
            issues.append(f'orphan quote-only sentence at S{j}: {stripped!r}')
            continue

        # Sentence consisting of an abbreviation-period pair (e.g. "mr .").
        if (len(tokens) == 2
                and tokens[0].strip('",.;:!?\'').lower() in _ABBREVIATIONS
                and tokens[1] == '.'):
            issues.append(
                f'orphan abbreviation fragment at S{j}: {stripped!r}'
            )
            continue

        # Stray closing quote at sentence start not resolved by cleanup.
        # Only flag when the previous sentence has an unbalanced quote count,
        # meaning this quote is a plausible close-match for its open quote.
        if (j > 0
                and tokens[0] in _ALL_QUOTES
                and not any(t in _ALL_QUOTES for t in tokens[1:])):
            prev_tokens = sentences[j - 1].strip().split()
            prev_quote_count = sum(
                1 for t in prev_tokens
                if t in _ALL_QUOTES and t not in _CONTRACTION_TOKENS
            )
            if prev_quote_count % 2 == 1:
                issues.append(
                    f'unresolved stray closing quote at S{j}: {stripped!r}'
                )
            continue

    # Consecutive quote-only sentences.
    for j in range(len(sentences) - 1):
        s0 = sentences[j].strip()
        s1 = sentences[j + 1].strip()
        t0 = s0.split()
        t1 = s1.split()
        if (t0 and t1
                and all(t in _ALL_QUOTES for t in t0)
                and all(t in _ALL_QUOTES for t in t1)):
            issues.append(
                f'consecutive quote-only sentences at S{j} / S{j + 1}'
            )

    return issues


def _cleanup_sentence_starts(sentences):
    """Post-process sentence list so no sentence starts with an isolated
    closing quote or with punctuation that belongs to the previous
    sentence (``;``, ``:``, ``,``).

    A leading ``\"`` or ``'`` is only treated as a stray closing quote
    when the sentence contains *no other quote character*.  Otherwise it
    is an opening quote and is left in place.
    """
    if len(sentences) <= 1:
        return sentences

    _ALL_QUOTES = _OPEN_QUOTE_TOKENS | _CLOSE_QUOTE_TOKENS

    def _context_quote_count(tokens_local, stop=None):
        if stop is None:
            stop = len(tokens_local)
        return sum(
            1 for idx in range(stop)
            if _quote_kind_at(tokens_local, idx) is not None
        )

    def _has_unmatched_quote(sent):
        return _context_quote_count(sent.split()) % 2 == 1

    result = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        tokens_local = sent.split()
        first_token = tokens_local[0]

        # ---- stray closing quote at sentence start ---------------------
        if first_token in _CLOSE_QUOTE_TOKENS:
            rest_tokens = tokens_local[1:]
            # Adjacent speaker boundary: the first quote is a stray
            # closing quote and the second is a new opening quote.
            # Eg: '... ended . " " no , it is ...'
            if (rest_tokens
                    and rest_tokens[0] in _ALL_QUOTES
                    and result
                    and _has_unmatched_quote(result[-1])):
                result[-1] = result[-1] + ' ' + first_token
                tokens_local = rest_tokens  # reprocess as new sentence
                first_token = tokens_local[0]
                rest_tokens = tokens_local[1:]
                # fall through to handle the second quote normally
            has_other_quote = any(
                _quote_kind_at(tokens_local, idx) is not None
                for idx in range(1, len(tokens_local))
            )
            if (not has_other_quote
                    and result
                    and _has_unmatched_quote(result[-1])):
                # Orphan closing quote — attach to previous sentence.
                result[-1] = result[-1] + ' ' + first_token
                rest_text = ' '.join(rest_tokens).strip()
                if rest_text:
                    result.append(rest_text)
                continue
            # Opening quote for a new speech unit — leave in place.

        # ---- stray opening quote at end of previous sentence -----------
        # After the main loop, merge trailing stand-alone quotes back.

        # ---- punctuation that never begins a sentence ------------------
        if first_token in (';', ':', ','):
            if result:
                result[-1] = result[-1] + ' ' + sent
            else:
                rest_text = ' '.join(tokens_local[1:]).strip()
                if rest_text:
                    result.append(rest_text)
            continue

        result.append(sent)

    # ---- second pass: trailing stray quotes at sentence ends -----------
    # A sentence like  "... narrative . "  where the final  "  has no
    # matching open quote in the same sentence should be cleaned up by
    # moving the quote to the next sentence (or dropping it if it's the
    # last sentence and clearly stray).
    cleaned = []
    open_quote_carry = False
    for i, sent in enumerate(result):
        tokens_local = sent.split()
        if not tokens_local:
            continue
        last_token = tokens_local[-1]
        if last_token in _CLOSE_QUOTE_TOKENS:
            # Count quotes in this sentence (excluding the trailing one)
            quote_count = _context_quote_count(tokens_local, len(tokens_local) - 1)
            if quote_count == 0 and i + 1 < len(result) and not open_quote_carry:
                # Stray trailing quote — move to next sentence's start.
                result[i + 1] = last_token + ' ' + result[i + 1]
                sent = ' '.join(tokens_local[:-1]).strip()
        if sent:
            cleaned.append(sent)
            total_quotes = _context_quote_count(sent.split())
            if total_quotes % 2 == 1:
                open_quote_carry = not open_quote_carry

    normalized = []
    carry_quote = None
    open_quote_carry = False
    for sent in cleaned:
        if carry_quote:
            sent = carry_quote + ' ' + sent
            carry_quote = None
        tokens_local = sent.split()
        if not tokens_local:
            continue

        if (tokens_local[0] in _ALL_QUOTES
                and normalized
                and open_quote_carry
                and normalized[-1].split()[-1] not in _ALL_QUOTES):
            normalized[-1] = normalized[-1] + ' ' + tokens_local[0]
            tokens_local = tokens_local[1:]
            sent = ' '.join(tokens_local).strip()
            open_quote_carry = False
            if not tokens_local:
                continue

        if (len(tokens_local) >= 2
                and tokens_local[0] in _ALL_QUOTES
                and tokens_local[1] in _ALL_QUOTES
                and normalized
                and open_quote_carry
                and normalized[-1].split()[-1] not in _ALL_QUOTES):
            normalized[-1] = normalized[-1] + ' ' + tokens_local[0]
            tokens_local = tokens_local[1:]
            sent = ' '.join(tokens_local).strip()

        if (len(tokens_local) >= 2
                and tokens_local[-1] in _ALL_QUOTES
                and tokens_local[-2] in _ALL_QUOTES):
            carry_quote = tokens_local[-1]
            sent = ' '.join(tokens_local[:-1]).strip()

        if sent:
            normalized.append(sent)
            total_quotes = _context_quote_count(sent.split())
            if total_quotes % 2 == 1:
                open_quote_carry = not open_quote_carry

    if carry_quote:
        if normalized:
            normalized[-1] = normalized[-1] + ' ' + carry_quote
        else:
            normalized.append(carry_quote)

    return normalized


def _split_sentences(text):
    """Compatibility alias for :func:`_split_story_sentences`."""
    return _split_story_sentences(text)


def _build_evidence_prompt(qa_records):
    """Build LLM prompt for evidence difficulty assessment of FairytaleQA pairs.

    Each record has: story_name, story_section, question, answer1, and metadata.
    """
    parts = []
    for i, rec in enumerate(qa_records):
        sentences = _split_story_sentences(rec["story_section"])
        sent_lines = "\n".join(f"  [S{j}] {s}" for j, s in enumerate(sentences))
        answer = rec["answer1"]
        if rec.get("answer2"):
            answer += f" / {rec['answer2']}"

        parts.append(
            f'--- QA Pair {i + 1} ---\n'
            f'Story: {rec["story_name"]}\n'
            f'Question: {rec["question"]}\n'
            f'Answer: {answer}\n'
            f'Metadata: local-or-sum={rec.get("local_or_sum", "N/A")}, '
            f'attribute={rec.get("attribute", "N/A")}, '
            f'ex-or-im={rec.get("ex_or_im", "N/A")}\n'
            f'Story section sentences:\n{sent_lines}'
        )

    qa_block = "\n\n".join(parts)

    return f"""You are an expert evidence analyst for narrative reading comprehension.

For each QA pair below, determine how many sentences from the story section
a reader MUST read to answer the question correctly.

## CRITICAL DISTINCTION

There are two different reasons extra sentences might be needed:

1. BACKGROUND CONTEXT: Extra sentences help understand the setting, characters,
   or situation. But the answer itself is directly stated in one sentence.

2. ANSWER IDENTIFICATION: The answer cannot be determined from a single sentence.
   The reader must combine information from multiple sentences to identify
   the correct answer.

Example of BACKGROUND CONTEXT (NOT Hard):
  Q: "What did the wolf do?"
  S3: "The wolf crept through the forest."
  S1: "Once upon a time, in a dark forest, there lived a wolf."
  -> S3 alone answers the question. S1 is background. answer_sentence_alone_sufficient=yes.

Example of ANSWER IDENTIFICATION (Hard):
  Q: "Why did the princess leave the castle?"
  S5: "She packed her bags that night."
  S3: "The king had forbidden her from seeing the young knight."
  S4: "The knight was banished to the eastern mountains."
  -> S5 says she left but not why. S3+S4 explain the motivation.
     answer_sentence_alone_sufficient=no, necessity_type=motivation_bridge.

## Evidence assessment fields

For each QA pair, provide:

1. answer_directly_found:
   Can the target answer, or a close paraphrase of it, be directly found in
   the provided text?
   - "yes": the answer text is explicitly present or directly paraphrased
   - "no": the answer must be inferred

2. answer_sentence_alone_sufficient:
   Is the most likely answer sentence alone enough to answer the question?
   - "yes": the sentence directly states the answer
   - "partial": the sentence gives a hint but needs one other sentence
   - "no": the sentence alone is ambiguous or insufficient

3. section_evidence_sufficient:
   Is the entire story section enough to answer the question?
   - "yes": the section fully contains the answer
   - "partial": the section helps but some inference is needed
   - "no": the question requires information beyond this section

4. full_context_needed:
   Does answering require the full story context (beyond the section)?
   - "yes": must read beyond the section
   - "partial": section is mostly sufficient
   - "no": section is fully sufficient

5. required_evidence_sentences:
   List of sentence indices [S0, S1, ...] that are REQUIRED to answer.

6. bridge_sentence_ids:
   Sentence indices that connect context to the answer. Empty if none.

7. num_required_sentences:
   How many sentences are required (len of required_evidence_sentences).

8. reasoning_operation:
   What kind of reasoning is needed?
   - "explicit_lookup": answer is directly stated
   - "temporal_order": answer depends on event sequence
   - "causal_chain": answer reached through cause-effect
   - "motivation": answer requires understanding character motivation
   - "character_state": answer about character feelings/beliefs
   - "disambiguation": multiple possible answers, need context to resolve
   - "summary_inference": answer requires summarizing multiple facts
   - "contrast": answer requires comparing/contrasting

9. bridge_removal_effect:
   If bridge sentences are removed:
   - "none": answer still findable
   - "harder": answer harder but possible
   - "ambiguous": answer becomes ambiguous
   - "unanswerable": answer cannot be determined

10. necessity_type:
   Why are extra sentences needed?
   - "background_context": extra sentences provide background only
   - "answer_identification": extra sentences needed to find the answer
   - "disambiguation": extra sentences resolve ambiguity
   - "causal_bridge": extra sentences provide causal chain
   - "temporal_bridge": extra sentences establish time sequence
   - "motivation_bridge": extra sentences explain character motivation
   - "summary_synthesis": answer requires synthesizing multiple facts

11. evidence_necessity_reason:
    One sentence explaining why.

## QA Pairs to assess

{qa_block}

## Output

Return a JSON object with key "assessments" containing a list, one entry per
QA pair.  Each entry MUST include ALL fields:
{{
  "qa_id": 1,
  "answer_directly_found": "no",
  "answer_sentence_alone_sufficient": "no",
  "section_evidence_sufficient": "yes",
  "full_context_needed": "no",
  "required_evidence_sentences": [2, 3, 5],
  "bridge_sentence_ids": [3],
  "num_required_sentences": 3,
  "reasoning_operation": "motivation",
  "bridge_removal_effect": "ambiguous",
  "necessity_type": "motivation_bridge",
  "evidence_necessity_reason": "S5 states the action but S3 and S4 explain
    the motivation. Without S3, the reader cannot determine why the character
    acted."
}}

Important:
- required_evidence_sentences must ONLY contain valid sentence indices from
  the story section shown above (0-based).
- Do NOT mark Hard merely because background context helps understand.
- If the answer is directly stated in one sentence, set answer_directly_found=yes.
- If the answer must be inferred, set answer_directly_found=no even when only
  one evidence sentence is needed.

Return ONLY the JSON object, no other text."""


def _parse_assessments(resp, num_qa):
    """Parse LLM response into a list of assessment dicts."""
    if not resp:
        return None

    for attempt_fn in [
        lambda r: json.loads(r),
        lambda r: json.loads(r[r.index("{"):r.rindex("}") + 1]),
        lambda r: json.loads(
            re.sub(r'^```(?:json)?\s*', '', r.strip(), flags=re.IGNORECASE)
            .rstrip('`').strip()
        ),
    ]:
        try:
            data = attempt_fn(resp)
            if isinstance(data, dict) and "assessments" in data:
                assessments = data["assessments"]
                if isinstance(assessments, list) and len(assessments) == num_qa:
                    return assessments
        except (json.JSONDecodeError, ValueError):
            pass

    return None


# Valid values
_VALID_ABILITY = {"yes", "partial", "no"}
_VALID_DIRECT = {"yes", "no"}
_VALID_REMOVAL = {"none", "harder", "ambiguous", "unanswerable"}
_VALID_NEC_TYPE = {
    "background_context", "answer_identification", "disambiguation",
    "causal_bridge", "temporal_bridge", "motivation_bridge", "summary_synthesis",
}
_VALID_REASONING_OP = {
    "explicit_lookup", "temporal_order", "causal_chain", "motivation",
    "character_state", "disambiguation", "summary_inference", "contrast",
}


def _validate_assessment(a, num_sentences):
    """Validate and normalize an assessment dict.

    Args:
        a: raw assessment dict from LLM
        num_sentences: total sentences in the story section

    Returns:
        (assessment_dict, contradiction_count)
    """
    contradictions = 0
    valid_ids = set(range(num_sentences))

    # Validate lists
    raw_req = a.get("required_evidence_sentences", [])
    raw_bridge = a.get("bridge_sentence_ids", [])
    if not isinstance(raw_req, list):
        raw_req = []
    if not isinstance(raw_bridge, list):
        raw_bridge = []

    # Clamp to valid sentence indices
    req_ids = sorted(set(int(x) for x in raw_req if isinstance(x, (int, float)) and int(x) in valid_ids))
    bridge_ids = sorted(set(int(x) for x in raw_bridge if isinstance(x, (int, float)) and int(x) in valid_ids and int(x) in req_ids))
    a["required_evidence_sentences"] = req_ids
    a["bridge_sentence_ids"] = bridge_ids
    a["num_required_sentences"] = len(req_ids)

    # Validate enums
    for key in ("answer_sentence_alone_sufficient", "section_evidence_sufficient", "full_context_needed"):
        if a.get(key) not in _VALID_ABILITY:
            a[key] = "partial"
    if a.get("answer_directly_found") not in _VALID_DIRECT:
        # Backward-compatible fallback for older prompt outputs.
        a["answer_directly_found"] = (
            "yes" if a.get("answer_sentence_alone_sufficient") == "yes" else "no"
        )
    if a.get("bridge_removal_effect") not in _VALID_REMOVAL:
        a["bridge_removal_effect"] = "harder"
    if a.get("necessity_type") not in _VALID_NEC_TYPE:
        a["necessity_type"] = "answer_identification"
    if a.get("reasoning_operation") not in _VALID_REASONING_OP:
        a["reasoning_operation"] = "explicit_lookup"
    if not isinstance(a.get("evidence_necessity_reason"), str):
        a["evidence_necessity_reason"] = ""

    # Fix contradictions without changing the evidence set. Under the current
    # definition, a single evidence sentence can still require inference.
    num_req = a["num_required_sentences"]

    # Contradiction: answer sentence alone is sufficient but multiple required
    # sentences are listed. Keep the evidence set and correct the diagnostic.
    if a["answer_sentence_alone_sufficient"] == "yes" and num_req > 1:
        contradictions += 1
        a["answer_sentence_alone_sufficient"] = "no"

    # If one direct evidence sentence is enough, the answer sentence alone is
    # sufficient. This does not apply to single-sentence implicit reasoning.
    if (
        num_req == 1
        and a["answer_directly_found"] == "yes"
        and a["answer_sentence_alone_sufficient"] != "yes"
    ):
        contradictions += 1
        a["answer_sentence_alone_sufficient"] = "yes"

    return a, contradictions


def classify_difficulty(assessment):
    """Classify Easy/Medium/Hard from the evidence assessment.

    Labels are consistent with the canonical definitions in
    dcqg.difficulty.definitions.DIFFICULTY_DEFINITIONS.
    """
    if assessment.get("section_evidence_sufficient") == "no":
        return "Invalid"
    if assessment.get("full_context_needed") == "yes":
        return "Invalid"

    direct = assessment.get("answer_directly_found")
    if direct not in ("yes", "no"):
        direct = (
            "yes"
            if assessment.get("answer_sentence_alone_sufficient") == "yes"
            else "no"
        )
    num_req = assessment.get("num_required_sentences", 1)
    if not isinstance(num_req, (int, float)):
        return "Invalid"
    num_req = int(num_req)

    # No required evidence is not a valid training label.  This can happen when
    # counterfactual verification removes all Stage A evidence candidates.
    if num_req <= 0:
        return "Invalid"

    if direct == "yes" and num_req == 1:
        return "Easy"

    if direct == "yes" and num_req >= 2:
        return "Medium"

    if direct == "no" and num_req == 1:
        return "Medium"

    if direct == "no" and num_req >= 2:
        return "Hard"

    return "Invalid"


class FairytaleEvidenceAuditor:
    """Audit evidence difficulty for FairytaleQA QA pairs."""

    def __init__(self, batch_size=10, model=None, max_retries=2, timeout=120):
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout

        cfg = get_api_config()
        self.api_url = cfg["SILICONFLOW_API_URL"]
        self.api_key = cfg["SILICONFLOW_API_KEY"]
        self.model = model or cfg["JUDGE_MODEL"]

    def audit_batch(self, qa_records):
        """Audit a batch of QA pairs.

        Returns list of candidate dicts with evidence assessments.
        """
        prompt, raw_resp, assessments, parse_ok = self._assess(qa_records)

        results = []
        for i, rec in enumerate(qa_records):
            sentences = _split_story_sentences(rec["story_section"])
            num_sents = len(sentences)
            split_issues = _check_split_anomalies(sentences)

            candidate = {
                "story_name": rec.get("story_name", ""),
                "story_section": rec.get("story_section", ""),
                "question": rec.get("question", ""),
                "answer": rec.get("answer1", ""),
                "answer1": rec.get("answer1", ""),
                "answer2": rec.get("answer2", ""),
                "local_or_sum": rec.get("local_or_sum", ""),
                "attribute": rec.get("attribute", ""),
                "ex_or_im": rec.get("ex_or_im", ""),
                "ex_or_im2": rec.get("ex_or_im2", ""),
                "split": rec.get("split", ""),
                "num_sentences_in_section": num_sents,
                "split_anomalies": split_issues,
                "needs_manual_check": bool(split_issues),
                # Trace fields
                "fairytale_evidence_prompt": prompt,
                "fairytale_evidence_raw": raw_resp or "",
                "fairytale_evidence_parse_ok": parse_ok,
                "fairytale_evidence_model": self.model,
            }

            if assessments and i < len(assessments):
                a, n_contra = _validate_assessment(assessments[i], num_sents)
                candidate.update({
                    "answer_directly_found": a["answer_directly_found"],
                    "answer_sentence_alone_sufficient": a["answer_sentence_alone_sufficient"],
                    "section_evidence_sufficient": a["section_evidence_sufficient"],
                    "full_context_needed": a["full_context_needed"],
                    "required_evidence_sentences": a["required_evidence_sentences"],
                    "bridge_sentence_ids": a["bridge_sentence_ids"],
                    "num_required_sentences": a["num_required_sentences"],
                    "reasoning_operation": a["reasoning_operation"],
                    "bridge_removal_effect": a["bridge_removal_effect"],
                    "necessity_type": a["necessity_type"],
                    "evidence_necessity_reason": a["evidence_necessity_reason"],
                    "assessment_status": "ok",
                    "fairytale_evidence_status": "ok",
                    "contradiction_count": n_contra,
                })
            else:
                candidate.update({
                    "answer_directly_found": "no",
                    "answer_sentence_alone_sufficient": "partial",
                    "section_evidence_sufficient": "partial",
                    "full_context_needed": "partial",
                    "required_evidence_sentences": [],
                    "bridge_sentence_ids": [],
                    "num_required_sentences": 0,
                    "reasoning_operation": "explicit_lookup",
                    "bridge_removal_effect": "harder",
                    "necessity_type": "background_context",
                    "evidence_necessity_reason": "assessment_failed",
                    "assessment_status": "llm_error",
                    "fairytale_evidence_status": "llm_error",
                    "contradiction_count": 0,
                })

            candidate["evidence_difficulty"] = classify_difficulty(candidate)
            results.append(candidate)

        return results

    def _assess(self, qa_records):
        """Call LLM to assess evidence necessity.

        Returns (prompt, raw_response, assessments_or_None, parse_ok).
        """
        prompt = _build_evidence_prompt(qa_records)

        for attempt in range(self.max_retries + 1):
            try:
                resp = call_openai_compatible(
                    prompt,
                    api_url=self.api_url,
                    api_key=self.api_key,
                    model=self.model,
                    temperature=0.0,
                    max_tokens=4000,
                    json_mode=True,
                    system="You are a precise evidence analyst for narrative QA. Return only valid JSON.",
                    timeout=self.timeout,
                )
                assessments = _parse_assessments(resp, len(qa_records))
                if assessments:
                    return (prompt, resp, assessments, True)
                if attempt == self.max_retries:
                    return (prompt, resp or "", None, False)
            except Exception as e:
                if attempt == self.max_retries:
                    return (prompt, f"ERROR: {e}", None, False)

        return (prompt, "", None, False)
