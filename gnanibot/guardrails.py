"""Guardrails for the Gnani.ai chatbot.

Input:   length, prompt-injection and abuse checks (before retrieval/LLM).
Context: scraped text is untrusted too, so strip injection-looking lines.
Output:  block leaked prompts and figures (prices, %) not present in the context.
"""
import re

MAX_INPUT_CHARS = 500

REFUSAL_INJECTION = "I can't help with that. Ask me something about Gnani.ai and its products."
REFUSAL_ABUSE = "Let's keep it respectful. Happy to answer questions about Gnani.ai."
REFUSAL_LONG = f"Please keep your question under {MAX_INPUT_CHARS} characters."
FALLBACK_UNKNOWN = ("I couldn't find that in the Gnani.ai website content I have. "
                    "Try rephrasing, or check https://www.gnani.ai directly.")
FALLBACK_UNGROUNDED = ("I wasn't able to verify that answer against the website content, "
                       "so I'd rather not guess. Please check https://www.gnani.ai.")

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions|prompts?|rules)",
    r"disregard\s+.{0,40}(instructions|rules|prompt)",
    r"forget\s+(everything|all|your)\s+.{0,30}(instructions|rules|prompt)",
    r"(reveal|show|print|repeat|leak|tell me)\s+.{0,30}(system|hidden|initial|original)\s+(prompt|instructions?|message)",
    r"\byou\s+are\s+now\b",
    r"\bpretend\s+(you\s+are|to\s+be)\b",
    r"\b(jailbreak|developer\s+mode|do\s+anything\s+now)\b",
    r"<\s*/?\s*(system|assistant)\s*>",
]
_INJECTION_RE = [re.compile(p, re.I) for p in INJECTION_PATTERNS]

# Starter list only: extend or replace with a proper classifier later.
ABUSE_WORDS = {"idiot", "stupid", "moron", "shut up", "fuck", "shit", "bitch", "asshole"}

_SMALLTALK_RE = re.compile(
    r"^\s*(hi|hello|hey|namaste|good\s+(morning|afternoon|evening)|thanks?|thank\s+you|bye)\b[\s!.?]*$",
    re.I,
)

# Money and percentage figures are the riskiest things to hallucinate.
_FIGURE_RE = re.compile(r"(?:[$₹€£]|rs\.?|inr|usd)\s?\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?\s?%", re.I)


def is_smalltalk(text: str) -> bool:
    return bool(_SMALLTALK_RE.match(text))


def check_input(text: str) -> tuple[bool, str]:
    """Return (ok, refusal_message). refusal_message is '' when ok."""
    if len(text) > MAX_INPUT_CHARS:
        return False, REFUSAL_LONG
    if any(r.search(text) for r in _INJECTION_RE):
        return False, REFUSAL_INJECTION
    low = text.lower()
    if any(w in low for w in ABUSE_WORDS):
        return False, REFUSAL_ABUSE
    return True, ""


def sanitize_context(text: str) -> str:
    """Drop lines of scraped text that look like instructions aimed at the model."""
    kept = [ln for ln in text.splitlines() if not any(r.search(ln) for r in _INJECTION_RE)]
    return "\n".join(kept)


def _norm(s: str) -> str:
    return re.sub(r"[\s,]", "", s.lower())


def check_output(answer: str, context: str, system_prompt: str) -> tuple[bool, str]:
    """Return (ok, reason). reason is '' when ok."""
    # 1. system prompt leak: any long verbatim run from the prompt
    words = system_prompt.split()
    for i in range(0, max(1, len(words) - 8)):
        if " ".join(words[i:i + 8]).lower() in answer.lower():
            return False, "possible system prompt leak"

    # 2. every price / percentage in the answer must appear in the context
    ctx = _norm(context)
    for fig in _FIGURE_RE.findall(answer):
        if _norm(fig) not in ctx:
            return False, f"ungrounded figure: {fig.strip()}"
    return True, ""