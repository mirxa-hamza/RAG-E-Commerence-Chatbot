"""
Offline checks for the ANSWER LENGTH contract. No network, no model, no key.

    python tests/test_answer_length_offline.py

The failure this guards is a pair, and fixing either half wrongly re-creates the other:

* **Too long.** Retrieval hands the model up to MAX_CONTEXT_CHARS of passages, and an open
  question ("tell me more about X") invites it to summarise every one of them - a page of
  bullets with a citation on each, where three sentences answered the question.
* **Too short, in the worst way.** The obvious cure is to cut LLM_MAX_TOKENS. That does
  not shorten the answer, it CUTS it: the model writes the same page and the transport stops
  mid-word. A truncated answer is strictly worse than a long one, because the reader cannot
  tell which facts were dropped.

So the budget is stated in the PROMPT (twice - once in the system prompt, once in the
reminder the model reads last) while the token ceiling stays generous. These checks pin that
split down, because "shorten the answer" is a change someone will reasonably try to make by
lowering the ceiling.
"""
import importlib
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Set before importing config: read at import time.
os.environ.setdefault("JWT_SECRET", "offline-test-secret-not-used-anywhere-real")
os.environ.pop("ANSWER_STYLE", None)
os.environ.pop("ANSWER_MAX_WORDS", None)
os.environ.pop("ANSWER_MAX_BULLETS", None)

from src.core import config  # noqa: E402
from src.ml import llm  # noqa: E402

PASSED = 0
FAILED = 0


def check(label, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {label}")
    else:
        FAILED += 1
        print(f"  [FAIL] {label}{('  -> ' + str(detail)) if detail else ''}")


def reload_config(**env):
    """
    Re-import config with these env vars set; return a SNAPSHOT of the answer-length
    settings, or the exception the import raised.

    A snapshot, not the module: `importlib.reload` mutates the existing module object in
    place, so restoring the environment and reloading again would rewrite the very values
    the caller is about to assert on - every check would read the defaults and pass
    regardless of what the override did.
    """
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update({k: str(v) for k, v in env.items()})
    try:
        reloaded = importlib.reload(config)
        return {
            "LLM_PROVIDER": reloaded.LLM_PROVIDER,
            "ANSWER_STYLE": reloaded.ANSWER_STYLE,
            "ANSWER_MAX_WORDS": reloaded.ANSWER_MAX_WORDS,
            "ANSWER_MAX_BULLETS": reloaded.ANSWER_MAX_BULLETS,
        }
    except Exception as exc:  # noqa: BLE001 - the raised error IS the thing under test
        return exc
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(config)


# ---------------------------------------------------------------- the settings
print("\nANSWER_STYLE - the budget setting")

check("defaults to 'concise'", config.ANSWER_STYLE == "concise", config.ANSWER_STYLE)
check("...which is a real budget, not unlimited",
      20 <= config.ANSWER_MAX_WORDS <= 200, config.ANSWER_MAX_WORDS)
check("a bullet cap exists too (20 three-word bullets obey a word budget and still dump)",
      config.ANSWER_MAX_BULLETS >= 1, config.ANSWER_MAX_BULLETS)

check("every named style maps to a word count",
      all(isinstance(v, int) and v > 0 for v in config.ANSWER_STYLES.values()),
      config.ANSWER_STYLES)
check("the styles are ordered brief < concise < standard < detailed",
      config.ANSWER_STYLES["brief"] < config.ANSWER_STYLES["concise"]
      < config.ANSWER_STYLES["standard"] < config.ANSWER_STYLES["detailed"],
      config.ANSWER_STYLES)

reloaded = reload_config(ANSWER_STYLE="detailed")
check("ANSWER_STYLE=detailed raises the budget",
      reloaded.get("ANSWER_MAX_WORDS") == config.ANSWER_STYLES["detailed"], reloaded)

reloaded = reload_config(ANSWER_STYLE="brief", ANSWER_MAX_WORDS="55")
check("an explicit ANSWER_MAX_WORDS overrides the style",
      reloaded.get("ANSWER_MAX_WORDS") == 55, reloaded)

check("a misspelt style is refused at import, not silently ignored",
      isinstance(reload_config(ANSWER_STYLE="short"), ValueError),
      reload_config(ANSWER_STYLE="short"))
check("...and the error names the valid styles",
      "concise" in str(reload_config(ANSWER_STYLE="short")))
check("an absurd budget is refused (a 5-word answer cannot carry a citation)",
      isinstance(reload_config(ANSWER_MAX_WORDS="5"), ValueError))


# ---------------------------------------------------------------- the prompt
print("\nThe budget reaches the model - in BOTH places")

check("the system prompt states the word budget",
      str(config.ANSWER_MAX_WORDS) in llm.SYSTEM_PROMPT)
check("the system prompt states the bullet cap",
      str(config.ANSWER_MAX_BULLETS) in llm.SYSTEM_PROMPT)

# The reminder is the half that matters most: by the time the model has read several
# thousand words of passages, a rule from the top of the prompt is a long way away, and
# everything it has just read is an invitation to summarise all of it.
check("the REMINDER repeats the word budget",
      str(config.ANSWER_MAX_WORDS) in llm.ANSWER_REMINDER, llm.ANSWER_REMINDER)
check("the reminder still carries the grounding rule (length must not displace it)",
      "only the CONTEXT" in llm.ANSWER_REMINDER)
check("the reminder still carries the citation rule",
      "Cite the document and page" in llm.ANSWER_REMINDER)

# Whitespace-normalised: the prompt is hard-wrapped at 90 columns, so a two-word phrase can
# have a newline and eight spaces of indent inside it.
lowered = " ".join(llm.SYSTEM_PROMPT.lower().split())
check("the prompt forbids walking through the passages one by one",
      "passage by passage" in lowered, "no anti-enumeration rule")
check("the prompt tells the model to offer to expand rather than pre-empt",
      "offer to go deeper" in lowered or "offer to expand" in lowered)
check("a broad question is named explicitly as still being in budget",
      "tell me about" in lowered, "the open-question case is not covered")
check("an explicit request for everything is allowed to exceed the budget",
      "list all" in lowered, "no escape hatch: the model cannot obey 'give me everything'")


# ---------------------------------------------------------------- the reminder's position
print("\nWhere the reminder sits")

chunks = [{"text": "Wymaro is a platform.", "source": "kb.pdf", "page_start": 1, "page_end": 1}]
messages = llm._messages("tell me more about wymaro", chunks, None)
user_turn = messages[-1]["content"]

check("the reminder is the LAST thing in the user turn, after the question",
      user_turn.rstrip().endswith(llm.ANSWER_REMINDER.rstrip()), user_turn[-120:])
check("...and comes after the question, not before it",
      user_turn.index("QUESTION:") < user_turn.index(llm.ANSWER_REMINDER[:40]))
check("the system prompt is a system message, not a content turn",
      messages[0]["role"] == "system" and messages[0]["content"] == llm.SYSTEM_PROMPT)


# ---------------------------------------------------------------- the token ceiling
print("\nThe token ceiling is NOT the length control")

# This is the regression that would undo the truncation fix. The completion budget is shared
# with a reasoning model's thinking, so it has to stay well clear of the answer's own size.
check("LLM_MAX_TOKENS is still generous enough that a compliant answer is never cut",
      config.LLM_MAX_TOKENS >= 1500, config.LLM_MAX_TOKENS)
check("...and is far above the word budget (a cut answer is worse than a long one)",
      config.LLM_MAX_TOKENS > config.ANSWER_MAX_WORDS * 4,
      (config.LLM_MAX_TOKENS, config.ANSWER_MAX_WORDS))


# ---------------------------------------------------- LLM_PROVIDER
print("\nLLM_PROVIDER")

check("the default is groq", config.LLM_PROVIDER == "groq", config.LLM_PROVIDER)
check("gemini is accepted",
      reload_config(LLM_PROVIDER="gemini", GEMINI_API_KEY="k").get("LLM_PROVIDER") == "gemini",
      reload_config(LLM_PROVIDER="gemini", GEMINI_API_KEY="k"))
err = reload_config(LLM_PROVIDER="vertex")
check("an unsupported provider is refused at import", isinstance(err, ValueError), err)
check("...and the message names the valid values",
      "groq" in str(err) and "gemini" in str(err), err)


# ------------------------------------------------- .env keys set more than once
print("\nDuplicate keys in .env")

# Why this is in THIS file: the symptom that led here was "I changed the setting and the
# behaviour did not change", and a repeated key is the other way that happens. python-dotenv
# keeps the LAST assignment and says nothing, so a .env that sets TOP_K=4 on line 27 and
# TOP_K=12 on line 180 runs 12 while its author reads the file and believes 4 - and every
# number measured afterwards is filed under the wrong setting. Found in this project's own
# .env, which is why it now warns.
import tempfile  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    env = Path(tmp) / ".env"

    env.write_text(
        "# a comment\n"
        "SEMANTIC_BREAKPOINT_PERCENTILE=95\n"
        "TOP_K=4\n"
        "\n"
        "SEMANTIC_BREAKPOINT_PERCENTILE=80\n"
        "GROQ_MODEL=openai/gpt-oss-20b   # trailing comment\n"
        "GROQ_MODEL=openai/gpt-oss-20b\n"
        "GROQ_API_KEY=gsk_first_secret_value\n"
        "GROQ_API_KEY=gsk_second_secret_value\n",
        encoding="utf-8",
    )
    found = config._find_duplicate_env_keys(env)
    joined = " | ".join(found)

    check("a key set twice with different values is reported",
          any("SEMANTIC_BREAKPOINT_PERCENTILE" in w for w in found), found)
    check("...and the message says which value actually wins",
          "'80' is what is running" in joined, joined)
    check("...and names the loser too, so it can be found and deleted",
          "'95'" in joined, joined)
    check("a key set twice with the SAME value is reported more quietly",
          any("GROQ_MODEL" in w and "same value" in w for w in found), found)
    check("a trailing comment is not mistaken for part of the value",
          "gpt-oss-20b   #" not in joined, joined)
    check("a key set once is not reported", not any("TOP_K" in w for w in found), found)
    check("a comment line is not parsed as a key", not any("#" in w[:20] for w in found))

    # A duplicated key still has to be reported - it is the same silent-override bug - but
    # the report must not carry the value, because these lines go to a log file.
    check("a duplicated SECRET is still reported",
          any("GROQ_API_KEY" in w for w in found), found)
    check("...but its value is never echoed into the warning",
          "gsk_first_secret_value" not in joined and "gsk_second_secret_value" not in joined,
          joined)

    env.write_text("TOP_K=4\nSEMANTIC_BREAKPOINT_PERCENTILE=95\n", encoding="utf-8")
    check("a clean .env produces no warnings", config._find_duplicate_env_keys(env) == [],
          config._find_duplicate_env_keys(env))

check("a missing .env is not an error", config._find_duplicate_env_keys(
    Path("/nonexistent-directory-for-this-test/.env")) == [])
check("config exposes the warnings for main.py to log (it cannot log itself)",
      isinstance(config.CONFIG_WARNINGS, list))


print("\n" + "=" * 60)
print(f"{PASSED} passed, {FAILED} failed")
print("=" * 60)
sys.exit(1 if FAILED else 0)
