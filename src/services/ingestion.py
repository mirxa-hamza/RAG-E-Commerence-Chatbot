"""
The only way documents get into this system.

Documents enter by being dropped into DATA_DIR by whoever runs the backend, or uploaded
through POST /upload from the web UI - which writes the file into that same folder.
Ingestion reads the folder directly; nothing is ever indexed straight out of a request
body.

Ingestion is fingerprinted: a file is (re-)ingested when it's new, or when its bytes have
changed since last time. Matching on filename alone meant an edited PDF was invisible
forever.

The work runs on a background thread (`start_job` / `job_status` below), so the API never
blocks on embedding a 900-page textbook. That is correct only because this app is a single
long-lived process - see the "single worker" note in core/ratelimit.py.
"""
import hashlib
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

from src.core.config import DATA_DIR
from src.core.logging import get_logger, timed
from src.ml.embeddings import split_to_token_limit
from src.services import answer_cache, manifest
from src.services.chunking import chunk_pages
from src.services.pdf import extract_pages
from src.services.vectorstore import add_chunks, delete_source

log = get_logger(__name__)

_HASH_CHUNK = 1024 * 1024  # 1MB reads while fingerprinting

# Cap on the per-job result/event lists, held in memory and returned on every status poll.
MAX_JOB_RESULTS = 500
MAX_JOB_EVENTS = 60

# Uploads live in data/users/<user_id>/<filename>, so a document's owner is recoverable
# from its path alone.
USERS_DIRNAME = "users"


def user_dir(user_id: str):
    """The folder a user's uploads live in."""
    return DATA_DIR / USERS_DIRNAME / str(user_id)


def owner_of(filename: str) -> Optional[str]:
    """
    Who a document belongs to: the user named by its path, or - for a file copied into
    data/ by hand or indexed by the CLI - the owner of record (the first account created).

    Ownerless documents are invisible to every filter, so a PDF dropped into data/ after
    the first signup would otherwise be indexed and then seen by nobody.
    """
    from src.services import ownership

    return owner_from_path(filename) or ownership.owner_of_record()


def owner_from_path(filename: str) -> Optional[str]:
    """
    'users/652.../book.pdf' -> '652...'; anything else -> None (an ownerless document).

    Ownership is derived from the path rather than trusted from the manifest, so a
    hand-edited or half-written manifest cannot hand one user another user's document.
    """
    parts = filename.split("/")
    if len(parts) >= 3 and parts[0] == USERS_DIRNAME and parts[1]:
        return parts[1]
    return None


# --------------------------------------------------------------------- helpers

def _pdf_filenames(user_id: Optional[str] = None) -> List[str]:
    """
    Every PDF under DATA_DIR, as paths relative to it, sorted.

    With a user_id, only that user's documents are listed - an upload triggers a scan, and
    walking every other account's documents to find one new file is wasted work that grows
    with the number of users.
    """
    if not DATA_DIR.is_dir():
        log.warning("Data folder %s does not exist - nothing to ingest.", DATA_DIR)
        return []

    if not user_id:
        roots = [DATA_DIR]
    else:
        # The user's own uploads...
        roots = [user_dir(user_id)]
        # ...plus, for the owner of record, the hand-copied files that live outside
        # users/. They belong to that account, so a scoped pass that skipped them would
        # "prune" every one of them as deleted.
        from src.services import ownership

        if ownership.owner_of_record() == user_id:
            roots.append(DATA_DIR)

    names = set()
    users_root = (DATA_DIR / USERS_DIRNAME).resolve()
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() != ".pdf":
                continue
            # When scanning DATA_DIR for the owner of record, skip other people's folders.
            if user_id and root == DATA_DIR and users_root in path.resolve().parents:
                if owner_from_path(path.relative_to(DATA_DIR).as_posix()) != user_id:
                    continue
            names.add(path.relative_to(DATA_DIR).as_posix())
    return sorted(names)


def _fingerprint(path) -> Dict:
    """sha256 + mtime + size of a file. The hash is what actually decides re-ingest."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_HASH_CHUNK), b""):
            digest.update(block)
    stat = os.stat(path)
    return {"sha256": digest.hexdigest(), "mtime": stat.st_mtime, "size": stat.st_size}


def _needs_ingest(filename: str, fingerprint: Dict) -> Optional[str]:
    """Returns None if already up to date, else the reason ('new' or 'changed')."""
    record = manifest.get(filename)
    if record is None:
        return "new"
    if record.get("sha256") != fingerprint["sha256"]:
        return "changed"
    return None


# --------------------------------------------------------------------- ingestion

def ingest_one(filename: str, *, reason: str = "new", fingerprint: Optional[Dict] = None,
               on_stage=None) -> Dict:
    """
    Extracts, chunks, embeds, and stores a single PDF.

    on_stage: optional callable(stage, done, total) where stage is "extracting" or
    "embedding". The UI turns this into a per-document progress bar.
    """
    fingerprint = fingerprint or _fingerprint(DATA_DIR / filename)
    owner = owner_of(filename)

    if on_stage:
        on_stage("extracting", 0, 0)
    _event("Reading pages", filename, owner)

    with timed(log, f"extract '{filename}'"):
        pages = extract_pages(str(DATA_DIR / filename))

    if not pages:
        _event("No extractable text - skipped (is it a scan?)", filename, owner, kind="warn")
        return {
            "filename": filename,
            "user_id": owner,
            "status": "skipped",
            "reason": "no extractable text - likely a scanned/image PDF (OCR is off)",
        }

    chunks = chunk_pages(pages)
    # The chunker counts words; the model counts tokens. Anything still over the window is
    # split here rather than being silently truncated at embedding time.
    chunks = split_to_token_limit(chunks)
    total_chunks = len(chunks)

    _event(f"Split {len(pages)} pages into {total_chunks} passages", filename, owner)
    log.info(
        "'%s': %d pages -> %d chunks (semantic chunking). Embedding - a large document "
        "takes several minutes, this is not a hang.",
        filename, len(pages), total_chunks,
    )

    # Replace, don't duplicate: drop any existing vectors for this document first.
    #
    # This runs for reason == "new" as well, which looks redundant but is not. A job killed
    # part-way (Ctrl+C, or `uvicorn --reload` restarting) leaves whatever chunks it had
    # already stored, while the manifest entry - written only once the whole file finishes -
    # never gets written. The next run therefore treats the file as "new" and re-embeds it
    # from scratch, so without this delete the abandoned chunks would pile up as duplicates
    # on every interrupted attempt. Unscoped by owner on purpose: it must remove any chunk
    # of this document, including ones stored before it had an owner.
    delete_source(filename)

    if on_stage:
        on_stage("embedding", 0, total_chunks)
    _event(f"Embedding and storing {total_chunks} passages", filename, owner)

    with timed(log, f"embed + store '{filename}'"):
        add_chunks(
            filename, chunks,
            on_progress=(lambda done, total: on_stage("embedding", done, total_chunks))
            if on_stage else None,
            user_id=owner,
        )

    # Answers built from the previous version of this document are now wrong.
    answer_cache.bump(owner)
    _event(f"Ready - {total_chunks} passages searchable", filename, owner, kind="done")

    manifest.put(
        filename,
        sha256=fingerprint["sha256"],
        mtime=fingerprint["mtime"],
        size=fingerprint["size"],
        pages=len(pages),
        chunks=total_chunks,
        user_id=owner,
    )

    return {
        "filename": filename,
        "user_id": owner,
        "status": "ingested",
        "reason": reason,
        "pages": len(pages),
        "chunks_stored": total_chunks,
    }


def prune_deleted(present: List[str], user_id: Optional[str] = None) -> List[Dict]:
    """
    Drops documents that are in the store but no longer on disk.

    Without this, deleting a PDF leaves its chunks searchable forever: the answer still
    quotes it and still cites it by name, with no way to tell from the app that it's gone.
    """
    removed = []
    for filename in manifest.sources(user_id):
        if filename in present:
            continue
        owner = owner_of(filename)
        log.info("'%s' is no longer present - removing it from the store.", filename)
        try:
            delete_source(filename)   # unscoped: see the note in ingest_one()
            manifest.remove(filename)
            answer_cache.bump(owner)
            _event("Removed - the file is no longer present", filename, owner)
            removed.append({"filename": filename, "user_id": owner, "status": "removed"})
        except Exception as exc:
            log.exception("Failed to remove '%s'", filename)
            removed.append({"filename": filename, "user_id": owner,
                            "status": "remove_failed", "error": str(exc)})
    return removed


def _plan_ingest(filenames: List[str], force: bool) -> List[Dict]:
    """
    For each filename, in order, decide skip/duplicate/already-stored/needs-ingest.

    Returns ONE ordered list, not two separate (decided, pending) lists - a pending entry
    carries {"pending": True, "filename", "reason", "fingerprint"}, everything else is a
    terminal result dict. Keeping it in scan order matters: callers append results as they
    go, and reordering would make a job's "results" list disagree with the folder listing
    (e.g. "book (2).pdf" before "book.pdf" - space sorts below period in ASCII).
    """
    plan: List[Dict] = []
    seen_hashes: Dict[tuple, str] = {}

    for filename in filenames:
        try:
            fingerprint = _fingerprint(DATA_DIR / filename)
        except (OSError, FileNotFoundError) as exc:
            log.warning("Could not read '%s': %s", filename, exc)
            plan.append({"filename": filename, "user_id": owner_of(filename),
                        "status": "failed", "error": str(exc)})
            continue

        # Same bytes under two names would be indexed twice and compete with itself for
        # every retrieval slot. Scoped PER OWNER: globally, two users uploading the same
        # textbook would leave the second one with a "skipped" document they can never see.
        owner = owner_of(filename)
        dedupe_key = (owner, fingerprint["sha256"])
        twin = seen_hashes.get(dedupe_key)
        if twin:
            log.warning("'%s' is byte-identical to '%s' - skipping the duplicate.", filename, twin)
            _event(f"Skipped - identical to '{twin}'", filename, owner, kind="warn")
            plan.append({"filename": filename, "user_id": owner, "status": "skipped",
                        "reason": f"duplicate of '{twin}'"})
            continue
        seen_hashes[dedupe_key] = filename

        reason = "forced" if force else _needs_ingest(filename, fingerprint)
        if reason is None:
            plan.append({"filename": filename, "user_id": owner, "status": "already_stored"})
            continue

        plan.append({
            "pending": True,
            "filename": filename,
            "reason": "changed" if reason in ("changed", "forced") else "new",
            "fingerprint": fingerprint,
        })

    return plan


def ingest_data_folder(force: bool = False, progress=None, stage=None,
                       user_id: Optional[str] = None) -> List[Dict]:
    """
    Ingests every PDF that is new or has changed, and removes stored documents whose file
    is gone. Safe to call repeatedly - unchanged files are reported, not re-embedded.
    force=True re-ingests everything (used after a reset).

    A failure on one document never stops the others: a single malformed PDF used to abort
    the whole job, leaving every file after it silently un-indexed.
    """
    filenames = _pdf_filenames(user_id)
    results: List[Dict] = list(prune_deleted(filenames, user_id=user_id))
    plan = _plan_ingest(filenames, force)

    pending_total = sum(1 for item in plan if item.get("pending"))
    pending_index = 0
    for item in plan:
        if not item.get("pending"):
            results.append(item)
            continue

        filename = item["filename"]
        if progress:
            progress(filename, pending_index, pending_total)
        pending_index += 1

        log.info("Ingesting '%s' (%s)...", filename, item["reason"])
        try:
            result = ingest_one(
                filename, reason=item["reason"], fingerprint=item["fingerprint"], on_stage=stage,
            )
        except Exception as exc:
            # Corrupt file, encrypted PDF, unreadable bytes, OOM on one monster document -
            # report it and carry on with the rest of the corpus.
            log.exception("Failed to ingest '%s'", filename)
            _event(f"Failed: {type(exc).__name__}", filename, owner_of(filename), kind="error")
            result = {"filename": filename, "user_id": owner_of(filename),
                      "status": "failed", "error": f"{type(exc).__name__}: {exc}"}

        log.info("%s", result)
        results.append(result)

    return results


# --------------------------------------------------------------------- background job

_lock = threading.Lock()
_state: Dict = {
    "state": "idle",          # idle | running | error
    "started_at": None,
    "finished_at": None,
    # Whose scan this is: a user id for an upload-triggered pass, None for a full one
    # (startup, the CLI, "sync"). This is what tells job_status() that a run belongs to the
    # caller even at moments when no file is being processed.
    "scope": None,
    "current_file": None,
    "files_done": 0,
    "files_total": 0,
    # Within the current document: "extracting" or "embedding", plus chunk counts, so the
    # UI can show real progress for one big book instead of a bar stuck at 0/1.
    "stage": None,
    "chunks_done": 0,
    "chunks_total": 0,
    "results": [],
    # A short, human-readable trail of what the pipeline actually did, newest last. The
    # progress bar answers "how far"; this answers "what is it doing", which is the question
    # people actually ask while watching a 900-page book index.
    "events": [],
    "error": None,
}

_rescan_requested = False
_queued_scopes: List[Optional[str]] = []


def _event(message: str, filename: Optional[str] = None, user_id: Optional[str] = None,
           kind: str = "info") -> None:
    """Appends one line to the activity trail, tagged with whose document it concerns."""
    with _lock:
        _state["events"].append({
            "at": time.time(),
            "kind": kind,              # info | done | warn | error
            "message": message,
            "file": filename,
            "user_id": user_id,
        })
        if len(_state["events"]) > MAX_JOB_EVENTS:
            del _state["events"][:-MAX_JOB_EVENTS]


def _run(force: bool, user_id: Optional[str] = None) -> None:
    def progress(filename: str, index: int, total: int) -> None:
        with _lock:
            _state["current_file"] = filename
            _state["files_done"] = index
            _state["files_total"] = total
            _state["stage"] = None
            _state["chunks_done"] = 0
            _state["chunks_total"] = 0

    def stage(name: str, done: int, total: int) -> None:
        with _lock:
            _state["stage"] = name
            _state["chunks_done"] = done
            _state["chunks_total"] = total

    try:
        results = ingest_data_folder(force=force, progress=progress, stage=stage,
                                     user_id=user_id)

        while True:
            wanted, scopes = _consume_rescan_request()
            if not wanted:
                break
            log.info("Files arrived during ingestion - scanning again (%s).",
                     ", ".join(scope or "all documents" for scope in scopes))
            for scope in scopes:
                results.extend(ingest_data_folder(force=False, progress=progress,
                                                  stage=stage, user_id=scope))
            if len(results) > MAX_JOB_RESULTS:
                dropped = len(results) - MAX_JOB_RESULTS
                results = results[-MAX_JOB_RESULTS:]
                log.info("Job result list trimmed; dropped %d older entries.", dropped)

        with _lock:
            _state.update(
                state="idle", current_file=None, stage=None, chunks_done=0, chunks_total=0,
                files_done=_state["files_total"], results=results, error=None,
                finished_at=time.time(),
            )
    except Exception as exc:  # a failed ingest must not kill the server
        log.exception("Ingestion job failed")
        with _lock:
            _state.update(state="error", current_file=None, error=str(exc),
                          finished_at=time.time())


def _consume_rescan_request() -> Tuple[bool, List[Optional[str]]]:
    """Returns (was one requested, the scopes to scan). Clears both."""
    global _rescan_requested, _queued_scopes
    with _lock:
        wanted, _rescan_requested = _rescan_requested, False
        scopes, _queued_scopes = _queued_scopes, []
    if None in scopes:
        return wanted, [None]
    return wanted, list(dict.fromkeys(scopes)) or [None]


# --------------------------------------------------------------------- public API

def job_status(user_id: Optional[str] = None) -> Dict:
    """The job's state, redacted to what `user_id` may know."""
    with _lock:
        snapshot = dict(_state)

    if user_id is None:
        return snapshot

    results = [r for r in snapshot.get("results", []) if r.get("user_id") == user_id]
    events = [e for e in snapshot.get("events", []) if e.get("user_id") in (user_id, None)]

    # The run belongs to the caller if it was started FOR them, or if the file being
    # processed right now is theirs.
    #
    # Checking only `current_file` was a bug: it is None at the start of a run, between
    # files, and for the whole finished state - so the counts were blanked at exactly the
    # moments the UI needed them, and the progress bar sat at 0% and then jumped to nothing.
    mine = (snapshot.get("scope") == user_id
            or owner_from_path(snapshot.get("current_file") or "") == user_id)

    snapshot["results"] = results
    snapshot["events"] = events
    if not mine:
        snapshot["current_file"] = None
        snapshot["stage"] = None
        snapshot["chunks_done"] = 0
        snapshot["chunks_total"] = 0
        snapshot["files_done"] = 0
        snapshot["files_total"] = 0
        snapshot["other_user_busy"] = snapshot["state"] == "running"
    return snapshot


def is_running(user_id: Optional[str] = None) -> bool:
    with _lock:
        return _state["state"] == "running"


def start_job(force: bool = False, user_id: Optional[str] = None) -> Dict:
    """
    Starts (or queues) an ingestion pass on a background thread.

    With a user_id the scan is scoped to that person's folder; without one it reconciles
    the whole data folder (startup, the CLI). A second start while one is already running
    queues a rescan rather than racing two passes against each other.
    """
    global _rescan_requested
    with _lock:
        if _state["state"] == "running":
            _rescan_requested = True
            _queued_scopes.append(user_id)
            return dict(_state)
        _state.update(
            state="running", scope=user_id, started_at=time.time(), finished_at=None,
            current_file=None, files_done=0, files_total=0, stage=None, chunks_done=0,
            chunks_total=0, results=[], events=[], error=None,
        )

    threading.Thread(target=_run, args=(force, user_id), daemon=True, name="ingest").start()
    return job_status()
