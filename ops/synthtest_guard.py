#!/usr/bin/env python3
"""Wipe-guard for synthesized-speech testing (2026-06-23, Dan's invariant).

Any synthesized-speech test (ops/acoustic_convo_driver.py et al.) lets Timmy HEAR
synthetic utterances on purpose, so they run the full production path and write
REAL rows: structured `facts` (the sole durable auto-writer, since
PERSIST_EXTRACTED_MEMORIES/COLD_SUMMARIES are off), possibly `episodes` on rollup
eviction, and a `speakers` row if an unknown guest voice auto-enrolls. The driver
does NOT clean up after itself.

Dan's invariant: ANY memory made from synthesized input MUST be wiped afterward.

Mechanism: every durable table has an integer `id` PK. Snapshot max(id) + count
BEFORE the test; afterward DELETE WHERE id > that max(id) and verify the table is
byte-identical to baseline (count, max id, and a content hash over the surviving
id<=baseline rows so we PROVE no pre-existing real memory was mutated/deleted).

Driver-free: shells out to `psql` (LT itself uses asyncpg; no psycopg here).

USAGE
  python ops/synthtest_guard.py snapshot  /tmp/lt_synthtest_baseline.json
  #   ... run the synthesized-speech test ...
  python ops/synthtest_guard.py cleanup   /tmp/lt_synthtest_baseline.json [--dry-run]
  python ops/synthtest_guard.py verify     /tmp/lt_synthtest_baseline.json   # check only

Exit 0 = clean / verified. Exit 2 = drift (a pre-baseline row changed) — investigate.
"""
import argparse, json, subprocess, sys

DSN = "postgresql://gearscodeandfire@localhost/little_timmy"
TABLES = ["facts", "episodes", "memories", "speakers"]

# Content columns only — EXCLUDES volatile read-side stats (access_count,
# accessed_at) and heavy/derived columns (embedding, content_tsv). This lets a
# read-only test that legitimately retrieves real memories bump access stats
# WITHOUT tripping the integrity hash, while still catching any change to actual
# memory content (value/subject/text/superseded_by/...).
STABLE_COLS = {
    "facts": "id,subject,predicate,value,source_memory_id,speaker_id,learned_at,"
             "confidence,superseded_by,sensitive,pii_category,source",
    "episodes": "id,span_start,span_end,created_at,text,token_count,content_hash,source",
    "memories": "id,type,content,speaker_id,created_at,metadata",
    "speakers": "id,name,voice_id,created_at",
}


def q(sql):
    """Run one SQL, return rows as list of tab-split field lists."""
    out = subprocess.run(["psql", DSN, "-At", "-F", "\t", "-c", sql],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip())
    return [line.split("\t") for line in out.stdout.splitlines() if line != ""]


def table_state(table, max_id):
    """(count_all, max_id_all, hash_of_rows_with_id<=max_id). The hash is the
    integrity anchor: if any pre-existing row's content changes, the hash moves."""
    count_all, max_all = q(f"SELECT count(*), coalesce(max(id),0) FROM {table};")[0]
    # content fingerprint of baseline rows only (id <= snapshot max), over
    # STABLE_COLS only so read-side access-stat bumps don't read as drift.
    cols = STABLE_COLS[table]
    h = q(f"SELECT coalesce(md5(string_agg(ROW({cols})::text, '|' ORDER BY id)),'EMPTY') "
          f"FROM {table} WHERE id <= {int(max_id)};")[0][0]
    return int(count_all), int(max_all), h


def cmd_snapshot(path):
    snap = {}
    for t in TABLES:
        count_all, max_all = q(f"SELECT count(*), coalesce(max(id),0) FROM {t};")[0]
        _, _, h = table_state(t, int(max_all))
        snap[t] = {"count": int(count_all), "max_id": int(max_all), "hash": h}
    json.dump(snap, open(path, "w"), indent=2)
    print(f"[snapshot] -> {path}")
    for t in TABLES:
        print(f"  {t:10s} count={snap[t]['count']:5d}  max_id={snap[t]['max_id']:5d}")
    return 0


def cmd_cleanup(path, dry_run):
    snap = json.load(open(path))
    for t in TABLES:
        base_max = int(snap[t]["max_id"])
        n_new = int(q(f"SELECT count(*) FROM {t} WHERE id > {base_max};")[0][0])
        if not n_new:
            print(f"  {t:10s} nothing new")
            continue
        if dry_run:
            cols = "id, subject, predicate, value" if t == "facts" else "id"
            rows = q(f"SELECT {cols} FROM {t} WHERE id > {base_max} ORDER BY id;")
            print(f"  {t:10s} WOULD delete {n_new} synthetic row(s): {rows}")
        else:
            q(f"DELETE FROM {t} WHERE id > {base_max};")
            print(f"  {t:10s} deleted {n_new} synthetic row(s)")
    print("[cleanup] done" if not dry_run else "[cleanup] dry-run, no deletes")
    return cmd_verify(path)


def cmd_verify(path):
    snap = json.load(open(path))
    print("[verify]")
    ok = True
    for t in TABLES:
        count, mx, h = table_state(t, int(snap[t]["max_id"]))
        drift = []
        if count != snap[t]["count"]:
            drift.append(f"count {snap[t]['count']}->{count}")
        if mx != snap[t]["max_id"]:
            drift.append(f"max_id {snap[t]['max_id']}->{mx} (synthetic rows remain!)")
        if h != snap[t]["hash"]:
            drift.append("BASELINE ROW MUTATED (hash changed)")
        if drift:
            ok = False
            print(f"  X {t:10s} {'; '.join(drift)}")
        else:
            print(f"  OK {t:10s} clean (count={count}, max_id={mx})")
    print("[verify] CLEAN — synthesized input fully wiped" if ok
          else "[verify] DRIFT — manual review required")
    return 0 if ok else 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["snapshot", "cleanup", "verify"])
    ap.add_argument("path")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.cmd == "snapshot":
        sys.exit(cmd_snapshot(a.path))
    elif a.cmd == "cleanup":
        sys.exit(cmd_cleanup(a.path, a.dry_run))
    else:
        sys.exit(cmd_verify(a.path))
