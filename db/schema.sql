CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Speaker identity (future: pyannote voice ID)
CREATE TABLE IF NOT EXISTS speakers (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    voice_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Seed reserved speakers at FIXED ids matching speaker.identifier._RESERVED_IDS
-- (dan=1, timmy=2). Explicit ids, not SERIAL, so the two sources of truth can't
-- diverge -- db/speakers.py reconciles facts.speaker_id FKs against those ids.
INSERT INTO speakers (id, name) VALUES (1, 'dan'), (2, 'timmy')
ON CONFLICT (id) DO NOTHING;

-- Memory types
DO $$ BEGIN
    CREATE TYPE memory_type AS ENUM (
        'episodic',
        'semantic',
        'procedural',
        'conversation_summary'
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Core memory table
CREATE TABLE IF NOT EXISTS memories (
    id SERIAL PRIMARY KEY,
    type memory_type NOT NULL,
    content TEXT NOT NULL,
    speaker_id INTEGER REFERENCES speakers(id),
    embedding vector(768),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    accessed_at TIMESTAMPTZ DEFAULT NOW(),
    access_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'::jsonb,
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

-- Structured facts (entity-attribute-value with provenance)
CREATE TABLE IF NOT EXISTS facts (
    id SERIAL PRIMARY KEY,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    value TEXT NOT NULL,
    source_memory_id INTEGER REFERENCES memories(id) ON DELETE SET NULL,
    speaker_id INTEGER REFERENCES speakers(id),
    learned_at TIMESTAMPTZ DEFAULT NOW(),
    confidence REAL DEFAULT 1.0,
    superseded_by INTEGER REFERENCES facts(id)
);

-- Provenance: which writer set the current value. 'tool' = explicit store_fact
-- route (a user-directed correction); 'extraction' = async background extractor.
-- Used by store_fact to stop the extractor from clobbering a newer explicit
-- correction (recency-gated precedence). See lt-store-fact-correction-clobbered
-- -by-extractor-race-2026-06-21. Existing rows backfill to 'extraction' (none of
-- them were tool-tagged, so none gain protection -- status quo preserved).
ALTER TABLE facts ADD COLUMN IF NOT EXISTS source TEXT DEFAULT 'extraction';

-- Add unique constraint for latest-wins upsert
-- (subject, predicate) where not yet superseded
CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_active
    ON facts (subject, predicate) WHERE superseded_by IS NULL;

-- Episodic memory (Session 0, 2026-06-20; docs/episodic-memory-plan.md).
-- Time-indexed rollup summaries for date-range recall ("what did we talk about
-- last Saturday"). DISTINCT from the `memories` vector tier: episodes carry a
-- real event-time SPAN (span_start..span_end) derived from turn timestamps and
-- are queried by RANGE OVERLAP, not similarity -- which sidesteps the
-- recency-blind ranker entirely. `embedding` is nullable and stays NULL until
-- Session 5 (vector restore, gated on dedup-at-write + recency decay); its
-- HNSW/FTS/trigram indexes are deliberately deferred to that session.
CREATE TABLE IF NOT EXISTS episodes (
    id SERIAL PRIMARY KEY,
    span_start TIMESTAMPTZ NOT NULL,
    span_end TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    text TEXT NOT NULL,
    token_count INTEGER,
    embedding vector(768),              -- nullable; filled in Session 5
    source JSONB DEFAULT '{}'::jsonb     -- provenance: turn count, trigger, etc.
);

-- Date-range overlap queries: episode [span_start, span_end] vs query window.
CREATE INDEX IF NOT EXISTS idx_episodes_span_start ON episodes (span_start);
CREATE INDEX IF NOT EXISTS idx_episodes_span_end ON episodes (span_end);

-- Session 5 (2026-06-20): vector restore on episodes (scale phase; flags default
-- OFF, see config.py). Idempotent ALTERs so this applies to the S0-created table
-- without a migration step (db/migrate.run re-executes this file on startup).
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS access_count INTEGER DEFAULT 0;
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS accessed_at TIMESTAMPTZ;
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS content_hash TEXT;
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;

-- Dedup-at-write FLOOR: exact content-hash. UNIQUE so a re-summarized verbatim
-- episode can't double-write (guards the rollup double-encode re-rot). Partial
-- so legacy NULL-hash rows (none today) don't collide.
CREATE UNIQUE INDEX IF NOT EXISTS idx_episodes_content_hash
    ON episodes (content_hash) WHERE content_hash IS NOT NULL;

-- Vector similarity (partial: only embedded rows; pgvector HNSW works from row 1).
CREATE INDEX IF NOT EXISTS idx_episodes_embedding
    ON episodes USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64) WHERE embedding IS NOT NULL;

-- FTS + trigram channels for recall_semantic.
CREATE INDEX IF NOT EXISTS idx_episodes_tsv ON episodes USING GIN (content_tsv);
CREATE INDEX IF NOT EXISTS idx_episodes_text_trgm ON episodes USING GIN (text gin_trgm_ops);

-- Vector similarity search (use HNSW for small datasets, works from row 1)
CREATE INDEX IF NOT EXISTS idx_memories_embedding
    ON memories USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- Full-text search
CREATE INDEX IF NOT EXISTS idx_memories_tsv ON memories USING GIN (content_tsv);

-- Trigram similarity
CREATE INDEX IF NOT EXISTS idx_memories_content_trgm ON memories USING GIN (content gin_trgm_ops);

-- Lookup indexes
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories (type);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts (subject);
CREATE INDEX IF NOT EXISTS idx_facts_subject_trgm ON facts USING GIN (subject gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_facts_predicate ON facts (predicate);
