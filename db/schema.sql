CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Speaker identity (future: pyannote voice ID)
CREATE TABLE IF NOT EXISTS speakers (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    voice_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

INSERT INTO speakers (name) VALUES ('dan') ON CONFLICT DO NOTHING;
INSERT INTO speakers (name) VALUES ('timmy') ON CONFLICT DO NOTHING;

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

-- Add unique constraint for latest-wins upsert
-- (subject, predicate) where not yet superseded
CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_active
    ON facts (subject, predicate) WHERE superseded_by IS NULL;

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
