-- Memories and their history (design §3, §4). A memory row is one version of one fact: a change
-- of fact (SUPERSEDE: a new value, a merge) closes the old row (valid_to, superseded_by) and opens
-- a new one; a correction (CORRECT: same fact, better wording) edits the row in place. Nothing governed is deleted
-- except by erase(user_id).
CREATE TABLE memories (
    id uuid PRIMARY KEY,
    namespace text NOT NULL REFERENCES namespaces (name) ON DELETE CASCADE,
    user_id text,
    agent_id text,
    session_id text,
    type text NOT NULL CHECK (type IN ('semantic', 'episodic', 'procedural', 'profile')),
    key text,
    content text NOT NULL CHECK (content <> ''),
    importance real NOT NULL CHECK (importance BETWEEN 0 AND 1),
    confidence real NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    valid_from timestamptz NOT NULL,
    valid_to timestamptz,
    -- Checked at commit: a supersede closes the old row (freeing the key's current slot) before
    -- the new row it points at exists.
    superseded_by uuid REFERENCES memories (id) DEFERRABLE INITIALLY DEFERRED,
    expires_at timestamptz,
    source text,
    metadata jsonb NOT NULL DEFAULT '{}',
    access_count integer NOT NULL DEFAULT 0,
    last_accessed_at timestamptz,
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    CHECK ((type = 'profile') = (key IS NOT NULL)),
    CHECK (valid_to IS NULL OR valid_to >= valid_from)
);

-- A scope's current memories (the common read).
CREATE INDEX memories_current ON memories (namespace, user_id, agent_id) WHERE valid_to IS NULL;
-- One current value per profile key and scope; a new value must close the old one first.
CREATE UNIQUE INDEX memories_profile_current
    ON memories (namespace, coalesce(user_id, ''), coalesce(agent_id, ''), key)
    WHERE key IS NOT NULL AND valid_to IS NULL;
CREATE INDEX memories_user ON memories (namespace, user_id);
-- erase clears links to the rows it deletes; the self-referencing FK checks them too.
CREATE INDEX memories_superseded_by ON memories (superseded_by) WHERE superseded_by IS NOT NULL;

-- Append-only history: every change to every memory, with why and who (design §3).
CREATE TABLE memory_events (
    id bigserial PRIMARY KEY,
    namespace text NOT NULL,
    memory_id uuid NOT NULL REFERENCES memories (id) ON DELETE CASCADE,
    user_id text,
    op text NOT NULL CHECK (op IN ('ADD', 'CORRECT', 'DELETE', 'SUPERSEDE', 'MERGE', 'EXPIRE')),
    before jsonb,
    after jsonb,
    reason text,
    produced_by text,
    at timestamptz NOT NULL
);
CREATE INDEX memory_events_memory ON memory_events (memory_id, id);

-- A write sent twice with the same key runs once; the second gets the first's result (design §4).
CREATE TABLE idempotency (
    namespace text NOT NULL REFERENCES namespaces (name) ON DELETE CASCADE,
    key text NOT NULL,
    user_id text,
    result jsonb NOT NULL,
    created_at timestamptz NOT NULL,
    PRIMARY KEY (namespace, key)
);
CREATE INDEX idempotency_user ON idempotency (namespace, user_id);
