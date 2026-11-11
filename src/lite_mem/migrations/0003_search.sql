-- Hybrid search (design §5): BM25 postings per memory (jieba terms) and its length in terms. The
-- vectors live in one table per dimension (memory_vectors_<dim>), created when a namespace is
-- bound, since pgvector columns have a fixed dimension.
ALTER TABLE memories ADD COLUMN terms integer NOT NULL DEFAULT 0;

CREATE TABLE memory_postings (
    namespace text NOT NULL,
    memory_id uuid NOT NULL REFERENCES memories (id) ON DELETE CASCADE,
    term text NOT NULL,
    tf integer NOT NULL CHECK (tf > 0),
    PRIMARY KEY (memory_id, term)
);
CREATE INDEX memory_postings_term ON memory_postings (namespace, term);
