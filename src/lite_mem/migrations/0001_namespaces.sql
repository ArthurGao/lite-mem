-- A namespace is bound to what built its indexes (design §4): the embedding model and dimension
-- (vectors) and the tokenizer (BM25 postings). Opening it with any other is refused, never mixed.
CREATE TABLE namespaces (
    name text PRIMARY KEY,
    embedding_provider text NOT NULL,
    embedding_model text NOT NULL,
    dim integer NOT NULL CHECK (dim > 0),
    tokenizer text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
