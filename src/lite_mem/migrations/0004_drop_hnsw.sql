-- S8: every vector search ranks exactly among the scope's rows (since S3: an HNSW scan filtered
-- after the join can miss in-scope memories), so the HNSW indexes were only a cost on each write.
-- The per-dimension tables are made at bind time, so their names are found, not listed.
DO $$
DECLARE
    found record;
BEGIN
    FOR found IN
        SELECT indexname FROM pg_indexes
        WHERE schemaname = current_schema() AND indexname LIKE 'memory\_vectors\_%\_hnsw'
    LOOP
        EXECUTE format('DROP INDEX %I', found.indexname);
    END LOOP;
END $$;
