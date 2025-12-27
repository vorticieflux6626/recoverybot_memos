-- Initialize PostgreSQL extensions for memOS
-- This script runs automatically when the PostgreSQL container starts

-- Enable pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable cryptographic functions for HIPAA compliance
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Enable btree_gin for advanced indexing
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- Enable pg_stat_statements for performance monitoring
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Create custom functions for memory operations
CREATE OR REPLACE FUNCTION calculate_memory_similarity(
    vector1 vector(1024),
    vector2 vector(1024),
    recovery_stage text DEFAULT NULL,
    therapeutic_weight float DEFAULT 1.0
) RETURNS float AS $$
BEGIN
    -- Calculate cosine similarity with therapeutic weighting
    RETURN (vector1 <=> vector2) * therapeutic_weight * 
           CASE 
               WHEN recovery_stage = 'crisis' THEN 1.2
               WHEN recovery_stage = 'early_recovery' THEN 1.1
               WHEN recovery_stage = 'maintenance' THEN 0.9
               ELSE 1.0
           END;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Create memory decay function for therapeutic relevance
CREATE OR REPLACE FUNCTION calculate_memory_decay(
    created_at timestamp,
    memory_importance float DEFAULT 0.5,
    is_milestone boolean DEFAULT false
) RETURNS float AS $$
DECLARE
    days_old integer;
    base_retention float;
    importance_factor float;
    milestone_factor float;
BEGIN
    -- Calculate days since creation
    days_old := EXTRACT(days FROM (NOW() - created_at));
    
    -- Base forgetting curve (modified Ebbinghaus curve)
    base_retention := exp(-days_old / 7.0);
    
    -- Importance multiplier
    importance_factor := 1 + memory_importance * 2;
    
    -- Milestone protection
    milestone_factor := CASE WHEN is_milestone THEN 10 ELSE 1 END;
    
    -- Recovery-specific adjustments
    IF days_old < 30 THEN
        -- Recent memories boost
        RETURN LEAST(base_retention * importance_factor * milestone_factor * 1.5, 1.0);
    ELSIF days_old > 365 THEN
        -- Long-term memories
        RETURN LEAST(base_retention * importance_factor * milestone_factor * 0.5, 1.0);
    ELSE
        RETURN LEAST(base_retention * importance_factor * milestone_factor, 1.0);
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Create audit trigger function for HIPAA compliance
CREATE OR REPLACE FUNCTION audit_memory_changes()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        INSERT INTO audit_logs (
            table_name, operation, user_id, old_values, 
            changed_at, ip_address, user_agent
        ) VALUES (
            TG_TABLE_NAME, TG_OP, OLD.user_id, row_to_json(OLD),
            NOW(), inet_client_addr(), current_setting('application_name', true)
        );
        RETURN OLD;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_logs (
            table_name, operation, user_id, old_values, new_values,
            changed_at, ip_address, user_agent
        ) VALUES (
            TG_TABLE_NAME, TG_OP, NEW.user_id, row_to_json(OLD), row_to_json(NEW),
            NOW(), inet_client_addr(), current_setting('application_name', true)
        );
        RETURN NEW;
    ELSIF TG_OP = 'INSERT' THEN
        INSERT INTO audit_logs (
            table_name, operation, user_id, new_values,
            changed_at, ip_address, user_agent
        ) VALUES (
            TG_TABLE_NAME, TG_OP, NEW.user_id, row_to_json(NEW),
            NOW(), inet_client_addr(), current_setting('application_name', true)
        );
        RETURN NEW;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Create therapeutic memory search function
CREATE OR REPLACE FUNCTION search_therapeutic_memories(
    p_user_id text,
    p_query_vector vector(1024),
    p_recovery_stage text DEFAULT NULL,
    p_memory_types text[] DEFAULT ARRAY['conversational', 'recovery'],
    p_limit integer DEFAULT 10,
    p_min_similarity float DEFAULT 0.7
) RETURNS TABLE (
    memory_id uuid,
    similarity_score float,
    therapeutic_relevance float,
    decay_factor float,
    final_score float
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        m.id,
        calculate_memory_similarity(m.embedding_vector, p_query_vector, p_recovery_stage) as similarity,
        m.therapeutic_relevance,
        calculate_memory_decay(m.created_at, m.therapeutic_relevance, m.crisis_level > 0.8) as decay,
        (calculate_memory_similarity(m.embedding_vector, p_query_vector, p_recovery_stage) * 
         m.therapeutic_relevance * 
         calculate_memory_decay(m.created_at, m.therapeutic_relevance, m.crisis_level > 0.8)) as final
    FROM memories m
    WHERE 
        m.user_id = p_user_id 
        AND m.is_deleted = false
        AND (p_memory_types IS NULL OR m.memory_type = ANY(p_memory_types))
        AND m.embedding_vector IS NOT NULL
        AND (m.embedding_vector <=> p_query_vector) < (1 - p_min_similarity)
    ORDER BY final DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Set up row-level security for multi-tenant isolation
ALTER TABLE IF EXISTS memories ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS user_memory_settings ENABLE ROW LEVEL SECURITY;

-- Grant necessary permissions
GRANT USAGE ON SCHEMA public TO memos_user;
GRANT CREATE ON SCHEMA public TO memos_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO memos_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO memos_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO memos_user;

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'memOS PostgreSQL initialization completed successfully';
    RAISE NOTICE 'Extensions enabled: vector, uuid-ossp, pgcrypto, btree_gin, pg_stat_statements';
    RAISE NOTICE 'Custom functions created for therapeutic memory operations';
END $$;