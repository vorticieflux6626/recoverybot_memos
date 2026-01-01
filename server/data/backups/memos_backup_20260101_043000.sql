--
-- PostgreSQL database dump
--

-- Dumped from database version 16.9 (Debian 16.9-1.pgdg120+1)
-- Dumped by pg_dump version 16.9 (Debian 16.9-1.pgdg120+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

ALTER TABLE IF EXISTS ONLY public.user_tasks DROP CONSTRAINT IF EXISTS user_tasks_user_quest_id_fkey;
ALTER TABLE IF EXISTS ONLY public.user_tasks DROP CONSTRAINT IF EXISTS user_tasks_task_id_fkey;
ALTER TABLE IF EXISTS ONLY public.user_quests DROP CONSTRAINT IF EXISTS user_quests_quest_id_fkey;
ALTER TABLE IF EXISTS ONLY public.user_achievements DROP CONSTRAINT IF EXISTS user_achievements_achievement_id_fkey;
ALTER TABLE IF EXISTS ONLY public.quest_tasks DROP CONSTRAINT IF EXISTS quest_tasks_quest_id_fkey;
DROP INDEX IF EXISTS public.ix_user_tasks_user_id;
DROP INDEX IF EXISTS public.ix_user_quests_user_id;
DROP INDEX IF EXISTS public.ix_user_achievements_user_id;
DROP INDEX IF EXISTS public.ix_ollama_model_specs_model_name;
DROP INDEX IF EXISTS public.ix_ollama_model_specs_base_model;
DROP INDEX IF EXISTS public.ix_memories_user_id;
DROP INDEX IF EXISTS public.idx_user_tasks_state;
DROP INDEX IF EXISTS public.idx_user_settings_enabled;
DROP INDEX IF EXISTS public.idx_user_quests_user_state;
DROP INDEX IF EXISTS public.idx_user_quests_completed;
DROP INDEX IF EXISTS public.idx_user_quest_stats_points;
DROP INDEX IF EXISTS public.idx_user_achievements_user;
DROP INDEX IF EXISTS public.idx_quests_category;
DROP INDEX IF EXISTS public.idx_memories_user_created;
DROP INDEX IF EXISTS public.idx_memories_type_privacy;
DROP INDEX IF EXISTS public.idx_memories_therapeutic_relevance;
DROP INDEX IF EXISTS public.idx_memories_recovery_stage;
DROP INDEX IF EXISTS public.idx_memories_crisis_level;
ALTER TABLE IF EXISTS ONLY public.user_tasks DROP CONSTRAINT IF EXISTS user_tasks_pkey;
ALTER TABLE IF EXISTS ONLY public.user_quests DROP CONSTRAINT IF EXISTS user_quests_pkey;
ALTER TABLE IF EXISTS ONLY public.user_quest_stats DROP CONSTRAINT IF EXISTS user_quest_stats_pkey;
ALTER TABLE IF EXISTS ONLY public.user_memory_settings DROP CONSTRAINT IF EXISTS user_memory_settings_pkey;
ALTER TABLE IF EXISTS ONLY public.user_achievements DROP CONSTRAINT IF EXISTS user_achievements_pkey;
ALTER TABLE IF EXISTS ONLY public.quests DROP CONSTRAINT IF EXISTS quests_pkey;
ALTER TABLE IF EXISTS ONLY public.quest_tasks DROP CONSTRAINT IF EXISTS quest_tasks_pkey;
ALTER TABLE IF EXISTS ONLY public.ollama_model_specs DROP CONSTRAINT IF EXISTS ollama_model_specs_pkey;
ALTER TABLE IF EXISTS ONLY public.memories DROP CONSTRAINT IF EXISTS memories_pkey;
ALTER TABLE IF EXISTS ONLY public.mem0migrations DROP CONSTRAINT IF EXISTS mem0migrations_pkey;
ALTER TABLE IF EXISTS ONLY public.mem0 DROP CONSTRAINT IF EXISTS mem0_pkey;
ALTER TABLE IF EXISTS ONLY public.alembic_version DROP CONSTRAINT IF EXISTS alembic_version_pkc;
ALTER TABLE IF EXISTS ONLY public.achievements DROP CONSTRAINT IF EXISTS achievements_pkey;
DROP TABLE IF EXISTS public.user_tasks;
DROP TABLE IF EXISTS public.user_quests;
DROP TABLE IF EXISTS public.user_quest_stats;
DROP TABLE IF EXISTS public.user_memory_settings;
DROP TABLE IF EXISTS public.user_achievements;
DROP TABLE IF EXISTS public.quests;
DROP TABLE IF EXISTS public.quest_tasks;
DROP TABLE IF EXISTS public.ollama_model_specs;
DROP TABLE IF EXISTS public.memories;
DROP TABLE IF EXISTS public.mem0migrations;
DROP TABLE IF EXISTS public.mem0;
DROP TABLE IF EXISTS public.alembic_version;
DROP TABLE IF EXISTS public.achievements;
DROP FUNCTION IF EXISTS public.search_therapeutic_memories(p_user_id text, p_query_vector public.vector, p_recovery_stage text, p_memory_types text[], p_limit integer, p_min_similarity double precision);
DROP FUNCTION IF EXISTS public.calculate_memory_similarity(vector1 public.vector, vector2 public.vector, recovery_stage text, therapeutic_weight double precision);
DROP FUNCTION IF EXISTS public.calculate_memory_decay(created_at timestamp without time zone, memory_importance double precision, is_milestone boolean);
DROP FUNCTION IF EXISTS public.audit_memory_changes();
DROP EXTENSION IF EXISTS vector;
DROP EXTENSION IF EXISTS "uuid-ossp";
DROP EXTENSION IF EXISTS pgcrypto;
DROP EXTENSION IF EXISTS pg_stat_statements;
DROP EXTENSION IF EXISTS btree_gin;
--
-- Name: btree_gin; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS btree_gin WITH SCHEMA public;


--
-- Name: EXTENSION btree_gin; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION btree_gin IS 'support for indexing common datatypes in GIN';


--
-- Name: pg_stat_statements; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS pg_stat_statements WITH SCHEMA public;


--
-- Name: EXTENSION pg_stat_statements; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION pg_stat_statements IS 'track planning and execution statistics of all SQL statements executed';


--
-- Name: pgcrypto; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA public;


--
-- Name: EXTENSION pgcrypto; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION pgcrypto IS 'cryptographic functions';


--
-- Name: uuid-ossp; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA public;


--
-- Name: EXTENSION "uuid-ossp"; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION "uuid-ossp" IS 'generate universally unique identifiers (UUIDs)';


--
-- Name: vector; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;


--
-- Name: EXTENSION vector; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION vector IS 'vector data type and ivfflat and hnsw access methods';


--
-- Name: audit_memory_changes(); Type: FUNCTION; Schema: public; Owner: memos_user
--

CREATE FUNCTION public.audit_memory_changes() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
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
$$;


ALTER FUNCTION public.audit_memory_changes() OWNER TO memos_user;

--
-- Name: calculate_memory_decay(timestamp without time zone, double precision, boolean); Type: FUNCTION; Schema: public; Owner: memos_user
--

CREATE FUNCTION public.calculate_memory_decay(created_at timestamp without time zone, memory_importance double precision DEFAULT 0.5, is_milestone boolean DEFAULT false) RETURNS double precision
    LANGUAGE plpgsql IMMUTABLE
    AS $$
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
$$;


ALTER FUNCTION public.calculate_memory_decay(created_at timestamp without time zone, memory_importance double precision, is_milestone boolean) OWNER TO memos_user;

--
-- Name: calculate_memory_similarity(public.vector, public.vector, text, double precision); Type: FUNCTION; Schema: public; Owner: memos_user
--

CREATE FUNCTION public.calculate_memory_similarity(vector1 public.vector, vector2 public.vector, recovery_stage text DEFAULT NULL::text, therapeutic_weight double precision DEFAULT 1.0) RETURNS double precision
    LANGUAGE plpgsql IMMUTABLE
    AS $$
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
$$;


ALTER FUNCTION public.calculate_memory_similarity(vector1 public.vector, vector2 public.vector, recovery_stage text, therapeutic_weight double precision) OWNER TO memos_user;

--
-- Name: search_therapeutic_memories(text, public.vector, text, text[], integer, double precision); Type: FUNCTION; Schema: public; Owner: memos_user
--

CREATE FUNCTION public.search_therapeutic_memories(p_user_id text, p_query_vector public.vector, p_recovery_stage text DEFAULT NULL::text, p_memory_types text[] DEFAULT ARRAY['conversational'::text, 'recovery'::text], p_limit integer DEFAULT 10, p_min_similarity double precision DEFAULT 0.7) RETURNS TABLE(memory_id uuid, similarity_score double precision, therapeutic_relevance double precision, decay_factor double precision, final_score double precision)
    LANGUAGE plpgsql
    AS $$
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
$$;


ALTER FUNCTION public.search_therapeutic_memories(p_user_id text, p_query_vector public.vector, p_recovery_stage text, p_memory_types text[], p_limit integer, p_min_similarity double precision) OWNER TO memos_user;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: achievements; Type: TABLE; Schema: public; Owner: memos_user
--

CREATE TABLE public.achievements (
    id uuid NOT NULL,
    title character varying(255) NOT NULL,
    description text,
    icon_url character varying(500),
    category character varying(50),
    criteria_type character varying(50),
    criteria_value integer,
    criteria_data json,
    badge_color character varying(7),
    is_active boolean,
    created_at timestamp without time zone
);


ALTER TABLE public.achievements OWNER TO memos_user;

--
-- Name: alembic_version; Type: TABLE; Schema: public; Owner: memos_user
--

CREATE TABLE public.alembic_version (
    version_num character varying(32) NOT NULL
);


ALTER TABLE public.alembic_version OWNER TO memos_user;

--
-- Name: mem0; Type: TABLE; Schema: public; Owner: memos_user
--

CREATE TABLE public.mem0 (
    id uuid NOT NULL,
    vector public.vector(1536),
    payload jsonb
);


ALTER TABLE public.mem0 OWNER TO memos_user;

--
-- Name: mem0migrations; Type: TABLE; Schema: public; Owner: memos_user
--

CREATE TABLE public.mem0migrations (
    id uuid NOT NULL,
    vector public.vector(1536),
    payload jsonb
);


ALTER TABLE public.mem0migrations OWNER TO memos_user;

--
-- Name: memories; Type: TABLE; Schema: public; Owner: memos_user
--

CREATE TABLE public.memories (
    id uuid NOT NULL,
    user_id character varying(255) NOT NULL,
    content_hash character varying(64) NOT NULL,
    encrypted_content bytea NOT NULL,
    content_summary text,
    memory_type character varying(50) NOT NULL,
    privacy_level character varying(50) NOT NULL,
    embedding_vector public.vector(1024),
    embedding_model character varying(100),
    recovery_stage character varying(50),
    therapeutic_relevance double precision,
    crisis_level double precision,
    source_conversation_id character varying(255),
    tags json,
    entities json,
    created_at timestamp without time zone NOT NULL,
    updated_at timestamp without time zone,
    accessed_at timestamp without time zone,
    expires_at timestamp without time zone,
    consent_given boolean NOT NULL,
    consent_date timestamp without time zone,
    retention_policy character varying(50),
    relevance_score double precision,
    quality_score double precision,
    access_count integer,
    is_deleted boolean,
    deleted_at timestamp without time zone,
    deletion_reason character varying(255)
);


ALTER TABLE public.memories OWNER TO memos_user;

--
-- Name: ollama_model_specs; Type: TABLE; Schema: public; Owner: memos_user
--

CREATE TABLE public.ollama_model_specs (
    id uuid NOT NULL,
    model_name character varying(255) NOT NULL,
    base_model character varying(100) NOT NULL,
    variant character varying(50),
    parameter_count double precision,
    file_size_gb double precision,
    vram_min_gb double precision,
    vram_recommended_gb double precision,
    context_window integer,
    context_window_extended integer,
    capabilities json,
    specialization character varying(100),
    speed_tier character varying(20),
    quantization character varying(50),
    description text,
    architecture character varying(100),
    license character varying(100),
    multimodal boolean,
    vision boolean,
    tags json,
    source_url character varying(500),
    last_scraped timestamp without time zone,
    scrape_successful boolean,
    created_at timestamp without time zone,
    updated_at timestamp without time zone
);


ALTER TABLE public.ollama_model_specs OWNER TO memos_user;

--
-- Name: quest_tasks; Type: TABLE; Schema: public; Owner: memos_user
--

CREATE TABLE public.quest_tasks (
    id uuid NOT NULL,
    quest_id uuid,
    title character varying(255) NOT NULL,
    description text,
    order_index integer NOT NULL,
    is_required boolean,
    verification_data json,
    created_at timestamp without time zone
);


ALTER TABLE public.quest_tasks OWNER TO memos_user;

--
-- Name: quests; Type: TABLE; Schema: public; Owner: memos_user
--

CREATE TABLE public.quests (
    id uuid NOT NULL,
    title character varying(255) NOT NULL,
    description text,
    category character varying(50) NOT NULL,
    points integer,
    min_recovery_stage character varying(50),
    max_active_days integer,
    cooldown_hours integer,
    prerequisites json,
    verification_type character varying(50),
    quest_metadata json,
    is_active boolean,
    created_at timestamp without time zone,
    updated_at timestamp without time zone
);


ALTER TABLE public.quests OWNER TO memos_user;

--
-- Name: user_achievements; Type: TABLE; Schema: public; Owner: memos_user
--

CREATE TABLE public.user_achievements (
    id uuid NOT NULL,
    user_id character varying(255) NOT NULL,
    achievement_id uuid,
    earned_at timestamp without time zone NOT NULL,
    points_awarded integer
);


ALTER TABLE public.user_achievements OWNER TO memos_user;

--
-- Name: user_memory_settings; Type: TABLE; Schema: public; Owner: memos_user
--

CREATE TABLE public.user_memory_settings (
    user_id character varying(255) NOT NULL,
    memory_enabled boolean NOT NULL,
    max_memories integer,
    retention_days integer,
    default_privacy_level character varying(50),
    auto_consent boolean,
    allow_clinical_memories boolean,
    allow_crisis_detection boolean,
    recovery_stage character varying(50),
    therapy_goals json,
    crisis_contacts json,
    retrieval_depth double precision,
    semantic_threshold double precision,
    include_low_relevance boolean,
    allow_care_team_access boolean,
    care_team_members json,
    family_sharing_enabled boolean,
    offline_sync_enabled boolean,
    push_notifications boolean,
    memory_insights_enabled boolean,
    settings_version integer,
    last_consent_date timestamp without time zone,
    consent_document_version character varying(50),
    created_at timestamp without time zone NOT NULL,
    updated_at timestamp without time zone,
    last_accessed timestamp without time zone
);


ALTER TABLE public.user_memory_settings OWNER TO memos_user;

--
-- Name: user_quest_stats; Type: TABLE; Schema: public; Owner: memos_user
--

CREATE TABLE public.user_quest_stats (
    user_id character varying(255) NOT NULL,
    total_points integer,
    current_streak_days integer,
    longest_streak_days integer,
    last_activity_date date,
    total_quests_completed integer,
    level character varying(50),
    weekly_points integer,
    monthly_points integer,
    created_at timestamp without time zone,
    updated_at timestamp without time zone
);


ALTER TABLE public.user_quest_stats OWNER TO memos_user;

--
-- Name: user_quests; Type: TABLE; Schema: public; Owner: memos_user
--

CREATE TABLE public.user_quests (
    id uuid NOT NULL,
    user_id character varying(255) NOT NULL,
    quest_id uuid,
    state character varying(50) NOT NULL,
    started_at timestamp without time zone NOT NULL,
    completed_at timestamp without time zone,
    verified_at timestamp without time zone,
    verified_by character varying(255),
    progress_data json,
    points_earned integer
);


ALTER TABLE public.user_quests OWNER TO memos_user;

--
-- Name: user_tasks; Type: TABLE; Schema: public; Owner: memos_user
--

CREATE TABLE public.user_tasks (
    id uuid NOT NULL,
    user_quest_id uuid,
    task_id uuid,
    user_id character varying(255) NOT NULL,
    state character varying(50) NOT NULL,
    completed_at timestamp without time zone,
    evidence_data json,
    created_at timestamp without time zone
);


ALTER TABLE public.user_tasks OWNER TO memos_user;

--
-- Data for Name: achievements; Type: TABLE DATA; Schema: public; Owner: memos_user
--

COPY public.achievements (id, title, description, icon_url, category, criteria_type, criteria_value, criteria_data, badge_color, is_active, created_at) FROM stdin;
\.


--
-- Data for Name: alembic_version; Type: TABLE DATA; Schema: public; Owner: memos_user
--

COPY public.alembic_version (version_num) FROM stdin;
0001_initial
\.


--
-- Data for Name: mem0; Type: TABLE DATA; Schema: public; Owner: memos_user
--

COPY public.mem0 (id, vector, payload) FROM stdin;
\.


--
-- Data for Name: mem0migrations; Type: TABLE DATA; Schema: public; Owner: memos_user
--

COPY public.mem0migrations (id, vector, payload) FROM stdin;
b3a69fc3-9a0f-45b6-8b42-eda63036d02d	[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]	{"type": "user_identity", "user_id": "b3a69fc3-9a0f-45b6-8b42-eda63036d02d"}
\.


--
-- Data for Name: memories; Type: TABLE DATA; Schema: public; Owner: memos_user
--

COPY public.memories (id, user_id, content_hash, encrypted_content, content_summary, memory_type, privacy_level, embedding_vector, embedding_model, recovery_stage, therapeutic_relevance, crisis_level, source_conversation_id, tags, entities, created_at, updated_at, accessed_at, expires_at, consent_given, consent_date, retention_policy, relevance_score, quality_score, access_count, is_deleted, deleted_at, deletion_reason) FROM stdin;
c22dc958-c9a8-4064-8e61-ffdcac10486a	final_test_storage	cbc06ef3e34813bc38e9444f4c40bb9d1d4d22b9ea3620325dcaf7fd4886b108	\\x674141414141426f6330305f4d56686e344a674d77436e41657244676d33443862656c56626f67513646427a3547696549644b33424c6136635165794434317a47572d5758344f4b747369416a36347a7875686b716a736a3670456a30394c326b675241447276466237556d75634e4d5f32646f434766374f417a646668454e6b554f6b675a33706f762d305567412d4432637574446579676c4367644a726742413d3d	This is the final test memory for Recovery Bot API.	conversational	balanced	[0.020351553,0.0050687934,-0.023834646,0.07094056,-0.062803365,-0.028001755,-0.00503228,-0.016026117,0.05708058,0.018094562,-0.0013519003,0.023260793,-0.01387277,-0.039548162,-0.07131996,0.008515181,-0.024183184,0.00387646,0.0033895476,-0.058588423,0.030761935,0.014567319,-0.042662065,-0.022228403,-0.018226126,0.00894788,-0.010063538,0.013400258,0.057081062,0.0016397858,-0.027342087,0.0011946874,0.006881724,-0.0699077,-0.02976461,-0.0072658686,0.04860528,0.0017711174,-0.008195156,-0.035091296,-0.006371767,-0.0046713687,0.018748308,-0.05962422,-0.098120965,-0.024754096,-0.0075822226,0.010402195,0.0094297305,-0.044028226,-0.018679777,-0.006095498,0.023004526,-0.011403182,0.023452915,-0.060149714,-0.029126495,0.005735015,-0.060272694,-0.0119092,0.07881504,0.016549923,0.0250395,-0.06008134,-0.0073993998,0.043978516,-0.011974988,0.007670914,-0.0061417413,-0.004947769,-0.04056392,0.022084527,-0.050014097,-0.065354876,-0.0073261107,0.020418778,-0.01073769,0.0270075,-0.011681882,0.02381047,-0.015448812,-0.006039274,-0.0009448628,-0.0006862147,-0.023947489,-0.048190065,0.004193709,-0.0026988762,0.019220658,-0.05066406,-0.03427374,0.02355697,-0.0089225685,0.081546046,-0.0059048804,0.010461718,-0.0051873485,0.008271232,0.004147153,-0.028076757,0.032435473,0.032719243,-0.036458917,0.014955382,-0.032761186,0.009024295,-0.0050518373,0.018620437,-0.056345865,-0.026127005,-0.006761262,0.018066349,0.010479331,0.009217107,-0.0618791,0.017652752,-0.052433323,0.0066857543,-0.008439468,0.0045206165,0.03086426,-0.015084463,-0.019493615,-0.053230777,-0.028025908,-0.030113403,0.02254301,0.03558104,0.0005183623,-0.0064187977,0.009245344,-0.0059673972,-0.004816459,0.026931781,0.00975797,-0.009705375,0.029116208,0.021902293,0.027541496,-0.027349526,0.023793481,-0.032657392,0.009645055,0.07741777,-0.0029321848,0.004224743,-0.019394487,-0.020260857,0.021850314,-0.009761707,-0.015054599,-0.017507778,0.041768387,0.015995016,0.010271122,-0.033688635,-0.035846714,0.039401475,-0.003130137,0.01083874,-0.032030348,0.02099768,-0.040722996,0.03222589,-0.020878416,0.003378999,0.024614269,-0.008744754,0.010358368,-0.0020015535,0.0389487,0.012113797,0.015541211,-0.00022183989,0.032468025,0.036101602,0.02274887,0.029163955,0.011142626,0.021788374,-0.02075468,-0.008322917,-0.018416416,0.036776926,-0.022385607,0.0095033,0.008376863,-0.030658541,-0.021444334,-0.012821882,0.015033021,0.047748256,-0.050317015,0.01507845,-0.016031018,0.04736124,-0.02467072,0.04515535,-0.015766218,-0.04939265,-0.040016394,0.025345342,0.0033119933,0.0069836285,-0.015081407,0.0067916643,0.057738006,0.039709788,0.007142591,-0.016229203,0.026138503,-0.016546445,-0.009414742,0.012019032,0.027798045,-0.050340768,-0.0042061363,0.021040607,0.019395804,0.014196918,0.0035467632,0.039413463,0.016105497,-0.018267838,0.031557582,0.020389503,-0.011272941,-0.004670948,-0.0014060574,-0.06605215,0.0012593312,0.022855971,-0.026817197,0.023935713,0.008991443,0.030008297,0.045587055,0.029969996,0.00035453148,0.012088646,-0.0022829394,0.014108879,0.0110012,0.015537031,0.03753162,-0.026794009,-0.028205767,-0.024074515,-0.020938871,0.008103687,-0.01151025,0.022443645,0.026773212,-0.0015891329,-0.028570147,-0.020444993,0.013968614,0.036035996,-0.026754256,0.026984416,0.006611362,-0.0006216823,0.017222593,0.0049189464,0.010191411,0.0123794265,0.039689593,-0.021578405,0.0073725106,-0.02627213,-0.009079404,-0.008356369,-0.035424545,-0.007126848,0.012531342,-0.005148506,0.026026173,-0.029257368,-0.013215829,-0.05033686,0.011186403,-0.040874995,0.020914517,0.041477017,0.040150173,0.0018687021,-0.0312455,-0.014824362,-0.0021406224,0.0076325405,-0.017925605,-0.008883179,-0.0234334,-0.022030989,0.008761988,-0.04507284,0.0109766675,0.041475665,-0.009298633,-0.026424035,0.006631564,-0.0033957115,-0.02639413,0.012979807,-0.024698779,0.04677492,0.06452013,0.03630908,0.04541194,0.032434378,-0.015169513,0.058784924,0.0045695473,0.013105811,-0.024348132,0.066020615,0.032433905,0.019993974,-0.04062525,0.0058262115,-0.021312008,-0.017350221,0.0010467494,-0.026770934,-0.012876316,-0.014896247,-0.024572728,-0.060892332,0.039760247,-0.057696085,-0.023302004,0.011368713,-0.034341592,0.032950796,-0.026300073,0.032189656,-0.052859485,-0.01181964,-0.04425981,0.009553324,0.0454451,-0.03559207,-0.046038922,0.0442253,-0.02180033,0.013710479,0.009137636,-0.013339263,0.011923093,-0.01253925,-0.013592438,0.01998478,0.010641734,0.012844447,0.030685436,0.019931387,-0.028189976,0.009018441,0.034367055,0.048646245,0.015770834,-0.026486825,0.014331492,0.0493344,-0.0068028844,-0.04628237,-0.015732877,0.049200486,0.019812966,-0.07275648,0.0067268503,-0.019069029,-0.00805387,-0.015373564,-0.0289803,-0.03337881,0.026292939,-0.035425372,0.051984116,-0.058642607,-0.0024393734,0.023728574,-0.007745625,0.041806463,0.022920134,-0.029920438,0.0061543733,0.024840659,-0.05743991,0.015394632,0.011226681,0.018521175,-0.022940895,-0.010783413,-0.03795328,-0.04109631,0.014923821,0.03420265,0.029608093,-0.045050003,0.0005034841,-0.03612331,0.0011518227,0.0369475,-0.034233455,-0.01167799,-0.05476577,0.040567912,0.036709704,0.024056898,-0.017774368,0.034365837,-0.04675186,0.018014213,-0.003325934,-0.0005972576,-0.027650991,0.025120523,6.7154106e-05,-0.05181435,-0.017856574,-0.018584728,-0.009424754,0.0014298379,0.0480551,-9.804412e-05,0.004262246,-0.0058788434,0.014076108,0.041419685,0.0069099176,-0.04324224,-0.0431525,-0.026996288,-0.05336997,0.045802865,0.01480351,-0.043180887,-0.023465555,0.0069602067,0.0430221,-0.010873023,-0.020122938,-0.021655858,-0.006809829,0.039355073,0.018957974,-0.0009949941,-0.0074169016,0.016220715,0.017028596,-0.053582393,0.024332125,-0.0337057,-0.003088033,0.019236796,-0.02079093,0.031239867,0.025475409,-0.016624438,-0.0060870517,-0.0301224,-0.011211708,-0.02492657,0.008071902,0.05865819,0.06278816,-0.008201411,-0.0048454544,-0.00663619,-0.028915532,0.004200577,0.004590049,-0.021356432,-0.010441244,-0.00074324675,0.052277524,-0.02477589,-0.001644297,-0.029631816,-0.017443875,0.0712127,0.033188865,0.003656019,0.011038042,-0.019863872,0.0050886516,-0.0105898,-0.026766974,0.011687528,0.034814753,-0.009436898,0.0060750693,0.019532895,-0.042040486,0.0052286456,-0.043438986,-0.01004008,0.031750336,0.00823334,0.027685825,-0.016839664,-0.053859737,0.049227808,-0.0087691555,0.0095007615,-0.02458144,-0.0015605988,-0.06840759,0.008845137,-0.0016962591,0.029158432,-0.013317093,0.016946938,0.029364401,-0.014447966,0.014577894,-0.045043517,-0.0375094,0.024471177,-0.0056946487,-0.027086,-0.022270149,0.004903714,-0.0060072076,0.03435571,0.038573712,-0.008118203,-0.046130933,-0.04165424,0.013358435,-0.068316,-0.0321748,-0.06271786,-0.016862333,0.0012199158,-0.003843648,0.017193124,-0.037288915,-0.037458315,-0.0050849114,0.008394372,0.005613818,-0.024234764,-0.01707432,-0.0014114338,0.0014971832,0.08791587,-0.022470351,0.030733954,0.009224978,0.04232624,-0.017398357,0.045239978,-0.020630853,-0.019698117,0.009928122,-0.03325435,0.024349516,0.062493596,-0.057503715,-0.0055472315,-0.019620778,0.018631894,-0.041979294,0.0041876794,-0.023964632,-0.018132225,0.004119482,-0.0030637337,0.0037208763,0.034759913,0.052392088,0.029337065,0.04776365,-0.03131697,-0.04339829,-0.03090758,-0.0531466,0.010450578,-0.030821595,-0.013687217,-0.0076550744,-0.017602155,0.036008302,-0.037002623,0.03284581,0.060601193,0.007383888,-0.020848716,-0.004381407,0.045668423,0.014322313,-0.039901685,-0.0468003,-0.024373952,-0.04103889,-0.0383884,-0.08711662,-0.060303424,-0.040093653,-0.022498576,0.048178766,-0.006519114,0.064630605,0.020032143,-0.04647068,-0.057135306,0.033221196,0.037189435,0.006854791,0.07063675,0.009786469,0.0054105036,-0.021813823,-0.049676914,0.0041763783,0.010934697,0.029020142,0.035937514,-0.023947926,0.011024159,0.013096471,-0.004140857,-0.074193604,-0.0054072826,-0.02774582,-0.0471784,-0.03625792,-0.019631922,-0.019632045,-0.0056332834,0.00024467395,0.012018432,-0.009425457,-0.0032818052,-0.0017161766,-0.013048275,-0.028062109,-0.006881528,0.07043629,-0.017016072,0.04918141,-0.071837075,-0.0416707,-0.044220313,0.009313596,0.0095190825,-0.01966578,-0.035154678,0.0035145953,0.012902498,0.08588867,-0.0293974,0.045629058,0.027619304,-0.036873966,-0.052960504,0.027077436,0.02057942,-0.052686468,0.04368561,-0.05474385,0.010667352,0.008534677,0.03489569,0.01640427,-0.05573203,-0.03383003,-0.051077053,-0.0062944833,0.00501128,-0.0035765574,0.0011552442,0.059519183,-0.022150435,-0.021495901,-0.008548918,0.013508823,-0.044363514,-0.040708926,-0.021608096,0.0036733793,-0.02310645,0.020903615,-0.034494214,-0.028853396,0.010935067,0.018457422,-0.0063022445,-0.03585834,0.012700553,0.01636778,0.0002901744,0.033853993,0.018206412,0.04175886,0.036050685,-0.011130828,0.012045824,0.0028801116,0.028301973,-0.01691891,0.018087586,-0.038894325,0.0018225984,-0.060124736,0.029432353,0.023003927,0.01570435,0.022734348,-0.006018356,0.011329916,-0.046689242,0.0056354385,-0.0060958504,-0.025648529,0.03444279,0.018136643,0.008955136,0.030126467,-0.024335936,-0.020183453,-0.049116865,0.015468474,0.043646384,0.0032179074,0.018680314,-0.03359818,0.00070408767,0.05015127,0.016169362,-0.028255716,-0.027263397,0.0011916647,-0.03197761,0.029831585,0.034187563,0.035528067,0.03993973,0.012326132,0.0015713726,-0.012564457,0.039019782,0.044749785,-0.012966107,-0.0220052,-0.064589374,-0.018412197,0.046212673,-0.029962227,-0.016317297,-0.009400768,0.033447687,-0.0030721978,0.034643408,-0.015946016,0.016671093,0.0029869876,0.014234385,0.048827015,-0.0556593,-0.029981252,-0.018449344,0.04449423,-0.018609112,-0.029546253,-1.5580774e-05,0.01788087,-0.003030732,0.010336749,-0.044776265,-0.0150810955,0.010556137,0.010762963,0.0250576,-0.011510658,-0.024529466,0.037933003,-0.01576435,0.02660637,0.017794982,-0.016448295,0.04052952,0.0196814,-0.0051794024,0.009863048,0.04920433,-0.017355792,0.04372566,0.009980017,-0.011284197,0.043221332,0.033642806,-0.030657219,0.062318366,0.051684897,0.0013112978,-0.045226432,0.049929474,-0.004623007,-0.06566432,-0.010010005,0.03449418,0.029619673,-0.037008367,0.034979906,0.0066368454,-0.00051235594,-0.012960256,-0.024598429,0.0007573724,-0.011883159,-0.037366845,-0.073262505,-0.0112790065,0.02863513,-0.025527876,-0.027147751,0.044206783,0.0036864267,0.021910155,0.026000213,-0.03094624,-0.048604466,0.041410834,-0.030419517,0.033564776,0.03878049,-0.007338657,0.02936498,-0.007443432,-0.008490855,0.008320837,-0.016548447,0.04031693,-0.014770455,-0.03166726,0.034874793,0.039775617,-0.023812596,0.029532855,-0.014002088,-0.013481746,-0.028448468,-0.037536707,-0.10068891,0.016108647,0.042040277,-0.002849165,0.020200323,0.031168979,-0.00093147403,0.043192625,-0.010985462,0.06946466,-0.017229075,0.04573373,0.032551713,-0.05444158,0.013044796,-0.0015064008,-0.049579345,-0.008293687,0.019789353,-0.010800203,0.053406022,-0.014686875,-0.03214709,0.0154160615,0.037990708,0.006580567,-0.04723877,0.03419275,0.011208539,-0.01265541,0.02593751,0.04015249,0.014075241,0.016618963,-0.021511812,0.02494668,-0.01927532,0.0009775689,0.0014112507,0.063116536,-0.0032193572,-0.046694748,-0.009329734,-0.04040496,0.0070318384,0.00496733,-0.009129946,-0.017016567,0.060529545,0.055457663,0.046023935,0.028457986,-0.034347802,-0.017772006,0.06152699,0.027124543,-0.005403756,0.043857537,0.016309807,0.01824643,0.027919572,-0.02540121,0.021049486,0.016954223,-0.032302033,-0.004860564,-0.007172595,-0.051976893,0.00040299405,0.009791685,0.032187924,0.0035328402,-0.011274837,-0.04601405,-0.06441893,-0.043761708,0.0060407273,-0.017529773,0.052751783,0.008962313,-0.04617083,0.010672868,-0.05352678,0.21477452,0.05268422,0.0041434644,0.0068616206,0.017564818,0.06206853,0.059607327,-0.025328858,-0.019482637,0.007599775,0.033566803,0.050691478,0.023164183,0.011779358,0.039631333,0.022377016,-0.04930799,0.04201381,0.018919325,-0.0039031527,-0.0029554877,0.022544434,0.011343474,0.057414785,0.009160035,0.010154204,-0.005147544,0.005971855,0.022874832,-0.018924499,0.044644568,-0.015417146,-0.0029764678,-0.012346256,-0.012083837,0.035580944,0.029298143,-0.08513143,0.006186525,-0.027377486,0.040685624,-0.010929178,-0.012750965,-0.013272201,0.0011446514,-0.003800536,-0.021747753,0.04541646,0.04973579,-0.020848686,0.018767353,-0.023826158,0.01723892,-0.03242198,-0.04238522,-0.018642347,-0.011645372,-0.02672372,0.010515563,-0.010028061,-2.0302617e-05,0.033637535,-0.015754776,0.03332652,-0.0051362766,0.011057739,-0.04033223,0.012389948,0.0033316386,-0.02521148,0.004591659,-0.010288262,-0.07778167,-0.00186925,0.012614236,0.02016552,-0.015957985,0.065655895,0.014549949,-0.044244718,0.005969824,0.0074312803,-0.011335405,-0.0045962855,-0.024529124,0.06009487,-0.058674343,0.03835013,-0.03284162,0.021459809,0.049766656,0.007370123,-0.012403516,0.007844214,0.038766436]	mxbai-embed-large	maintenance	0.9	0	final_test	["final", "test"]	{"locations": [], "organizations": [], "persons": [], "dates": []}	2025-07-13 06:07:59.158222	2025-07-13 06:07:59.158222	2025-07-13 06:07:59.16712	2032-07-11 06:07:59.158222	t	2025-07-13 06:07:59.158222	7_years	0.5	0.5	0	f	\N	\N
\.


--
-- Data for Name: ollama_model_specs; Type: TABLE DATA; Schema: public; Owner: memos_user
--

COPY public.ollama_model_specs (id, model_name, base_model, variant, parameter_count, file_size_gb, vram_min_gb, vram_recommended_gb, context_window, context_window_extended, capabilities, specialization, speed_tier, quantization, description, architecture, license, multimodal, vision, tags, source_url, last_scraped, scrape_successful, created_at, updated_at) FROM stdin;
f451b722-0070-42fe-8e57-0998a891589e	bge-m3:latest	bge-m3	latest	\N	1.0781666310504079	1.2937999572604895	1.6819399444386363	8000	\N	["function_calling"]	embedding	fast	\N	BGE-M3:latest is optimized for high-context reasoning tasks leveraging its 8,000 token context window and function calling capabilities. Key strengths include advanced multi-linguality, robust multi-functionality, and a specialized embedding layer. While offering strong performance, the model exhibits a moderate trade-off between inference speed and output quality, representing a specialization over broader general-purpose language modeling.	\N	\N	f	f	[]	https://ollama.com/library/bge-m3	2025-12-28 04:46:34.4873	t	2025-12-28 04:46:35.288799	2025-12-28 04:46:35.288809
1f94d41c-7dce-4333-9752-41f87a3f9442	functiongemma:latest	functiongemma	latest	\N	0.28014849592000246	0.33617819510400293	0.4370316536352038	301000000	\N	["function_calling"]	general_purpose	fast	\N	FunctionGemma prioritizes robust function calling via a fine-tuned Gemma 3 270M base model, leveraging a 301,000,000 token context window. Key capabilities include function execution and contextual reasoning, supported by its function-calling specialization. While optimized for function-driven tasks, this model represents a specialization rather than a general-purpose LLM, potentially exhibiting slower inference speeds compared to broader models.	\N	\N	f	f	[]	https://ollama.com/library/functiongemma	2025-12-28 04:46:36.765452	t	2025-12-28 04:46:36.766606	2025-12-28 04:46:36.76661
70672de5-c2ac-44f7-8623-bb7e48c3b799	functiongemma:270m-it-q8_0	functiongemma	270m-it-q8_0	\N	0.28014849592000246	0.33617819510400293	0.4370316536352038	301000000	\N	["function_calling"]	general_purpose	fast	\N	FunctionGemma:270m-it-q8_0 is a function-calling model optimized for executing complex instructions via its 301M token context window. Key capabilities include function invocation and extended context understanding, supported by a quantized (Q8) inference format. This model represents a specialization of the Gemma 3 270M architecture, prioritizing function execution over broad general-purpose capabilities, potentially impacting inference speed compared to larger models.	\N	\N	f	f	[]	https://ollama.com/library/functiongemma	2025-12-28 04:46:36.769675	t	2025-12-25 07:29:21.943597	2025-12-28 04:46:36.770233
a50b2e76-92f0-4ffe-ad2a-5ce52bc92fa3	functiongemma:270m	functiongemma	270m	\N	0.28014849592000246	0.33617819510400293	0.4370316536352038	270000000	\N	["function_calling"]	general_purpose	fast	\N	FunctionGemma:270m is optimized for function-calling applications leveraging a 270M parameter base model and a 270,000,000 token context window. Key capabilities include function invocation and contextual understanding, supported by a function-calling specialization. This model represents a balance between contextual awareness and computational efficiency, prioritizing quality within a large context window over potentially faster, more generalized models.	\N	\N	f	f	[]	https://ollama.com/library/functiongemma	2025-12-28 04:46:36.772159	t	2025-12-25 07:29:22.445772	2025-12-28 04:46:37.413309
be308b7b-97f9-4164-8c23-e09cda5c87be	qwen3-embedding:0.6b-fp16	qwen3-embedding	0.6b-fp16	0.6	1.1153797591105103	1.3384557109326123	1.739992424212396	40000	\N	["chat"]	embedding	fast	\N	Qwen3-embedding:0.6b-fp16 generates high-quality text embeddings optimized for semantic similarity search and retrieval. Key capabilities include a 40,000 token context window and 0.6 billion parameters, supporting chat functionality alongside embedding generation. This model represents a specialized embedding solution, prioritizing embedding quality over broader general-purpose language model performance.	\N	\N	f	f	[]	https://ollama.com/library/qwen3-embedding	2025-12-28 04:46:38.759936	t	2025-12-28 04:46:38.760645	2025-12-28 04:46:38.760648
c2a06fdb-c37e-4b51-8655-f85d73ba48ff	qwen3-embedding:0.6b-q8_0	qwen3-embedding	0.6b-q8_0	0.6	0.5952556226402521	0.7143067471683026	0.9285987713187933	40000	\N	["chat"]	embedding	fast	\N	Qwen3-embedding:0.6b-q8_0 generates high-quality text embeddings optimized for semantic similarity search and retrieval. Key capabilities include a 40,000 token context window and Q8 quantization, enabling efficient representation of large text corpora. While prioritizing embedding quality, this model represents a specialization over a general-purpose language model, potentially exhibiting slower inference speeds compared to larger models.	\N	\N	f	f	[]	https://ollama.com/library/qwen3-embedding	2025-12-28 04:46:40.162981	t	2025-12-28 04:46:40.164414	2025-12-28 04:46:40.164421
9f5475e1-a3c7-48ce-ab42-344c2fe6e742	qwen3-embedding:4b-fp16	qwen3-embedding	4b-fp16	4	7.497044360265136	8.996453232318162	11.695389202013612	40000	\N	["chat"]	embedding	fast	\N	Qwen3-embedding:4b-fp16 generates high-quality text embeddings optimized for semantic similarity search and retrieval. Key capabilities include a 40,000 token context window and 4.0 billion parameters, supporting chat functionality alongside embedding generation. This model represents a specialized embedding solution, prioritizing embedding quality over real-time inference speed, suitable for large-scale knowledge graph construction and dense vector search.	\N	\N	f	f	[]	https://ollama.com/library/qwen3-embedding	2025-12-28 04:46:41.542844	t	2025-12-28 04:46:41.543872	2025-12-28 04:46:41.543877
c8b42cb8-3cbe-46f8-9d96-6c47a5eefc06	qwen3-embedding:4b-q8_0	qwen3-embedding	4b-q8_0	4	3.9857444232329726	4.782893307879567	6.217761300243437	40000	\N	["chat"]	embedding	fast	\N	Qwen3-embedding:4b-q8_0 generates high-quality text embeddings optimized for semantic similarity search and retrieval. Key capabilities include a 40,000 token context window and 4.0 billion parameters, supporting chat functionality alongside embedding generation. This model represents a specialized embedding solution, prioritizing embedding quality over general-purpose language modeling speed and breadth.	\N	\N	f	f	[]	https://ollama.com/library/qwen3-embedding	2025-12-28 04:46:42.855053	t	2025-12-28 04:46:42.85724	2025-12-28 04:46:42.857252
4852b22e-a71a-40f5-a8e0-64985194b78a	qwen3-embedding:4b-q4_K_M	qwen3-embedding	4b-q4_K_M	4	2.3252368355169892	2.790284202620387	3.6273694634065032	40000	\N	["chat"]	embedding	fast	\N	Qwen3-embedding:4b-q4_K_M generates high-quality text embeddings optimized for semantic search and retrieval. Key capabilities include a 40,000-token context window and 4.0 billion parameters, facilitating nuanced representation learning. This model represents a specialization for embedding tasks, potentially sacrificing some speed compared to more general-purpose models, but offering superior performance within its focused domain.	\N	\N	f	f	[]	https://ollama.com/library/qwen3-embedding	2025-12-28 04:46:44.324974	t	2025-12-28 04:46:44.32641	2025-12-28 04:46:44.326416
95fd9ed8-58d6-4ad7-af93-54c97f2bd4d2	qwen3-embedding:8b-q4_K_M	qwen3-embedding	8b-q4_K_M	8	4.3556142533198	5.226737103983759	6.794758235178888	40000	\N	["chat"]	embedding	fast	\N	Qwen3-embedding:8b-q4_K_M generates high-quality text embeddings optimized for semantic search and retrieval. Key capabilities include a 40,000 token context window and 8.0 billion parameters, facilitating nuanced representation learning. This model represents a specialization for embedding tasks, potentially sacrificing some speed compared to more general-purpose models, while maintaining strong performance.	\N	\N	f	f	[]	https://ollama.com/library/qwen3-embedding	2025-12-28 04:46:45.698032	t	2025-12-28 04:46:45.699597	2025-12-28 04:46:45.699607
6a9766f9-6d1c-49ad-9bab-935138f4737a	qwen3-embedding:8b-q8_0	qwen3-embedding	8b-q8_0	8	7.494451559148729	8.993341870978474	11.691344432272016	40000	\N	["chat"]	embedding	fast	\N	Qwen3-embedding:8b-q8_0 generates high-quality text embeddings optimized for semantic search and retrieval. Key capabilities include a 40,000 token context window and 8.0B parameters, supporting chat functionality alongside embedding generation. This model represents a specialized embedding solution, prioritizing embedding quality over general-purpose language modeling speed and broader task versatility.	\N	\N	f	f	[]	https://ollama.com/library/qwen3-embedding	2025-12-28 04:46:47.105087	t	2025-12-28 04:46:47.10687	2025-12-28 04:46:47.106877
db771905-b8d6-4aa2-b731-e8da787ca5b8	qwen3-embedding:8b-fp16	qwen3-embedding	8b-fp16	8	14.101300990208983	16.92156118825078	21.998029544726016	40000	\N	["chat"]	embedding	fast	\N	Qwen3-embedding:8b-fp16 generates high-quality text embeddings optimized for semantic similarity tasks. Key capabilities include a 40,000 token context window and 8.0 billion parameters, facilitating nuanced representation learning. This model represents a specialized embedding model, prioritizing contextual depth over broad general-purpose language understanding, and operates in FP16 precision for efficient inference.	\N	\N	f	f	[]	https://ollama.com/library/qwen3-embedding	2025-12-28 04:46:48.488895	t	2025-12-28 04:46:48.490364	2025-12-28 04:46:48.490371
fcae76b2-9e14-4de3-b598-d0187ccfdf7d	qwen3-embedding:latest	qwen3-embedding	latest	\N	4.3556142533198	5.226737103983759	6.794758235178888	40000	\N	["chat"]	embedding	fast	\N	Qwen3 Embedding generates high-fidelity text embeddings optimized for semantic similarity search and retrieval. Key capabilities include generating embeddings for 40,000 token contexts and supporting chat functionality. While prioritizing embedding quality, the model’s performance may be less efficient than general-purpose embedding models, representing a specialization over broad applicability.	\N	\N	f	f	[]	https://ollama.com/library/qwen3-embedding	2025-12-28 04:46:49.774055	t	2025-12-28 04:46:50.560072	2025-12-28 04:46:50.560075
ba17fd7b-798e-49e3-846d-52f422e1cb14	bge-large:latest	bge-large	latest	\N	0.6244816156104207	0.7493779387325048	0.9741913203522563	671000000	\N	["chat"]	embedding	fast	\N	This BAAI bge-large model specializes in high-dimensional text embedding generation, leveraging a 671M token context window for nuanced representation learning. Key capabilities include vector similarity search and semantic clustering, followed by contextualized embedding generation. While offering high-quality embeddings, inference speed is comparatively slower than general-purpose language models due to its large context window and specialized training; best suited for applications prioritizing embedding accuracy over real-time responsiveness.	\N	\N	f	f	[]	https://ollama.com/library/bge-large	2025-12-28 04:46:52.020445	t	2025-12-28 04:46:52.676024	2025-12-28 04:46:52.676026
13b3deaa-c65b-4802-a310-fec99293ebb9	all-minilm:33m	all-minilm	33m	\N	0.06269655004143715	0.07523586004972457	0.09780661806464194	33000000	\N	["chat"]	general_purpose	fast	\N	all-minilm:33m is a large language model optimized for general-purpose conversational applications, leveraging a 33 million token context window for extended dialogue and complex reasoning. Key capabilities include text generation, chat interaction, and semantic understanding, prioritized by contextual awareness. While offering strong performance, the model’s large size introduces computational demands, representing a trade-off between quality and inference speed; it’s best suited for applications requiring extensive contextual processing rather than specialized, narrow domains.	\N	\N	f	f	[]	https://ollama.com/library/all-minilm	2025-12-28 04:46:54.294067	t	2025-12-28 04:46:54.295395	2025-12-28 04:46:54.295407
7033f7d5-9eb0-4407-a74e-cbb8364ad8ef	all-minilm:latest	all-minilm	latest	\N	0.04280451312661171	0.05136541575193405	0.06677504047751426	46000000	\N	["chat"]	general_purpose	fast	\N	This all-minilm:latest model is optimized for general-purpose conversational AI, leveraging a 46M token context window and exhibiting strong chat capabilities alongside robust embedding generation.  Its primary strength lies in contextual understanding and response generation, followed by effective semantic similarity matching and text completion.  While offering high-quality results, inference speed may be impacted relative to smaller models due to the large context window, representing a tradeoff between quality and computational efficiency.	\N	\N	f	f	[]	https://ollama.com/library/all-minilm	2025-12-28 04:46:55.804551	t	2025-12-28 04:46:56.462669	2025-12-28 04:46:56.462672
5c90e51c-739a-4f71-b642-14ba6b4c5a05	snowflake-arctic-embed2:latest	snowflake-arctic-embed2	latest	\N	1.0806105267256498	1.2967326320707797	1.6857524216920137	8000	\N	["multilingual"]	embedding	fast	\N	Snowflake Arctic Embed 2.0 is a high-quality multilingual embedding model optimized for semantic representation learning. Key capabilities include robust performance across multiple languages (8,000 token context window) and scalable inference, prioritizing embedding quality over real-time speed. This specialization focuses on embedding generation, potentially sacrificing lower-latency applications compared to more general-purpose embedding models.	\N	\N	f	f	[]	https://ollama.com/library/snowflake-arctic-embed2	2025-12-28 04:46:57.728606	t	2025-12-28 04:46:57.730435	2025-12-28 04:46:57.73045
ea11de07-14bd-47a9-8368-35d9fc759333	nomic-embed-text:latest	nomic-embed-text	latest	\N	0.25546406395733356	0.3065568767488003	0.39852393977344036	274000000	\N	["chat"]	embedding	fast	\N	nomic-embed-text:latest is a high-performance embedding model optimized for generating dense vector representations of text data, leveraging a 274,000,000 token context window for enhanced semantic understanding. Key capabilities include contextualized embedding generation and chat functionality, with a primary focus on quality over speed due to the extensive context window. This model represents a specialized approach to text embedding, prioritizing nuanced representation over broad generality.	\N	\N	f	f	[]	https://ollama.com/library/nomic-embed-text	2025-12-28 04:46:59.866892	t	2025-12-28 04:47:00.49311	2025-12-28 04:47:00.493114
a0103b86-be45-409d-8dd4-41eda04435f8	embeddinggemma:latest	embeddinggemma	latest	\N	0.5791670801118016	0.6950004961341619	0.9035006449744105	622000000	\N	["chat"]	embedding	fast	\N	EmbeddingGemma is a high-quality embedding model optimized for generating dense vector representations of text. Key capabilities include semantic similarity search and contextualized embedding generation, leveraging a 622,000,000 token context window. While offering strong performance, the model’s specialized focus may result in reduced generalization ability compared to broader, more general-purpose embedding models.	\N	\N	f	f	[]	https://ollama.com/library/embeddinggemma	2025-12-28 04:47:01.801837	t	2025-12-28 04:47:02.430666	2025-12-28 04:47:02.430671
241a23c2-5165-4e30-b345-1908fc74f60e	deepseek-r1:14b-qwen-distill-q8_0	deepseek-r1	14b-qwen-distill-q8_0	14	14.623254134319723	17.547904961183665	22.812276449538764	128000	\N	["reasoning", "vision"]	reasoning	medium	\N	DeepSeek-R1:14b-Qwen-Distill-Q8_0 is a high-performance reasoning model leveraging a 14.0B parameter architecture and 128k token context window. Primary capabilities include complex reasoning and vision understanding, complemented by strong contextual awareness. This model prioritizes quality over speed, representing a specialized reasoning model with a larger parameter footprint compared to more general-purpose models.	\N	\N	t	t	[]	https://ollama.com/library/deepseek-r1	2025-12-28 04:47:02.433441	t	2025-12-26 22:45:50.917339	2025-12-28 04:47:02.433799
568f89e9-31b6-451f-a33d-6a5702e1606f	deepseek-r1:32b	deepseek-r1	32b	32	18.487999037839472	22.185598845407366	28.841278499029578	128000	\N	["reasoning", "vision"]	reasoning	slow	\N	DeepSeek-R1:32B is a high-performance reasoning model leveraging a 32.0B parameter architecture and 128,000 token context window for complex logical inference. Key capabilities include advanced reasoning, vision/image understanding, and a specialized focus on reasoning tasks. While offering strong performance, the model’s large size may result in slower inference speeds compared to smaller models, prioritizing quality over speed.	\N	\N	t	t	[]	https://ollama.com/library/deepseek-r1	2025-12-28 04:47:02.435354	t	2025-12-25 07:29:47.482279	2025-12-28 04:47:02.435674
6d557b61-282c-400c-b420-63af436ca580	deepseek-r1:8b	deepseek-r1	8b	8	4.866510673426092	5.83981280811131	7.591756650544703	128000	\N	["reasoning", "vision"]	reasoning	fast	\N	DeepSeek-R1:8B is a high-performance reasoning model leveraging a 128k token context window for complex logical inference and problem-solving. Key capabilities include advanced reasoning, vision with image understanding, and a substantial 8.0B parameter size. While prioritizing quality through extensive training, the model’s large context window may introduce latency considerations compared to more streamlined models, specializing in reasoning tasks over broad general-purpose applications.	\N	\N	t	t	[]	https://ollama.com/library/deepseek-r1	2025-12-28 04:47:02.437059	t	2025-12-25 07:29:47.985764	2025-12-28 04:47:03.059788
48b14d3c-0ee2-4c15-9e15-4a374cb13596	ministral-3:8b	ministral-3	8b	8	5.608644910156727	6.730373892188072	8.749486059844495	256000	\N	["chat"]	general_purpose	fast	\N	Ministral-3:8B is a general-purpose language model optimized for chat applications, leveraging 8.0 billion parameters and a 256,000 token context window. Key capabilities include conversational response generation and text completion, supported by a broad parameter count. While offering strong performance, the model’s architecture prioritizes context length over inference speed, representing a trade-off between quality and computational efficiency.	\N	\N	f	f	[]	https://ollama.com/library/ministral-3	2025-12-28 04:47:03.061834	t	2025-12-25 22:25:07.632553	2025-12-28 04:47:03.06226
2c03c8cd-25e1-4300-b433-75f03e758ccb	ministral-3:3b	ministral-3	3b	3	2.7509786263108253	3.3011743515729903	4.291526657044888	256000	\N	["chat"]	general_purpose	fast	\N	Ministral-3:3B is a general-purpose language model optimized for chat applications, leveraging 3.0 billion parameters and a 256,000 token context window. Key capabilities include conversational response generation and text completion, supported by a broad architectural design. While prioritizing context length, performance may exhibit a trade-off compared to smaller models, and the model represents a general-purpose solution rather than a highly specialized one.	\N	\N	f	f	[]	https://ollama.com/library/ministral-3	2025-12-28 04:47:03.063539	t	2025-12-25 22:25:08.255629	2025-12-28 04:47:03.681384
94d16ade-b4fd-4e51-b79a-002f7d37444b	devstral-small-2:24b-instruct-2512-q8_0	devstral-small-2	24b-instruct-2512-q8_0	24	24.130331412889063	28.956397695466876	37.64331700410694	384000	\N	["code", "function_calling", "agentic"]	code	medium	\N	This 24B parameter model specializes in code-centric tasks, leveraging a 384k token context window for sophisticated code exploration and manipulation. Key capabilities include code generation, function calling, and agentic workflows, optimized for software engineering applications.  The model exhibits a specialization in code execution, potentially impacting general-purpose reasoning performance compared to more broadly trained models.	\N	\N	f	f	[]	https://ollama.com/library/devstral-small-2	2025-12-28 04:47:03.683243	t	2025-12-25 07:29:19.962444	2025-12-28 04:47:03.683624
c0c291cb-99bc-4307-afe2-e6be4eca6b2b	devstral-small-2:24b	devstral-small-2	24b	24	14.135031121782959	16.96203734613955	22.050648549981414	384000	\N	["code", "function_calling", "agentic"]	code	medium	\N	devstral-small-2:24b is a 24B parameter model optimized for agentic software engineering, leveraging a 384k token context window and function calling. Key capabilities include code generation, complex code manipulation, and tool utilization, with a specialization focused on codebase exploration and editing.  This model exhibits a trade-off between contextual understanding and inference speed, prioritizing quality over rapid response times within a code-centric environment.	\N	\N	f	f	[]	https://ollama.com/library/devstral-small-2	2025-12-28 04:47:03.685032	t	2025-12-25 07:29:20.466047	2025-12-28 04:47:04.33332
f5edfa80-815d-41ca-ad86-810f20af471d	olmo-3:7b	olmo-3	7b	7	4.164885323494673	4.9978623881936075	6.49722110465169	64000	\N	["chat"]	general_purpose	fast	\N	Olmo-3:7B is a general-purpose language model optimized for extended context understanding, leveraging a 64,000 token context window. Primary capabilities include chat functionality and broad text generation, supported by 7.0 billion parameters. While prioritizing contextual coherence, the model exhibits a moderate trade-off between response quality and inference speed, representing a balance between general-purpose utility and specialized performance.	\N	\N	f	f	[]	https://ollama.com/library/olmo-3	2025-12-28 04:47:04.335225	t	2025-12-25 07:29:21.159043	2025-12-28 04:47:05.001401
126eb753-abef-4329-8b22-fef7d04bf3b9	olmo-3.1:32b	olmo-3.1	32b	32	18.144442421384156	21.773330905660988	28.305330177359284	64000	\N	["chat"]	general_purpose	slow	\N	Olmo-3.1:32b is a general-purpose language model optimized for extended context understanding, leveraging a 64,000 token context window and 32 billion parameters. Key capabilities include chat functionality and broad knowledge retrieval, though performance may be influenced by context length. This model represents a balance between model size and capability, suitable for diverse applications requiring both quality and extended contextual reasoning.	\N	\N	f	f	[]	https://ollama.com/library/olmo-3.1	2025-12-28 04:47:05.003158	t	2025-12-25 07:29:21.815306	2025-12-28 04:47:05.624112
2aa9e607-5b4f-475f-b93e-dff8c6a671d0	functiongemma:270m-it-fp16	functiongemma	270m-it-fp16	\N	0.5141801126301289	0.6170161351561546	0.802120975703001	301000000	\N	["function_calling"]	general_purpose	fast	\N	FunctionGemma:270m-it-fp16 is optimized for function calling via a fine-tuned Gemma 3 270M base model, leveraging a 301,000,000 token context window. Key capabilities include function invocation and contextual reasoning, supported by FP16 precision. This model represents a specialization focused on function calling, potentially impacting computational efficiency compared to broader general-purpose models.	\N	\N	f	f	[]	https://ollama.com/library/functiongemma	2025-12-28 04:46:35.292255	t	2025-12-25 07:29:21.941451	2025-12-28 04:46:35.293663
78eec361-8ca5-4997-be58-1d7bbb0ff6fd	snowflake-arctic-embed2:568m	snowflake-arctic-embed2	568m	\N	1.0806105267256498	1.2967326320707797	1.6857524216920137	568000000	\N	["multilingual"]	embedding	fast	\N	Snowflake Arctic Embed 2.0:568m is a high-context multilingual embedding model optimized for semantic representation learning. Key capabilities include a 568 million token context window and robust multilingual support, alongside strong English performance. This model prioritizes context length and scale, potentially impacting inference speed relative to more specialized embedding models; it’s best suited for applications demanding extensive contextual understanding and broad language coverage.	\N	\N	f	f	[]	https://ollama.com/library/snowflake-arctic-embed2	2025-12-28 04:46:57.73523	t	2025-12-25 07:29:33.382182	2025-12-28 04:46:58.356342
05ce3d92-8f19-499e-85f3-369d0ceda2a5	nemotron-3-nano:30b	nemotron-3-nano	30b	30	22.60500087030232	27.126001044362784	35.26380135767162	1000000	\N	["agentic"]	general_purpose	slow	\N	Nemotron-3-nano:30b is a high-performance, agentic language model optimized for complex reasoning and task execution. Key capabilities include 1,000,000 token context window, 30 billion parameters, and agentic behavior, facilitating interactive problem-solving. While prioritizing efficiency, this model represents a general-purpose specialization, potentially exhibiting a trade-off in raw output quality compared to larger, more narrowly-tuned models.	\N	\N	f	f	[]	https://ollama.com/library/nemotron-3-nano	2025-12-28 04:47:05.625913	t	2025-12-25 07:29:23.086282	2025-12-28 04:47:06.246237
9572b488-e32b-4761-aa23-c2ef23b96582	nemotron-mini:4b	nemotron-mini	4b	4	2.5121518839150667	3.01458226069808	3.918956938907504	4000	\N	["function_calling"]	general_purpose	fast	\N	nemotron-mini:4b is optimized for efficient function calling and RAG QA applications. Key capabilities include 4.0B parameters, a 4,000-token context window, and robust function invocation. This model represents a balance between general-purpose utility and performance, prioritizing speed over absolute output quality, suitable for rapid prototyping and iterative development.	\N	\N	f	f	[]	https://ollama.com/library/nemotron-mini	2025-12-28 04:47:06.248364	t	2025-12-25 07:29:23.711124	2025-12-28 04:47:06.877511
7da23cb9-1548-4002-95ea-9ed61567d735	qwen3-coder:30b	qwen3-coder	30b	30	17.28227432910353	20.738729194924236	26.96034795340151	256000	\N	["code", "agentic"]	code	slow	\N	qwen3-coder:30b is a high-performance model optimized for code generation and agentic task execution, leveraging a 30B parameter architecture and 256K token context window. Key capabilities include code completion, code understanding, and agentic reasoning, supported by a specialized codebase.  While prioritizing quality and context length, inference speed may be impacted compared to smaller models; this model represents a specialized solution for complex coding and agentic workflows.	\N	\N	f	f	[]	https://ollama.com/library/qwen3-coder	2025-12-28 04:47:06.879795	t	2025-12-25 07:29:24.337356	2025-12-28 04:47:07.513703
554579da-6bdc-4190-8509-b6962e5f9f1f	qwen3-vl:32b	qwen3-vl	32b	32	19.474232383072376	23.36907885968685	30.379802517592907	256000	\N	["vision"]	vision	slow	\N	qwen3-vl:32b is a high-performance vision-language model optimized for complex visual understanding tasks. Key capabilities include image captioning, visual question answering, and object detection, leveraging a 32.0B parameter architecture and a 256,000 token context window. While offering superior accuracy, inference speed may be impacted relative to smaller models due to its large size and specialized vision focus.	\N	\N	t	t	[]	https://ollama.com/library/qwen3-vl	2025-12-28 04:47:07.515582	t	2025-12-25 07:29:24.493616	2025-12-28 04:47:07.515866
4077f88c-8d88-45da-8dac-f1d2fdc0a544	qwen3-vl:8b	qwen3-vl	8b	8	5.718707925640047	6.862449510768056	8.921184363998472	256000	\N	["vision"]	vision	fast	\N	qwen3-vl:8b is a high-performance vision-language model optimized for complex visual understanding tasks. Key capabilities include image captioning, visual question answering, and visual reasoning utilizing a 256,000 token context window and 8.0B parameters. While offering strong performance, inference speed may be impacted relative to smaller models, and the model’s specialization in vision-language tasks represents a trade-off compared to more general-purpose language models.	\N	\N	t	t	[]	https://ollama.com/library/qwen3-vl	2025-12-28 04:47:07.517159	t	2025-12-25 07:29:24.495561	2025-12-28 04:47:07.517449
c3421c50-fb25-4c26-b826-b6880f10a2fa	qwen3-vl:4b	qwen3-vl	4b	4	3.0693003302440047	3.6831603962928057	4.788108515180648	256000	\N	["vision"]	vision	fast	\N	qwen3-vl:4b is a high-performance vision-language model optimized for complex visual understanding tasks. Key capabilities include image captioning, visual question answering, and object detection, leveraging a 4.0B parameter base model and 256,000 token context window. While offering strong performance, inference speed may be impacted relative to smaller models due to its larger size and specialized vision adaptation.	\N	\N	t	t	[]	https://ollama.com/library/qwen3-vl	2025-12-28 04:47:07.518663	t	2025-12-25 07:29:24.497248	2025-12-28 04:47:07.518901
bbbfb438-d224-487d-aa00-e3458f86902a	qwen3-vl:2b	qwen3-vl	2b	2	1.759752339683473	2.111702807620168	2.7452136499062183	256000	\N	["vision"]	vision	fast	\N	qwen3-vl:2b is a vision-language model optimized for complex visual understanding tasks. Key capabilities include image captioning, visual question answering, and object detection, leveraging a 2.0B parameter base model and a 256,000 token context window. While offering high accuracy within a specialized vision context, inference speed may be slower compared to more general-purpose models; this model prioritizes visual reasoning depth over raw processing velocity.	\N	\N	t	t	[]	https://ollama.com/library/qwen3-vl	2025-12-28 04:47:07.520124	t	2025-12-25 07:29:25.000622	2025-12-28 04:47:08.216385
a339157b-830e-4b9a-9c57-063bcff4367a	gemma3:12b	gemma3	12b	12	7.589524847455323	9.107429816946388	11.839658762030304	128000	\N	["chat"]	general_purpose	medium	\N	gemma3:12b is a high-performance chat model optimized for general-purpose conversational AI. Key capabilities include 12.0B parameters, a 128,000 token context window, and robust text generation. While offering strong quality, inference speed may be impacted relative to smaller models due to its size; this model balances broad applicability with computational demands.	\N	\N	f	f	[]	https://ollama.com/library/gemma3	2025-12-28 04:47:08.218438	t	2025-12-25 07:29:25.154935	2025-12-28 04:47:08.218839
01b4530a-d715-4cc2-9309-b11e09adaae6	gemma3:27b	gemma3	27b	27	16.202160102315247	19.442592122778297	25.275369759611788	128000	\N	["chat"]	general_purpose	medium	\N	gemma3:27b is a high-performance chat model optimized for complex conversational tasks. Key capabilities include 128k token context understanding, 27 billion parameters, and general-purpose reasoning.  While offering superior quality, inference speed is constrained by the large model size, representing a trade-off between computational efficiency and output fidelity.	\N	\N	f	f	[]	https://ollama.com/library/gemma3	2025-12-28 04:47:08.22039	t	2025-12-25 07:29:25.159614	2025-12-28 04:47:08.220723
6205f6df-3abd-40da-9279-2c949e3eefc0	gemma3:4b	gemma3	4b	4	3.1095014922320843	3.731401790678501	4.850822327882051	128000	\N	["chat"]	general_purpose	fast	\N	gemma3:4b is a high-performing chat model optimized for general-purpose conversational AI. Key capabilities include 128k token context understanding, 4.0 billion parameters, and robust dialogue generation.  This model prioritizes quality over speed, representing a general-purpose solution with a larger context window compared to more streamlined, specialized models.	\N	\N	f	f	[]	https://ollama.com/library/gemma3	2025-12-28 04:47:08.222128	t	2025-12-25 07:29:25.665482	2025-12-28 04:47:08.851912
653ad6b3-9db2-44b1-a9cc-91dabf91c63f	qwen3:32b	qwen3	32b	32	18.813883726485074	22.576660471782088	29.349658613316716	40000	\N	["chat"]	general_purpose	slow	\N	Qwen3:32B is a high-performance language model optimized for extended context understanding and complex reasoning, leveraging a 40,000 token context window. Key capabilities include chat functionality and general-purpose text generation, alongside dense model architecture.  While offering strong quality, inference speed may be impacted relative to smaller models; this model represents a balance between generality and specialized performance.	\N	\N	f	f	[]	https://ollama.com/library/qwen3	2025-12-28 04:47:08.854032	t	2025-12-25 07:29:25.808899	2025-12-28 04:47:08.854422
2db46ad4-12fb-454d-9f07-7c24a8be95aa	qwen3:8b	qwen3	8b	8	4.866521958261728	5.839826349914074	7.591774254888296	40000	\N	["chat"]	general_purpose	fast	\N	Qwen3:8B is a high-performance language model optimized for extended context understanding and complex reasoning tasks, leveraging a 40,000 token context window. Key capabilities include chat interactions and general-purpose text generation, supported by 8.0 billion parameters. While prioritizing quality through a dense architecture, inference speed may be impacted compared to smaller models; this model represents a balance between generality and specialized performance.	\N	\N	f	f	[]	https://ollama.com/library/qwen3	2025-12-28 04:47:08.855828	t	2025-12-25 07:29:25.811045	2025-12-28 04:47:08.856203
05c41a18-e6cb-4a8c-9297-887c3566c2ae	qwen3:14b	qwen3	14b	14	8.639133130200207	10.366959756240249	13.477047683112323	40000	\N	["chat"]	general_purpose	medium	\N	Qwen3:14B is a high-performance language model optimized for extended context understanding and complex reasoning, leveraging a 40,000 token context window. Key capabilities include chat functionality and general-purpose text generation, alongside dense model architecture. While offering strong quality, inference speed may be impacted compared to smaller models; this model represents a balance between generality and specialized performance.	\N	\N	f	f	[]	https://ollama.com/library/qwen3	2025-12-28 04:47:08.857705	t	2025-12-25 07:29:26.313452	2025-12-28 04:47:09.475523
0af937ed-9b51-4c00-85f1-97dad10c854f	gpt-oss:120b	gpt-oss	120b	120	60.806229904294014	72.96747588515281	94.85771865069866	128000	\N	["code", "reasoning", "agentic"]	reasoning	slow	\N	This 120B-parameter model specializes in advanced reasoning and agentic task execution, leveraging a 128,000 token context window. Key capabilities include code generation, complex reasoning, and agentic behavior, though performance may be influenced by context window length.  Primarily suited for applications demanding high-quality, multi-turn reasoning and agentic interactions, potentially exhibiting slower inference speeds compared to more generalized models.	\N	\N	f	f	[]	https://ollama.com/library/gpt-oss	2025-12-28 04:47:09.478874	t	2025-12-25 07:29:26.43073	2025-12-28 04:47:09.479679
4be9a1e9-d2ee-4c58-b3bc-cea33a0c40c6	gpt-oss:20b	gpt-oss	20b	20	12.833786978386343	15.40054437406361	20.020707686282694	128000	\N	["code", "reasoning", "agentic"]	reasoning	medium	\N	GPT-OSS:20B is a large language model optimized for complex reasoning and agentic task execution. Key capabilities include code generation, advanced reasoning, and 128k token context understanding.  This model prioritizes quality of reasoning over inference speed, representing a specialization in sophisticated problem-solving compared to broader, more general capabilities.	\N	\N	f	f	[]	https://ollama.com/library/gpt-oss	2025-12-28 04:47:09.482466	t	2025-12-25 07:29:26.932114	2025-12-28 04:47:10.056958
51c562fa-af9e-4300-98d0-de6b2d51e8ee	lukaspetrik/gemma3-tools:1b	lukaspetrik/gemma3-tools	1b	1	0.7593276742845774	0.9111932091414928	1.1845511718839408	\N	\N	[]	general_purpose	fast	\N	This Gemma 3-tools model (1.0B parameters) demonstrates general-purpose language capabilities, primarily suited for text generation and completion tasks. Key functionalities include text generation, summarization, and question answering, though performance may be impacted by the unknown context window and inherent tradeoffs between model size and inference speed.  It represents a broadly applicable model, prioritizing generality over specialized performance.	\N	\N	f	f	[]	https://ollama.com/library/lukaspetrik/gemma3-tools	2025-12-28 04:47:10.060576	f	2025-12-25 07:29:27.510645	2025-12-28 04:47:10.728615
b4415318-0182-4ebd-9904-740e54679ce0	llama4:128x17b	llama4	128x17b	17	228.02884100470692	273.6346092056483	355.7249919673428	1000000	\N	["vision"]	vision	medium	\N	Llama 4:128x17b is a large language model optimized for complex multimodal reasoning, leveraging a 1,000,000 token context window and 17.0B parameters to achieve superior image understanding and generation capabilities alongside standard text processing. While offering high-quality results, inference speed may be impacted compared to smaller models, and its specialization in vision tasks represents a trade-off against broader general-purpose language capabilities. This model is best suited for applications demanding advanced visual understanding and multimodal interaction.	\N	\N	t	t	[]	https://ollama.com/library/llama4	2025-12-28 04:47:10.730664	t	2025-12-25 07:29:27.639268	2025-12-28 04:47:10.730989
5624e482-7aab-43ce-a988-a6bd7c423e01	llama4:16x17b	llama4	16x17b	17	62.805472428910434	75.36656691469251	97.97653698910027	10000000	\N	["vision"]	vision	medium	\N	Llama 4:16x17b is a large multimodal model primarily designed for complex reasoning and generation across text and vision inputs. Key capabilities include 10,000,000 token context understanding, vision/image understanding, and specialized vision processing.  This model prioritizes quality over speed, exhibiting a significant computational cost compared to more generalized models, and is best suited for applications demanding high-fidelity multimodal interactions.	\N	\N	t	t	[]	https://ollama.com/library/llama4	2025-12-28 04:47:10.732512	t	2025-12-25 07:29:28.142528	2025-12-28 04:47:11.352802
abcc56c2-01a7-4720-99ad-f16fb41cceef	aya-expanse:8b	aya-expanse	8b	8	4.709698372520506	5.6516380470246075	7.34712946113199	8000	\N	["multilingual"]	general_purpose	fast	\N	aya-expanse:8b is a large language model optimized for multilingual text generation and understanding, leveraging 8.0B parameters and an 8,000-token context window. Key capabilities include multilingual proficiency, general-purpose reasoning, and substantial contextual awareness. While offering strong performance across diverse languages, the model prioritizes quality over speed and represents a general-purpose specialization, potentially less efficient than models fine-tuned for specific language pairs.	\N	\N	f	f	[]	https://ollama.com/library/aya-expanse	2025-12-28 04:47:11.354833	t	2025-12-25 07:29:28.254629	2025-12-28 04:47:11.355168
f51539c5-45be-4f4a-a19f-72a5268356c3	aya-expanse:32b	aya-expanse	32b	32	18.44097944535315	22.12917533442378	28.767927934750915	8000	\N	["multilingual"]	general_purpose	slow	\N	aya-expanse:32b is a high-performance language model optimized for multilingual text generation and understanding. Leveraging 32.0B parameters and an 8,000-token context window, it demonstrates strong performance across 23 languages, followed by robust general-purpose capabilities. While prioritizing quality, inference speed may be impacted compared to smaller models; this model offers a balance between generality and specialized multilingual proficiency.	\N	\N	f	f	[]	https://ollama.com/library/aya-expanse	2025-12-28 04:47:11.35718	t	2025-12-25 07:29:28.758464	2025-12-28 04:47:12.008519
f86af56e-4a33-4f80-9a7d-d8f65c273aed	mxbai-embed-large:latest	mxbai-embed-large	latest	\N	0.6236280249431729	0.7483536299318075	0.9728597189113498	670000000	\N	["chat"]	embedding	fast	\N	mxbai-embed-large:latest generates high-fidelity embeddings optimized for semantic similarity search and retrieval within expansive contexts. Key capabilities include dense vector representation and contextualized embedding generation, leveraging a 670,000,000 token context window. While prioritizing embedding quality, the model’s computational demands may result in slower inference speeds compared to more generalized embedding models; it represents a specialized solution for large-scale knowledge graph construction and retrieval.	\N	\N	f	f	[]	https://ollama.com/library/mxbai-embed-large	2025-12-28 04:47:12.010706	t	2025-12-25 07:29:29.371073	2025-12-28 04:47:12.620948
a8125956-9143-4fc5-bac2-8f1bc00fcf5c	granite3-guardian:2b	granite3-guardian	2b	2	2.5091071352362633	3.010928562283516	3.914207130968571	8000	\N	["chat"]	general_purpose	fast	\N	granite3-guardian:2b is a large language model optimized for risk detection within conversational prompts and responses. Key capabilities include prompt toxicity analysis (high effectiveness), response safety assessment (medium effectiveness), and general conversational chat functionality. This model offers a balance between accuracy and processing speed, prioritizing quality over rapid inference, and represents a general-purpose specialization rather than a highly focused domain expert.	\N	\N	f	f	[]	https://ollama.com/library/granite3-guardian	2025-12-28 04:47:12.622792	t	2025-12-25 07:29:29.751921	2025-12-28 04:47:12.623224
3a79966c-1e4d-4774-bd51-dc3c244b17e3	granite3-guardian:8b	granite3-guardian	8b	8	5.399315828457475	6.4791789941489695	8.42293269239366	8000	\N	["chat"]	general_purpose	fast	\N	granite3-guardian:8b is a large language model optimized for risk detection within conversational prompts and responses. Key capabilities include prompt toxicity identification (high effectiveness), response safety assessment (medium effectiveness), and general conversational chat functionality. This 8B parameter model with an 8K context window offers a balance between accuracy and speed, prioritizing safety over absolute generative quality, and is suitable for applications requiring proactive risk mitigation in dynamic dialogue.	\N	\N	f	f	[]	https://ollama.com/library/granite3-guardian	2025-12-28 04:47:12.624576	t	2025-12-25 07:29:30.255383	2025-12-28 04:47:13.264301
58c4bd2d-5941-4f4d-bd19-192db8c3de75	smollm2:1.7b	smollm2	1.7b	1.7	1.6954061882570386	2.0344874259084462	2.6448336536809802	8000	\N	["chat"]	general_purpose	fast	\N	Smollm2:1.7b is a general-purpose language model optimized for chat applications, leveraging 1.7 billion parameters and an 8,000 token context window. Key capabilities include conversational response generation and contextual understanding, supported by a broad parameterization. This model represents a balance between quality and computational efficiency, offering a tradeoff between speed and output complexity compared to smaller models, and a general specialization versus highly focused fine-tuning.	\N	\N	f	f	[]	https://ollama.com/library/smollm2	2025-12-28 04:47:13.266316	t	2025-12-25 07:29:30.375798	2025-12-28 04:47:13.266764
d72c39da-8ed0-4592-86b2-f0d6ed95cadd	smollm2:360m	smollm2	360m	\N	0.6757364720106125	0.810883766412735	1.0541488963365555	360000000	\N	["chat"]	general_purpose	fast	\N	Smollm2:360m is a high-context language model optimized for extended conversational applications, leveraging a 360 million parameter architecture and 360,000,000 token context window. Key capabilities include chat functionality and general-purpose text generation, though performance may be influenced by context length. This model represents a balance between model size and contextual capacity, prioritizing conversational depth over raw inference speed.	\N	\N	f	f	[]	https://ollama.com/library/smollm2	2025-12-28 04:47:13.268307	t	2025-12-25 07:29:30.378032	2025-12-28 04:47:13.268595
0e26b06a-3a21-4722-9997-f87a3d848261	smollm2:135m	smollm2	135m	\N	0.25229404866695404	0.3027528584003448	0.3935787159204483	135000000	\N	["chat"]	general_purpose	fast	\N	Smollm2:135m is a high-context language model optimized for extended conversational applications, leveraging a 135,000,000 token context window. Key capabilities include chat functionality and general-purpose text generation, with performance prioritized over computational efficiency. This model represents a balance between context length and parameter size, offering a pragmatic solution for tasks requiring substantial contextual understanding.	\N	\N	f	f	[]	https://ollama.com/library/smollm2	2025-12-28 04:47:13.27006	t	2025-12-25 07:29:30.881367	2025-12-28 04:47:13.892879
9b8c4456-c04e-4ced-a37c-77d108c68331	opencoder:8b	opencoder	8b	8	4.410805343650281	5.292966412380338	6.880856336094439	8000	\N	["code", "chat", "multilingual"]	code	fast	\N	opencoder:8b is a code-specialized LLM optimized for code generation and understanding, leveraging 8.0B parameters and an 8,000-token context window. Key capabilities include code generation, chat (English & Chinese), and multilingual support, prioritizing code performance. This model represents a balance between specialization and generality, potentially exhibiting slightly lower response quality compared to broader general-purpose models due to its focused training.	\N	\N	f	f	[]	https://ollama.com/library/opencoder	2025-12-28 04:47:13.895086	t	2025-12-25 07:29:31.012578	2025-12-28 04:47:13.895459
c5907f64-9f13-48f9-a5f9-28a8353f845f	opencoder:1.5b	opencoder	1.5b	1.5	1.3204128136858344	1.5844953764230012	2.0598439893499014	4000	\N	["code", "chat", "multilingual"]	code	fast	\N	opencoder:1.5b is a code-focused LLM optimized for code generation and understanding, leveraging a 4,000-token context window and 1.5 billion parameters. Key capabilities include code completion, code translation (English & Chinese), and general chat functionality. While offering strong code performance, the model’s specialization may result in reduced effectiveness compared to more general-purpose LLMs with larger parameter counts.	\N	\N	f	f	[]	https://ollama.com/library/opencoder	2025-12-28 04:47:13.896753	t	2025-12-25 07:29:31.515802	2025-12-28 04:47:14.537404
f04c73ac-dbb9-4d29-a58d-64e9680f5379	tulu3:8b	tulu3	8b	8	4.582841868512332	5.499410242214799	7.149233314879238	128000	\N	["code", "instruction"]	code	fast	\N	Tulu3:8B is a code-specialized instruction-following model optimized for complex code generation and understanding. Key capabilities include code generation (primary), instruction following, and general text comprehension. This model exhibits a trade-off between contextual understanding (128k tokens) and inference speed, prioritizing code specialization over broad generality.	\N	\N	f	f	[]	https://ollama.com/library/tulu3	2025-12-28 04:47:14.539388	t	2025-12-25 07:29:31.639891	2025-12-28 04:47:14.53972
808c2ef3-968a-41ee-ac19-7909d6f88362	tulu3:70b	tulu3	70b	70	39.60029551014304	47.52035461217165	61.77646099582315	128000	\N	["code", "instruction"]	code	slow	\N	Tulu3:70B is a high-performance code generation and instruction-following model. Key capabilities include code generation (primary) and instruction execution, supported by a 128,000 token context window. While prioritizing code specialization, the model exhibits a trade-off between computational speed and broader generality, suitable for applications demanding precise code output.	\N	\N	f	f	[]	https://ollama.com/library/tulu3	2025-12-28 04:47:14.541169	t	2025-12-25 07:29:32.142493	2025-12-28 04:47:15.16866
36479e1c-298d-4f5c-ade2-0637ef144716	sailor2:1b	sailor2	1b	1	0.983663422986865	1.180396107584238	1.5345149398595095	32000	\N	["multilingual"]	general_purpose	fast	\N	Sailor2:1B is a multilingual language model optimized for general-purpose text generation across diverse Southeast Asian languages. Key capabilities include a 32,000 token context window and support for multiple languages, alongside 1.0 billion parameters. While offering a balance of performance and efficiency, this model prioritizes generality over specialized performance and may exhibit slightly lower quality compared to larger variants.	\N	\N	f	f	[]	https://ollama.com/library/sailor2	2025-12-28 04:47:15.170715	t	2025-12-25 07:29:32.261569	2025-12-28 04:47:15.171113
85e7f87a-467a-47bf-90c7-1169397631ae	sailor2:8b	sailor2	8b	8	4.882864099927247	5.859436919912696	7.617267995886505	32000	\N	["multilingual"]	general_purpose	fast	\N	Sailor2:8B is a high-capacity, multilingual language model optimized for general-purpose tasks, leveraging a 32,000 token context window. Key capabilities include multilingual support and a substantial 8.0 billion parameters, facilitating strong performance across diverse applications. While offering robust quality, the model’s size may introduce latency considerations compared to smaller models, representing a trade-off between computational demands and output fidelity.	\N	\N	f	f	[]	https://ollama.com/library/sailor2	2025-12-28 04:47:15.172626	t	2025-12-25 07:29:32.263227	2025-12-28 04:47:15.172958
8c6592db-6c32-477d-9c6a-b3c25fcf7884	sailor2:20b	sailor2	20b	20	10.824186427518725	12.98902371302247	16.885730826929212	32000	\N	["multilingual"]	general_purpose	medium	\N	Sailor2:20B is a large language model optimized for general-purpose tasks, leveraging a 32,000 token context window and 20 billion parameters. Key capabilities include multilingual support and robust performance across diverse text generation and understanding applications. While offering high quality, the model’s size may introduce latency considerations compared to smaller variants, representing a tradeoff between computational cost and output fidelity.	\N	\N	f	f	[]	https://ollama.com/library/sailor2	2025-12-28 04:47:15.174308	t	2025-12-25 07:29:32.765986	2025-12-28 04:47:15.794514
8478e8e7-4906-44cd-9947-969316afabf9	llama3.3:70b	llama3.3	70b	70	39.60022136196494	47.520265634357926	61.77634532466531	128000	\N	["chat"]	general_purpose	slow	\N	Llama 3.3 70B is a high-performance language model optimized for general-purpose conversational AI tasks. Key capabilities include 128,000 token context window support and 70 billion parameters, enabling complex dialogue generation. While offering strong performance, the model’s scale may introduce latency considerations compared to smaller models, representing a tradeoff between quality and inference speed.	\N	\N	f	f	[]	https://ollama.com/library/llama3.3	2025-12-28 04:47:15.796864	t	2025-12-25 07:29:34.243017	2025-12-28 04:47:16.425706
ab5c7b33-eb93-461f-a555-66062b8d04d6	exaone3.5:2.4b	exaone3.5	2.4b	2.4	1.5319636100903153	1.8383563321083782	2.389863231740892	32000	\N	["instruction"]	general_purpose	fast	\N	EXAONE 3.5:2.4b is a general-purpose instruction-tuned language model designed for broad text generation tasks. Key capabilities include instruction following and a 32,000 token context window, alongside 2.4 billion parameters. While offering strong performance, the model’s size may result in slower inference speeds compared to smaller models, prioritizing quality over speed.	\N	\N	f	f	[]	https://ollama.com/library/exaone3.5	2025-12-28 04:47:16.427656	t	2025-12-25 07:29:34.390862	2025-12-28 04:47:16.427962
78aff1a9-37fd-456e-8976-65abb0a1a903	granite-embedding:30m	granite-embedding	30m	\N	0.058240074664354324	0.06988808959722519	0.09085451647639274	30000000	\N	["code", "multilingual"]	embedding	fast	\N	granite-embedding:30m is a dense biencoder embedding model optimized for generating high-quality contextual embeddings. Key capabilities include code and multilingual text embedding, alongside a 30,000,000 token context window. This model prioritizes embedding quality over inference speed and is specialized for English-language applications, representing a tradeoff between generality and performance.	\N	\N	f	f	[]	https://ollama.com/library/granite-embedding	2025-12-28 04:47:17.055739	t	2025-12-25 07:29:35.530398	2025-12-28 04:47:17.671776
becbfdff-bc87-474b-9bfe-cd6c5aa2366a	granite3.1-moe:3b	granite3.1-moe	3b	3	1.8954159282147884	2.274499113857746	2.9568488480150696	128000	\N	["chat"]	general_purpose	fast	\N	The granite3.1-moe:3b model is optimized for low-latency chat applications leveraging a 3.0B parameter Mixture-of-Experts architecture. Key capabilities include a 128,000 token context window and general-purpose conversational responses. While prioritizing speed, the model represents a general-purpose specialization compared to more focused MoE models, offering a balance between responsiveness and broader knowledge.	\N	\N	f	f	[]	https://ollama.com/library/granite3.1-moe	2025-12-28 04:47:17.673756	t	2025-12-25 07:29:35.672487	2025-12-28 04:47:17.674095
8b2d733b-1a1b-417e-a52f-469700c74f6b	granite3.1-moe:1b	granite3.1-moe	1b	1	1.3245764020830393	1.589491682499647	2.0663391872495414	128000	\N	["chat"]	general_purpose	fast	\N	The granite3.1-moe:1b model is optimized for chat applications leveraging a 1.0B parameter Mixture-of-Experts architecture with a 128,000 token context window. Key capabilities include conversational response generation and general-purpose text understanding, supported by its MoE design for efficient scaling. While prioritizing low latency, the model represents a general-purpose specialization, potentially sacrificing some quality compared to models trained solely on broader datasets.	\N	\N	f	f	[]	https://ollama.com/library/granite3.1-moe	2025-12-28 04:47:17.67545	t	2025-12-25 07:29:36.175593	2025-12-28 04:47:18.304833
b8be6e5d-e6c7-42e2-921c-5f8c2e6aaa22	granite3.1-dense:2b	granite3.1-dense	2b	2	1.4618873801082373	1.7542648561298846	2.28054431296885	128000	\N	["chat"]	general_purpose	fast	\N	Granite3.1-dense:2b is a general-purpose dense LLM optimized for chat applications, leveraging 2.0 billion parameters and a 128,000 token context window. Key capabilities include conversational interaction and text generation, supported by extensive training on 12+ trillion tokens. While prioritizing efficiency, the model represents a balance between model size and performance, offering broader applicability compared to highly specialized models.	\N	\N	f	f	[]	https://ollama.com/library/granite3.1-dense	2025-12-28 04:47:18.306905	t	2025-12-25 07:29:36.306477	2025-12-28 04:47:18.307229
c28faba9-9e30-4433-aa71-3bb9ed0a2ab9	granite3.1-dense:8b	granite3.1-dense	8b	8	4.648821255192161	5.578585506230593	7.252161158099771	128000	\N	["chat"]	general_purpose	fast	\N	Granite3.1-dense:8b is a general-purpose dense LLM optimized for chat applications, leveraging 8.0B parameters and a 128,000 token context window. Key capabilities include conversational response generation and text understanding, though performance may be influenced by context length. This model represents a balance between efficiency and broad applicability, prioritizing speed over potentially higher quality outputs in extremely complex scenarios.	\N	\N	f	f	[]	https://ollama.com/library/granite3.1-dense	2025-12-28 04:47:18.308452	t	2025-12-25 07:29:36.808607	2025-12-28 04:47:18.928911
ce303eb1-b74a-41a5-81bb-4a0ed43c5b2a	falcon3:10b	falcon3	10b	10	5.855722369626164	7.026866843551397	9.134926896616816	32000	\N	["code", "math"]	code	medium	\N	Falcon3:10B is a high-performance language model optimized for code generation and mathematical problem-solving, leveraging a 32,000 token context window. Key capabilities include code, math, and general language understanding, with a focus on efficiency due to its 10.0B parameter size. While highly effective for specialized coding and mathematical tasks, the model’s performance may exhibit a slight trade-off compared to larger models in broader, less specialized contexts.	\N	\N	f	f	[]	https://ollama.com/library/falcon3	2025-12-28 04:47:18.931085	t	2025-12-25 07:29:37.433014	2025-12-28 04:47:19.549668
06628da6-a523-4165-b58a-efffa20f681c	dolphin3:8b	dolphin3	8b	8	4.582812754437327	5.499375305324793	7.149187896922231	128000	\N	["code", "math", "function_calling", "agentic", "instruction"]	code	fast	\N	dolphin3:8b is a code-focused instruct-tuned model leveraging 8.0B parameters and a 128,000 token context window. Key capabilities include code generation, mathematical reasoning, function calling, and agentic behavior, prioritized for optimal performance. This model represents a specialization within the broader Llama 3.1 ecosystem, offering a balance between code-specific efficiency and general instruction-following capabilities.	\N	\N	f	f	[]	https://ollama.com/library/dolphin3	2025-12-28 04:47:19.551948	t	2025-12-25 07:29:38.055549	2025-12-28 04:47:20.200836
c419e99e-4dd6-41e4-ae32-236d4f251c97	gemma3n:e4b	gemma3n	e4b	4	7.0292401276528835	8.43508815318346	10.965614599138497	32000	\N	["chat"]	general_purpose	fast	\N	Gemma 3n:e4b is a general-purpose language model optimized for conversational applications. Key capabilities include chat functionality and a 32,000 token context window, leveraging 4.0 billion parameters. While designed for efficient execution on consumer devices, this model represents a tradeoff between model size and potential performance, prioritizing generality over specialized task optimization.	\N	\N	f	f	[]	https://ollama.com/library/gemma3n	2025-12-28 04:47:20.20352	t	2025-12-25 07:29:38.186931	2025-12-28 04:47:20.20383
a1129a80-d090-4427-941b-2a0090a36b79	gemma3n:e2b	gemma3n	e2b	2	5.235538410022855	6.282646092027425	8.167439919635653	32000	\N	["chat"]	general_purpose	fast	\N	Gemma 3n:e2b is a general-purpose language model optimized for conversational applications. Key capabilities include 32,000 token context window processing and chat functionality, alongside 2.0 billion parameters. While designed for efficient execution on consumer devices, this model represents a tradeoff between context length and potential performance compared to larger models, offering broad applicability rather than deep specialization.	\N	\N	f	f	[]	https://ollama.com/library/gemma3n	2025-12-28 04:47:20.206513	t	2025-12-25 07:29:38.689692	2025-12-28 04:47:20.83066
1aa93e72-6a8c-4fff-bf53-3d7169f46827	mistral-small3.2:24b-instruct-2506-q8_0	mistral-small3.2	24b-instruct-2506-q8_0	24	24.130341436713934	28.95640972405672	37.64333264127374	128000	\N	["function_calling", "instruction"]	function_calling	medium	\N	This Mistral Small variant (24B parameters) is optimized for complex instruction following and function calling within a 128K token context window. Key capabilities include robust instruction adherence and enhanced function calling support, alongside a substantial parameter count.  The model represents a specialization focused on high-quality, context-rich interactions, potentially impacting inference speed compared to more generalized models.	\N	\N	f	f	[]	https://ollama.com/library/mistral-small3.2	2025-12-28 04:47:20.832408	t	2025-12-25 07:29:38.800773	2025-12-28 04:47:20.832724
6cfa8401-c868-4930-ab85-f1ef2a3b5f70	mistral-small3.2:24b	mistral-small3.2	24b	24	14.135041145607829	16.962049374729393	22.05066418714821	128000	\N	["function_calling", "instruction"]	function_calling	medium	\N	Mistral-small3.2:24b is a high-quality language model optimized for instruction following and function calling, leveraging a 24.0B parameter base. Key capabilities include a 128,000 token context window and robust function calling support, alongside standard instruction-tuned performance.  This model represents a specialized variant prioritizing quality over speed, suitable for complex reasoning and task execution requiring extensive contextual understanding.	\N	\N	f	f	[]	https://ollama.com/library/mistral-small3.2	2025-12-28 04:47:20.833851	t	2025-12-25 07:29:39.306221	2025-12-28 04:47:21.457388
924b6a21-72f1-4a3c-9b99-606c69c54d6c	devstral:24b-small-2505-q8_0	devstral	24b-small-2505-q8_0	24	23.334099274128675	28.00091912895441	36.40119486764073	128000	\N	["code", "agentic"]	code	medium	\N	Devstral:24b-small-2505-q8_0 is a code generation model optimized for agentic coding tasks, leveraging a 24.0B parameter base. Key capabilities include code completion, code generation, and agentic reasoning, with a 128,000 token context window.  The model represents a balance between quality and inference speed, exhibiting a specialization in code-related applications while operating with quantized (Q8_0) weights.	\N	\N	f	f	[]	https://ollama.com/library/devstral	2025-12-28 04:47:21.459433	t	2025-12-25 07:29:39.432886	2025-12-28 04:47:21.459723
a1a55a43-b65d-47d8-9512-607a2cb0dc78	devstral:24b	devstral	24b	24	13.349510652944446	16.019412783533333	20.825236618593333	128000	\N	["code", "agentic"]	code	medium	\N	Devstral:24b is a high-performance model optimized for code generation and agentic tasks, leveraging a 128k token context window. Key capabilities include code completion, agentic reasoning, and complex code understanding, with a prioritization of accuracy over inference speed. This model exhibits specialization in code-related domains, representing a trade-off against broader general-purpose language understanding.	\N	\N	f	f	[]	https://ollama.com/library/devstral	2025-12-28 04:47:21.461032	t	2025-12-25 07:29:39.936071	2025-12-28 04:47:22.086491
61e61745-b54f-46cb-8c25-64c97f6eaae1	minicpm-v:8b	minicpm-v	8b	8	5.0979093331843615	6.117491199821234	7.952738559767604	32000	\N	["vision"]	vision	fast	\N	minicpm-v:8b is a multimodal LLM optimized for vision-language understanding, leveraging a 32K token context window and 8.0B parameters. Key capabilities include image understanding and multimodal reasoning, followed by text generation. This model represents a specialization in vision tasks, potentially sacrificing some generality compared to broader LLMs, and may exhibit a moderate speed-quality tradeoff during inference.	\N	\N	t	t	[]	https://ollama.com/library/minicpm-v	2025-12-28 04:47:22.088637	t	2025-12-25 07:29:40.565873	2025-12-28 04:47:22.711748
23d86252-2921-45dd-8b14-cc4fa176e224	reader-lm:1.5b	reader-lm	1.5b	1.5	0.8707558121532202	1.0449069745838642	1.3583790669590234	935000000	\N	["chat"]	general_purpose	fast	\N	reader-lm:1.5b is optimized for converting HTML to Markdown, leveraging a 935,000,000 token context window for high-fidelity content transformation. Key capabilities include Markdown generation and semantic structure preservation, followed by HTML parsing and tokenization. While offering strong quality, the model’s general-purpose design may necessitate iterative refinement for highly specialized HTML formats compared to a dedicated, specialized converter.	\N	\N	f	f	[]	https://ollama.com/library/reader-lm	2025-12-28 04:47:22.713838	t	2025-12-25 07:29:40.931232	2025-12-28 04:47:22.714236
7e7346a7-fd19-4735-8cba-4bd93ef9c0b2	reader-lm:0.5b	reader-lm	0.5b	0.5	0.32798095885664225	0.39357715062797066	0.5116502958163619	352000000	\N	["chat"]	general_purpose	fast	\N	reader-lm:0.5b is a large language model optimized for converting HTML to Markdown, leveraging a 352,000,000 token context window and 0.5 billion parameters for high-fidelity output. Key capabilities include Markdown syntax generation and semantic structure preservation, followed by content formatting and tokenization. While prioritizing quality, the model exhibits a moderate processing latency and represents a general-purpose solution suitable for diverse content transformation needs.	\N	\N	f	f	[]	https://ollama.com/library/reader-lm	2025-12-28 04:47:22.715751	t	2025-12-25 07:29:41.434239	2025-12-28 04:47:23.337471
bd4976f3-34cf-4022-969a-fc5aaa5c8e0f	solar-pro:22b	solar-pro	22b	22	12.396110350266099	14.875332420319317	19.337932146415113	4000	\N	["chat"]	general_purpose	medium	\N	Solar Pro:22b is a general-purpose LLM optimized for conversational applications, leveraging 22.0B parameters and a 4,000-token context window. Key capabilities include text generation, question answering, and dialogue management, prioritizing quality over rapid inference due to its model size. This model represents a balance between broad applicability and computational demands, suitable for tasks requiring nuanced understanding and extended context.	\N	\N	f	f	[]	https://ollama.com/library/solar-pro	2025-12-28 04:47:23.339744	t	2025-12-25 07:29:42.774179	2025-12-28 04:47:23.956298
3420c7aa-b096-492e-8bf8-2ca9ecc61072	magistral:24b	magistral	24b	24	13.349504401907325	16.01940528228879	20.825226866975427	39000	\N	["reasoning"]	reasoning	medium	\N	Magistral:24b is a high-performance reasoning model leveraging 24.0B parameters and a 39,000 token context window. Key capabilities include complex reasoning, followed by detailed information retrieval and logical deduction. This model prioritizes reasoning accuracy over inference speed, representing a specialization within the broader LLM landscape.	\N	\N	f	f	[]	https://ollama.com/library/magistral	2025-12-28 04:47:23.959572	t	2025-12-25 07:29:43.403101	2025-12-28 04:47:24.819678
29bf5432-bc9f-4bb0-94ac-68d648a3ae33	phi4-mini:3.8b	phi4-mini	3.8b	3.8	2.320741092786193	2.784889311343431	3.6203561047464605	128000	\N	["reasoning", "math", "multilingual", "function_calling"]	reasoning	fast	\N	Phi4-mini:3.8b is optimized for complex reasoning tasks, leveraging 3.8 billion parameters and a 128,000 token context window to achieve high-quality results. Key capabilities include advanced reasoning, mathematical computation, and multilingual processing, complemented by supported function calling.  This model prioritizes specialized reasoning performance over broad generality, potentially exhibiting slower inference speeds compared to larger models.	\N	\N	f	f	[]	https://ollama.com/library/phi4-mini	2025-12-28 04:47:24.821998	t	2025-12-25 07:29:44.014228	2025-12-28 04:47:25.436558
812a0def-527f-420a-8c5d-b7b6d71772d8	deepcoder:14b	deepcoder	14b	14	8.370832671411335	10.044999205693601	13.058498967401682	128000	\N	["code"]	code	medium	\N	DeepCoder:14B is a code generation model leveraging 14 billion parameters and a 128k token context window, prioritizing code completion and generation. Key capabilities include code synthesis, code translation, and code documentation, with a specialization focused on software development tasks.  Performance represents a trade-off between output quality and inference speed, optimized for applications demanding high-fidelity code solutions.	\N	\N	f	f	[]	https://ollama.com/library/deepcoder	2025-12-28 04:47:25.438414	t	2025-12-25 07:29:44.627637	2025-12-28 04:47:26.047631
5775e7ec-ba3f-4655-bfd2-09f10531e772	granite3.3:8b	granite3.3	8b	8	4.603426580317318	5.524111896380782	7.181345465295016	128000	\N	["reasoning", "instruction"]	reasoning	fast	\N	Granite3.3:8B is a 8.0B parameter language model optimized for reasoning and instruction-following, leveraging a 128,000 token context window. Key capabilities include advanced reasoning and precise instruction adherence, supported by a substantial model size. While offering high quality, inference speed may be impacted relative to smaller models due to its specialized focus.	\N	\N	f	f	[]	https://ollama.com/library/granite3.3	2025-12-28 04:47:26.049535	t	2025-12-25 07:29:44.745637	2025-12-28 04:47:26.04988
f2823e00-43a2-44fd-8cc5-0e27750eea9c	granite3.3:2b	granite3.3	2b	2	1.439192925579846	1.727031510695815	2.2451409639045594	128000	\N	["reasoning", "instruction"]	reasoning	fast	\N	granite3.3:2b is a 2.0B parameter language model optimized for reasoning tasks, leveraging a 128,000 token context window and demonstrating strong instruction-following performance.  Key capabilities include reasoning and instruction execution, supported by a fine-tuned architecture.  While prioritizing reasoning accuracy, the model exhibits a moderate context window size, potentially impacting speed compared to models with larger context windows.	\N	\N	f	f	[]	https://ollama.com/library/granite3.3	2025-12-28 04:47:26.050944	t	2025-12-25 07:29:45.249322	2025-12-28 04:47:26.671933
b91e60bd-331a-4dfe-935b-30a325ff6327	exaone-deep:32b	exaone-deep	32b	32	18.01528310868889	21.61833973042667	28.10384164955467	32000	\N	["code", "reasoning", "math"]	reasoning	slow	\N	EXAONE Deep:32b is a high-capacity model optimized for complex reasoning tasks, leveraging 32.0B parameters and a 32,000 token context window. Primary capabilities include mathematical reasoning, code generation, and general reasoning, with demonstrated performance exceeding smaller models. This model prioritizes quality of reasoning over inference speed and exhibits a specialization towards sophisticated problem-solving rather than broad general-purpose capabilities.	\N	\N	f	f	[]	https://ollama.com/library/exaone-deep	2025-12-28 04:47:26.674199	t	2025-12-25 07:29:45.362898	2025-12-28 04:47:26.67456
6d3debfe-1104-4601-ad8b-82423bf6833f	exaone3.5:7.8b	exaone3.5	7.8b	7.8	4.443027877248824	5.331633452698588	6.931123488508165	32000	\N	["instruction"]	general_purpose	fast	\N	EXAONE 3.5:7.8B is a high-capacity, instruction-tuned language model optimized for complex text generation and reasoning. Key capabilities include a 32,000 token context window and 7.8 billion parameters, facilitating nuanced understanding and extended output. While offering superior quality, the model’s size may introduce latency considerations compared to smaller models; it represents a general-purpose solution rather than a highly specialized one.	\N	\N	f	f	[]	https://ollama.com/library/exaone3.5	2025-12-28 04:47:16.429211	t	2025-12-25 07:29:34.392893	2025-12-28 04:47:16.429569
5f973f46-658d-4290-a9f5-1e4aa008ac29	exaone3.5:32b	exaone3.5	32b	32	18.01528283394873	21.618339400738478	28.10384122096002	32000	\N	["instruction"]	general_purpose	slow	\N	EXAONE 3.5:32b is a high-capacity, instruction-tuned language model optimized for complex text generation tasks. Key capabilities include instruction following, a 32,000-token context window, and 32.0B parameters, enabling strong performance across diverse applications. While offering superior quality, the model’s size may introduce latency considerations compared to smaller models, representing a general-purpose specialization with a focus on quality over speed.	\N	\N	f	f	[]	https://ollama.com/library/exaone3.5	2025-12-28 04:47:16.430984	t	2025-12-25 07:29:34.895885	2025-12-28 04:47:17.052291
dc7b7f9e-69df-405a-9848-5431e2831044	granite-embedding:278m	granite-embedding	278m	\N	0.5241272049024701	0.6289526458829641	0.8176384396478533	278000000	\N	["code", "multilingual"]	embedding	fast	\N	The granite-embedding:278m model specializes in generating high-quality dense embeddings for semantic similarity tasks. Key capabilities include multilingual embedding generation and code embedding, alongside a substantial 278,000,000 token context window. While offering superior embedding quality, the model’s large context window may introduce a performance trade-off compared to smaller embedding models, and its primary focus is on embedding generation rather than general-purpose text generation.	\N	\N	f	f	[]	https://ollama.com/library/granite-embedding	2025-12-28 04:47:17.054116	t	2025-12-25 07:29:35.027055	2025-12-28 04:47:17.054446
5f6ccb6c-efbe-487f-935e-bf07ceac7e1c	exaone-deep:7.8b	exaone-deep	7.8b	7.8	4.443028151988983	5.331633782386779	6.931123917102814	32000	\N	["code", "reasoning", "math"]	reasoning	fast	\N	EXAONE Deep:7.8B is a high-capacity model optimized for complex reasoning tasks, demonstrating strong performance in code generation, mathematical problem-solving, and general reasoning.  Featuring 7.8 billion parameters and a 32,000 token context window, it prioritizes quality of reasoning over inference speed.  This model specializes in advanced reasoning applications while maintaining broad capabilities across code, math, and general reasoning domains.	\N	\N	f	f	[]	https://ollama.com/library/exaone-deep	2025-12-28 04:47:26.676028	t	2025-12-25 07:29:45.86587	2025-12-28 04:47:27.299549
271c0025-52c3-47ac-9700-98b5d16ca3f8	phi4-reasoning:14b	phi4-reasoning	14b	14	10.353981734253466	12.42477808110416	16.152211505435407	32000	\N	["reasoning"]	reasoning	medium	\N	Phi 4 reasoning:14b is a high-performance reasoning model leveraging 14.0B parameters and a 32,000 token context window to achieve competitive results on complex reasoning benchmarks. Key capabilities include advanced logical deduction and strategic problem-solving, complemented by general language understanding. While prioritizing reasoning accuracy, the model may exhibit slower inference speeds compared to models with fewer parameters, representing a specialization over broader generality.	\N	\N	f	f	[]	https://ollama.com/library/phi4-reasoning	2025-12-28 04:47:27.301527	t	2025-12-25 07:29:46.486646	2025-12-28 04:47:27.922865
0a852d49-162e-4329-8773-61a2981da511	dolphin-mixtral:8x7b	dolphin-mixtral	8x7b	7	24.627535050734878	29.55304206088185	38.41895467914641	32000	\N	["code"]	code	fast	\N	dolphin-mixtral:8x7b is a code-optimized Mixtral mixture-of-experts model leveraging 7.0B parameters and a 32,000 token context window. Primary capabilities include code generation and understanding, followed by general text generation. This model prioritizes quality over speed due to its large size and specialized training, making it suitable for complex coding tasks and potentially less efficient for broader natural language applications.	\N	\N	f	f	[]	https://ollama.com/library/dolphin-mixtral	2025-12-28 04:47:27.924886	t	2025-12-25 07:29:47.109922	2025-12-28 04:47:28.546685
f1ee8b3c-84d1-4939-821f-1a591443742b	qwen2.5vl:7b	qwen2.5vl	7b	7	5.559293419122696	6.671152102947235	8.672497733831406	125000	\N	["vision"]	vision	fast	\N	qwen2.5vl:7b is a high-performance vision-language model optimized for complex visual understanding tasks. Key capabilities include 125,000 token context window, 7.0B parameters, and integrated vision/image understanding. While offering strong accuracy, inference speed may be impacted compared to smaller models due to its large parameter size and extensive context window; it represents a specialized model for vision-language applications rather than a general-purpose language model.	\N	\N	t	t	[]	https://ollama.com/library/qwen2.5vl	2025-12-28 04:47:28.54868	t	2025-12-25 07:29:48.110808	2025-12-28 04:47:28.549002
0dbfa1ba-76e1-47dd-8d3f-88cf1f31ad89	qwen2.5vl:7b-fp16	qwen2.5vl	7b-fp16	7	15.453094092197716	18.543712910637257	24.106826783828435	125000	\N	["vision"]	vision	fast	\N	qwen2.5vl:7b-fp16 is a high-performance vision-language model optimized for complex visual understanding tasks. Key capabilities include 125,000 token context length, 7.0B parameters, and native vision processing with image understanding. This model prioritizes accuracy and detail over inference speed, representing a specialized model for applications demanding sophisticated visual reasoning.	\N	\N	t	t	[]	https://ollama.com/library/qwen2.5vl	2025-12-28 04:47:28.550296	t	2025-12-25 07:29:48.112753	2025-12-28 04:47:28.550551
ff92f649-be41-4618-97ff-84e770bb8a50	qwen2.5vl:7b-q8_0	qwen2.5vl	7b-q8_0	7	8.758300451561809	10.50996054187417	13.662948704436422	125000	\N	["vision"]	vision	fast	\N	qwen2.5vl:7b-q8_0 is a high-performance vision-language model optimized for complex visual understanding tasks. Key capabilities include 125,000 token context length, 7.0B parameters, and integrated vision processing for image analysis. While prioritizing image understanding, this model represents a specialization over broader language capabilities and may exhibit a moderate trade-off in inference speed compared to more general-purpose models.	\N	\N	t	t	[]	https://ollama.com/library/qwen2.5vl	2025-12-28 04:47:28.551758	t	2025-12-25 07:29:48.114278	2025-12-28 04:47:28.551989
8a9c99f9-fbe0-4abf-90cc-bf8b48bb513a	qwen2.5vl:32b	qwen2.5vl	32b	32	19.706143678165972	23.647372413799165	30.741584137938915	125000	\N	["vision"]	vision	slow	\N	qwen2.5vl:32b is a high-performance vision-language model optimized for complex visual understanding and generation tasks. Key capabilities include 125,000 token context length, 32.0B parameters, and integrated vision/image understanding, alongside specialized vision processing.  This model prioritizes output quality over inference speed, representing a specialized model for detailed visual analysis rather than broad, general-purpose language modeling.	\N	\N	t	t	[]	https://ollama.com/library/qwen2.5vl	2025-12-28 04:47:28.553132	t	2025-12-25 07:29:48.617006	2025-12-28 04:47:29.16669
71bc13aa-d7a5-4476-9c39-ba868caa7e38	command-r7b:7b	command-r7b	7b	7	4.709727315232158	5.651672778278589	7.347174611762165	8000	\N	["chat"]	general_purpose	fast	\N	Command-r7b:7b is a high-performance chat model optimized for rapid inference, leveraging 7.0 billion parameters and an 8,000-token context window. Key capabilities include conversational AI and general-purpose text generation, though it represents a trade-off between speed and potentially higher output quality compared to larger models. This model offers a balance between efficiency and broad applicability, suitable for resource-constrained environments and diverse AI application development.	\N	\N	f	f	[]	https://ollama.com/library/command-r7b	2025-12-28 04:47:29.168795	t	2025-12-25 07:29:49.251015	2025-12-28 04:47:29.789847
f668f89b-8ace-4b7c-8a0c-45234fe9e7ab	llama3.2-vision:11b	llama3.2-vision	11b	11	7.2797659654170275	8.735719158500432	11.356434906050563	128000	\N	["reasoning", "vision", "instruction"]	vision	medium	\N	Llama 3.2-vision:11b is a vision-specialized large language model optimized for complex image reasoning tasks. Key capabilities include image understanding, visual question answering, and generative image captioning, leveraging a 128k token context window. This model prioritizes accuracy and detail over inference speed, representing a specialized solution for applications demanding high-quality visual analysis.	\N	\N	t	t	[]	https://ollama.com/library/llama3.2-vision	2025-12-28 04:47:29.791693	t	2025-12-25 07:29:49.870932	2025-12-28 04:47:30.413297
506c5a4b-c358-450e-b39a-1908c7c36ec3	granite3.2-vision:2b	granite3.2-vision	2b	2	2.2704270342364907	2.724512441083789	3.5418661734089256	16000	\N	["vision"]	vision	fast	\N	Granite3.2-vision:2b is a vision-language model optimized for automated visual document understanding, leveraging a 2.0B parameter architecture and 16,000 token context window. Key capabilities include image understanding and object detection within visual inputs, followed by structured data extraction from charts, diagrams, and tables. While prioritizing accuracy in specialized visual document processing, inference speed may be impacted compared to more general-purpose vision models.	\N	\N	t	t	[]	https://ollama.com/library/granite3.2-vision	2025-12-28 04:47:30.416347	t	2025-12-25 07:29:50.489226	2025-12-28 04:47:31.038764
b6a0cf4b-f462-447a-a278-aa125896d9bb	openthinker:32b	openthinker	32b	32	18.488008065149188	22.185609678179024	28.841292581632732	32000	\N	["reasoning", "vision"]	reasoning	slow	\N	OpenTHinker:32B is a reasoning model optimized for complex problem-solving, leveraging a 32B parameter architecture and 32,000 token context window. Key capabilities include advanced reasoning, vision with image understanding, and a specialized focus on deductive and inductive reasoning processes. While prioritizing quality through extensive distillation, the model exhibits a moderate computational cost compared to more general-purpose models, making it best suited for applications demanding robust logical inference.	\N	\N	t	t	[]	https://ollama.com/library/openthinker	2025-12-28 04:47:31.040538	t	2025-12-25 07:29:50.604798	2025-12-28 04:47:31.040848
bdc1077b-3677-403a-a99a-bf39ea5d760f	openthinker:7b	openthinker	7b	7	4.36146302241832	5.233755626901984	6.803882314972579	32000	\N	["reasoning", "vision"]	reasoning	fast	\N	OpenTHinker 7B is a reasoning model optimized for complex, multi-turn dialogues and knowledge-intensive tasks. Key capabilities include advanced reasoning, vision with image understanding, and a 32,000 token context window. While prioritizing high-quality reasoning, the model’s performance may be slightly impacted by the large context window, representing a trade-off between context length and inference speed.	\N	\N	t	t	[]	https://ollama.com/library/openthinker	2025-12-28 04:47:31.042277	t	2025-12-25 07:29:51.108046	2025-12-28 04:47:31.664739
211c1080-816f-4422-965f-748a9b2b92fe	qwq:32b	qwq	32b	32	18.488010072149336	22.1856120865792	28.841295712552963	40000	\N	["reasoning"]	reasoning	slow	\N	QwQ:32b is a reasoning-focused language model leveraging 32.0B parameters and a 40,000 token context window, demonstrating strong performance in complex logical inference and problem-solving. Key capabilities include advanced reasoning, followed by contextual understanding and text generation.  While prioritizing reasoning accuracy, the model exhibits a moderate computational cost compared to more general-purpose models, representing a specialization over broad applicability.	\N	\N	f	f	[]	https://ollama.com/library/qwq	2025-12-28 04:47:31.666774	t	2025-12-25 07:29:51.714274	2025-12-28 04:47:32.300172
2789f7fc-1c88-48f1-814c-bfd5be499fef	cogito:32b	cogito	32b	32	18.48537180200219	22.18244616240263	28.837180011123422	128000	\N	["reasoning", "vision"]	reasoning	slow	\N	Cogito:32b is a high-performance reasoning model leveraging a 128k token context window and 32B parameters, demonstrating superior performance on benchmarks compared to comparable open models. Key capabilities include advanced reasoning and vision/image understanding, with a specialization towards rigorous logical deduction.  While prioritizing quality, the model’s architecture may exhibit a moderate latency profile compared to more general-purpose models.	\N	\N	t	t	[]	https://ollama.com/library/cogito	2025-12-28 04:47:32.301904	t	2025-12-25 07:29:51.838555	2025-12-28 04:47:32.302248
fbcf6078-3077-4625-a420-f09bae952dfc	cogito:8b	cogito	8b	8	4.582803647965193	5.499364377558231	7.1491736908257	128000	\N	["reasoning", "vision"]	reasoning	fast	\N	Cogito:8B is a high-quality reasoning model leveraging a 128k token context window and 8.0B parameters, demonstrating superior performance on benchmarks compared to comparable open models. Key capabilities include advanced reasoning and vision/image understanding, complemented by a specialization focused on complex reasoning tasks. While prioritizing reasoning accuracy, the model’s large context window may introduce latency considerations relative to more general-purpose models.	\N	\N	t	t	[]	https://ollama.com/library/cogito	2025-12-28 04:47:32.303381	t	2025-12-25 07:29:52.341379	2025-12-28 04:47:32.942612
dce3801c-bce9-46c6-a20f-3b486f2a56a2	falcon:40b	falcon	40b	40	22.17335907649249	26.608030891790985	34.59044015932828	40000	\N	["chat"]	general_purpose	slow	\N	falcon:40b is a large language model optimized for conversational AI, leveraging a 40B parameter architecture and 40,000 token context window to deliver high-quality chat responses. Key capabilities include text generation, summarization, and interactive dialogue, supported by a general-purpose specialization.  While offering strong performance, inference speed may be impacted relative to smaller models due to its substantial size.	\N	\N	f	f	[]	https://ollama.com/library/falcon	2025-12-28 04:47:32.944634	t	2025-12-25 07:29:52.969395	2025-12-28 04:47:33.445281
\.


--
-- Data for Name: quest_tasks; Type: TABLE DATA; Schema: public; Owner: memos_user
--

COPY public.quest_tasks (id, quest_id, title, description, order_index, is_required, verification_data, created_at) FROM stdin;
b1f9d7ec-9812-465c-836b-b98cd1dea2c2	00cbeaf5-b4d2-4db7-9788-dd9a6d55cf82	Write down three things you're grateful for	Take a moment to reflect on positive aspects of your life	0	t	{}	2025-07-13 06:24:33.623311
24abf577-0ab8-4830-966f-44149383db90	49855915-b148-4248-88a4-3c4cc8c927a1	Rate your mood on a scale of 1-10	\N	0	t	{}	2025-07-13 06:24:33.630531
5765be6e-c832-4b5a-9f04-82c74fee5331	49855915-b148-4248-88a4-3c4cc8c927a1	Journal about your day	Write at least 3 sentences about your recovery journey today	1	t	{}	2025-07-13 06:24:33.630534
bd0760c1-7754-438b-8e89-a385a35599c6	49855915-b148-4248-88a4-3c4cc8c927a1	Set an intention for tomorrow	\N	2	f	{}	2025-07-13 06:24:33.630535
94ae55a1-d8f7-4f27-9823-893e2da5da7e	dafdc555-a8f0-4c0d-a271-d50c7752287f	Plan a healthy meal	\N	0	t	{}	2025-07-13 06:24:33.636356
1c6eb437-f9ef-4885-85df-74f063ef407d	dafdc555-a8f0-4c0d-a271-d50c7752287f	Shop for ingredients	\N	1	t	{}	2025-07-13 06:24:33.636359
eaccb659-456d-4397-baff-8bca5dad77c4	dafdc555-a8f0-4c0d-a271-d50c7752287f	Prepare and enjoy the meal	Take a photo of your prepared meal	2	t	{"requires_photo": true}	2025-07-13 06:24:33.636361
baada657-d180-48ad-8ba1-70d8837fbc71	cbaa99da-a785-4e2c-bf56-0511b0f2fc50	Find a meeting to attend	Use the app to locate a meeting near you	0	t	{}	2025-07-13 06:24:33.64123
18d5e4e9-9066-463c-812c-a67054d033f2	cbaa99da-a785-4e2c-bf56-0511b0f2fc50	Attend the meeting	\N	1	t	{}	2025-07-13 06:24:33.641233
461fd938-707e-4097-ba80-adb6e05fab34	cbaa99da-a785-4e2c-bf56-0511b0f2fc50	Share during the meeting	Optional: Share your experience if comfortable	2	f	{}	2025-07-13 06:24:33.641235
fca233d3-581d-4358-9e48-b1a7eed0cb4d	cbaa99da-a785-4e2c-bf56-0511b0f2fc50	Reflect on the experience	Journal about what you learned	3	t	{}	2025-07-13 06:24:33.641236
d875c7da-2d69-4fae-b93c-cc8777fd0e50	62cee89c-f94b-41d3-8250-2ab236dbe883	Day 1: 20 minutes of physical activity	Walking, running, yoga, or any exercise	0	t	{}	2025-07-13 06:24:33.645891
41d651f9-3073-481a-afeb-a9762f292baf	62cee89c-f94b-41d3-8250-2ab236dbe883	Day 3: 20 minutes of physical activity	\N	1	t	{}	2025-07-13 06:24:33.645894
8ebf035d-b042-46a4-af96-057fa08cb4cd	62cee89c-f94b-41d3-8250-2ab236dbe883	Day 5: 20 minutes of physical activity	\N	2	t	{}	2025-07-13 06:24:33.645895
9424441b-72a3-4eb5-832c-a22a8feedd38	62cee89c-f94b-41d3-8250-2ab236dbe883	Bonus: Try a new type of exercise	\N	3	f	{}	2025-07-13 06:24:33.645897
ab8a3295-422a-47da-b344-f642b33302b0	02be82ec-0b91-4392-bb39-c8c4f5af21cb	Reflect on your journey	Write about your experience over the past 30 days	0	t	{}	2025-07-13 06:24:33.650758
e231bb92-ebc1-40f2-8185-af0faf5fbed5	02be82ec-0b91-4392-bb39-c8c4f5af21cb	Share with someone you trust	Tell someone about your achievement	1	t	{}	2025-07-13 06:24:33.650766
39361085-6659-47cd-a303-0ffca17d17e5	02be82ec-0b91-4392-bb39-c8c4f5af21cb	Reward yourself	Do something special (non-substance related) to celebrate	2	t	{}	2025-07-13 06:24:33.65077
423459b7-a6b3-4f5a-97d9-3b0ea00c1d2d	fc97d70f-b2e4-44c5-912b-05cca940908f	Write a letter to your past self	\N	0	t	{}	2025-07-13 06:24:33.656809
0ba301fd-34d0-49b2-b94f-07d8171673b6	fc97d70f-b2e4-44c5-912b-05cca940908f	Create a recovery timeline	Document your journey with key moments	1	t	{}	2025-07-13 06:24:33.656814
070f0192-2d6b-4ab4-b708-5706edebd997	fc97d70f-b2e4-44c5-912b-05cca940908f	Plan your next 90 days	\N	2	t	{}	2025-07-13 06:24:33.656817
0e7a82d5-874f-46a4-a615-108ce3f574b8	dede3ef1-fb76-42eb-8b59-d96f43dadbc9	List all sources of income	\N	0	t	{}	2025-07-13 06:24:33.662512
300dac0d-fbc5-4c56-9cce-b3750e66d53a	dede3ef1-fb76-42eb-8b59-d96f43dadbc9	Track expenses for one week	\N	1	t	{}	2025-07-13 06:24:33.662516
19706374-e72d-4470-8883-0cdf781d0b17	dede3ef1-fb76-42eb-8b59-d96f43dadbc9	Create categories for spending	\N	2	t	{}	2025-07-13 06:24:33.662518
19fa7e3a-b492-4e45-a091-78863de2ee2f	dede3ef1-fb76-42eb-8b59-d96f43dadbc9	Set one savings goal	\N	3	t	{}	2025-07-13 06:24:33.662519
bed5ed12-b84a-4c10-8d60-db2e7f112230	c2f903e4-5466-4eca-88e8-78f2ca2bc314	Update or create your resume	\N	0	t	{}	2025-07-13 06:24:33.667437
59c23eab-7d89-432d-980f-844061276e7f	c2f903e4-5466-4eca-88e8-78f2ca2bc314	Find 3 job openings that interest you	\N	1	t	{}	2025-07-13 06:24:33.667441
1eaee7bc-a9c0-4a1a-9ce6-5dae94244df8	c2f903e4-5466-4eca-88e8-78f2ca2bc314	Submit at least one application	\N	2	t	{}	2025-07-13 06:24:33.667443
e445b604-6e4d-44c9-985e-51e733d5ce50	c2f903e4-5466-4eca-88e8-78f2ca2bc314	Practice interview questions	\N	3	f	{}	2025-07-13 06:24:33.667446
40394e55-f2d8-4340-9f3c-a5bf4be63371	d06e2e3b-be69-4978-972f-83bb4522499a	Reach out to someone from your meeting	\N	0	t	{}	2025-07-13 06:24:33.672004
d0324df0-cb70-4b31-9b60-a573c37b0949	d06e2e3b-be69-4978-972f-83bb4522499a	Exchange contact information	\N	1	t	{}	2025-07-13 06:24:33.672008
607cd933-ecbf-4707-8c87-977dc02570ac	d06e2e3b-be69-4978-972f-83bb4522499a	Check in with them this week	\N	2	t	{}	2025-07-13 06:24:33.672011
92aca466-1177-4ea8-a8b1-736defff94e4	5f76c756-0e9d-4d6a-a7cc-a1b61eee5270	Volunteer for a service position	Greeter, coffee maker, or cleanup	0	t	{}	2025-07-13 06:24:33.677392
99e8de99-b479-4359-b318-a029f2e84165	5f76c756-0e9d-4d6a-a7cc-a1b61eee5270	Complete your service commitment	\N	1	t	{}	2025-07-13 06:24:33.677396
e940b55d-9a54-48a5-b9fc-8a3365c0ed86	5f76c756-0e9d-4d6a-a7cc-a1b61eee5270	Reflect on the experience	\N	2	t	{}	2025-07-13 06:24:33.677399
6d8d9e6a-59b7-4b24-8cc9-cb8f31948ed9	98cc47da-7942-4744-b891-86f802ae198b	List warning signs of crisis	\N	0	t	{}	2025-07-13 06:24:33.690017
4df9048f-2c2c-42de-9330-ace9f6bc22a3	98cc47da-7942-4744-b891-86f802ae198b	Identify coping strategies	\N	1	t	{}	2025-07-13 06:24:33.690026
b6d9aaf1-8237-4d42-acba-bd2c8f151b6f	98cc47da-7942-4744-b891-86f802ae198b	List emergency contacts	Include sponsor, counselor, crisis hotline	2	t	{}	2025-07-13 06:24:33.690032
c8aba8f7-c5db-4c16-8da5-d5b9c017f460	98cc47da-7942-4744-b891-86f802ae198b	Share plan with trusted person	\N	3	t	{}	2025-07-13 06:24:33.690037
6f512dd2-47cf-43e6-9907-1753df2bb8c7	ff5c0560-dbc0-4b8a-99eb-86389f02ba5b	Day 1: 5-minute breathing exercise	\N	0	t	{}	2025-07-13 06:24:33.700457
55113173-b130-4f61-9ab2-77f8371e3eb7	ff5c0560-dbc0-4b8a-99eb-86389f02ba5b	Day 2: Body scan meditation	\N	1	t	{}	2025-07-13 06:24:33.700467
8bcfb97d-b2ae-490b-b822-bfe99068299e	ff5c0560-dbc0-4b8a-99eb-86389f02ba5b	Day 3: Mindful walking	\N	2	t	{}	2025-07-13 06:24:33.700472
7db53cb1-f2ff-4470-b82f-9714f03ffc13	ff5c0560-dbc0-4b8a-99eb-86389f02ba5b	Day 4: Loving-kindness meditation	\N	3	t	{}	2025-07-13 06:24:33.700477
f24e9aed-b11c-433e-9f3e-bc4444921a03	ff5c0560-dbc0-4b8a-99eb-86389f02ba5b	Day 5: Mindful eating	\N	4	t	{}	2025-07-13 06:24:33.700482
2b5b2461-4dd0-4b3b-9cc4-25af548e1e09	ff5c0560-dbc0-4b8a-99eb-86389f02ba5b	Day 6: Gratitude meditation	\N	5	t	{}	2025-07-13 06:24:33.700487
b19ecbf9-eacb-4598-90cc-648a7f639549	ff5c0560-dbc0-4b8a-99eb-86389f02ba5b	Day 7: Choose your favorite practice	\N	6	t	{}	2025-07-13 06:24:33.700492
0fcc8def-c4a5-4e9a-8e3d-63e5a82057c4	af0c12be-c0c2-43dd-a92b-d18b6c0cecac	Set consistent sleep/wake times	\N	0	t	{}	2025-07-13 06:24:33.708854
62a97da2-2eb0-4844-859f-01e861648408	af0c12be-c0c2-43dd-a92b-d18b6c0cecac	Create bedtime routine	No screens 30 minutes before bed	1	t	{}	2025-07-13 06:24:33.708859
c21cf74a-e1f2-41ed-b4da-43587680c6df	af0c12be-c0c2-43dd-a92b-d18b6c0cecac	Track sleep quality for 7 nights	\N	2	t	{}	2025-07-13 06:24:33.708861
0f336113-5e43-40c7-9faf-353a32bf175c	af0c12be-c0c2-43dd-a92b-d18b6c0cecac	Adjust environment	Temperature, darkness, comfort	3	f	{}	2025-07-13 06:24:33.708862
fe5fcd4a-a0b2-494b-af89-9fd2715ed097	3d5503a2-8f6d-4444-9730-d41c981cb9a5	Try a new spiritual practice	Prayer, meditation, nature walk, etc.	0	t	{}	2025-07-13 06:24:33.715997
94a6dc72-bbdf-4bd7-a7d3-baf9e38cb539	3d5503a2-8f6d-4444-9730-d41c981cb9a5	Read spiritual literature	Any text that inspires you	1	t	{}	2025-07-13 06:24:33.716004
1cf81fc6-1da0-4ed5-807d-141f1b6387c2	3d5503a2-8f6d-4444-9730-d41c981cb9a5	Reflect on your spiritual journey	\N	2	t	{}	2025-07-13 06:24:33.716007
ae9f35c1-2bdf-4b19-8da8-a2e7e1f8c28c	68b075a9-b58f-4694-85d0-6ca9684d39f2	List resentments you're holding	\N	0	t	{}	2025-07-13 06:24:33.725502
46466cf8-210a-4876-a6da-7a60d011462f	68b075a9-b58f-4694-85d0-6ca9684d39f2	Write a forgiveness letter (don't send)	\N	1	t	{}	2025-07-13 06:24:33.725511
88c10fcb-5031-44ac-9c38-face986a9903	68b075a9-b58f-4694-85d0-6ca9684d39f2	Practice self-forgiveness meditation	\N	2	t	{}	2025-07-13 06:24:33.725517
dcb8eb27-5a47-4907-92d1-f25c17b52dbf	68b075a9-b58f-4694-85d0-6ca9684d39f2	Share with sponsor or counselor	\N	3	f	{}	2025-07-13 06:24:33.725521
\.


--
-- Data for Name: quests; Type: TABLE DATA; Schema: public; Owner: memos_user
--

COPY public.quests (id, title, description, category, points, min_recovery_stage, max_active_days, cooldown_hours, prerequisites, verification_type, quest_metadata, is_active, created_at, updated_at) FROM stdin;
00cbeaf5-b4d2-4db7-9788-dd9a6d55cf82	Morning Gratitude	Start your day with gratitude by listing three things you're thankful for	daily	50	\N	1	20	[]	self_report	{"time_of_day": "morning", "duration_minutes": 5}	t	2025-07-13 06:24:33.619391	2025-07-13 06:24:33.619402
49855915-b148-4248-88a4-3c4cc8c927a1	Daily Check-In	Complete your daily recovery check-in and mood assessment	daily	75	\N	1	20	[]	self_report	{"includes_mood_tracking": true}	t	2025-07-13 06:24:33.629472	2025-07-13 06:24:33.629475
dafdc555-a8f0-4c0d-a271-d50c7752287f	Healthy Meal Planning	Plan and prepare a nutritious meal	daily	60	\N	1	12	[]	photo_evidence	{}	t	2025-07-13 06:24:33.635417	2025-07-13 06:24:33.635422
cbaa99da-a785-4e2c-bf56-0511b0f2fc50	Meeting Attendance	Attend a recovery support meeting this week	weekly	150	\N	7	144	[]	self_report	{"meeting_types": ["AA", "NA", "SMART Recovery", "Other"]}	t	2025-07-13 06:24:33.640409	2025-07-13 06:24:33.640412
62cee89c-f94b-41d3-8250-2ab236dbe883	Physical Wellness Week	Complete physical activities 3 times this week	weekly	200	\N	7	144	[]	self_report	{}	t	2025-07-13 06:24:33.645139	2025-07-13 06:24:33.645142
02be82ec-0b91-4392-bb39-c8c4f5af21cb	30 Days of Recovery	Celebrate reaching 30 days in recovery	milestone	500	30_days	\N	0	[]	auto_verify	{"milestone_days": 30, "celebration": true}	t	2025-07-13 06:24:33.649898	2025-07-13 06:24:33.649901
fc97d70f-b2e4-44c5-912b-05cca940908f	90 Days Strong	Commemorate 90 days of recovery	milestone	1000	90_days	\N	0	[]	auto_verify	{"milestone_days": 90}	t	2025-07-13 06:24:33.655789	2025-07-13 06:24:33.655794
dede3ef1-fb76-42eb-8b59-d96f43dadbc9	Budget Basics	Create a simple monthly budget	life_skills	150	\N	\N	0	[]	self_report	{}	t	2025-07-13 06:24:33.661688	2025-07-13 06:24:33.661692
c2f903e4-5466-4eca-88e8-78f2ca2bc314	Job Application Workshop	Prepare and submit a job application	life_skills	250	\N	\N	0	[]	self_report	{"career_focused": true}	t	2025-07-13 06:24:33.666608	2025-07-13 06:24:33.666611
d06e2e3b-be69-4978-972f-83bb4522499a	Recovery Buddy	Connect with another person in recovery	community	100	\N	\N	0	[]	self_report	{}	t	2025-07-13 06:24:33.67118	2025-07-13 06:24:33.671183
5f76c756-0e9d-4d6a-a7cc-a1b61eee5270	Service Work	Volunteer to help at a recovery meeting or event	community	200	\N	\N	0	[]	self_report	{"service_type": "meeting_support"}	t	2025-07-13 06:24:33.676559	2025-07-13 06:24:33.676563
98cc47da-7942-4744-b891-86f802ae198b	Crisis Safety Plan	Create a personal crisis intervention plan	emergency	300	\N	\N	0	[]	self_report	{"crisis_preparedness": true}	t	2025-07-13 06:24:33.688115	2025-07-13 06:24:33.688123
ff5c0560-dbc0-4b8a-99eb-86389f02ba5b	Mindfulness Journey	Practice mindfulness meditation for 7 days	wellness	175	\N	10	0	[]	self_report	{}	t	2025-07-13 06:24:33.698387	2025-07-13 06:24:33.698395
af0c12be-c0c2-43dd-a92b-d18b6c0cecac	Sleep Hygiene Challenge	Improve your sleep habits over one week	wellness	150	\N	7	0	[]	self_report	{}	t	2025-07-13 06:24:33.707846	2025-07-13 06:24:33.70785
3d5503a2-8f6d-4444-9730-d41c981cb9a5	Spiritual Exploration	Explore spiritual practices that resonate with you	spiritual	125	\N	\N	0	[]	self_report	{}	t	2025-07-13 06:24:33.714541	2025-07-13 06:24:33.714547
68b075a9-b58f-4694-85d0-6ca9684d39f2	Forgiveness Practice	Work on forgiveness - of self and others	spiritual	200	\N	\N	0	[]	self_report	{"emotional_intensity": "high"}	t	2025-07-13 06:24:33.723677	2025-07-13 06:24:33.723685
\.


--
-- Data for Name: user_achievements; Type: TABLE DATA; Schema: public; Owner: memos_user
--

COPY public.user_achievements (id, user_id, achievement_id, earned_at, points_awarded) FROM stdin;
\.


--
-- Data for Name: user_memory_settings; Type: TABLE DATA; Schema: public; Owner: memos_user
--

COPY public.user_memory_settings (user_id, memory_enabled, max_memories, retention_days, default_privacy_level, auto_consent, allow_clinical_memories, allow_crisis_detection, recovery_stage, therapy_goals, crisis_contacts, retrieval_depth, semantic_threshold, include_low_relevance, allow_care_team_access, care_team_members, family_sharing_enabled, offline_sync_enabled, push_notifications, memory_insights_enabled, settings_version, last_consent_date, consent_document_version, created_at, updated_at, last_accessed) FROM stdin;
\.


--
-- Data for Name: user_quest_stats; Type: TABLE DATA; Schema: public; Owner: memos_user
--

COPY public.user_quest_stats (user_id, total_points, current_streak_days, longest_streak_days, last_activity_date, total_quests_completed, level, weekly_points, monthly_points, created_at, updated_at) FROM stdin;
test_user_361765a0	0	0	0	\N	0	newcomer	0	0	2025-07-13 16:05:07.24757	2025-07-13 16:05:07.247577
test_user_ce12432d	0	0	0	\N	0	newcomer	0	0	2025-07-13 16:12:00.628041	2025-07-13 16:12:00.62805
test_user	0	0	0	\N	0	newcomer	0	0	2025-07-13 16:12:44.981337	2025-07-13 16:12:44.981346
test_user_123	0	0	0	\N	0	newcomer	0	0	2025-07-13 16:36:14.931694	2025-07-13 16:37:38.89306
test_user_888	0	0	0	\N	0	newcomer	0	0	2025-07-13 18:09:06.797164	2025-07-13 18:09:06.796111
test_user_999	150	1	1	2025-07-13	1	seeker	150	150	2025-07-13 18:10:37.457306	2025-07-13 18:13:28.074878
testuser123	0	0	0	\N	0	newcomer	0	0	2025-07-16 03:34:42.310427	2025-07-16 03:34:42.310448
cafdf613	0	0	0	\N	0	newcomer	0	0	2025-07-16 04:10:00.463091	2025-07-16 04:10:53.246682
sparkone	0	0	0	\N	0	newcomer	0	0	2025-07-15 00:45:29.898285	2025-07-16 04:38:17.185266
cafdf613-1101-4ec4-802e-19e533236e89	0	0	0	\N	0	newcomer	0	0	2025-07-16 03:55:55.942899	2025-07-16 16:49:10.933977
\.


--
-- Data for Name: user_quests; Type: TABLE DATA; Schema: public; Owner: memos_user
--

COPY public.user_quests (id, user_id, quest_id, state, started_at, completed_at, verified_at, verified_by, progress_data, points_earned) FROM stdin;
8730cc5d-2096-41bb-8298-6cc7ede2b397	test_user_123	00cbeaf5-b4d2-4db7-9788-dd9a6d55cf82	assigned	2025-07-13 16:36:14.922854	\N	\N	\N	{}	0
25641190-fc56-40ce-8af1-075a3e68ebc9	test_user_123	49855915-b148-4248-88a4-3c4cc8c927a1	assigned	2025-07-13 16:37:38.880347	\N	\N	\N	{}	0
e582667c-9fe1-4d47-bc1a-9407c68f8113	test_user_888	dafdc555-a8f0-4c0d-a271-d50c7752287f	assigned	2025-07-13 18:09:06.786052	\N	\N	\N	{}	0
e347eb44-6dd0-4dfe-a8f2-4a18378c3b2b	test_user_999	cbaa99da-a785-4e2c-bf56-0511b0f2fc50	completed	2025-07-13 18:10:37.446404	2025-07-13 18:13:28.064621	\N	\N	{}	150
79d3c029-0200-44c9-b564-ad46f6e65684	sparkone	00cbeaf5-b4d2-4db7-9788-dd9a6d55cf82	assigned	2025-07-15 00:45:29.879607	\N	\N	\N	{}	0
aaee6d22-e02a-4314-99ca-84ba83493aa5	cafdf613	00cbeaf5-b4d2-4db7-9788-dd9a6d55cf82	assigned	2025-07-16 04:10:36.161387	\N	\N	\N	{}	0
ee2a3523-68e3-4339-9898-8017c83c4ce9	cafdf613	49855915-b148-4248-88a4-3c4cc8c927a1	assigned	2025-07-16 04:10:53.238535	\N	\N	\N	{}	0
0a573d89-ba0a-4b7f-8f1b-5fdddea4e03c	cafdf613	dafdc555-a8f0-4c0d-a271-d50c7752287f	assigned	2025-07-16 04:10:53.239436	\N	\N	\N	{}	0
cdf0ef01-acde-41d8-b1e1-dea956844374	cafdf613-1101-4ec4-802e-19e533236e89	3d5503a2-8f6d-4444-9730-d41c981cb9a5	assigned	2025-07-16 04:35:06.554489	\N	\N	\N	{}	0
171959f4-2154-46e3-bf0d-db80ef78d862	sparkone	3d5503a2-8f6d-4444-9730-d41c981cb9a5	assigned	2025-07-16 04:38:17.178144	\N	\N	\N	{}	0
71aa4e46-a4f0-4c82-8007-7824bb6bdb8f	cafdf613-1101-4ec4-802e-19e533236e89	02be82ec-0b91-4392-bb39-c8c4f5af21cb	assigned	2025-07-16 05:19:30.227712	\N	\N	\N	{}	0
5b53a984-b3d9-4725-b0fe-a2aeb105c1a7	cafdf613-1101-4ec4-802e-19e533236e89	00cbeaf5-b4d2-4db7-9788-dd9a6d55cf82	assigned	2025-07-16 05:38:02.050056	\N	\N	\N	{}	0
ba24d491-c868-45b3-a33c-d94d6574df21	cafdf613-1101-4ec4-802e-19e533236e89	49855915-b148-4248-88a4-3c4cc8c927a1	assigned	2025-07-16 05:38:32.228239	\N	\N	\N	{}	0
741f0623-2e0c-4369-832a-8b651a46dde2	cafdf613-1101-4ec4-802e-19e533236e89	dafdc555-a8f0-4c0d-a271-d50c7752287f	assigned	2025-07-16 14:39:15.682084	\N	\N	\N	{}	0
9028991b-dbb3-4b5a-8543-1b793e4d2d62	cafdf613-1101-4ec4-802e-19e533236e89	cbaa99da-a785-4e2c-bf56-0511b0f2fc50	assigned	2025-07-16 15:52:23.926907	\N	\N	\N	{}	0
5b05fab5-860f-4740-8db8-4ff9f2c81faf	cafdf613-1101-4ec4-802e-19e533236e89	62cee89c-f94b-41d3-8250-2ab236dbe883	assigned	2025-07-16 16:26:04.839935	\N	\N	\N	{}	0
14fc62be-740f-4760-9d1b-43653e0af8f0	cafdf613-1101-4ec4-802e-19e533236e89	fc97d70f-b2e4-44c5-912b-05cca940908f	assigned	2025-07-16 16:26:26.700993	\N	\N	\N	{}	0
c89ddec4-c817-482f-abb6-c4fd882b77ef	cafdf613-1101-4ec4-802e-19e533236e89	dede3ef1-fb76-42eb-8b59-d96f43dadbc9	assigned	2025-07-16 16:33:13.315782	\N	\N	\N	{}	0
58854169-8c7f-45f5-81ed-30f49fb1d800	cafdf613-1101-4ec4-802e-19e533236e89	c2f903e4-5466-4eca-88e8-78f2ca2bc314	assigned	2025-07-16 16:35:14.154574	\N	\N	\N	{}	0
26ba2d18-38e9-48de-a20e-389b37daccf9	cafdf613-1101-4ec4-802e-19e533236e89	d06e2e3b-be69-4978-972f-83bb4522499a	assigned	2025-07-16 16:49:10.925785	\N	\N	\N	{}	0
\.


--
-- Data for Name: user_tasks; Type: TABLE DATA; Schema: public; Owner: memos_user
--

COPY public.user_tasks (id, user_quest_id, task_id, user_id, state, completed_at, evidence_data, created_at) FROM stdin;
ddd2e7c8-2168-43a8-a9fb-7d518f2839cf	\N	b1f9d7ec-9812-465c-836b-b98cd1dea2c2	test_user_123	pending	\N	{}	2025-07-13 16:36:14.927432
74476054-4e4f-436a-b888-663317772b87	\N	24abf577-0ab8-4830-966f-44149383db90	test_user_123	pending	\N	{}	2025-07-13 16:37:38.886826
63c33618-b0c2-4b59-bd79-7a7e360264e2	\N	5765be6e-c832-4b5a-9f04-82c74fee5331	test_user_123	pending	\N	{}	2025-07-13 16:37:38.886848
0c2888bd-96a9-4bb4-abe1-ffb9c34f32ca	\N	bd0760c1-7754-438b-8e89-a385a35599c6	test_user_123	pending	\N	{}	2025-07-13 16:37:38.886855
0f7f2707-6248-4283-82a3-5db2a84cccff	\N	94ae55a1-d8f7-4f27-9823-893e2da5da7e	test_user_888	pending	\N	{}	2025-07-13 18:09:06.792378
b9d0bed5-1a37-49c5-82b7-8aeb80d71c4a	\N	1c6eb437-f9ef-4885-85df-74f063ef407d	test_user_888	pending	\N	{}	2025-07-13 18:09:06.792398
509ad305-b1ce-423b-b984-e1a4e50eeca6	\N	eaccb659-456d-4397-baff-8bca5dad77c4	test_user_888	pending	\N	{}	2025-07-13 18:09:06.792406
48c593d4-4506-499c-a77f-d4e7e623467f	e347eb44-6dd0-4dfe-a8f2-4a18378c3b2b	baada657-d180-48ad-8ba1-70d8837fbc71	test_user_999	completed	2025-07-13 18:10:58.674214	{"evidence": {"note": "I found a local AA meeting and attended it today"}}	2025-07-13 18:10:37.452289
49d0ada6-cd97-4111-aab8-2f4d2dae1446	e347eb44-6dd0-4dfe-a8f2-4a18378c3b2b	18d5e4e9-9066-463c-812c-a67054d033f2	test_user_999	completed	2025-07-13 18:13:23.207151	{"evidence": {"note": "Found a sponsor at the meeting"}}	2025-07-13 18:10:37.452307
57f7e3dc-d3cf-4f89-aa17-6d0b4dac0302	e347eb44-6dd0-4dfe-a8f2-4a18378c3b2b	461fd938-707e-4097-ba80-adb6e05fab34	test_user_999	completed	2025-07-13 18:13:25.757704	{"evidence": {"note": "Shared my experience with the group"}}	2025-07-13 18:10:37.452312
006b4705-1f04-49c5-8951-e45ce72472e4	e347eb44-6dd0-4dfe-a8f2-4a18378c3b2b	fca233d3-581d-4358-9e48-b1a7eed0cb4d	test_user_999	completed	2025-07-13 18:13:28.064501	{"evidence": {"note": "Committed to attend next weeks meeting"}}	2025-07-13 18:10:37.452316
bd275409-9512-4da0-a24e-a908853b8c0d	79d3c029-0200-44c9-b564-ad46f6e65684	b1f9d7ec-9812-465c-836b-b98cd1dea2c2	sparkone	pending	\N	{}	2025-07-15 00:45:29.890962
bb52ed03-bfba-4c8a-aefb-b525d9aaf289	aaee6d22-e02a-4314-99ca-84ba83493aa5	b1f9d7ec-9812-465c-836b-b98cd1dea2c2	cafdf613	pending	\N	{}	2025-07-16 04:10:36.166382
3a3fdd8b-1aee-48e7-a4bf-26b471433964	ee2a3523-68e3-4339-9898-8017c83c4ce9	24abf577-0ab8-4830-966f-44149383db90	cafdf613	pending	\N	{}	2025-07-16 04:10:53.241569
2a233e6f-d172-483c-8d21-c27bd685bb36	ee2a3523-68e3-4339-9898-8017c83c4ce9	5765be6e-c832-4b5a-9f04-82c74fee5331	cafdf613	pending	\N	{}	2025-07-16 04:10:53.241579
2ddc51fa-d5b1-4264-910c-09b956afb681	ee2a3523-68e3-4339-9898-8017c83c4ce9	bd0760c1-7754-438b-8e89-a385a35599c6	cafdf613	pending	\N	{}	2025-07-16 04:10:53.241587
bb5a16da-e533-451d-8f00-39d6f9470c1a	0a573d89-ba0a-4b7f-8f1b-5fdddea4e03c	94ae55a1-d8f7-4f27-9823-893e2da5da7e	cafdf613	pending	\N	{}	2025-07-16 04:10:53.243854
fbeb915b-acc1-4312-9a80-b95fdab20567	0a573d89-ba0a-4b7f-8f1b-5fdddea4e03c	1c6eb437-f9ef-4885-85df-74f063ef407d	cafdf613	pending	\N	{}	2025-07-16 04:10:53.24386
f49fe10a-c1c7-46d2-b130-f2017b0d56f8	0a573d89-ba0a-4b7f-8f1b-5fdddea4e03c	eaccb659-456d-4397-baff-8bca5dad77c4	cafdf613	pending	\N	{}	2025-07-16 04:10:53.243864
96170ce5-e114-4086-a81c-5bcdb37eefdf	cdf0ef01-acde-41d8-b1e1-dea956844374	fe5fcd4a-a0b2-494b-af89-9fd2715ed097	cafdf613-1101-4ec4-802e-19e533236e89	pending	\N	{}	2025-07-16 04:35:06.558164
87dc563d-559d-434a-a8b1-a24435ba7799	cdf0ef01-acde-41d8-b1e1-dea956844374	94a6dc72-bbdf-4bd7-a7d3-baf9e38cb539	cafdf613-1101-4ec4-802e-19e533236e89	pending	\N	{}	2025-07-16 04:35:06.558174
6ed4101e-325c-4b8d-a4dd-a53daefacc19	cdf0ef01-acde-41d8-b1e1-dea956844374	1cf81fc6-1da0-4ed5-807d-141f1b6387c2	cafdf613-1101-4ec4-802e-19e533236e89	pending	\N	{}	2025-07-16 04:35:06.558181
7eeb3479-67d4-4c16-b112-bf998050222d	171959f4-2154-46e3-bf0d-db80ef78d862	fe5fcd4a-a0b2-494b-af89-9fd2715ed097	sparkone	pending	\N	{}	2025-07-16 04:38:17.182145
c14dce56-8c82-4ef0-95c2-a998539ab49d	171959f4-2154-46e3-bf0d-db80ef78d862	94a6dc72-bbdf-4bd7-a7d3-baf9e38cb539	sparkone	pending	\N	{}	2025-07-16 04:38:17.182163
b567545d-03d4-4336-a606-328e4015d7d2	171959f4-2154-46e3-bf0d-db80ef78d862	1cf81fc6-1da0-4ed5-807d-141f1b6387c2	sparkone	pending	\N	{}	2025-07-16 04:38:17.182175
f55b503c-8f6e-4ea6-bf9f-2748423efa08	71aa4e46-a4f0-4c82-8007-7824bb6bdb8f	ab8a3295-422a-47da-b344-f642b33302b0	cafdf613-1101-4ec4-802e-19e533236e89	pending	\N	{}	2025-07-16 05:19:30.231424
71ac3256-2607-4abe-ac37-7d01141f8c6f	71aa4e46-a4f0-4c82-8007-7824bb6bdb8f	e231bb92-ebc1-40f2-8185-af0faf5fbed5	cafdf613-1101-4ec4-802e-19e533236e89	pending	\N	{}	2025-07-16 05:19:30.231441
ba504f2f-fc4f-4009-8716-d55ebd212aa9	71aa4e46-a4f0-4c82-8007-7824bb6bdb8f	39361085-6659-47cd-a303-0ffca17d17e5	cafdf613-1101-4ec4-802e-19e533236e89	pending	\N	{}	2025-07-16 05:19:30.231452
818681ac-f613-471c-96a4-ae0eed37f310	5b53a984-b3d9-4725-b0fe-a2aeb105c1a7	b1f9d7ec-9812-465c-836b-b98cd1dea2c2	cafdf613-1101-4ec4-802e-19e533236e89	pending	\N	{}	2025-07-16 05:38:02.052583
d06dd861-9500-4c22-bf88-99a38ee987f8	ba24d491-c868-45b3-a33c-d94d6574df21	24abf577-0ab8-4830-966f-44149383db90	cafdf613-1101-4ec4-802e-19e533236e89	pending	\N	{}	2025-07-16 05:38:32.230558
3b3f19b5-d9db-46f7-8233-cab56afee579	ba24d491-c868-45b3-a33c-d94d6574df21	5765be6e-c832-4b5a-9f04-82c74fee5331	cafdf613-1101-4ec4-802e-19e533236e89	pending	\N	{}	2025-07-16 05:38:32.230567
adaad66e-7096-425f-8069-256830cf3704	ba24d491-c868-45b3-a33c-d94d6574df21	bd0760c1-7754-438b-8e89-a385a35599c6	cafdf613-1101-4ec4-802e-19e533236e89	pending	\N	{}	2025-07-16 05:38:32.230573
6378807e-9645-4f0d-8171-e6f669815436	741f0623-2e0c-4369-832a-8b651a46dde2	94ae55a1-d8f7-4f27-9823-893e2da5da7e	cafdf613-1101-4ec4-802e-19e533236e89	pending	\N	{}	2025-07-16 14:39:15.684134
7f8daf88-9bc6-4a27-b74b-b3fd645bd79b	741f0623-2e0c-4369-832a-8b651a46dde2	1c6eb437-f9ef-4885-85df-74f063ef407d	cafdf613-1101-4ec4-802e-19e533236e89	pending	\N	{}	2025-07-16 14:39:15.68414
07146fca-a6ab-4e25-ace0-0aad6dc40d81	741f0623-2e0c-4369-832a-8b651a46dde2	eaccb659-456d-4397-baff-8bca5dad77c4	cafdf613-1101-4ec4-802e-19e533236e89	pending	\N	{}	2025-07-16 14:39:15.684144
58b36970-2688-42ec-b1db-99b475a95149	9028991b-dbb3-4b5a-8543-1b793e4d2d62	baada657-d180-48ad-8ba1-70d8837fbc71	cafdf613-1101-4ec4-802e-19e533236e89	pending	\N	{}	2025-07-16 15:52:23.931829
dd81e5dc-5437-4c5d-b826-a18dab00e573	9028991b-dbb3-4b5a-8543-1b793e4d2d62	18d5e4e9-9066-463c-812c-a67054d033f2	cafdf613-1101-4ec4-802e-19e533236e89	pending	\N	{}	2025-07-16 15:52:23.931859
5524b409-49d5-4cd7-acc9-be71b076fcce	9028991b-dbb3-4b5a-8543-1b793e4d2d62	461fd938-707e-4097-ba80-adb6e05fab34	cafdf613-1101-4ec4-802e-19e533236e89	pending	\N	{}	2025-07-16 15:52:23.931884
4ec2b5bc-f4d9-4a1e-bffc-f53ac45fe055	9028991b-dbb3-4b5a-8543-1b793e4d2d62	fca233d3-581d-4358-9e48-b1a7eed0cb4d	cafdf613-1101-4ec4-802e-19e533236e89	pending	\N	{}	2025-07-16 15:52:23.931909
ddf143d4-d649-40cd-84c1-607df88a7322	5b05fab5-860f-4740-8db8-4ff9f2c81faf	d875c7da-2d69-4fae-b93c-cc8777fd0e50	cafdf613-1101-4ec4-802e-19e533236e89	pending	\N	{}	2025-07-16 16:26:04.843565
7ceefa5b-4724-4390-93cf-1ab550f3f208	5b05fab5-860f-4740-8db8-4ff9f2c81faf	41d651f9-3073-481a-afeb-a9762f292baf	cafdf613-1101-4ec4-802e-19e533236e89	pending	\N	{}	2025-07-16 16:26:04.843577
38a8c176-b4e9-430c-89b9-a8ff8b9afe4a	5b05fab5-860f-4740-8db8-4ff9f2c81faf	8ebf035d-b042-46a4-af96-057fa08cb4cd	cafdf613-1101-4ec4-802e-19e533236e89	pending	\N	{}	2025-07-16 16:26:04.843584
32152c43-715c-4558-8b7e-7179d039d42a	5b05fab5-860f-4740-8db8-4ff9f2c81faf	9424441b-72a3-4eb5-832c-a22a8feedd38	cafdf613-1101-4ec4-802e-19e533236e89	pending	\N	{}	2025-07-16 16:26:04.84359
ec053b50-4dde-40c8-92b4-f8a1eda30a66	14fc62be-740f-4760-9d1b-43653e0af8f0	423459b7-a6b3-4f5a-97d9-3b0ea00c1d2d	cafdf613-1101-4ec4-802e-19e533236e89	pending	\N	{}	2025-07-16 16:26:26.705222
c608d410-e5f3-456f-b396-bb9ba7b7d494	14fc62be-740f-4760-9d1b-43653e0af8f0	0ba301fd-34d0-49b2-b94f-07d8171673b6	cafdf613-1101-4ec4-802e-19e533236e89	pending	\N	{}	2025-07-16 16:26:26.705236
99edc50d-447a-4e7f-b3fe-ee684fa94c38	14fc62be-740f-4760-9d1b-43653e0af8f0	070f0192-2d6b-4ab4-b708-5706edebd997	cafdf613-1101-4ec4-802e-19e533236e89	pending	\N	{}	2025-07-16 16:26:26.705244
2d97a341-2516-45ba-8ba1-4fee8f942289	c89ddec4-c817-482f-abb6-c4fd882b77ef	0e7a82d5-874f-46a4-a615-108ce3f574b8	cafdf613-1101-4ec4-802e-19e533236e89	pending	\N	{}	2025-07-16 16:33:13.317364
3dac68e4-01d6-4123-942b-e9287820af52	c89ddec4-c817-482f-abb6-c4fd882b77ef	300dac0d-fbc5-4c56-9cce-b3750e66d53a	cafdf613-1101-4ec4-802e-19e533236e89	pending	\N	{}	2025-07-16 16:33:13.31737
0a967fb6-5df3-4c40-818b-a78ed557309e	c89ddec4-c817-482f-abb6-c4fd882b77ef	19706374-e72d-4470-8883-0cdf781d0b17	cafdf613-1101-4ec4-802e-19e533236e89	pending	\N	{}	2025-07-16 16:33:13.317373
5a18e056-50b0-4617-be7b-473358611dd4	c89ddec4-c817-482f-abb6-c4fd882b77ef	19fa7e3a-b492-4e45-a091-78863de2ee2f	cafdf613-1101-4ec4-802e-19e533236e89	pending	\N	{}	2025-07-16 16:33:13.317377
eeb863ea-0f3b-4ab0-8173-40c3a18e7e4e	58854169-8c7f-45f5-81ed-30f49fb1d800	bed5ed12-b84a-4c10-8d60-db2e7f112230	cafdf613-1101-4ec4-802e-19e533236e89	pending	\N	{}	2025-07-16 16:35:14.156521
a74c6d02-ef7f-45e8-b920-105f0c2f4a3c	58854169-8c7f-45f5-81ed-30f49fb1d800	59c23eab-7d89-432d-980f-844061276e7f	cafdf613-1101-4ec4-802e-19e533236e89	pending	\N	{}	2025-07-16 16:35:14.156531
d18a381a-3caf-46d8-988e-c9f223ecd54e	58854169-8c7f-45f5-81ed-30f49fb1d800	1eaee7bc-a9c0-4a1a-9ce6-5dae94244df8	cafdf613-1101-4ec4-802e-19e533236e89	pending	\N	{}	2025-07-16 16:35:14.156539
f25d7a19-fb11-418a-9746-2a1c619fd17e	58854169-8c7f-45f5-81ed-30f49fb1d800	e445b604-6e4d-44c9-985e-51e733d5ce50	cafdf613-1101-4ec4-802e-19e533236e89	pending	\N	{}	2025-07-16 16:35:14.156547
0e961285-5a6b-4ae2-8cf8-33d866caf05d	26ba2d18-38e9-48de-a20e-389b37daccf9	40394e55-f2d8-4340-9f3c-a5bf4be63371	cafdf613-1101-4ec4-802e-19e533236e89	pending	\N	{}	2025-07-16 16:49:10.930081
49880bfa-239e-47d1-8b05-f2124578f875	26ba2d18-38e9-48de-a20e-389b37daccf9	d0324df0-cb70-4b31-9b60-a573c37b0949	cafdf613-1101-4ec4-802e-19e533236e89	pending	\N	{}	2025-07-16 16:49:10.930095
b1f3e3a3-d7e4-4567-b34f-1a27d9a3f39c	26ba2d18-38e9-48de-a20e-389b37daccf9	607cd933-ecbf-4707-8c87-977dc02570ac	cafdf613-1101-4ec4-802e-19e533236e89	pending	\N	{}	2025-07-16 16:49:10.930103
\.


--
-- Name: achievements achievements_pkey; Type: CONSTRAINT; Schema: public; Owner: memos_user
--

ALTER TABLE ONLY public.achievements
    ADD CONSTRAINT achievements_pkey PRIMARY KEY (id);


--
-- Name: alembic_version alembic_version_pkc; Type: CONSTRAINT; Schema: public; Owner: memos_user
--

ALTER TABLE ONLY public.alembic_version
    ADD CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num);


--
-- Name: mem0 mem0_pkey; Type: CONSTRAINT; Schema: public; Owner: memos_user
--

ALTER TABLE ONLY public.mem0
    ADD CONSTRAINT mem0_pkey PRIMARY KEY (id);


--
-- Name: mem0migrations mem0migrations_pkey; Type: CONSTRAINT; Schema: public; Owner: memos_user
--

ALTER TABLE ONLY public.mem0migrations
    ADD CONSTRAINT mem0migrations_pkey PRIMARY KEY (id);


--
-- Name: memories memories_pkey; Type: CONSTRAINT; Schema: public; Owner: memos_user
--

ALTER TABLE ONLY public.memories
    ADD CONSTRAINT memories_pkey PRIMARY KEY (id);


--
-- Name: ollama_model_specs ollama_model_specs_pkey; Type: CONSTRAINT; Schema: public; Owner: memos_user
--

ALTER TABLE ONLY public.ollama_model_specs
    ADD CONSTRAINT ollama_model_specs_pkey PRIMARY KEY (id);


--
-- Name: quest_tasks quest_tasks_pkey; Type: CONSTRAINT; Schema: public; Owner: memos_user
--

ALTER TABLE ONLY public.quest_tasks
    ADD CONSTRAINT quest_tasks_pkey PRIMARY KEY (id);


--
-- Name: quests quests_pkey; Type: CONSTRAINT; Schema: public; Owner: memos_user
--

ALTER TABLE ONLY public.quests
    ADD CONSTRAINT quests_pkey PRIMARY KEY (id);


--
-- Name: user_achievements user_achievements_pkey; Type: CONSTRAINT; Schema: public; Owner: memos_user
--

ALTER TABLE ONLY public.user_achievements
    ADD CONSTRAINT user_achievements_pkey PRIMARY KEY (id);


--
-- Name: user_memory_settings user_memory_settings_pkey; Type: CONSTRAINT; Schema: public; Owner: memos_user
--

ALTER TABLE ONLY public.user_memory_settings
    ADD CONSTRAINT user_memory_settings_pkey PRIMARY KEY (user_id);


--
-- Name: user_quest_stats user_quest_stats_pkey; Type: CONSTRAINT; Schema: public; Owner: memos_user
--

ALTER TABLE ONLY public.user_quest_stats
    ADD CONSTRAINT user_quest_stats_pkey PRIMARY KEY (user_id);


--
-- Name: user_quests user_quests_pkey; Type: CONSTRAINT; Schema: public; Owner: memos_user
--

ALTER TABLE ONLY public.user_quests
    ADD CONSTRAINT user_quests_pkey PRIMARY KEY (id);


--
-- Name: user_tasks user_tasks_pkey; Type: CONSTRAINT; Schema: public; Owner: memos_user
--

ALTER TABLE ONLY public.user_tasks
    ADD CONSTRAINT user_tasks_pkey PRIMARY KEY (id);


--
-- Name: idx_memories_crisis_level; Type: INDEX; Schema: public; Owner: memos_user
--

CREATE INDEX idx_memories_crisis_level ON public.memories USING btree (crisis_level DESC) WHERE (crisis_level > (0.0)::double precision);


--
-- Name: idx_memories_recovery_stage; Type: INDEX; Schema: public; Owner: memos_user
--

CREATE INDEX idx_memories_recovery_stage ON public.memories USING btree (recovery_stage) WHERE (recovery_stage IS NOT NULL);


--
-- Name: idx_memories_therapeutic_relevance; Type: INDEX; Schema: public; Owner: memos_user
--

CREATE INDEX idx_memories_therapeutic_relevance ON public.memories USING btree (therapeutic_relevance DESC) WHERE (is_deleted = false);


--
-- Name: idx_memories_type_privacy; Type: INDEX; Schema: public; Owner: memos_user
--

CREATE INDEX idx_memories_type_privacy ON public.memories USING btree (memory_type, privacy_level) WHERE (is_deleted = false);


--
-- Name: idx_memories_user_created; Type: INDEX; Schema: public; Owner: memos_user
--

CREATE INDEX idx_memories_user_created ON public.memories USING btree (user_id, created_at DESC);


--
-- Name: idx_quests_category; Type: INDEX; Schema: public; Owner: memos_user
--

CREATE INDEX idx_quests_category ON public.quests USING btree (category) WHERE (is_active = true);


--
-- Name: idx_user_achievements_user; Type: INDEX; Schema: public; Owner: memos_user
--

CREATE INDEX idx_user_achievements_user ON public.user_achievements USING btree (user_id);


--
-- Name: idx_user_quest_stats_points; Type: INDEX; Schema: public; Owner: memos_user
--

CREATE INDEX idx_user_quest_stats_points ON public.user_quest_stats USING btree (total_points DESC);


--
-- Name: idx_user_quests_completed; Type: INDEX; Schema: public; Owner: memos_user
--

CREATE INDEX idx_user_quests_completed ON public.user_quests USING btree (completed_at) WHERE (completed_at IS NOT NULL);


--
-- Name: idx_user_quests_user_state; Type: INDEX; Schema: public; Owner: memos_user
--

CREATE INDEX idx_user_quests_user_state ON public.user_quests USING btree (user_id, state);


--
-- Name: idx_user_settings_enabled; Type: INDEX; Schema: public; Owner: memos_user
--

CREATE INDEX idx_user_settings_enabled ON public.user_memory_settings USING btree (memory_enabled, updated_at DESC);


--
-- Name: idx_user_tasks_state; Type: INDEX; Schema: public; Owner: memos_user
--

CREATE INDEX idx_user_tasks_state ON public.user_tasks USING btree (user_id, state);


--
-- Name: ix_memories_user_id; Type: INDEX; Schema: public; Owner: memos_user
--

CREATE INDEX ix_memories_user_id ON public.memories USING btree (user_id);


--
-- Name: ix_ollama_model_specs_base_model; Type: INDEX; Schema: public; Owner: memos_user
--

CREATE INDEX ix_ollama_model_specs_base_model ON public.ollama_model_specs USING btree (base_model);


--
-- Name: ix_ollama_model_specs_model_name; Type: INDEX; Schema: public; Owner: memos_user
--

CREATE UNIQUE INDEX ix_ollama_model_specs_model_name ON public.ollama_model_specs USING btree (model_name);


--
-- Name: ix_user_achievements_user_id; Type: INDEX; Schema: public; Owner: memos_user
--

CREATE INDEX ix_user_achievements_user_id ON public.user_achievements USING btree (user_id);


--
-- Name: ix_user_quests_user_id; Type: INDEX; Schema: public; Owner: memos_user
--

CREATE INDEX ix_user_quests_user_id ON public.user_quests USING btree (user_id);


--
-- Name: ix_user_tasks_user_id; Type: INDEX; Schema: public; Owner: memos_user
--

CREATE INDEX ix_user_tasks_user_id ON public.user_tasks USING btree (user_id);


--
-- Name: quest_tasks quest_tasks_quest_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: memos_user
--

ALTER TABLE ONLY public.quest_tasks
    ADD CONSTRAINT quest_tasks_quest_id_fkey FOREIGN KEY (quest_id) REFERENCES public.quests(id) ON DELETE CASCADE;


--
-- Name: user_achievements user_achievements_achievement_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: memos_user
--

ALTER TABLE ONLY public.user_achievements
    ADD CONSTRAINT user_achievements_achievement_id_fkey FOREIGN KEY (achievement_id) REFERENCES public.achievements(id);


--
-- Name: user_quests user_quests_quest_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: memos_user
--

ALTER TABLE ONLY public.user_quests
    ADD CONSTRAINT user_quests_quest_id_fkey FOREIGN KEY (quest_id) REFERENCES public.quests(id);


--
-- Name: user_tasks user_tasks_task_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: memos_user
--

ALTER TABLE ONLY public.user_tasks
    ADD CONSTRAINT user_tasks_task_id_fkey FOREIGN KEY (task_id) REFERENCES public.quest_tasks(id);


--
-- Name: user_tasks user_tasks_user_quest_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: memos_user
--

ALTER TABLE ONLY public.user_tasks
    ADD CONSTRAINT user_tasks_user_quest_id_fkey FOREIGN KEY (user_quest_id) REFERENCES public.user_quests(id) ON DELETE CASCADE;


--
-- Name: SCHEMA public; Type: ACL; Schema: -; Owner: pg_database_owner
--

GRANT ALL ON SCHEMA public TO memos_user;


--
-- PostgreSQL database dump complete
--

