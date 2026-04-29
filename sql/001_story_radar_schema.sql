BEGIN;

CREATE TABLE IF NOT EXISTS story_clusters (
    cluster_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    topics JSONB NOT NULL DEFAULT '[]'::jsonb,
    countries JSONB NOT NULL DEFAULT '[]'::jsonb,
    source_count INTEGER NOT NULL DEFAULT 1,
    document_count INTEGER NOT NULL DEFAULT 1,
    first_seen_at TIMESTAMPTZ NOT NULL,
    last_seen_at TIMESTAMPTZ NOT NULL,
    freshness_score DOUBLE PRECISION,
    novelty_score DOUBLE PRECISION,
    newsroom_labels JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cluster_entities (
    cluster_id TEXT NOT NULL REFERENCES story_clusters(cluster_id) ON DELETE CASCADE,
    entity_name TEXT NOT NULL,
    entity_type TEXT NOT NULL DEFAULT 'unknown',
    salience DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    PRIMARY KEY (cluster_id, entity_name)
);

CREATE TABLE IF NOT EXISTS cluster_documents (
    document_id TEXT PRIMARY KEY,
    cluster_id TEXT NOT NULL REFERENCES story_clusters(cluster_id) ON DELETE CASCADE,
    source_name TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    url TEXT NOT NULL DEFAULT '',
    published_at TIMESTAMPTZ NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS performance_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    captured_at TIMESTAMPTZ NOT NULL,
    rolling_ctr_index DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    rolling_subscription_index DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    breaking_mode BOOLEAN NOT NULL DEFAULT FALSE,
    consumer_alert_mode BOOLEAN NOT NULL DEFAULT FALSE,
    section_heat JSONB NOT NULL DEFAULT '{}'::jsonb,
    entity_heat JSONB NOT NULL DEFAULT '{}'::jsonb,
    topic_heat JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS bild_coverage_matches (
    coverage_match_id BIGSERIAL PRIMARY KEY,
    cluster_id TEXT NOT NULL REFERENCES story_clusters(cluster_id) ON DELETE CASCADE,
    article_id TEXT NOT NULL,
    coverage_type TEXT NOT NULL,
    matched_title TEXT NOT NULL,
    title_overlap DOUBLE PRECISION NOT NULL,
    entity_overlap DOUBLE PRECISION NOT NULL,
    topic_overlap DOUBLE PRECISION NOT NULL,
    freshness_delta_minutes INTEGER NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    missing_angle_tokens JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS gap_assessments (
    gap_assessment_id BIGSERIAL PRIMARY KEY,
    cluster_id TEXT NOT NULL REFERENCES story_clusters(cluster_id) ON DELETE CASCADE,
    scoring_run_id TEXT NOT NULL,
    coverage_status TEXT NOT NULL CHECK (coverage_status IN (
        'already_covered',
        'partially_covered',
        'not_covered',
        'angle_gap',
        'follow_up'
    )),
    gap_score DOUBLE PRECISION NOT NULL,
    coverage_confidence DOUBLE PRECISION NOT NULL,
    best_match_article_id TEXT,
    missing_angle JSONB NOT NULL DEFAULT '[]'::jsonb,
    follow_up_potential DOUBLE PRECISION NOT NULL DEFAULT 0,
    reason TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ml_scores (
    ml_score_id BIGSERIAL PRIMARY KEY,
    cluster_id TEXT NOT NULL REFERENCES story_clusters(cluster_id) ON DELETE CASCADE,
    scoring_run_id TEXT NOT NULL,
    ranker_version TEXT NOT NULL,
    relevance_score DOUBLE PRECISION NOT NULL,
    expected_interest DOUBLE PRECISION NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    feature_values JSONB NOT NULL DEFAULT '{}'::jsonb,
    score_components JSONB NOT NULL DEFAULT '{}'::jsonb,
    reasons JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS llm_scores (
    llm_score_id BIGSERIAL PRIMARY KEY,
    cluster_id TEXT NOT NULL REFERENCES story_clusters(cluster_id) ON DELETE CASCADE,
    scoring_run_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    relevance_score DOUBLE PRECISION NOT NULL,
    expected_interest DOUBLE PRECISION NOT NULL,
    gap_score DOUBLE PRECISION NOT NULL,
    urgency_score DOUBLE PRECISION NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    suppressed BOOLEAN NOT NULL DEFAULT FALSE,
    suppressed_reason TEXT NOT NULL DEFAULT '',
    why_relevant TEXT NOT NULL,
    why_now TEXT NOT NULL,
    why_gap TEXT NOT NULL,
    recommended_angle TEXT NOT NULL,
    raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    latency_ms INTEGER,
    estimated_cost_usd NUMERIC(10, 4),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS final_rankings (
    final_ranking_id BIGSERIAL PRIMARY KEY,
    cluster_id TEXT NOT NULL REFERENCES story_clusters(cluster_id) ON DELETE CASCADE,
    scoring_run_id TEXT NOT NULL,
    model_variant TEXT NOT NULL CHECK (model_variant IN ('ml', 'llm', 'hybrid')),
    final_rank INTEGER NOT NULL,
    final_score DOUBLE PRECISION NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    suppressed BOOLEAN NOT NULL DEFAULT FALSE,
    suppression_reason TEXT NOT NULL DEFAULT '',
    ranking_reason TEXT NOT NULL,
    explainability JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (scoring_run_id, model_variant, cluster_id)
);

CREATE TABLE IF NOT EXISTS evaluation_labels (
    evaluation_label_id BIGSERIAL PRIMARY KEY,
    cluster_id TEXT NOT NULL REFERENCES story_clusters(cluster_id) ON DELETE CASCADE,
    label_source TEXT NOT NULL,
    editorial_decision TEXT NOT NULL,
    outcome_label TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS feedback_events (
    feedback_event_id BIGSERIAL PRIMARY KEY,
    cluster_id TEXT NOT NULL REFERENCES story_clusters(cluster_id) ON DELETE CASCADE,
    editor_id TEXT NOT NULL,
    action TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_story_clusters_last_seen_at ON story_clusters(last_seen_at DESC);
CREATE INDEX IF NOT EXISTS idx_story_clusters_topics ON story_clusters USING GIN (topics);
CREATE INDEX IF NOT EXISTS idx_cluster_entities_name ON cluster_entities(entity_name);
CREATE INDEX IF NOT EXISTS idx_cluster_documents_cluster_id ON cluster_documents(cluster_id);
CREATE INDEX IF NOT EXISTS idx_bild_coverage_matches_cluster_id ON bild_coverage_matches(cluster_id, confidence DESC);
CREATE INDEX IF NOT EXISTS idx_gap_assessments_run_variant ON gap_assessments(scoring_run_id, coverage_status);
CREATE INDEX IF NOT EXISTS idx_ml_scores_run ON ml_scores(scoring_run_id, relevance_score DESC);
CREATE INDEX IF NOT EXISTS idx_llm_scores_run ON llm_scores(scoring_run_id, relevance_score DESC);
CREATE INDEX IF NOT EXISTS idx_final_rankings_run_variant_rank ON final_rankings(scoring_run_id, model_variant, final_rank);
CREATE INDEX IF NOT EXISTS idx_feedback_events_cluster_id ON feedback_events(cluster_id, created_at DESC);

COMMIT;
