BEGIN;

ALTER TABLE IF EXISTS story_clusters DROP COLUMN IF EXISTS graphd_score;
ALTER TABLE IF EXISTS story_clusters DROP COLUMN IF EXISTS graphd_reason;
ALTER TABLE IF EXISTS ml_scores DROP COLUMN IF EXISTS graphd_score;
ALTER TABLE IF EXISTS llm_scores DROP COLUMN IF EXISTS graphd_score;
ALTER TABLE IF EXISTS gap_assessments DROP COLUMN IF EXISTS graphd_score;
ALTER TABLE IF EXISTS final_rankings DROP COLUMN IF EXISTS graphd_score;
ALTER TABLE IF EXISTS final_rankings DROP COLUMN IF EXISTS graphd_rank;
ALTER TABLE IF EXISTS final_rankings DROP COLUMN IF EXISTS graphd_bucket;
ALTER TABLE IF EXISTS cluster_documents DROP COLUMN IF EXISTS graphd_score;
ALTER TABLE IF EXISTS evaluation_labels DROP COLUMN IF EXISTS graphd_baseline_bucket;

DROP VIEW IF EXISTS story_radar_rankings_with_graphd;
DROP MATERIALIZED VIEW IF EXISTS story_radar_graphd_leaderboard;
DROP INDEX IF EXISTS idx_story_clusters_graphd_score;
DROP INDEX IF EXISTS idx_final_rankings_graphd_rank;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'feature_flags'
    ) THEN
        DELETE FROM feature_flags
        WHERE name IN (
            'story_radar_graphd_fallback',
            'story_radar_graphd_ui',
            'story_radar_graphd_experiment'
        );
    END IF;
END $$;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'experiment_arms'
    ) THEN
        DELETE FROM experiment_arms
        WHERE experiment_name = 'story_radar_relevance'
          AND arm_key IN ('graphd_only', 'graphd_hybrid', 'graphd_fallback');
    END IF;
END $$;

COMMIT;
