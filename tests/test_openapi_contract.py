from pathlib import Path
import re

import yaml


OPENAPI_PATH = Path(__file__).resolve().parents[1] / "push-balancer-api-v3.1.0.yaml"
ROUTERS_DIR = Path(__file__).resolve().parents[1] / "app" / "routers"


def load_openapi() -> dict:
    return yaml.safe_load(OPENAPI_PATH.read_text(encoding="utf-8"))


def iter_operations(document: dict):
    for path, path_item in document.get("paths", {}).items():
        for method, operation in path_item.items():
            if method in {"get", "post", "put", "patch", "delete"}:
                yield path, method, operation


def normalize_router_path(path: str) -> str:
    return re.sub(r"\{([^}:]+):[^}]+\}", r"{\1}", path)


def load_router_paths() -> set[str]:
    paths: set[str] = set()
    pattern = re.compile(r'@router\.(?:get|post|put|patch|delete)\("([^"]+)"')
    for router_file in ROUTERS_DIR.glob("*.py"):
        for match in pattern.finditer(router_file.read_text(encoding="utf-8")):
            route = match.group(1)
            if route.startswith("/api/"):
                paths.add(normalize_router_path(route))
    return paths


def test_openapi_uses_supported_version():
    document = load_openapi()
    assert document["openapi"] == "3.1.0"


def test_openapi_documents_critical_public_routes():
    document = load_openapi()
    paths = document["paths"]

    expected_paths = {
        "/api/health",
        "/api/articles",
        "/api/pushes",
        "/api/pushes/sync",
        "/api/predictions/feedback",
        "/api/research-rules",
        "/api/check-plus",
        "/api/ml-model",
        "/api/gbrt-model",
        "/api/tagesplan",
        "/api/tagesplan/history",
        "/api/tagesplan/suggestions",
        "/api/tagesplan/log-suggestions",
        "/api/push-title-generations",
    }

    missing = expected_paths - set(paths)
    assert not missing, f"Missing documented paths: {sorted(missing)}"


def test_every_router_path_is_documented_in_openapi():
    document = load_openapi()
    openapi_paths = set(document["paths"])
    router_paths = load_router_paths()

    missing = sorted(router_paths - openapi_paths)
    assert not missing, f"Undocumented router paths: {missing}"


def test_every_operation_has_required_metadata():
    document = load_openapi()

    for path, method, operation in iter_operations(document):
        assert operation.get("operationId"), f"{method.upper()} {path} misses operationId"
        assert operation.get("tags"), f"{method.upper()} {path} misses tags"
        assert operation.get("summary"), f"{method.upper()} {path} misses summary"
        assert operation.get("description"), f"{method.upper()} {path} misses description"


def test_every_operation_defines_an_error_response():
    document = load_openapi()

    for path, method, operation in iter_operations(document):
        responses = operation.get("responses", {})
        has_error = any(
            code == "default" or (str(code).isdigit() and int(code) >= 400)
            for code in responses
        )
        assert has_error, f"{method.upper()} {path} misses an error response"


def test_problem_schema_is_available_for_standardized_errors():
    document = load_openapi()
    schemas = document["components"]["schemas"]
    responses = document["components"]["responses"]

    assert "Problem" in schemas
    assert responses["InternalServerError"]["content"]["application/problem+json"][
        "schema"
    ]["$ref"] == "#/components/schemas/Problem"


def test_removed_compatibility_operations_are_absent_from_openapi():
    document = load_openapi()

    removed_paths = {
        "/api/push/{path}",
        "/api/competitors",
        "/api/sport-competitors",
        "/api/forschung",
        "/api/learnings",
        "/api/adobe/traffic",
        "/api/ml/status",
        "/api/ml/monitoring",
        "/api/ml/retrain",
        "/api/ml/monitoring/tick",
        "/api/predict-batch",
        "/api/gbrt/status",
        "/api/gbrt/model.json",
        "/api/gbrt/retrain",
        "/api/gbrt/force-promote",
    }

    remaining = removed_paths & set(document["paths"])
    assert not remaining, f"Removed compatibility paths still documented: {sorted(remaining)}"


def test_runtime_no_longer_contains_compatibility_deprecation_helpers():
    main_py = (Path(__file__).resolve().parents[1] / "app" / "main.py").read_text(encoding="utf-8")

    assert "_DEPRECATED_COMPATIBILITY_EXACT_PATHS" not in main_py
    assert "_DEPRECATED_COMPATIBILITY_PREFIXES" not in main_py
    assert "_apply_runtime_headers" not in main_py
    assert '@app.get("/push-balancer.html"' not in main_py


def test_openapi_no_longer_defines_deprecation_headers():
    document = load_openapi()
    headers = document.get("components", {}).get("headers", {})
    assert "Deprecation" not in headers
    assert "Sunset" not in headers


def test_frontend_code_uses_editorial_one_package_imports_only():
    frontend_src = Path(__file__).resolve().parents[1] / "frontend" / "src"
    violations: list[str] = []

    for source_file in frontend_src.rglob("*.ts*"):
        if "editorial-one-ui-shim" in source_file.parts:
            continue

        content = source_file.read_text(encoding="utf-8")
        if "@/editorial-one-ui-shim" in content or "editorial-one-ui-shim/" in content:
            violations.append(str(source_file.relative_to(frontend_src.parent)))

    assert not violations, f"Direct shim imports are not allowed: {violations}"


def test_frontend_entrypoint_imports_public_editorial_one_package_names():
    main_tsx = (
        Path(__file__).resolve().parents[1] / "frontend" / "src" / "main.tsx"
    ).read_text(encoding="utf-8")

    assert "@spring-media/editorial-one-ui" in main_tsx
    assert "@spring-media/editorial-one-ui/fonts.css" in main_tsx
    assert "editorial-one-ui-shim" not in main_tsx


def test_frontend_shim_aliases_stay_configured_in_vite_and_tsconfig():
    repo_root = Path(__file__).resolve().parents[1]
    vite_config = (repo_root / "frontend" / "vite.config.ts").read_text(encoding="utf-8")
    tsconfig = (repo_root / "frontend" / "tsconfig.app.json").read_text(encoding="utf-8")

    assert "@spring-media/editorial-one-ui/fonts.css" in vite_config
    assert "./src/editorial-one-ui-shim/fonts.css" in vite_config
    assert "@spring-media/editorial-one-ui" in vite_config
    assert "./src/editorial-one-ui-shim/index.tsx" in vite_config

    assert '"@spring-media/editorial-one-ui"' in tsconfig
    assert '"./src/editorial-one-ui-shim/index.tsx"' in tsconfig
    assert '"@spring-media/editorial-one-ui/*"' in tsconfig
    assert '"./src/editorial-one-ui-shim/*"' in tsconfig


def test_frontend_package_does_not_claim_real_editorial_one_dependency_without_tokenized_install():
    package_json = (
        Path(__file__).resolve().parents[1] / "frontend" / "package.json"
    ).read_text(encoding="utf-8")

    assert '"@spring-media/editorial-one-ui"' not in package_json


def test_runtime_environment_variables_are_documented_for_handover():
    repo_root = Path(__file__).resolve().parents[1]
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    env_example = (repo_root / ".env.example").read_text(encoding="utf-8")
    render_config = yaml.safe_load((repo_root / "render.yaml").read_text(encoding="utf-8"))

    documented_everywhere = {
        "PAID_EXTERNAL_APIS_ENABLED",
        "BACKGROUND_AUTOMATIONS_ENABLED",
        "HEALTH_ACTIVE_CHECKS_ENABLED",
        "ECONOMY_MODE",
        "PUSH_LIVE_FETCH_ENABLED",
        "LIVE_FEED_FALLBACK_ENABLED",
        "RESEARCH_EXTERNAL_CONTEXT_ENABLED",
        "ARTICLE_PREDICTION_ENRICHMENT_ENABLED",
        "TAGESPLAN_ON_DEMAND_BUILD_ENABLED",
        "OPENAI_TITLE_GENERATION_ENABLED",
        "OPENAI_TITLE_GENERATION_MODEL",
        "OPENAI_TITLE_GENERATION_TIMEOUT_S",
        "OPENAI_TITLE_GENERATION_MAX_TOKENS",
        "OPENAI_BACKFILL_ENABLED",
        "OPENAI_PREDICTION_SCORING_ENABLED",
        "OPENAI_PREDICTION_SCORING_MODEL",
        "OPENAI_PREDICTION_SCORING_TIMEOUT_S",
        "OPENAI_PREDICTION_SCORING_MAX_TOKENS",
        "OPENAI_PREDICTION_SCORING_CACHE_TTL_S",
        "PUSH_API_BASE",
        "DB_PATH",
        "PUSH_DB_MAX_ROWS",
        "ADMIN_API_KEY",
        "INTERNAL_ACCESS_ENABLED",
        "INTERNAL_ACCESS_ALLOWED_CIDRS",
        "INTERNAL_ACCESS_EXEMPT_PATHS",
        "PUSH_SYNC_SECRET",
        "ADOBE_TRAFFIC_ENABLED",
        "NPM_TOKEN",
    }

    render_env_keys = {
        item["key"]
        for service in render_config.get("services", [])
        for item in service.get("envVars", [])
        if "key" in item
    }

    for variable in documented_everywhere:
        assert variable in readme, f"{variable} missing from README handover docs"
        assert f"{variable}=" in env_example, f"{variable} missing from .env.example"
        if variable != "NPM_TOKEN":
            assert variable in render_env_keys, f"{variable} missing from render.yaml"


def test_repo_does_not_track_legacy_frontend_html_artifacts():
    repo_root = Path(__file__).resolve().parents[1]

    assert not (repo_root / "push-balancer.html").exists()
    assert not (repo_root / "app" / "legacy_push_balancer.html").exists()


def test_dockerfile_does_not_copy_removed_legacy_frontend_html():
    dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text(
        encoding="utf-8"
    )

    assert "COPY push-balancer.html ./push-balancer.html" not in dockerfile


def test_app_main_uses_spa_compat_frontend_without_file_based_legacy_helper():
    app_main = (Path(__file__).resolve().parents[1] / "app" / "main.py").read_text(
        encoding="utf-8"
    )

    assert "_legacy_frontend_path" not in app_main
    assert "_legacy_frontend_response" not in app_main


def test_stable_openapi_schemas_include_descriptions_and_examples():
    document = load_openapi()
    schemas = document["components"]["schemas"]

    required_schema_properties = {
        "HealthCheck": {"ok", "error"},
        "HealthResponse": {"status", "uptime", "checks", "costControls"},
        "CompetitorItem": {"title", "url", "pubDate", "outlet", "outletColor", "isGap", "isExklusiv", "isHot", "outlets"},
        "CompetitorSummary": {"total", "gaps", "exklusiv", "hot"},
        "CompetitorResponse": {"items", "summary", "fetchedAt"},
        "GenerateTitleRequest": {"url", "title", "category"},
        "Push": {"id", "title", "channel", "sentAt", "recipients", "opened", "openRate", "predictedOR", "url"},
        "PushDaySummary": {"count", "avgOR", "topOR", "recipients"},
        "LearningItem": {"id", "text", "impact", "createdAt"},
        "ResearchInsightsResponse": {"learnings", "experiments", "abTest"},
        "ResearchRule": {"id", "category", "rule", "confidence", "supportCount", "createdAt"},
        "ResearchRulesResponse": {"version", "rules", "rollingAccuracy", "generatedAt"},
        "AdobeTrafficEntry": {"hour", "pageviews", "visitors"},
        "AdobeTrafficArticle": {"title", "url", "pageviews"},
        "AdobeTrafficResponse": {"hourly", "topArticles", "fetchedAt"},
        "TagesplanSlot": {"hour", "label", "predictedOR", "actualOR", "pushed", "pushedTitle", "pushedAt", "isGoldenHour", "recommendation"},
        "TagesplanResponse": {"date", "mode", "slots", "goldenHour", "avgOR", "pushedCount", "mae", "trainedOnRows", "loading"},
        "TagesplanRetroSummary": {"avgOR", "totalPushes", "avgMAE"},
        "TagesplanRetroDay": {"date", "avgOR", "pushedCount", "mae", "slots"},
        "TagesplanRetroResponse": {"days"},
        "TagesplanSuggestion": {"hour", "title", "url", "score", "predictedOR"},
        "MlStatusResponse": {"modelVersion", "trainedAt", "mae", "rmse", "r2", "trainingRows", "features", "isEnsemble", "advisoryOnly", "actionAllowed"},
        "MlMonitoringPrediction": {"id", "predictedOR", "actualOR", "error", "timestamp"},
        "MlMonitoringResponse": {"recentPredictions", "rollingMAE", "drift"},
        "JobResponse": {"ok", "message", "jobId"},
        "GbrtStatusResponse": {"active", "modelVersion", "mae", "trainingRows", "features", "lastRetrain"},
    }

    for schema_name, property_names in required_schema_properties.items():
        properties = schemas[schema_name]["properties"]
        for property_name in property_names:
            property_schema = properties[property_name]
            assert property_schema.get("description"), (
                f"{schema_name}.{property_name} misses description"
            )
            if property_schema.get("type") in {"string", "integer", "number", "boolean"}:
                assert "example" in property_schema, (
                    f"{schema_name}.{property_name} misses example"
                )

    research_properties = schemas["HealthResponse"]["properties"]["research"]["properties"]
    for property_name in {"version", "lastUpdate"}:
        property_schema = research_properties[property_name]
        assert property_schema.get("description"), (
            f"HealthResponse.research.{property_name} misses description"
        )
        assert "example" in property_schema, (
            f"HealthResponse.research.{property_name} misses example"
        )

    cost_control_properties = schemas["HealthResponse"]["properties"]["costControls"]["properties"]
    for property_name in {
        "paidExternalApisEnabled",
        "openaiTitleGenerationEnabled",
        "openaiPredictionScoringEnabled",
        "openaiBackfillEnabled",
        "adobeTrafficEnabled",
    }:
        property_schema = cost_control_properties[property_name]
        assert property_schema.get("description"), (
            f"HealthResponse.costControls.{property_name} misses description"
        )
        assert "example" in property_schema, (
            f"HealthResponse.costControls.{property_name} misses example"
        )
