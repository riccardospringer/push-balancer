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
