"""Consumer-facing contract and secret-safe smoke client tests."""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path
from urllib.error import HTTPError

import yaml

from app.score_main import app

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "openapi" / "score-api-v1.yaml"
SMOKE_PATH = ROOT / "scripts" / "smoke_score_api.py"
CMS_ID = "0123456789abcdef01234567"
API_KEY = "synthetic-score-key"
ENGAGEMENT_BREAKDOWN = {
    "kind": "engagement",
    "relevance": 30,
    "urgency": 0,
    "curiosity": 7.6,
    "freshness": 11.7,
    "timing": 6,
    "titleBoost": 3,
    "breaking": 0,
    "research": 0,
    "pushHistory": 0,
    "topicSaturation": 0,
}


def _load_smoke_module():
    spec = importlib.util.spec_from_file_location("smoke_score_api", SMOKE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _SyntheticResponse:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, limit: int) -> bytes:
        return self.body[:limit]


class _SyntheticOpener:
    def __init__(self, result):
        self.result = result
        self.request = None
        self.timeout = None

    def open(self, request, timeout):
        self.request = request
        self.timeout = timeout
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _environment() -> dict[str, str]:
    return {
        "NEXT_PUSH_BALANCER_URL": "https://score.example.invalid",
        "SCORE_API_KEY": API_KEY,
        "CMS_ID": CMS_ID,
    }


def _matches_pair_schema_branch(payload: object, schema: dict) -> bool:
    """Evaluate the Draft 2020-12 keywords used by the pair-only oneOf branches."""
    if isinstance(payload, dict) and any(
        field not in payload for field in schema.get("required", [])
    ):
        return False
    if "type" in schema and schema["type"] == "null" and payload is not None:
        return False
    if "anyOf" in schema and not any(
        _matches_pair_schema_branch(payload, branch) for branch in schema["anyOf"]
    ):
        return False
    if "not" in schema and _matches_pair_schema_branch(payload, schema["not"]):
        return False
    if isinstance(payload, dict):
        for field, field_schema in schema.get("properties", {}).items():
            if field in payload and not _matches_pair_schema_branch(payload[field], field_schema):
                return False
    return True


def _matches_details_one_of(payload: dict, response_schema: dict) -> bool:
    return (
        sum(
            _matches_pair_schema_branch(payload, branch)
            for branch in response_schema["oneOf"]
        )
        == 1
    )


def test_minimal_openapi_contains_only_the_score_consumer_surface():
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))

    assert contract["openapi"] == "3.1.0"
    assert set(contract["paths"]) == {
        "/api/v1/scores/batch",
        "/api/v1/scores/{cms_id}",
    }
    operation = contract["paths"]["/api/v1/scores/{cms_id}"]["get"]
    assert operation["operationId"] == "getScoreByCmsId"
    assert operation["security"] == [{"scoreApiKey": []}]
    assert set(operation["responses"]) == {"200", "401", "404", "422", "500", "502", "503"}
    assert contract["components"]["securitySchemes"]["scoreApiKey"] == {
        "type": "apiKey",
        "in": "header",
        "name": "X-Score-Key",
        "description": (
            "Dedicated server-side credential. It must come from the approved secret "
            "manager and must never be embedded in browser code or a URL."
        ),
    }
    batch_operation = contract["paths"]["/api/v1/scores/batch"]["post"]
    assert batch_operation["operationId"] == "getScoresByCmsIds"
    assert batch_operation["security"] == [{"scoreApiKey": []}]
    assert set(batch_operation["responses"]) == {
        "200",
        "401",
        "422",
        "429",
        "502",
        "503",
    }


def test_minimal_openapi_matches_runtime_success_schema_and_path_constraints():
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    runtime = app.openapi()
    contract_operation = contract["paths"]["/api/v1/scores/{cms_id}"]["get"]
    runtime_operation = runtime["paths"]["/api/v1/scores/{cms_id}"]["get"]

    contract_parameter = contract_operation["parameters"][0]
    runtime_parameter = runtime_operation["parameters"][0]
    for key in ("name", "in", "required"):
        assert contract_parameter[key] == runtime_parameter[key]
    for key in ("minLength", "maxLength", "pattern"):
        assert contract_parameter["schema"][key] == runtime_parameter["schema"][key]

    contract_schema = contract["components"]["schemas"]["ArticleScoreResponse"]
    runtime_schema = runtime["components"]["schemas"]["ArticleScoreResponse"]
    assert contract_schema["required"] == runtime_schema["required"]
    assert contract_schema["oneOf"] == runtime_schema["oneOf"]
    assert contract_schema["additionalProperties"] is True
    assert runtime_schema["additionalProperties"] is True
    for field in contract_schema["required"]:
        assert contract_schema["properties"][field]["type"] == runtime_schema["properties"][field][
            "type"
        ]
    assert set(contract_schema["properties"]) == {
        "cmsId",
        "score",
        "scoredAt",
        "scoreBreakdown",
        "orFactor",
    }
    assert set(contract["components"]["schemas"]["EngagementScoreBreakdownResponse"]["properties"]) == {
        "kind",
        "relevance",
        "urgency",
        "curiosity",
        "freshness",
        "timing",
        "titleBoost",
        "breaking",
        "research",
        "pushHistory",
        "topicSaturation",
    }
    assert set(contract["components"]["schemas"]["SportScoreBreakdownResponse"]["properties"]) == {
        "kind",
        "sportRelevance",
        "timing",
        "drama",
        "freshness",
    }


def test_score_details_one_of_accepts_absent_null_and_enriched_pairs():
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    runtime = app.openapi()
    response_schemas = [
        contract["components"]["schemas"]["ArticleScoreResponse"],
        runtime["components"]["schemas"]["ArticleScoreResponse"],
    ]
    valid_payloads = [
        {},
        {"scoreBreakdown": None, "orFactor": None},
        {"scoreBreakdown": ENGAGEMENT_BREAKDOWN, "orFactor": 1.06},
    ]

    for response_schema in response_schemas:
        assert len(response_schema["oneOf"]) == 3
        for payload in valid_payloads:
            assert _matches_details_one_of(payload, response_schema)


def test_score_details_one_of_rejects_partial_and_mixed_pairs():
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    runtime = app.openapi()
    response_schemas = [
        contract["components"]["schemas"]["ArticleScoreResponse"],
        runtime["components"]["schemas"]["ArticleScoreResponse"],
    ]
    invalid_payloads = [
        {"scoreBreakdown": ENGAGEMENT_BREAKDOWN},
        {"orFactor": 1.06},
        {"scoreBreakdown": ENGAGEMENT_BREAKDOWN, "orFactor": None},
        {"scoreBreakdown": None, "orFactor": 1.06},
    ]

    for response_schema in response_schemas:
        for payload in invalid_payloads:
            assert not _matches_details_one_of(payload, response_schema)


def test_batch_contract_is_bounded_exact_and_matches_runtime():
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    runtime = app.openapi()
    contract_request = contract["components"]["schemas"]["BatchScoreRequest"]
    runtime_request = runtime["components"]["schemas"]["BatchScoreRequest"]

    assert contract_request["additionalProperties"] is False
    assert runtime_request["additionalProperties"] is False
    for schema in (contract_request, runtime_request):
        cms_ids = schema["properties"]["cmsIds"]
        assert cms_ids["minItems"] == 1
        assert cms_ids["maxItems"] == 500
        assert cms_ids["items"]["pattern"] == "^[0-9a-fA-F]{24}$"

    for schema_name in {
        "BatchFoundScoreResponse",
        "BatchNotFoundScoreResponse",
        "BatchScoreResponse",
    }:
        contract_schema = contract["components"]["schemas"][schema_name]
        runtime_schema = runtime["components"]["schemas"][schema_name]
        assert contract_schema["additionalProperties"] is False
        assert runtime_schema["additionalProperties"] is False
        assert set(contract_schema["properties"]) == set(runtime_schema["properties"])

    found_contract = contract["components"]["schemas"]["BatchFoundScoreResponse"]
    found_runtime = runtime["components"]["schemas"]["BatchFoundScoreResponse"]
    valid_pairs = [
        {"scoreBreakdown": None, "orFactor": None},
        {"scoreBreakdown": ENGAGEMENT_BREAKDOWN, "orFactor": 1.06},
    ]
    invalid_pairs = [
        {"scoreBreakdown": ENGAGEMENT_BREAKDOWN, "orFactor": None},
        {"scoreBreakdown": None, "orFactor": 1.06},
    ]
    for schema in (found_contract, found_runtime):
        for payload in valid_pairs:
            assert _matches_details_one_of(payload, schema)
        for payload in invalid_pairs:
            assert not _matches_details_one_of(payload, schema)


def test_smoke_client_sends_secret_in_header_and_prints_only_score():
    smoke = _load_smoke_module()
    response = _SyntheticResponse(
        b'{"cmsId":"0123456789abcdef01234567","score":87.4,'
        b'"scoredAt":"2026-07-15T12:00:00"}'
    )
    opener = _SyntheticOpener(response)
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = smoke.main(_environment(), opener, stdout, stderr)

    assert exit_code == 0
    assert stdout.getvalue() == "87.4\n"
    assert stderr.getvalue() == ""
    assert opener.request.get_header("X-score-key") == API_KEY
    assert API_KEY not in opener.request.full_url
    assert opener.timeout == 35.0


def test_smoke_client_redacts_identifier_key_and_error_body():
    smoke = _load_smoke_module()
    error = HTTPError(
        f"https://score.example.invalid/api/v1/scores/{CMS_ID}",
        404,
        f"synthetic error containing {CMS_ID} and {API_KEY}",
        {},
        None,
    )
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = smoke.main(_environment(), _SyntheticOpener(error), stdout, stderr)

    assert exit_code == 2
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == "no current score is available\n"
    assert CMS_ID not in stderr.getvalue()
    assert API_KEY not in stderr.getvalue()


def test_smoke_client_rejects_non_tls_remote_base_url_before_request():
    smoke = _load_smoke_module()
    environment = _environment() | {"NEXT_PUSH_BALANCER_URL": "http://score.example.invalid"}
    opener = _SyntheticOpener(AssertionError("network must not run"))
    stderr = io.StringIO()

    exit_code = smoke.main(environment, opener, io.StringIO(), stderr)

    assert exit_code == 3
    assert stderr.getvalue() == "score API smoke configuration is invalid\n"
    assert opener.request is None


def test_smoke_client_fails_closed_on_mismatched_response_id():
    smoke = _load_smoke_module()
    response = _SyntheticResponse(
        b'{"cmsId":"different","score":87.4,"scoredAt":"2026-07-15T12:00:00"}'
    )
    stderr = io.StringIO()

    exit_code = smoke.main(_environment(), _SyntheticOpener(response), io.StringIO(), stderr)

    assert exit_code == 3
    assert stderr.getvalue() == "score API response violates the v1 contract\n"
    assert CMS_ID not in stderr.getvalue()
    assert API_KEY not in stderr.getvalue()
