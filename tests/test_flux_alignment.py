from pathlib import Path
import ast
import re

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _load_yaml(path: str) -> dict:
    return yaml.safe_load(_read(path))


def test_version_file_uses_semver():
    version = _read("VERSION").strip()
    assert re.fullmatch(r"\d+\.\d+\.\d+", version)


def test_docker_workflow_matches_flux_release_expectations():
    workflow = _read(".github/workflows/docker-build.yaml")

    assert "name: Docker Build and Push" in workflow
    assert "runs-on: ubuntu-24.04-arm" in workflow
    assert "platforms: linux/arm64" in workflow
    assert "IMAGE_NAME: ${{ github.repository }}" in workflow
    assert "cat VERSION" in workflow
    assert "deploy/flux-examples/stg/push-balancer.yaml" in workflow
    assert "deploy/flux-examples/prd/push-balancer.yaml" in workflow
    assert "github.event.pull_request.number || github.ref" in workflow


def test_container_starts_the_configured_score_app_without_path_access_logs():
    dockerfile = _read("Dockerfile")

    assert "${ASGI_APP:-app.score_main:app}" in dockerfile
    assert "USER appuser" in dockerfile
    assert "exec su" not in dockerfile
    assert "--no-access-log" in dockerfile
    assert "--no-proxy-headers" in dockerfile


def test_release_workflow_publishes_matching_image_and_chart():
    workflow = _read(".github/workflows/docker-build.yaml")

    assert "Prepare exact build or release tree" in workflow
    assert "git pull --rebase" not in workflow
    assert "Skipping stale main event" in workflow
    assert "is_version_bump" in workflow
    assert "steps.release.outputs.release_sha" in workflow
    assert "Reuse an existing immutable version image" in workflow
    assert "Existing semver image does not match" in workflow
    assert "steps.existing_image.outputs.exists != 'true'" in workflow
    assert "helm package helm" in workflow
    assert "helm push" in workflow
    assert "helm pull" in workflow
    assert "diff -rq" in workflow
    assert "--version \"$VERSION\"" in workflow
    assert "steps.final_image.outputs.digest" in workflow


def test_chart_metadata_and_values_track_the_repo_version():
    version = _read("VERSION").strip()
    chart = _load_yaml("helm/Chart.yaml")
    values = _load_yaml("helm/values.yaml")

    assert chart["name"] == "next-push-balancer-chart"
    assert chart["version"] == version
    assert chart["appVersion"] == version
    assert values["image"] == "ghcr.io/spring-media/next-push.balancer"
    assert values["tag"] == version
    assert values["digest"] == ""
    assert values["replicaCount"] == 1
    assert values["config"]["ASGI_APP"] == "app.score_main:app"
    assert values["config"]["PUSH_API_BASE"] == ""
    assert "URL_API_BASE" in values["config"]
    assert values["config"]["PUSH_LIVE_FETCH_ENABLED"] == "false"
    assert values["config"]["LIVE_FEED_FALLBACK_ENABLED"] == "false"
    assert values["config"]["INTERNAL_ACCESS_ENABLED"] == "true"
    assert values["persistence"]["enabled"] is False
    assert values["securityContext"]["readOnlyRootFilesystem"] is True
    assert values["temporaryDirectory"] == {
        "enabled": True,
        "mountPath": "/tmp",
        "sizeLimit": "64Mi",
    }
    assert values["ingress"]["enabled"] is False
    assert values["ingress"]["rolloutGateApproved"] is False
    assert values["ingress"]["requireTls"] is True
    assert values["ingress"]["platformManagedTls"] is False
    assert values["networkPolicy"]["enabled"] is True


def test_chart_templates_expose_flux_image_policy_marker():
    deployment = _read("helm/templates/deployment.yaml")
    image_repository = _read("helm/templates/ImageRepository.yaml")
    image_policy = _read("helm/templates/ImagePolicy.yaml")

    assert '# {"$imagepolicy":' in deployment
    assert ".Values.digest" in deployment
    assert "immutable image digest is required" in _read("helm/templates/ingress.yaml")
    assert "kind: ImageRepository" in image_repository
    assert "kind: ImagePolicy" in image_policy


def test_chart_extra_env_supports_least_privilege_secret_key_projection():
    deployment = _read("helm/templates/deployment.yaml")
    values = _load_yaml("helm/values.yaml")

    assert "toYaml . | nindent 12" in deployment
    assert values["extraEnv"] == []


def test_chart_external_secret_supports_data_from_extract_and_rewrite():
    template = _read("helm/templates/secret.yaml")
    values = _load_yaml("helm/values.yaml")

    assert ".Values.externalSecret.dataFrom" in template
    assert "dataFrom:" in template
    assert values["externalSecret"]["dataFrom"] == []


def test_chart_contains_fail_closed_isolated_egress_proxy():
    values = _load_yaml("helm/values.yaml")
    template = _read("helm/templates/egress-proxy.yaml")
    fixture = _load_yaml("tests/fixtures/egress-proxy-values.yaml")

    assert values["egressProxy"]["enabled"] is False
    assert values["egressProxy"]["allowedConnectHosts"] == []
    assert values["egressProxy"]["networkPolicy"] == {
        "dns": {"cidr": "", "namespace": "", "podLabels": {}},
        "upstreamCidrs": [],
    }
    assert fixture["egressProxy"]["enabled"] is True
    assert "automountServiceAccountToken: false" in template
    assert "readOnlyRootFilesystem: true" in template
    assert "app/egress_proxy.py" in template
    assert "exact CONNECT hostname is required" in template
    assert "approved upstream egress is required" in template
    internal_ingress = _load_yaml("tests/fixtures/internal-ingress-values.yaml")
    assert internal_ingress["ingress"]["platformManagedTls"] is True
    assert internal_ingress["ingress"]["className"] == "skipper"


def test_runtime_and_chart_use_the_same_fixed_proxy_host_allowlist():
    runtime_tree = ast.parse(_read("app/egress_proxy.py"))
    runtime_hosts = None
    for node in runtime_tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "_PERMITTED_HOSTS" for target in node.targets):
            continue
        assert isinstance(node.value, ast.Call)
        runtime_hosts = set(ast.literal_eval(node.value.args[0]))
        break

    template = _read("helm/templates/egress-proxy.yaml")
    helm_allowlist = re.search(r'\(list ((?:"[^"]+"\s*)+)\)', template)
    assert helm_allowlist is not None
    chart_hosts = set(re.findall(r'"([^"]+)"', helm_allowlist.group(1)))
    assert runtime_hosts == chart_hosts == {
        "api.stg.editorial.one",
        "api.editorial.one",
        "push-balancer.onrender.com",
        "www.bild.de",
    }


def test_flux_examples_reference_published_chart_and_setter_strategy():
    for env_name, host_fragment in (("stg", "push-balancer-stg.example.invalid"), ("prd", "push-balancer.example.invalid")):
        helm_release = _load_yaml(f"deploy/flux-examples/{env_name}/push-balancer.yaml")
        helm_repo = _load_yaml(f"deploy/flux-examples/{env_name}/ghcr.yaml")
        automation = _load_yaml(
            f"deploy/flux-examples/{env_name}/push-balancer-automation.yaml"
        )
        helm_release_text = _read(f"deploy/flux-examples/{env_name}/push-balancer.yaml")

        assert helm_release["kind"] == "HelmRelease"
        assert helm_release["spec"]["chart"]["spec"]["chart"] == "next-push-balancer-chart"
        assert helm_release["spec"]["chart"]["spec"]["version"] == _read("VERSION").strip()
        assert helm_release["spec"]["chart"]["spec"]["sourceRef"]["kind"] == "HelmRepository"
        assert helm_release["spec"]["chart"]["spec"]["sourceRef"]["name"] == "next-helm-charts"
        assert helm_release["spec"]["values"]["image"] == "ghcr.io/spring-media/next-push.balancer"
        assert str(helm_release["spec"]["values"]["tag"]) == _read("VERSION").strip()
        assert helm_release["spec"]["values"]["digest"] == ""
        assert helm_release["spec"]["values"]["flux"]["enabled"] is True
        values = helm_release["spec"]["values"]
        config = values["config"]
        assert values["replicaCount"] == 1
        assert values["persistence"]["enabled"] is False
        assert config["ASGI_APP"] == "app.score_main:app"
        assert config["PUSH_API_BASE"] == ""
        assert config["URL_API_BASE"].startswith("https://")
        assert config["BACKGROUND_AUTOMATIONS_ENABLED"] == "false"
        assert config["PUSH_LIVE_FETCH_ENABLED"] == "false"
        assert config["LIVE_FEED_FALLBACK_ENABLED"] == "false"
        assert config["PAID_EXTERNAL_APIS_ENABLED"] == "false"
        assert config["INTERNAL_ACCESS_ENABLED"] == "true"
        assert values["ingress"]["enabled"] is False
        assert values["ingress"]["rolloutGateApproved"] is False
        assert values["ingress"]["requireTls"] is True
        assert values["ingress"]["tls"] == []
        ingress_paths = values["ingress"]["hosts"][0]["paths"]
        assert ingress_paths == [{"path": "/api/v1/scores", "pathType": "Prefix"}]
        secret_keys = {
            item["secretKey"]
            for item in values["externalSecret"]["data"]
        }
        assert secret_keys == {"SCORE_API_KEY", "URL_API_KEY"}
        assert values["networkPolicy"] == {
            "enabled": True,
            "policyTypes": ["Ingress", "Egress"],
            "ingress": [],
            "egress": [],
        }
        assert host_fragment in helm_release_text
        assert '# {"$imagepolicy": "bildnext:push-balancer:tag"}' in helm_release_text

        assert helm_repo["kind"] == "HelmRepository"
        assert helm_repo["metadata"]["name"] == "next-helm-charts"
        assert helm_repo["spec"]["type"] == "oci"
        assert helm_repo["spec"]["url"] == "oci://ghcr.io/spring-media"

        assert automation["kind"] == "ImageUpdateAutomation"
        assert automation["spec"]["update"]["strategy"] == "Setters"
        assert automation["spec"]["update"]["path"] == f"./{env_name}/push-balancer"


def test_readmes_document_flux_alignment_artifacts():
    root_readme = _read("README.md")
    deploy_readme = _read("deploy/README.md")

    assert "Deployment (Flux/CD)" in root_readme
    assert "https://github.com/spring-media/next-push.balancer" in root_readme
    assert "deploy/flux-examples/" in root_readme
    assert ".github/workflows/docker-build.yaml" in root_readme
    assert "matching Helm-chart" in root_readme

    assert "https://github.com/spring-media/bildnext-flux-cd" in deploy_readme
    assert "ghcr.io/spring-media/next-push.balancer" in deploy_readme
    assert "Simple Deployment Path" in deploy_readme
    assert "Deployment Checklist" in deploy_readme
    assert 'What "Follows The Examples" Means' in deploy_readme


@pytest.mark.parametrize("env_name", ["stg", "prd"])
def test_teams_runtime_examples_pin_the_channel_contract(env_name):
    """Die Teams-Laufzeit auf Next: volle App, Persistenz, Secrets, kein Ingress."""
    document = yaml.safe_load(
        _read(f"deploy/flux-examples/{env_name}/push-balancer-teams.yaml")
    )

    assert document["kind"] == "HelmRelease"
    assert document["metadata"]["name"] == "push-balancer-teams"
    values = document["spec"]["values"]

    config = values["config"]
    assert config["ASGI_APP"] == "app.main:app"
    assert config["PUSH_TEAMS_ALERTS_ENABLED"] == "true"
    assert config["PUSH_LIVE_FETCH_ENABLED"] == "true"
    assert config["PUSH_BALANCER_SCORE_API_ENABLED"] == "true"
    assert config["DB_PATH"] == "/data/.push_history.db"
    assert "/api/ready" in config["INTERNAL_ACCESS_EXEMPT_PATHS"]

    assert values["persistence"]["enabled"] is True
    assert values["persistence"]["mountPath"] == "/data"
    assert values["ingress"]["enabled"] is False

    secret_keys = {
        entry["secretKey"] for entry in values["externalSecret"]["data"]
    }
    assert "PUSH_TEAMS_WEBHOOK_URL" in secret_keys
    assert "PUSH_BALANCER_SCORE_API_KEY" in secret_keys

    # Kein interner Hostname im oeffentlichen Beispiel.
    assert config["PUSH_API_BASE"] == ""
    assert config["PUSH_BALANCER_SCORE_API_BASE_URL"] == ""

    # Image-Automation aktualisiert auch dieses Release ueber den Marker.
    raw = _read(f"deploy/flux-examples/{env_name}/push-balancer-teams.yaml")
    assert '{"$imagepolicy": "bildnext:push-balancer:tag"}' in raw
