"""Render tests for the optional AWS VPC CNI ClusterNetworkPolicy controls."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CHART = ROOT / "helm"
FIXTURE = ROOT / "tests" / "fixtures" / "cluster-network-policy-values.yaml"
EGRESS_FIXTURE = ROOT / "tests" / "fixtures" / "egress-proxy-values.yaml"
API_VERSION = "networking.k8s.aws/v1alpha1"


def _render(*extra_args: str, release: str = "score-api", namespace: str = "synthetic-ns"):
    result = subprocess.run(
        [
            "helm",
            "template",
            release,
            str(CHART),
            "--namespace",
            namespace,
            *extra_args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [document for document in yaml.safe_load_all(result.stdout) if document]


def _render_failure(*extra_args: str) -> str:
    result = subprocess.run(
        ["helm", "template", "score-api", str(CHART), *extra_args],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    return result.stderr


def _cnps(documents: list[dict]) -> list[dict]:
    return [document for document in documents if document.get("kind") == "ClusterNetworkPolicy"]


def _rule(policy: dict, name: str) -> dict:
    return next(rule for rule in policy["spec"]["egress"] if rule["name"] == name)


def _app_deployment(documents: list[dict]) -> dict:
    return next(
        document
        for document in documents
        if document.get("kind") == "Deployment"
        and document["metadata"]["labels"].get("app.kubernetes.io/component") != "egress-proxy"
    )


def test_cluster_network_policies_are_disabled_by_default():
    assert _cnps(_render()) == []


def test_application_pod_never_automounts_a_service_account_token():
    pod_spec = _app_deployment(_render())["spec"]["template"]["spec"]

    assert pod_spec["automountServiceAccountToken"] is False
    assert "initContainers" not in pod_spec


def test_cluster_network_policy_requires_the_secretless_startup_gate():
    pod_spec = _app_deployment(_render("--values", str(FIXTURE)))["spec"]["template"]["spec"]
    main_container = pod_spec["containers"][0]
    init_containers = pod_spec["initContainers"]

    assert pod_spec["automountServiceAccountToken"] is False
    assert len(init_containers) == 1
    gate = init_containers[0]
    assert gate["name"] == "egress-policy-gate"
    assert gate["image"] == main_container["image"]
    assert gate["image"] == (
        "ghcr.io/spring-media/next-push.balancer@"
        "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    )
    assert gate["command"] == ["python", "-I", "-B", "/app/app/egress_policy_gate.py"]
    assert gate["args"] == ["score-api-next-push-balancer-chart-egress-proxy", "3128", "3129"]
    assert gate["securityContext"] == {
        "allowPrivilegeEscalation": False,
        "capabilities": {"drop": ["ALL"]},
        "readOnlyRootFilesystem": True,
        "runAsNonRoot": True,
        "runAsUser": 1000,
        "runAsGroup": 1000,
        "seccompProfile": {"type": "RuntimeDefault"},
    }
    for forbidden in ("env", "envFrom", "volumeMounts"):
        assert forbidden not in gate
    assert all(
        "serviceAccountToken" not in volume.get("projected", {}) for volume in pod_spec["volumes"]
    )


def test_same_proxy_service_exposes_secretless_sentinel_only_when_cnp_is_enabled():
    default_service = next(
        document
        for document in _render("--values", str(EGRESS_FIXTURE))
        if document.get("kind") == "Service" and document["metadata"]["name"].endswith("egress-proxy")
    )
    cnp_service = next(
        document
        for document in _render("--values", str(FIXTURE))
        if document.get("kind") == "Service" and document["metadata"]["name"].endswith("egress-proxy")
    )
    assert [port["port"] for port in default_service["spec"]["ports"]] == [3128]
    assert [port["port"] for port in cnp_service["spec"]["ports"]] == [3128, 3129]

    app_policy = next(
        document
        for document in _render("--values", str(FIXTURE))
        if document.get("kind") == "NetworkPolicy" and document["metadata"]["name"].endswith("to-egress-proxy")
    )
    app_ports = {
        rule_port["port"]
        for rule in app_policy["spec"]["egress"]
        for rule_port in rule.get("ports", [])
    }
    assert {3128, 3129}.issubset(app_ports)


def test_cluster_network_policy_template_keeps_its_own_fail_closed_gates():
    template = (CHART / "templates" / "aws-cluster-networkpolicy.yaml").read_text(encoding="utf-8")

    for gate in (
        "the isolated egress proxy is required",
        "an immutable sha256 image digest is required",
        "the application default-deny NetworkPolicy is required",
        "must select Ingress and Egress",
        "direct application egress rules are forbidden",
        "exactly one target-cluster DNS peer is required",
        "approved proxy upstream egress is required",
    ):
        assert gate in template


def test_cluster_network_policies_render_exact_disjoint_app_and_proxy_controls():
    policies = _cnps(_render("--values", str(FIXTURE)))

    assert len(policies) == 4
    assert {policy["apiVersion"] for policy in policies} == {API_VERSION}
    assert {policy["spec"]["tier"] for policy in policies} == {"Admin"}
    assert {policy["spec"]["priority"] for policy in policies} == {900, 901}
    names = {policy["metadata"]["name"] for policy in policies}
    assert len(names) == 4
    assert all(name.startswith("push-balancer-synthetic-score-api-") for name in names)
    assert all(len(name) <= 63 for name in names)

    by_suffix = {
        suffix: next(policy for policy in policies if policy["metadata"]["name"].endswith(suffix))
        for suffix in ("app-accept", "app-deny", "proxy-accept", "proxy-deny")
    }
    app_subject = by_suffix["app-accept"]["spec"]["subject"]["pods"]
    assert app_subject == by_suffix["app-deny"]["spec"]["subject"]["pods"]
    assert app_subject["namespaceSelector"]["matchLabels"] == {
        "kubernetes.io/metadata.name": "synthetic-ns"
    }
    assert app_subject["podSelector"] == {
        "matchLabels": {
            "app.kubernetes.io/name": "next-push-balancer-chart",
            "app.kubernetes.io/instance": "score-api",
        },
        "matchExpressions": [
            {
                "key": "app.kubernetes.io/component",
                "operator": "DoesNotExist",
            }
        ],
    }

    proxy_subject = by_suffix["proxy-accept"]["spec"]["subject"]["pods"]
    assert proxy_subject == by_suffix["proxy-deny"]["spec"]["subject"]["pods"]
    assert proxy_subject["namespaceSelector"] == app_subject["namespaceSelector"]
    assert proxy_subject["podSelector"]["matchLabels"] == {
        "app.kubernetes.io/name": "next-push-balancer-chart-egress-proxy",
        "app.kubernetes.io/instance": "score-api",
        "app.kubernetes.io/component": "egress-proxy",
    }
    assert app_subject["podSelector"] != proxy_subject["podSelector"]

    app_proxy = _rule(by_suffix["app-accept"], "accept-egress-proxy")
    assert app_proxy["to"] == [{"pods": proxy_subject}]
    assert app_proxy["ports"] == [{"portNumber": {"port": 3128, "protocol": "TCP"}}]

    for suffix in ("app-accept", "proxy-accept"):
        dns = _rule(by_suffix[suffix], "accept-dns")
        assert dns["to"] == [
            {
                "pods": {
                    "namespaceSelector": {
                        "matchLabels": {"kubernetes.io/metadata.name": "kube-system"}
                    },
                    "podSelector": {"matchLabels": {"k8s-app": "kube-dns"}},
                }
            }
        ]
        assert dns["ports"] == [
            {"portNumber": {"port": 53, "protocol": "UDP"}},
            {"portNumber": {"port": 53, "protocol": "TCP"}},
        ]

    proxy_https = _rule(by_suffix["proxy-accept"], "accept-https")
    assert proxy_https == {
        "name": "accept-https",
        "action": "Accept",
        "to": [{"networks": ["0.0.0.0/0"]}],
        "ports": [{"portNumber": {"port": 443, "protocol": "TCP"}}],
    }

    for suffix in ("app-deny", "proxy-deny"):
        assert by_suffix[suffix]["spec"]["priority"] == 901
        assert by_suffix[suffix]["spec"]["egress"] == [
            {
                "name": "deny-all-ipv4",
                "action": "Deny",
                "to": [{"networks": ["0.0.0.0/0"]}],
            }
        ]


def test_cluster_scoped_names_change_with_release_namespace():
    first = {
        policy["metadata"]["name"]
        for policy in _cnps(
            _render(
                "--values",
                str(FIXTURE),
                release="score-api",
                namespace="synthetic-namespace-one",
            )
        )
    }
    second = {
        policy["metadata"]["name"]
        for policy in _cnps(
            _render(
                "--values",
                str(FIXTURE),
                release="score-api",
                namespace="synthetic-namespace-two",
            )
        )
    }

    assert first.isdisjoint(second)
    assert all(len(name) <= 63 for name in first | second)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            ["--set", "egressProxy.enabled=false"],
            "the isolated egress proxy is required",
        ),
        (
            ["--set-string", "digest="],
            "an immutable image digest is required",
        ),
        (
            ["--set", "networkPolicy.enabled=false"],
            "the application default-deny NetworkPolicy is required",
        ),
        (
            ["--set", "networkPolicy.policyTypes={Ingress}"],
            "must select Egress",
        ),
        (
            [
                "--set-string",
                "networkPolicy.egress[0].to[0].ipBlock.cidr=203.0.113.1/32",
            ],
            "direct application egress rules are forbidden",
        ),
        (
            ["--set", "egressProxy.allowedConnectHosts={api.stg.editorial.one}"],
            "the fixed gate target www.bild.de must be allowlisted",
        ),
    ],
)
def test_cluster_network_policy_rollout_gates_fail_closed(overrides: list[str], message: str):
    stderr = _render_failure("--values", str(FIXTURE), *overrides)
    assert message in stderr
