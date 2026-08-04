{{/*
Expand the name of the chart.
*/}}
{{- define "push-balancer.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "push-balancer.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "push-balancer.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels.
*/}}
{{- define "push-balancer.labels" -}}
helm.sh/chart: {{ include "push-balancer.chart" . }}
{{ include "push-balancer.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels.
*/}}
{{- define "push-balancer.selectorLabels" -}}
app.kubernetes.io/name: {{ include "push-balancer.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/* Selector labels used only by the isolated egress proxy workload. */}}
{{- define "push-balancer.egressProxySelectorLabels" -}}
app.kubernetes.io/name: {{ printf "%s-egress-proxy" (include "push-balancer.name" .) | trunc 63 | trimSuffix "-" }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: egress-proxy
{{- end }}

{{/* Cluster-local name for the isolated egress proxy. */}}
{{- define "push-balancer.egressProxyName" -}}
{{- printf "%s-egress-proxy" (include "push-balancer.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create the name of the service account to use.
*/}}
{{- define "push-balancer.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "push-balancer.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
ConfigMap name.
*/}}
{{- define "push-balancer.configMapName" -}}
{{- default (printf "%s-config" (include "push-balancer.fullname" .)) .Values.configMapName }}
{{- end }}

{{/*
Secret name.
*/}}
{{- define "push-balancer.secretName" -}}
{{- default (printf "%s-secret" (include "push-balancer.fullname" .)) .Values.secretName }}
{{- end }}
