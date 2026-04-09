# Privacy Agent Rules

Apply privacy-by-design and GDPR-aware engineering whenever work touches personal data, analytics, tracking, logging, monitoring, external APIs, cloud services, AI tooling, or employee-related data.

## Core rules

1. Process personal data only for a defined purpose and only as necessary.
2. Use synthetic, dummy, or effectively pseudonymised data for development, tests, demos, prompts, and screenshots.
3. Never send secrets, tokens, session identifiers, private keys, raw production logs, or raw personal data to external tools or models.
4. Prefer privacy-friendly defaults, short retention, and the least intrusive architecture.
5. If legal basis, controller/processor roles, transfer path, or deletion capability is unclear: stop and escalate.

## Hard stops

Escalate to a human before implementation if:

- sensitive or employee data is introduced or newly processed
- new tracking, profiling, scoring, or monitoring is added
- external AI, cloud, analytics, or support providers receive personal data
- international transfer may occur
- deletion, access, export, or correction handling is unclear
- a potential privacy incident or disclosure is detected

## Axel Springer escalation

- Privacy Manager
- Product Owner / System Owner
- Data Protection Officer
- Legal / Group Legal

## Required output

For privacy-relevant changes, provide a `PRIVACY NOTE` covering purpose, data categories, data subjects, legal basis, roles, recipients/transfers, retention, safeguards, and required approvals.
