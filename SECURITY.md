# Security Policy

## Supported version

Security fixes are applied to the `main` branch.

## Reporting a vulnerability

Do not publish API keys, confidential legal documents, or vulnerability details in a public issue. Use GitHub's private vulnerability-reporting feature if it is enabled, or contact the repository owner privately through the contact method on their GitHub profile.

Include the affected commit, a minimal reproduction, expected impact, and suggested mitigation. Do not include real client data.

## Credential incident procedure

If a credential is committed:

1. Revoke or rotate it at the provider immediately.
2. Remove it from the current branch.
3. Inspect workflows, logs, and provider usage for abuse.
4. Rewrite Git history only after coordinating with collaborators and forks.
5. Assume history rewriting does not remove copies already cloned or forked.

The repository audit detects common Hugging Face, OpenAI, and Groq token formats, but automated scanning is not proof that a repository contains no secrets.
