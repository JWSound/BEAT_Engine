# Extraction provenance

Source: https://github.com/JWSound/boundary-lab
Source branch: dev
Source commit: 01988025adeabec50a692891deb35ba7fec29aa4
Filtered head: 04876c88199a28b5f52daf1273e06a4b5f201aa4

A fresh independent clone was filtered with git-filter-repo. The source repository
was not rewritten. 105 relevant historical commits were retained. Original commit
IDs map to rewritten IDs in extraction-commit-map.txt; zero IDs denote commits
without retained content.

Retained: Julia source and environments, numerical fixtures/tests, wire contracts,
worker transport, their independent Python tests, contract documentation, and
licensing. Application-specific assertions stay in Boundary Lab. Runtime assets
are located under src/beat_engine so wheels and contributor checkouts use the same
public path API. Root project scaffolding and the public EngineWorker API were
added after filtering.

The source-request Julia entrypoint is retained for existing reference and Deploy
comparisons. It is not the physical-system public request contract. Specialized
historical scripts which reference application-only fixtures remain extended
research tooling; the portable gates must never rely on those paths.
