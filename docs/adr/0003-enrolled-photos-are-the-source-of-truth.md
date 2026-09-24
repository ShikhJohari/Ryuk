---
status: accepted
---

# Enrolled photos are the source of truth; embeddings are derived per model

Embeddings from different recognition models cannot be compared, and Ryuk compares three of them, with the live monitor switching between them. So Ryuk keeps every enrolled photo and computes its embedding under every recognition model at enrollment. Switching the active model is then instant, and adding a model later means recomputing embeddings from the photos. Enrollment costs roughly 20 ms per photo across all three models on the M4, so there is no reason to compute embeddings lazily.

## Considered options

- Store embeddings only and block a model switch until every person of interest is re-enrolled from photos the operator supplies again. Rejected: a switch would be slow and manual.
- Store photos, but compute embeddings for a model only when it becomes active. Rejected: switching would stall the live monitor for a cost that is trivial up front.

## Consequences

Ryuk stores face photos, not just embeddings, which is the more sensitive data. Purge must erase the photos along with everything made from them, and the report's ethics section says so. Every sighting records the model and threshold that produced it, because sightings from different models are not comparable either.
