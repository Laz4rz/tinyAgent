# Universal Implementation Rules

High-level engineering rules that should transfer across projects.

## 1) Priorities
1. Optimize for clarity first, then flexibility, then cleverness.
2. Prefer explicit behavior over implicit magic.
3. Keep code easy to read top-to-bottom without hidden control flow.
4. Choose designs that are teachable to another engineer in one pass.

## 2) Complexity Discipline
1. Start with the simplest implementation that can work.
2. Add abstractions only when there is repeated, real pressure.
3. Avoid speculative architecture and "just in case" layers.
4. Remove unnecessary indirection when it no longer pays for itself.

## 3) Contracts and Validation
1. Fail fast on internal invariant violations.
2. Validate external inputs at boundaries, not everywhere.
3. Prefer explicit errors over silent fallbacks.
4. Use types and schema checks to make invalid states hard to represent.

## 4) Error Handling
1. Do not swallow errors silently.
2. Avoid broad catch-all exception handling around core logic.
3. Catch narrowly, add useful context, and re-raise or return structured errors.
4. At external boundaries (network, filesystem, SDK, OS), handle failures clearly and visibly.

## 5) Architecture Ownership
1. Keep shared behavior in shared layers; keep integrations provider-specific.
2. Keep entrypoint files focused on orchestration, not utility clutter.
3. Put setup/configuration logic in dedicated modules.
4. Keep parsing/formatting responsibilities close to the integration that owns them.

## 6) State and History
1. Maintain one clear source of truth for runtime state.
2. If state changes affect behavior, make those changes inspectable.
3. Avoid dual representations that can drift.
4. Preserve protocol-native structures where possible instead of flattening semantics.

## 7) Configuration
1. Use explicit, deterministic config locations and load order.
2. Ask users for required setup once, then persist it.
3. Avoid hidden defaults for critical behavior.
4. Keep secrets centralized and named consistently.

## 8) Developer Experience and UX
1. Make runtime output easy to scan and semantically clear.
2. Prefer consistent interaction patterns over one-off special cases.
3. Show meaningful progress at slow operations.
4. Treat logs/debug artifacts as product features: readable, structured, and useful.

## 9) Testing Strategy
1. Keep fast tests for core logic and architecture seams.
2. Isolate slow or environment-dependent tests.
3. Add tests for regressions and contract expectations, not SDK internals.
4. Make test execution easy to scope for fast feedback loops.

## 10) Code Review Heuristics
1. Prioritize hidden complexity and behavioral risk.
2. Look for drift between intended architecture and actual ownership.
3. Prefer fewer moving parts when behavior is equivalent.
4. Require clear rationale for each new abstraction.

## 11) Documentation Rules
1. Keep guidance durable and principle-based, not changelog-like.
2. Capture decision heuristics and non-goals explicitly.
3. Update docs when stable preferences become clear.
4. Keep examples concrete, but avoid overfitting to one repository.

## 12) Decision Rule
When multiple solutions work, choose the one that is explicit, testable, and easiest to maintain by the next engineer.
