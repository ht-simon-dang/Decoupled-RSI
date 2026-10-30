"""Agent invocation helper — the single path that enforces schema-valid I/O.

Runs the full contract cycle: validate_request → plan → execute → validate_result,
persisting the AgentPlan and AgentResult. Both request and result are Pydantic
models, so all agent inputs/outputs are schema-validated (acceptance #8).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..agents.base import AgentError, RetryableAgentError
from ..contracts.agent import AgentRequest, AgentResult

if TYPE_CHECKING:
    from .context import RunContext


def invoke_agent(
    ctx: RunContext,
    agent_id: str,
    action: str,
    *,
    inputs: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    task_id: str | None = None,
) -> AgentResult:
    agent = ctx.registry.get_agent(agent_id)
    request = AgentRequest(
        run_id=ctx.run.id,
        task_id=task_id,
        agent_id=agent_id,
        action=action,
        inputs=inputs or {},
        params=params or {},
        deterministic=ctx.settings.deterministic,
        seed=ctx.run.seed,
        correlation_id=ctx.correlation_id,
    )

    # 1) validate request
    req_issues = agent.validate_request(request)
    errors = [i for i in req_issues if i.severity == "error"]
    if errors:
        raise AgentError(f"{agent_id}.{action} invalid request: {errors[0].message}")

    # 2) plan (inspectable, persisted)
    plan = agent.plan(request, ctx)
    ctx.store.put_object("agent_plan", plan, run_id=ctx.run.id)

    # 3) execute
    with ctx.telemetry.span(f"agent.{agent_id}.{action}", agent=agent_id, action=action):
        try:
            result = agent.execute(request, ctx)
        except RetryableAgentError:
            raise
        except AgentError:
            raise
        except Exception as exc:
            raise AgentError(f"{agent_id}.{action} failed: {type(exc).__name__}: {exc}") from exc

    # 4) validate result (independent of execute)
    result.validation_issues.extend(agent.validate_result(request, result, ctx))
    if not result.explanation:
        result.explanation = agent.explain(result)

    # 5) persist result + budget accounting
    ctx.store.put_object("agent_result", result, run_id=ctx.run.id)
    ctx.run.budget.tokens_spent += result.tokens_used
    ctx.run.budget.external_calls_spent += result.external_calls
    ctx.telemetry.incr("agent.invocations")
    ctx.telemetry.incr("agent.tokens", result.tokens_used)

    if not result.ok:
        first = next((i for i in result.validation_issues if i.severity == "error"), None)
        raise AgentError(
            f"{agent_id}.{action} produced invalid result: {first.message if first else 'unknown'}"
        )
    return result
