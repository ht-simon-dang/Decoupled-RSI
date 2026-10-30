"""Explicit task/run state machines with a legal-transition table.

The workflow is *not* a chat loop: every task moves through a fixed set of
states via validated transitions. Illegal transitions raise, which is how we
catch orchestration bugs early.
"""

from __future__ import annotations

from ..contracts.enums import RunState, TaskState

# Legal task transitions. Absent edges are illegal.
_TASK_TRANSITIONS: dict[TaskState, frozenset[TaskState]] = {
    TaskState.PENDING: frozenset(
        {TaskState.READY, TaskState.BLOCKED, TaskState.CANCELLED, TaskState.SUPERSEDED}
    ),
    TaskState.READY: frozenset(
        {
            TaskState.RUNNING,
            TaskState.AWAITING_APPROVAL,
            TaskState.BLOCKED,
            TaskState.CANCELLED,
            TaskState.SUPERSEDED,
        }
    ),
    TaskState.RUNNING: frozenset(
        {
            TaskState.VALIDATING,
            TaskState.SUCCEEDED,
            TaskState.FAILED_RETRYABLE,
            TaskState.FAILED_FINAL,
            TaskState.CANCELLED,
        }
    ),
    TaskState.VALIDATING: frozenset(
        {TaskState.SUCCEEDED, TaskState.FAILED_RETRYABLE, TaskState.FAILED_FINAL, TaskState.CANCELLED}
    ),
    TaskState.FAILED_RETRYABLE: frozenset({TaskState.READY, TaskState.FAILED_FINAL, TaskState.CANCELLED}),
    TaskState.AWAITING_APPROVAL: frozenset(
        {TaskState.READY, TaskState.RUNNING, TaskState.CANCELLED, TaskState.FAILED_FINAL}
    ),
    TaskState.BLOCKED: frozenset({TaskState.READY, TaskState.CANCELLED, TaskState.SUPERSEDED}),
    # terminal states
    TaskState.SUCCEEDED: frozenset({TaskState.SUPERSEDED}),
    TaskState.FAILED_FINAL: frozenset({TaskState.SUPERSEDED, TaskState.READY}),
    TaskState.CANCELLED: frozenset(),
    TaskState.SUPERSEDED: frozenset(),
}

_RUN_TRANSITIONS: dict[RunState, frozenset[RunState]] = {
    RunState.CREATED: frozenset({RunState.RUNNING, RunState.CANCELLED}),
    RunState.RUNNING: frozenset(
        {RunState.AWAITING_APPROVAL, RunState.PAUSED, RunState.SUCCEEDED, RunState.FAILED, RunState.CANCELLED}
    ),
    RunState.AWAITING_APPROVAL: frozenset({RunState.RUNNING, RunState.CANCELLED, RunState.FAILED}),
    RunState.PAUSED: frozenset({RunState.RUNNING, RunState.CANCELLED}),
    RunState.SUCCEEDED: frozenset(),
    RunState.FAILED: frozenset({RunState.RUNNING}),  # allow resume after failure
    RunState.CANCELLED: frozenset(),
}


class IllegalTransitionError(RuntimeError):
    pass


def can_transition_task(src: TaskState, dst: TaskState) -> bool:
    if src == dst:
        return True
    return dst in _TASK_TRANSITIONS.get(src, frozenset())


def assert_task_transition(src: TaskState, dst: TaskState) -> None:
    if not can_transition_task(src, dst):
        raise IllegalTransitionError(f"illegal task transition {src} -> {dst}")


def can_transition_run(src: RunState, dst: RunState) -> bool:
    if src == dst:
        return True
    return dst in _RUN_TRANSITIONS.get(src, frozenset())


def assert_run_transition(src: RunState, dst: RunState) -> None:
    if not can_transition_run(src, dst):
        raise IllegalTransitionError(f"illegal run transition {src} -> {dst}")


def is_terminal_task(state: TaskState) -> bool:
    return state in {TaskState.SUCCEEDED, TaskState.FAILED_FINAL, TaskState.CANCELLED, TaskState.SUPERSEDED}
