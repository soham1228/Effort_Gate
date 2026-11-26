import pytest
from effortgate.agent_loop import AgentPlan, AgentStep, StepPhase, StepStatus


def test_agent_plan_valid_dag_and_readiness():
    plan = AgentPlan(
        goal="Implement feature X with tests",
        steps=[
            AgentStep(
                id="step_1",
                title="Analyze requirements",
                phase=StepPhase.PLAN,
                instruction="Identify affected files.",
            ),
            AgentStep(
                id="step_2",
                title="Modify module",
                phase=StepPhase.ACTION,
                instruction="Apply code changes.",
                depends_on=["step_1"],
            ),
            AgentStep(
                id="step_3",
                title="Run verification",
                phase=StepPhase.VERIFICATION,
                instruction="Verify unit tests pass.",
                depends_on=["step_2"],
            ),
            AgentStep(
                id="step_4",
                title="Synthesize summary",
                phase=StepPhase.SYNTHESIS,
                instruction="Summarize changes.",
                depends_on=["step_3"],
            ),
        ],
    )
    plan.validate_dag()
    assert not plan.is_complete()
    assert not plan.has_failed()

    # Initially only step_1 is ready
    ready = plan.get_ready_steps()
    assert [s.id for s in ready] == ["step_1"]

    # Complete step_1 -> step_2 becomes ready
    plan.steps[0].status = StepStatus.COMPLETED
    ready = plan.get_ready_steps()
    assert [s.id for s in ready] == ["step_2"]

    # Complete remaining steps
    plan.steps[1].status = StepStatus.COMPLETED
    plan.steps[2].status = StepStatus.COMPLETED
    plan.steps[3].status = StepStatus.COMPLETED
    assert plan.is_complete()
    assert not plan.has_failed()


def test_agent_plan_cycle_detection():
    plan = AgentPlan(
        goal="Cyclic plan",
        steps=[
            AgentStep(id="a", title="Step A", instruction="Do A", depends_on=["b"]),
            AgentStep(id="b", title="Step B", instruction="Do B", depends_on=["a"]),
        ],
    )
    with pytest.raises(ValueError, match="Cycle detected"):
        plan.validate_dag()


def test_agent_plan_unknown_dependency():
    plan = AgentPlan(
        goal="Missing dep",
        steps=[
            AgentStep(id="a", title="Step A", instruction="Do A", depends_on=["nonexistent"]),
        ],
    )
    with pytest.raises(ValueError, match="unknown step"):
        plan.validate_dag()


def test_agent_plan_duplicate_id():
    plan = AgentPlan(
        goal="Duplicate IDs",
        steps=[
            AgentStep(id="a", title="Step 1", instruction="Do 1"),
            AgentStep(id="a", title="Step 2", instruction="Do 2"),
        ],
    )
    with pytest.raises(ValueError, match="Duplicate step ID"):
        plan.validate_dag()


def test_agent_plan_exceeds_max_steps():
    steps = [
        AgentStep(id=f"s{i}", title=f"Step {i}", instruction=f"Do {i}")
        for i in range(6)
    ]
    plan = AgentPlan(goal="Too many steps", steps=steps, max_steps=5)
    with pytest.raises(ValueError, match="exceeds maximum allowed steps"):
        plan.validate_dag()
