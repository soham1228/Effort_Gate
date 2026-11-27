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


@pytest.mark.asyncio
async def test_generate_plan_parses_fenced_json():
    import json
    from unittest.mock import AsyncMock, patch
    from effortgate.agent_loop import generate_plan

    plan_json = json.dumps({
        "goal": "Refactor auth module",
        "steps": [
            {"id": "step_1", "title": "Inspect auth", "phase": "plan", "instruction": "Read auth.py", "depends_on": []},
            {"id": "step_2", "title": "Update auth", "phase": "action", "instruction": "Edit auth.py", "depends_on": ["step_1"]},
        ],
    })
    mock_resp = {"choices": [{"message": {"content": f"```json\n{plan_json}\n```"}}]}
    req = {"model": "anthropic/claude-3-5-sonnet", "messages": [{"role": "user", "content": "Refactor auth"}]}

    with patch("effortgate.agent_loop.litellm.acompletion", new=AsyncMock(return_value=mock_resp)):
        plan = await generate_plan(req)

    assert plan.goal == "Refactor auth module"
    assert len(plan.steps) == 2
    assert plan.steps[1].depends_on == ["step_1"]


@pytest.mark.asyncio
async def test_generate_plan_fallback_on_invalid_json():
    from unittest.mock import AsyncMock, patch
    from effortgate.agent_loop import generate_plan

    mock_resp = {"choices": [{"message": {"content": "Not valid JSON at all!"}}]}
    req = {"model": "anthropic/claude-3-5-sonnet", "messages": [{"role": "user", "content": "Build a CLI"}]}

    with patch("effortgate.agent_loop.litellm.acompletion", new=AsyncMock(return_value=mock_resp)):
        plan = await generate_plan(req)

    assert len(plan.steps) == 2
    assert plan.steps[0].id == "step_1"
    assert plan.steps[1].depends_on == ["step_1"]


@pytest.mark.asyncio
async def test_execute_step_chaining_and_status():
    from unittest.mock import AsyncMock, patch
    from effortgate.agent_loop import execute_step

    plan = AgentPlan(
        goal="Build feature",
        steps=[
            AgentStep(
                id="step_1",
                title="Analyze",
                phase=StepPhase.PLAN,
                instruction="Analyze files",
                status=StepStatus.COMPLETED,
                result="Found main.py",
            ),
            AgentStep(
                id="step_2",
                title="Implement",
                phase=StepPhase.ACTION,
                instruction="Edit main.py",
                depends_on=["step_1"],
            ),
        ],
    )
    req = {"model": "anthropic/claude-3-5-sonnet", "messages": []}
    mock_resp = {"choices": [{"message": {"content": "Edited main.py successfully"}}]}

    with patch("effortgate.agent_loop.litellm.acompletion", new=AsyncMock(return_value=mock_resp)) as mock_call:
        out = await execute_step(plan.steps[1], plan, req)

    assert out == "Edited main.py successfully"
    assert plan.steps[1].status == StepStatus.COMPLETED
    assert plan.steps[1].result == "Edited main.py successfully"
    # Verify context from step_1 was chained into step_2's prompt
    sent_messages = mock_call.call_args.kwargs["messages"]
    assert "Found main.py" in sent_messages[-1]["content"]


@pytest.mark.asyncio
async def test_run_agent_loop_end_to_end():
    import json
    from unittest.mock import AsyncMock, patch
    from effortgate.agent_loop import run_agent_loop

    class MockChunk:
        def __init__(self, text: str):
            self.text = text

        def model_dump(self, **kwargs):
            return {"choices": [{"delta": {"content": self.text}}]}

    async def mock_stream():
        yield MockChunk("Final unified response.")

    plan_json = json.dumps({
        "goal": "Test end-to-end",
        "steps": [
            {"id": "step_1", "title": "Plan", "phase": "plan", "instruction": "Do step 1", "depends_on": []},
            {"id": "step_2", "title": "Verify", "phase": "verification", "instruction": "Do step 2", "depends_on": ["step_1"]},
        ],
    })

    responses = [
        {"choices": [{"message": {"content": plan_json}}]},       # 1. Planner
        {"choices": [{"message": {"content": "Step 1 output"}}]}, # 2. Step 1
        {"choices": [{"message": {"content": "Step 2 output"}}]}, # 3. Step 2
        mock_stream(),                                             # 4. Synthesizer stream
    ]

    req = {"model": "anthropic/claude-3-5-sonnet", "messages": [{"role": "user", "content": "Run full loop"}]}

    with patch("effortgate.agent_loop.litellm.acompletion", new=AsyncMock(side_effect=responses)) as mock_call:
        chunks = []
        async for chunk in run_agent_loop(req):
            chunks.append(chunk)

    joined = "".join(chunks)
    assert mock_call.await_count == 4
    assert "[AGENT] Plan created: Test end-to-end (2 steps)" in joined
    assert "[AGENT] Executing step_1: Plan..." in joined
    assert "[AGENT] Executing step_2: Verify..." in joined
    assert "Final unified response." in joined
    assert "data: [DONE]" in joined
