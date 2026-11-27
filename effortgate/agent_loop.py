import json
import logging
from enum import Enum
from typing import AsyncGenerator, Dict, Any, List, Optional, Set
from pydantic import BaseModel, Field
import litellm

logger = logging.getLogger(__name__)


class StepStatus(str, Enum):
    """Execution status of a single step in an AgentPlan DAG."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StepPhase(str, Enum):
    """Lifecycle phase of a step in the Plan-and-Execute loop."""
    PLAN = "plan"
    ACTION = "action"
    VERIFICATION = "verification"
    SYNTHESIS = "synthesis"


class AgentStep(BaseModel):
    """Represents a single node in the multi-step execution DAG."""
    id: str = Field(..., description="Unique step identifier (e.g., 'step_1').")
    title: str = Field(..., description="Short human-readable summary of the step.")
    phase: StepPhase = Field(default=StepPhase.ACTION, description="Execution phase of this step.")
    instruction: str = Field(..., description="Actionable instruction or prompt for this step.")
    depends_on: List[str] = Field(default_factory=list, description="IDs of prerequisite steps.")
    status: StepStatus = Field(default=StepStatus.PENDING)
    result: Optional[str] = Field(default=None, description="Output produced by this step.")
    error: Optional[str] = Field(default=None, description="Error message if this step failed.")


class AgentPlan(BaseModel):
    """Directed Acyclic Graph (DAG) representing a multi-step agent execution plan."""
    goal: str = Field(..., description="High-level objective for the agent run.")
    steps: List[AgentStep] = Field(default_factory=list, description="Ordered list of DAG steps.")
    max_steps: int = Field(default=5, ge=1, description="Maximum allowed steps for safety.")
    timeout_seconds: float = Field(default=120.0, gt=0.0, description="Overall execution timeout.")

    def validate_dag(self) -> None:
        """
        Validates that:
        1. Step count does not exceed max_steps.
        2. Step IDs are unique.
        3. All depends_on references point to valid step IDs.
        4. The dependency graph contains no cycles.
        """
        if len(self.steps) > self.max_steps:
            raise ValueError(f"Plan exceeds maximum allowed steps ({len(self.steps)} > {self.max_steps}).")

        step_ids: Set[str] = set()
        for step in self.steps:
            if step.id in step_ids:
                raise ValueError(f"Duplicate step ID detected: '{step.id}'.")
            step_ids.add(step.id)

        adj: Dict[str, List[str]] = {step.id: list(step.depends_on) for step in self.steps}
        for step_id, deps in adj.items():
            for dep in deps:
                if dep not in step_ids:
                    raise ValueError(f"Step '{step_id}' depends on unknown step '{dep}'.")

        # Cycle detection using 3-color DFS (0=unvisited, 1=visiting, 2=visited)
        state: Dict[str, int] = {step_id: 0 for step_id in step_ids}

        def _visit(node: str) -> None:
            state[node] = 1
            for dep in adj[node]:
                if state[dep] == 1:
                    raise ValueError(f"Cycle detected involving step '{node}' and '{dep}'.")
                if state[dep] == 0:
                    _visit(dep)
            state[node] = 2

        for step_id in step_ids:
            if state[step_id] == 0:
                _visit(step_id)

    def get_ready_steps(self) -> List[AgentStep]:
        """Returns PENDING steps whose prerequisite dependencies are all COMPLETED."""
        status_by_id = {step.id: step.status for step in self.steps}
        ready: List[AgentStep] = []
        for step in self.steps:
            if step.status != StepStatus.PENDING:
                continue
            if all(status_by_id.get(dep) == StepStatus.COMPLETED for dep in step.depends_on):
                ready.append(step)
        return ready

    def is_complete(self) -> bool:
        """Returns True when no steps remain PENDING or IN_PROGRESS."""
        terminal = {StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED}
        return all(step.status in terminal for step in self.steps)

    def has_failed(self) -> bool:
        """Returns True if any step has FAILED."""
        return any(step.status == StepStatus.FAILED for step in self.steps)

def _extract_content(response: Any) -> str:
    """Extracts assistant text content from a LiteLLM response object or dict."""
    if isinstance(response, dict):
        choices = response.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            return str(msg.get("content") or "")
        return ""
    choices = getattr(response, "choices", None)
    if choices:
        msg = getattr(choices[0], "message", None)
        if msg is not None:
            return str(getattr(msg, "content", "") or "")
    return ""


def _strip_json_fences(raw_text: str) -> str:
    """Strips optional markdown ```json ... ``` code fences from LLM output."""
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _build_llm_kwargs(
    request_data: Dict[str, Any],
    messages: List[Dict[str, Any]],
    stream: bool = False,
) -> Dict[str, Any]:
    """Builds a clean litellm.acompletion kwargs dict without mutating request_data."""
    allowed_keys = {
        "model",
        "api_key",
        "api_base",
        "temperature",
        "max_tokens",
        "top_p",
        "timeout",
    }
    kwargs: Dict[str, Any] = {k: v for k, v in request_data.items() if k in allowed_keys and v is not None}
    kwargs["messages"] = messages
    kwargs["stream"] = stream
    return kwargs


async def generate_plan(request_data: Dict[str, Any], max_steps: int = 5) -> AgentPlan:
    """
    Prompts the model to produce a structured JSON AgentPlan DAG.
    Falls back to a safe 2-step plan if the model returns unparseable output.
    """
    user_messages = request_data.get("messages", [])
    last_user_prompt = ""
    for msg in reversed(user_messages):
        if msg.get("role") == "user":
            last_user_prompt = str(msg.get("content", ""))
            break

    planner_system = (
        "You are an expert engineering planner. Break the user's request into a concise "
        f"directed acyclic graph (DAG) of 2 to {max_steps} steps.\n"
        "Respond ONLY with valid JSON matching this schema:\n"
        "{\n"
        '  "goal": "short summary of objective",\n'
        '  "steps": [\n'
        '    {"id": "step_1", "title": "...", "phase": "plan|action|verification|synthesis", '
        '"instruction": "...", "depends_on": []}\n'
        "  ]\n"
        "}"
    )

    planner_messages = [
        {"role": "system", "content": planner_system},
        *user_messages,
    ]
    kwargs = _build_llm_kwargs(request_data, planner_messages, stream=False)

    try:
        response = await litellm.acompletion(**kwargs)
        raw_content = _strip_json_fences(_extract_content(response))
        parsed = json.loads(raw_content)
        plan = AgentPlan(
            goal=parsed.get("goal") or last_user_prompt[:120] or "Execute user task",
            steps=[AgentStep(**s) for s in parsed.get("steps", [])],
            max_steps=max_steps,
        )
        if not plan.steps:
            raise ValueError("Planner returned zero steps.")
        plan.validate_dag()
        return plan
    except Exception as exc:
        logger.warning(f"Planner JSON parsing/validation failed ({exc}); using fallback 2-step plan.")
        fallback = AgentPlan(
            goal=last_user_prompt[:120] or "Execute user task",
            max_steps=max_steps,
            steps=[
                AgentStep(
                    id="step_1",
                    title="Analyze and execute task",
                    phase=StepPhase.ACTION,
                    instruction=last_user_prompt or "Complete the requested task thoroughly.",
                    depends_on=[],
                ),
                AgentStep(
                    id="step_2",
                    title="Verify solution",
                    phase=StepPhase.VERIFICATION,
                    instruction="Review the solution for correctness, edge cases, and completeness.",
                    depends_on=["step_1"],
                ),
            ],
        )
        fallback.validate_dag()
        return fallback


async def execute_step(
    step: AgentStep,
    plan: AgentPlan,
    request_data: Dict[str, Any],
) -> str:
    """
    Executes a single AgentStep using context chained from completed prerequisite steps.
    Updates step.status and step.result (or step.error on failure).
    """
    step.status = StepStatus.IN_PROGRESS

    completed_context_parts: List[str] = []
    step_map = {s.id: s for s in plan.steps}
    context_ids = step.depends_on if step.depends_on else [
        s.id for s in plan.steps if s.status == StepStatus.COMPLETED
    ]
    for dep_id in context_ids:
        dep_step = step_map.get(dep_id)
        if dep_step and dep_step.status == StepStatus.COMPLETED and dep_step.result:
            completed_context_parts.append(f"[{dep_step.id}: {dep_step.title}]\n{dep_step.result}")

    context_block = "\n\n".join(completed_context_parts)
    step_prompt = (
        f"Overall Goal: {plan.goal}\n"
        f"Current Step ({step.id} - {step.phase.value}): {step.title}\n"
        f"Instruction: {step.instruction}\n"
    )
    if context_block:
        step_prompt += f"\nContext from previous steps:\n{context_block}\n"

    step_messages = [
        {"role": "system", "content": "You are executing one step of a multi-step engineering plan. Be concise, accurate, and actionable."},
        {"role": "user", "content": step_prompt},
    ]
    kwargs = _build_llm_kwargs(request_data, step_messages, stream=False)

    try:
        response = await litellm.acompletion(**kwargs)
        result = _extract_content(response)
        step.result = result
        step.status = StepStatus.COMPLETED
        return result
    except Exception as exc:
        step.error = str(exc)
        step.status = StepStatus.FAILED
        raise


async def synthesize_response(
    plan: AgentPlan,
    request_data: Dict[str, Any],
    stream: bool = True,
) -> Any:
    """
    Synthesizes all completed step outputs into a unified final response or code diff.
    """
    step_summaries: List[str] = []
    for step in plan.steps:
        if step.status == StepStatus.COMPLETED and step.result:
            step_summaries.append(f"### {step.id}: {step.title}\n{step.result}")
        elif step.status == StepStatus.FAILED:
            step_summaries.append(f"### {step.id}: {step.title} (FAILED)\nError: {step.error}")

    synthesis_prompt = (
        f"Goal: {plan.goal}\n\n"
        "Combine the following executed step results into a single, cohesive final response "
        "(including unified code diffs or instructions where appropriate):\n\n"
        + "\n\n".join(step_summaries)
    )

    synth_messages = [
        *request_data.get("messages", []),
        {"role": "user", "content": synthesis_prompt},
    ]
    kwargs = _build_llm_kwargs(request_data, synth_messages, stream=stream)
    return await litellm.acompletion(**kwargs)


async def run_agent_loop(request_data: Dict[str, Any]) -> AsyncGenerator[str, None]:
    """
    Executes the AGENT tier structured Plan-and-Execute loop.
    1. Generate structured DAG plan via Planner.
    2. Execute ready DAG steps sequentially with context chaining.
    3. Synthesize and stream the unified final response via SSE.
    """
    def emit_chunk(text: str) -> str:
        return f"data: {json.dumps({'choices': [{'delta': {'content': text}}]})}\n\n"

    yield emit_chunk("[AGENT] Starting planning phase...\n")

    try:
        plan = await generate_plan(request_data)
        yield emit_chunk(f"[AGENT] Plan created: {plan.goal} ({len(plan.steps)} steps)\n")

        while not plan.is_complete():
            ready_steps = plan.get_ready_steps()
            if not ready_steps:
                break
            for step in ready_steps:
                yield emit_chunk(f"[AGENT] Executing {step.id}: {step.title}...\n")
                await execute_step(step, plan, request_data)

        yield emit_chunk("[AGENT] Synthesizing final response...\n\n")
        response = await synthesize_response(plan, request_data, stream=True)
        async for chunk in response:
            data = chunk.model_dump(exclude_unset=True)
            yield f"data: {json.dumps(data)}\n\n"

        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Error in agent execution: {str(e)}")
        error_msg = {"error": {"message": str(e), "type": "agent_internal_error", "param": None, "code": "500"}}
        yield f"data: {json.dumps(error_msg)}\n\n"
        yield "data: [DONE]\n\n"
