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

async def run_agent_loop(request_data: Dict[str, Any]) -> AsyncGenerator[str, None]:
    """
    Executes the AGENT tier structured Plan-and-Execute loop.
    1. Emit planning progress chunks via SSE.
    2. Prompt planner to generate a structured 3-step action DAG.
    3. Execute steps and emit unified summary.
    """
    def emit_chunk(text: str) -> str:
        return f"data: {json.dumps({'choices': [{'delta': {'content': text}}]})}\n\n"
        
    yield emit_chunk("[AGENT] Starting planning phase...\n")
    
    # In a fully realized system, we would:
    # - Run a specific prompt to generate the plan
    # - Parse the plan DAG
    # - Execute each step using litellm
    # - Handle retries for failures
    
    # For now, we simulate the structured loop by making a single larger call 
    # but injecting a planning instruction, and streaming it.
    
    messages = request_data.get("messages", [])
    
    # Inject planning instructions
    planning_instruction = (
        "You are in AGENT mode. Follow this structured process:\n"
        "1. Create a brief 3-step plan.\n"
        "2. Execute the steps logically.\n"
        "3. Provide a unified summary with code diffs.\n"
    )
    
    # Try to insert into system prompt
    inserted = False
    for msg in messages:
        if msg.get("role") == "system":
            msg["content"] = str(msg["content"]) + "\n\n" + planning_instruction
            inserted = True
            break
            
    if not inserted:
        messages.insert(0, {"role": "system", "content": planning_instruction})
        
    request_data["messages"] = messages
    
    yield emit_chunk("[AGENT] Executing steps...\n\n")
    
    try:
        response = await litellm.acompletion(**request_data)
        async for chunk in response:
            data = chunk.model_dump(exclude_unset=True)
            yield f"data: {json.dumps(data)}\n\n"
            
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Error in agent execution: {str(e)}")
        error_msg = {"error": {"message": str(e), "type": "agent_internal_error", "param": None, "code": "500"}}
        yield f"data: {json.dumps(error_msg)}\n\n"
        yield "data: [DONE]\n\n"
