"""
title: Quagga Progress Translator
author: Odoma
author_url: https://github.com/open-webui
funding_url: https://github.com/open-webui
version: 0.2
Function Name: QuaggaProgressPipe
"""

from pydantic import BaseModel, Field
from typing import Optional


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0, description="Priority level for the filter operations."
        )

    def __init__(self):
        self.valves = self.Valves()
        self._last_action = "kg_query"

    async def stream(
        self, event: dict, __event_emitter__: Optional[callable] = None
    ) -> Optional[dict]:
        if not isinstance(event, dict):
            return event

        chunk_type = event.get("x_chunk_type")

        if chunk_type == "progress":
            if __event_emitter__:
                description = event.get("x_progress", "")
                if not description:
                    choices = event.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        description = delta.get("content", "")
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "action": self._last_action,
                            "description": description,
                            "done": False,
                        },
                    }
                )
            return None

        if chunk_type == "tool_call":
            choices = event.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                if "Querying knowledge graph" in content:
                    self._last_action = "kg_query"
                elif "Validating" in content:
                    self._last_action = "kg_validate"
                elif "Selecting" in content:
                    self._last_action = "kg_select"
                elif "Checking" in content:
                    self._last_action = "kg_describe"
                else:
                    self._last_action = "kg_query"

        if chunk_type == "final":
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "action": self._last_action,
                            "description": "Complete",
                            "done": True,
                            "hidden": True,
                        },
                    }
                )

        return event
