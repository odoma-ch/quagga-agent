"""
title: Chat ID Injector
author: open-webui
author_url: https://github.com/open-webui
funding_url: https://github.com/open-webui
version: 0.1
Function Name: ChatIDInlet
"""

from pydantic import BaseModel, Field
from typing import Optional


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0, description="Priority level for the filter operations."
        )
        pass

    def __init__(self):
        self.valves = self.Valves()
        pass

    def inlet(self, body: dict, __metadata__: Optional[dict] = None) -> dict:
        if __metadata__ is None:
            __metadata__ = {}
        chat_id = __metadata__.get("chat_id", "")
        body["chat_id"] = chat_id
        return body
