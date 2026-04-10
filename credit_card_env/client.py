from __future__ import annotations
from typing import Any
import requests

class CreditCardEnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session() # Faster for multiple requests

    @staticmethod
    def _unwrap_observation(payload: dict[str, Any]) -> dict[str, Any]:
        # This handles the "nested observation" bug perfectly
        observation = payload.get("observation", payload)
        if isinstance(observation, dict):
            nested = observation.get("observation")
            if isinstance(nested, dict):
                return nested
        return observation

    def reset(self, task_id: str) -> dict[str, Any]:
        response = self.session.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id},
            timeout=10
        )
        response.raise_for_status()
        return self._unwrap_observation(response.json())

    def step(self, action: int) -> dict[str, Any]:
        response = self.session.post(
            f"{self.base_url}/step",
            json={"action": action},
            timeout=10
        )
        response.raise_for_status()
        payload = response.json()
        
        # We unwrap the observation part but keep reward/done at the top level
        if "observation" in payload:
            payload["observation"] = self._unwrap_observation(payload["observation"])
        
        return payload