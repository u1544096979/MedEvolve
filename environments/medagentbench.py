"""Native MedAgentBench actions for the pinned upstream implementation.

The bundled interface declarations come from Stanford Machine Learning Group's
MedAgentBench repository, commit 99260117137b09f04837a8c18d18a1107efa55ae.
The upstream repository is licensed under MIT; see its LICENSE file. The action
feedback and parsing rules below follow src/server/tasks/medagentbench/__init__.py.
"""

import json
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
from agents import Tool

from contracts import JsonObject
from environments.errors import PatientAccessError
from environments.fhir import FHIRClient, owners, patient_id, references


class NativeTools(list[Tool]):
    """List-compatible marker carrying the context for native text actions."""

    def __init__(self, endpoint: str, patient_ids: list[str], functions: list[JsonObject]):
        super().__init__()
        self.endpoint = endpoint.rstrip("/") + "/"
        self.patient_ids = sorted({patient_id(value) for value in patient_ids})
        self.functions = functions
        self._client = FHIRClient(self.endpoint, self.patient_ids)

    async def execute(self, action: str) -> JsonObject:
        """Execute exactly one upstream-format GET, POST, or FINISH action."""
        original = action
        action = action.replace("```tool_code", "").replace("```", "").strip()
        if action.startswith("GET"):
            raw_url = action[3:].strip()
            parts = urlsplit(raw_url)
            base = urlsplit(self.endpoint)
            target = (
                parts
                if parts.scheme or parts.netloc
                else urlsplit(self.endpoint + raw_url.lstrip("/"))
            )
            if (target.scheme, target.netloc) != (base.scheme, base.netloc) or not (
                target.path == base.path.rstrip("/")
                or target.path.startswith(base.path)
            ):
                raise PatientAccessError("FHIR request leaves this instance")
            relative = target.path[len(base.path):]
            resource_type = relative.split("/", 1)[0]
            if not resource_type or "/" in relative:
                raise PatientAccessError("Only FHIR search URLs are supported")
            query_params = parse_qsl(target.query, keep_blank_values=True)
            selected = set()
            for key in ("patient", "subject"):
                for _, value in (pair for pair in query_params if pair[0] == key):
                    selected.update(
                        patient_id(part.strip()) for part in value.split(",")
                    )
            if selected and not selected.issubset(self.patient_ids):
                raise PatientAccessError(
                    "FHIR request selects a patient outside the allowed scope"
                )
            if resource_type == "Patient" and not selected:
                query_params.append(("_id", ",".join(self.patient_ids)))
            elif resource_type != "Patient" and not selected:
                query_params.append(
                    ("patient", ",".join(self.patient_ids))
                )
            query_params.append(("_format", "json"))
            request_url = urlunsplit(
                (base.scheme, base.netloc, target.path, urlencode(query_params), "")
            )
            request = {"action": original, "url": request_url, "payload": None}
            try:
                async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
                    response = await client.get(
                        request_url, headers={"Accept": "application/fhir+json"}
                    )
                    response.raise_for_status()
                    result = (
                        response.json()
                        if response.headers.get("Content-Type") == "application/json"
                        else response.text
                    )
            except Exception as exc:
                return {
                    "kind": "GET",
                    "request": request,
                    "reply": f"Error in sending the GET request: {exc}",
                    "failure_category": "environment_error",
                }
            if isinstance(result, dict) and result.get("resourceType") == "Bundle":
                entries = result.get("entry", [])
                if resource_type == "Patient":
                    result["entry"] = [
                        entry for entry in entries
                        if entry.get("resource", {}).get("resourceType") == "Patient"
                        and entry.get("resource", {}).get("id") in self.patient_ids
                    ]
                else:
                    for entry in entries:
                        resource = entry.get("resource", {})
                        self._client._check_resource(
                            resource, selected or set(self.patient_ids)
                        )
            return {
                "kind": "GET",
                "request": request,
                "reply": (
                    "Here is the response from the GET request:\n"
                    f"{result}. Please call FINISH if you have got answers for all "
                    "the questions and finished all the requested tasks"
                ),
                "data": result,
            }

        if action.startswith("POST"):
            lines = action.split("\n")
            raw_url = lines[0][4:].strip()
            payload_text = "\n".join(lines[1:])
            parts = urlsplit(raw_url)
            base = urlsplit(self.endpoint)
            target = (
                parts
                if parts.scheme or parts.netloc
                else urlsplit(self.endpoint + raw_url.lstrip("/"))
            )
            if (target.scheme, target.netloc) != (base.scheme, base.netloc) or not target.path.startswith(base.path):
                raise PatientAccessError("FHIR request leaves this instance")
            resource_type = target.path[len(base.path):].strip("/")
            request = {
                "action": original,
                "url": urlunsplit(
                    (base.scheme, base.netloc, target.path, target.query, "")
                ),
                "payload": None,
            }
            try:
                payload = json.loads(payload_text)
            except (json.JSONDecodeError, TypeError):
                return {"kind": "POST", "request": request, "reply": "Invalid POST request"}
            request["payload"] = payload
            if not isinstance(payload, dict):
                raise PatientAccessError("POST payload must identify exactly one allowed patient")
            try:
                target_patients = owners(payload)
            except (KeyError, TypeError, AttributeError) as exc:
                raise PatientAccessError(
                    "POST payload must identify exactly one allowed patient"
                ) from exc
            if len(target_patients) != 1 or not target_patients.issubset(self.patient_ids):
                raise PatientAccessError("POST must target exactly one allowed patient")
            for ref in references(payload):
                if ref.startswith("Patient/") and patient_id(ref) not in self.patient_ids:
                    raise PatientAccessError(
                        "POST references a patient outside the allowed scope"
                    )
            return {
                "kind": "POST",
                "request": request,
                "reply": (
                    "POST request accepted and executed successfully. Please call "
                    "FINISH if you have got answers for all the questions and finished "
                    "all the requested tasks"
                ),
            }

        if action.startswith("FINISH(") and action.endswith(")"):
            raw_answers = action[len("FINISH("):-1]
            try:
                answers = json.loads(raw_answers)
            except json.JSONDecodeError:
                return {"kind": "error", "request": {"action": original}, "reply": "Invalid FINISH request"}
            if not isinstance(answers, list):
                return {"kind": "error", "request": {"action": original}, "reply": "Invalid FINISH request"}
            return {"kind": "FINISH", "request": {"action": original}, "reply": "FINISH", "answers": answers}
        return {"kind": "error", "request": {"action": original}, "reply": "Invalid action"}


def load_functions() -> list[JsonObject]:
    return json.loads(
        Path(__file__)
        .with_name("medagentbench-functions.json")
        .read_text(encoding="utf-8")
    )
