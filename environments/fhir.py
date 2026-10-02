"""Patient-scoped native FHIR transport for PhysicianBench."""

import base64
import copy
import re
from collections.abc import Iterator
from urllib.parse import urljoin, urlsplit

import httpx

from contracts import JsonObject
from environments.errors import PatientAccessError

FHIR_ID = re.compile(r"[A-Za-z0-9\-.]{1,64}\Z")
SHARED_TYPES = {
    "Practitioner",
    "PractitionerRole",
    "Organization",
    "Location",
    "HealthcareService",
    "Medication",
    "Substance",
}


def patient_id(value: str) -> str:
    identifier = value.removeprefix("Patient/")
    if not FHIR_ID.fullmatch(identifier):
        raise PatientAccessError(
            "Patient must be a local FHIR ID or Patient/<id> reference"
        )
    return identifier


def references(value: object) -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "reference" and isinstance(item, str):
                yield item
            else:
                yield from references(item)
    elif isinstance(value, list):
        for item in value:
            yield from references(item)


def owners(resource: dict) -> set[str]:
    if resource.get("resourceType") == "Patient":
        return {patient_id(resource["id"])}
    result = set()
    for field in ("subject", "patient", "beneficiary", "for"):
        for ref in references(resource.get(field)):
            if ref.startswith("Patient/"):
                result.add(patient_id(ref))
    if resource.get("resourceType") == "Appointment":
        for ref in references(resource.get("participant")):
            if ref.startswith("Patient/"):
                result.add(patient_id(ref))
    return result


def decode_document_notes(resource: dict) -> dict:
    decoded = copy.deepcopy(resource)
    if resource.get("resourceType") == "DocumentReference":
        for content in decoded.get("content", []):
            attachment = content.get("attachment", {})
            if attachment.get("data") and attachment.get("contentType", "").startswith(
                "text/"
            ):
                attachment["data"] = base64.b64decode(
                    attachment["data"], validate=True
                ).decode("utf-8")
    return decoded


class FHIRClient:
    def __init__(self, endpoint: str, patient_ids: list[str]):
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("endpoint must be a native FHIR HTTP(S) base URL")
        self.endpoint = endpoint.rstrip("/") + "/"
        self.patient_ids = frozenset(patient_id(p) for p in patient_ids)
        if not self.patient_ids:
            raise PatientAccessError("At least one allowed patient ID is required")

    def _allowed(self, identifier: str) -> str:
        identifier = patient_id(identifier)
        if identifier not in self.patient_ids:
            raise PatientAccessError(
                f"Patient {identifier} is outside the allowed patient scope"
            )
        return identifier

    def _check_resource(
        self, resource: dict, expected: set[str], require_owner: bool = True
    ) -> None:
        resource_owners = owners(resource)
        if (require_owner and not resource_owners) or not resource_owners <= expected:
            raise PatientAccessError(
                "FHIR resource is outside the selected patient scope"
            )
        # References to other patients must not leak through nested or contained records.
        for ref in references(resource):
            if ref.startswith("Patient/") and patient_id(ref) not in expected:
                raise PatientAccessError("FHIR resource references another patient")
        for contained in resource.get("contained", []):
            self._check_resource(contained, expected, require_owner=False)

    def _page_url(self, next_link: str) -> str:
        url = urljoin(self.endpoint, next_link)
        target, base = urlsplit(url), urlsplit(self.endpoint)
        if (target.scheme, target.netloc) != (base.scheme, base.netloc) or not (
            target.path == base.path.rstrip("/") or target.path.startswith(base.path)
        ):
            raise PatientAccessError("FHIR pagination link leaves this instance")
        return url

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
            response = await client.request(
                method,
                urljoin(self.endpoint, path),
                headers={
                    "Accept": "application/fhir+json",
                    "Content-Type": "application/fhir+json",
                },
                **kwargs,
            )
            response.raise_for_status()
            return response.json()

    async def _validate_reference(
        self, ref: str, expected: set[str], visited: set[str]
    ) -> None:
        if ref.startswith("#"):
            return
        parts = ref.split("/")
        if len(parts) != 2 or not parts[0].isalnum() or not FHIR_ID.fullmatch(parts[1]):
            raise PatientAccessError(
                "Only local FHIR resource references are supported"
            )
        resource_type, identifier = parts
        if resource_type == "Patient":
            if self._allowed(identifier) not in expected:
                raise PatientAccessError(
                    "Reference belongs to another selected patient"
                )
            return
        if resource_type in SHARED_TYPES or ref in visited:
            return
        visited.add(ref)
        resource = await self._request("GET", ref)
        if (
            resource.get("resourceType") != resource_type
            or resource.get("id") != identifier
        ):
            raise PatientAccessError(
                "FHIR server returned a different referenced resource"
            )
        self._check_resource(resource, expected)
        for linked in references(resource):
            await self._validate_reference(linked, expected, visited)

    async def search(
        self,
        resource_type: str,
        params: dict,
        page_limit: int,
        decode_notes: bool = False,
    ) -> JsonObject:
        if page_limit < 1 or params.get("_count", 1) < 1:
            raise ValueError("count and page_limit must be positive")
        params = dict(params)
        selected = {
            self._allowed(params[key])
            for key in ("patient", "subject")
            if params.get(key)
        }
        if len(selected) > 1:
            raise PatientAccessError("patient and subject must select the same patient")
        selected = selected or set(self.patient_ids)
        if resource_type == "Patient":
            params["_id"] = ",".join(sorted(selected))
        else:
            params.pop("subject", None)
            params["patient"] = ",".join(sorted(selected))
        for field, ref_type in (
            ("encounter", "Encounter"),
            ("requester", "Practitioner"),
            ("performer", "Practitioner"),
        ):
            if params.get(field):
                ref = (
                    params[field]
                    if "/" in params[field]
                    else f"{ref_type}/{params[field]}"
                )
                await self._validate_reference(ref, selected, set())
        result = await self._collect(resource_type, params, page_limit, selected)
        if decode_notes:
            result["entries"] = [decode_document_notes(r) for r in result["entries"]]
        return result

    async def read(self, source_id: str) -> JsonObject:
        parts = source_id.split("/")
        if len(parts) != 2 or not parts[0].isalnum() or not FHIR_ID.fullmatch(parts[1]):
            raise PatientAccessError("Evidence must be a local ResourceType/id reference")
        resource = await self._request("GET", source_id)
        if resource.get("resourceType") != parts[0] or resource.get("id") != parts[1]:
            raise PatientAccessError("FHIR server returned a different evidence resource")
        self._check_resource(resource, set(self.patient_ids))
        return decode_document_notes(resource)

    async def _collect(
        self,
        path: str,
        params: dict,
        page_limit: int | None,
        selected: set[str],
        everything: bool = False,
    ) -> JsonObject:
        entries = []
        pages = 0
        total = None
        url = urljoin(self.endpoint, path)
        seen = set()
        async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
            while url and (page_limit is None or pages < page_limit):
                if url in seen:
                    raise ValueError("FHIR server returned a pagination cycle")
                seen.add(url)
                response = await client.get(
                    url, params=params, headers={"Accept": "application/fhir+json"}
                )
                response.raise_for_status()
                bundle = response.json()
                if bundle.get("resourceType") != "Bundle":
                    raise ValueError("Expected a native FHIR Bundle")
                pages += 1
                total = bundle.get("total", total)
                for entry in bundle.get("entry", []):
                    resource = entry.get("resource", {})
                    if not everything and resource.get("resourceType") != path:
                        continue
                    # $everything may include shared supporting records; it must never include
                    # another patient's records. Clinical entries must have a patient owner.
                    self._check_resource(
                        resource,
                        selected,
                        require_owner=resource.get("resourceType") not in SHARED_TYPES,
                    )
                    entries.append(resource)
                next_link = next(
                    (
                        link["url"]
                        for link in bundle.get("link", [])
                        if link.get("relation") == "next"
                    ),
                    None,
                )
                url = self._page_url(next_link) if next_link else ""
                params = None
        return {"entries": entries, "total": total, "pages": pages}

    async def create(self, resource: dict) -> JsonObject:
        selected = owners(resource)
        if len(selected) != 1:
            raise PatientAccessError("A write must target exactly one allowed patient")
        self._allowed(next(iter(selected)))
        self._check_resource(resource, selected)
        visited: set[str] = set()
        for ref in references(resource):
            await self._validate_reference(ref, selected, visited)
        # POST is deliberately not retried: a lost response must not duplicate a clinical action.
        created = await self._request("POST", resource["resourceType"], json=resource)
        self._check_resource(created, selected)
        return created

    async def snapshot(self, identifier: str) -> list[dict]:
        identifier = self._allowed(identifier)
        # R4 Patient/$everything retains linked supporting records and follows all pages.
        result = await self._collect(
            f"Patient/{identifier}/$everything",
            {"_count": 200},
            None,
            {identifier},
            everything=True,
        )
        if not any(
            r.get("resourceType") == "Patient" and r.get("id") == identifier
            for r in result["entries"]
        ):
            raise ValueError(f"Patient/$everything did not return Patient/{identifier}")
        return result["entries"]
