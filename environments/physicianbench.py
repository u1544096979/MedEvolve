"""Native PhysicianBench FHIR tools adapted to async, instance-bound HTTP.

Derived from tools/fhir_api_functions.py and agent/tool_registry.py at PhysicianBench c7efa8fd5b1e4744ada50668efe4b7e84023cbb0.
Licensed under Apache-2.0; the license text is included below.
Modified: HTTP transport, pagination and patient access checks are supplied by FHIRClient.
Native tool names, clinical parameters, resource payloads and referral rule are retained.

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""

from typing import Any

from environments.fhir import FHIRClient


async def fhir_condition_search_problems(
    client: FHIRClient,
    subject: str | None = None,
    patient: str | None = None,
    clinical_status: str | None = None,
    category: str | None = None,
    code: str | None = None,
    onset_date: str | None = None,
    count: int = 10,
    page_limit: int = 1,
) -> dict[str, Any]:
    """Query FHIR R4 Condition search for Problems (USCDI compliant).
    Retrieves problems from a patient's chart including problem list data across all encounters.
    Uses HTTP GET {api_base}/Condition?subject={subject}&patient={patient}&clinical-status={clinical-status}&category={category}

    Args:
        subject: FHIR subject reference (Patient/12345)
        patient: FHIR Patient id or reference (e.g., '12345')
        clinical_status: Clinical status filter (e.g., 'active', 'inactive', 'resolved')
        category: FHIR 'category' token (e.g., 'problem-list-item')
        code: Condition code filter (ICD-10, SNOMED, etc.)
        count: _count page size (server may cap)
        page_limit: max pages to follow via Bundle.link[rel=next]

    Returns:
        A dict with 'entries' (list of Condition resources for problems), 'total' (if available),
            and 'pages' (how many Bundle pages retrieved).
    """
    params = {}
    if subject:
        params["subject"] = subject
    if patient:
        params["patient"] = patient
    if clinical_status:
        params["clinical-status"] = clinical_status
    if category:
        params["category"] = category
    if code:
        params["code"] = code
    if onset_date:
        params["onset-date"] = onset_date
    if count:
        params["_count"] = count
    return await client.search("Condition", params, page_limit, decode_notes=False)


async def fhir_observation_search_labs(
    client: FHIRClient,
    patient: str | None = None,
    subject: str | None = None,
    category: str | None = None,
    code: str | None = None,
    date: str | None = None,
    count: int = 50,
    page_limit: int = 2,
) -> dict[str, Any]:
    """Query FHIR R4 Observation search for Labs (USCDI compliant).
    Returns component level data for lab results. Observation resources contain component results
    that are typically referenced by DiagnosticReport resources.
    Uses HTTP GET {api_base}/Observation?patient={patient}&subject={subject}&category={category}&code={code}&date={date}

    Args:
        patient: FHIR Patient id or reference (e.g., '12345')
        subject: FHIR subject reference (Patient/12345)
        category: FHIR 'category' token (e.g., 'laboratory')
        code: FHIR code filter (e.g., LOINC codes for specific lab components)
        date: The date range for when the Observation was taken.
        count: _count page size (server may cap)
        page_limit: max pages to follow via Bundle.link[rel=next]

    Returns:
        A dict with 'entries' (list of Observation resources for labs), 'total' (if available),
            and 'pages' (how many Bundle pages retrieved).

        Note:
            - Returns component level data for lab results
            - Limited to microbiology sensitivities and textual observations
            - Most systems expect LOINC codes for component identification
            - Observation.code may be empty if no LOINC code is found
            - Consider using DiagnosticReport.Search to identify lab procedures first
    """
    params = {}
    if patient:
        params["patient"] = patient
    if subject:
        params["subject"] = subject
    if category:
        params["category"] = category
    if code:
        params["code"] = code
    if date:
        params["date"] = date
    if count:
        params["_count"] = count
    return await client.search("Observation", params, page_limit, decode_notes=False)


async def fhir_observation_search_vitals(
    client: FHIRClient,
    patient: str | None = None,
    subject: str | None = None,
    code: str | None = None,
    date: str | None = None,
    category: str | None = None,
    count: int = 50,
    page_limit: int = 2,
) -> dict[str, Any]:
    """Query FHIR R4 Observation search for Vitals (USCDI compliant).
    Retrieves vital sign data from a patient's chart, as well as any other non-duplicable
    data found in the patient's flowsheets across all encounters.
    Uses HTTP GET {api_base}/Observation?patient={patient}&subject={subject}&code={code}&date={date}&category={category}

    Args:
        patient: FHIR Patient id or reference (e.g., '12345')
        subject: FHIR subject reference (Patient/12345)
        code: FHIR code filter (e.g., LOINC codes for specific vital signs)
        date: The date range for when the Observation was taken.
        category: FHIR 'category' token (e.g., 'vital-signs')
        count: _count page size (server may cap)
        page_limit: max pages to follow via Bundle.link[rel=next]

    Returns:
        A dict with 'entries' (list of Observation resources for vitals), 'total' (if available),
            and 'pages' (how many Bundle pages retrieved).

        Note:
            - Retrieves vital sign data and non-duplicable flowsheet data across all encounters
            - This API may behave differently when used in a patient-facing context
            - Common vital sign codes include blood pressure, heart rate, temperature, etc.
            - Consider filtering by category='vital-signs' for most relevant results
    """
    params = {}
    if patient:
        params["patient"] = patient
    if subject:
        params["subject"] = subject
    if code:
        params["code"] = code
    if date:
        params["date"] = date
    if category:
        params["category"] = category
    if count:
        params["_count"] = count
    return await client.search("Observation", params, page_limit, decode_notes=False)


async def fhir_patient_search_demographics(
    client: FHIRClient,
    address: str | None = None,
    address_city: str | None = None,
    address_postalcode: str | None = None,
    address_state: str | None = None,
    birthdate: str | None = None,
    family: str | None = None,
    gender: str | None = None,
    given: str | None = None,
    identifier: str | None = None,
    name: str | None = None,
    own_name: str | None = None,
    own_prefix: str | None = None,
    partner_name: str | None = None,
    partner_prefix: str | None = None,
    telecom: str | None = None,
    legal_sex: str | None = None,
    count: int = 50,
    page_limit: int = 6,
) -> dict[str, Any]:
    """Query FHIR R4 Patient search for Demographics (USCDI compliant).
    Allows filtering or searching for patients based on demographic parameters and retrieves
    patient demographic information from patient charts for each matching patient record.
    Uses HTTP GET {api_base}/Patient?address={address}&address-city={address-city}&...

    Args:
        address: Patient address search parameter
        address_city: Patient address city search parameter
        address_postalcode: Patient address postal code search parameter
        address_state: Patient address state search parameter
        birthdate: Patient birth date (e.g., '1990-01-01' or date range)
        family: Patient family name search parameter
        gender: Patient gender (e.g., 'male', 'female', 'other', 'unknown')
        given: Patient given name search parameter
        identifier: Patient identifier search parameter
        name: Patient name search parameter (family and/or given)
        own_name: Patient's own name search parameter
        own_prefix: Patient's own name prefix search parameter
        partner_name: Patient's partner name search parameter
        partner_prefix: Patient's partner name prefix search parameter
        telecom: Patient telecom (phone, email) search parameter
        legal_sex: Patient legal sex parameter
        count: _count page size (server may cap)
        page_limit: max pages to follow via Bundle.link[rel=next]

    Returns:
        A dict with 'entries' (list of Patient resources with demographics), 'total' (if available),
            and 'pages' (how many Bundle pages retrieved).

        Note:
            - Retrieves patient demographic information from patient charts
            - Does not respect the same filtering as MyChart (except careProvider parameter)
            - Supports comprehensive demographic search across multiple parameters
            - Consider privacy and access control requirements when using patient search
    """
    params = {}
    if address:
        params["address"] = address
    if address_city:
        params["address-city"] = address_city
    if address_postalcode:
        params["address-postalcode"] = address_postalcode
    if address_state:
        params["address-state"] = address_state
    if birthdate:
        params["birthdate"] = birthdate
    if family:
        params["family"] = family
    if gender:
        params["gender"] = gender
    if given:
        params["given"] = given
    if identifier:
        params["identifier"] = identifier
    if name:
        params["name"] = name
    if own_name:
        params["own-name"] = own_name
    if own_prefix:
        params["own-prefix"] = own_prefix
    if partner_name:
        params["partner-name"] = partner_name
    if partner_prefix:
        params["partner-prefix"] = partner_prefix
    if telecom:
        params["telecom"] = telecom
    if legal_sex:
        params["legal-sex"] = legal_sex
    if count:
        params["_count"] = count
    return await client.search("Patient", params, page_limit, decode_notes=False)


async def fhir_procedure_search_orders(
    client: FHIRClient,
    date: str | None = None,
    patient: str | None = None,
    subject: str | None = None,
    category: str | None = None,
    count: int = 50,
    page_limit: int = 2,
) -> dict[str, Any]:
    """Query FHIR R4 Procedure search for Orders (USCDI compliant).
    Retrieves completed procedures performed on or with a patient as part of care provision.
    Includes surgeries, endoscopies, biopsies, counseling, physiotherapy, and other procedures.
    Uses HTTP GET {api_base}/Procedure?date={date}&patient={patient}&subject={subject}&category={category}

    Args:
        date: Date or period that the procedure was performed, using the FHIR date parameter format.
        patient: FHIR Patient id or reference (e.g., '12345')
        subject: FHIR subject reference (Patient/12345)
        category: FHIR 'category' token for procedure classification
        count: _count page size (server may cap)
        page_limit: max pages to follow via Bundle.link[rel=next]

    Returns:
        A dict with 'entries' (list of Procedure resources for orders), 'total' (if available),
            and 'pages' (how many Bundle pages retrieved).

        Note:
            - Designed for high-level summarization of procedure occurrences
            - Only completed procedures are returned when searching
            - Not intended for specific procedure log documentation
            - Includes surgeries, endoscopies, biopsies, counseling, physiotherapy, etc.
            - Corresponds with procedures performed as part of care provision
    """
    params = {}
    if date:
        params["date"] = date
    if patient:
        params["patient"] = patient
    if subject:
        params["subject"] = subject
    if category:
        params["category"] = category
    if count:
        params["_count"] = count
    return await client.search("Procedure", params, page_limit, decode_notes=False)


async def fhir_medication_request_search_orders(
    client: FHIRClient,
    patient: str | None = None,
    subject: str | None = None,
    category: str | None = None,
    status: str | None = None,
    authoredon: str | None = None,
    date: str | None = None,
    intent: str | None = None,
    count: int = 50,
    page_limit: int = 2,
) -> dict[str, Any]:
    """Query FHIR R4 MedicationRequest search for Signed Medication Orders (USCDI compliant).
    Retrieves medication orders based on a patient and optionally status or category.
    Returns various types of medications including inpatient-ordered medications, clinic-administered
    medications (CAMS), patient-reported medications, and reconciled medications from Care Everywhere
    and other external sources.
    Uses HTTP GET {api_base}/MedicationRequest?patient={patient}&subject={subject}&category={category}&status={status}&authoredon={authoredon}&date={date}&intent={intent}

    Args:
        patient: FHIR Patient id or reference (e.g., '12345')
        subject: FHIR subject reference (Patient/12345)
        category: FHIR 'category' token for medication classification
        status: Medication request status (e.g., 'active', 'completed', 'cancelled', 'draft')
        authoredon: Date when the prescription was authored (e.g., '2023-01-01' or date range)
        date: The medication administration date (e.g., '2023-01-01' or date range)
        intent: Intent of the medication request (e.g., 'order', 'plan', 'proposal')
        count: _count page size (server may cap)
        page_limit: max pages to follow via Bundle.link[rel=next]

    Returns:
        A dict with 'entries' (list of MedicationRequest resources), 'total' (if available),
            and 'pages' (how many Bundle pages retrieved).

        Note:
            - Returns various types of medications including inpatient-ordered, clinic-administered (CAMS),
              patient-reported, and reconciled medications from external sources
            - R4 version returns patient-reported medications with reportedBoolean element set to True
            - If informant is known for patient-reported medications, it's specified in reportedReference element
            - Supports Backend Systems, Non-OAuth 2.0, Clinicians/Administrative Users, and Patients
            - Industry-Standard Level 1 API subset of USCDI
    """
    params = {}
    if patient:
        params["patient"] = patient
    if subject:
        params["subject"] = subject
    if category:
        params["category"] = category
    if status:
        params["status"] = status
    if authoredon:
        params["authoredon"] = authoredon
    if date:
        params["date"] = date
    if intent:
        params["intent"] = intent
    if count:
        params["_count"] = count
    return await client.search(
        "MedicationRequest", params, page_limit, decode_notes=False
    )


async def fhir_document_reference_search_clinical_notes(
    client: FHIRClient,
    category: str | None = None,
    date: str | None = None,
    docstatus: str | None = None,
    encounter: str | None = None,
    patient: str | None = None,
    period: str | None = None,
    subject: str | None = None,
    type: str | None = None,
    count: int = 5,
    page_limit: int = 2,
) -> dict[str, Any]:
    """Query FHIR R4 DocumentReference search for Clinical Notes (USCDI compliant).
    Finds information about clinical notes for a patient. DocumentReference resources contain
    metadata about clinical notes, including references to Binary resources that contain the
    actual note content.
    Uses HTTP GET {api_base}/DocumentReference?category={category}&date={date}&docstatus={docstatus}&encounter={encounter}&patient={patient}&period={period}&subject={subject}&type={type}

    Args:
        category: FHIR 'category' token for document classification (e.g., 'clinical-note')
        date: Date filter for when the document was created (e.g., '2023-01-01' or date range)
        docstatus: Document status filter (e.g., 'preliminary', 'final', 'amended', 'entered-in-error')
        encounter: FHIR Encounter reference associated with the document
        patient: FHIR Patient id or reference (e.g., '12345')
        period: Period filter for document timeframe
        subject: FHIR subject reference (Patient/12345)
        type: FHIR 'type' token for specific note types (e.g., LOINC codes for note types)
        count: _count page size (server may cap)
        page_limit: max pages to follow via Bundle.link[rel=next]

    Returns:
        A dict with 'entries' (list of DocumentReference resources for clinical notes),
            'total' (if available), and 'pages' (how many Bundle pages retrieved).
    """
    params = {}
    if category:
        params["category"] = category
    if date:
        params["date"] = date
    if docstatus:
        params["docstatus"] = docstatus
    if encounter:
        params["encounter"] = encounter
    if patient:
        params["patient"] = patient
    if period:
        params["period"] = period
    if subject:
        params["subject"] = subject
    if type:
        params["type"] = type
    if count:
        params["_count"] = count
    return await client.search(
        "DocumentReference", params, page_limit, decode_notes=True
    )


async def fhir_communication_create_message(
    client: FHIRClient,
    patient_reference: str,
    message_text: str,
    recipient_reference: str | None = None,
    based_on_service_request: str | None = None,
    part_of_task: str | None = None,
    in_response_to_communication: str | None = None,
    encounter_reference: str | None = None,
    sender_reference: str | None = None,
    status: str = "in-progress",
) -> dict[str, Any]:
    """Create a secure message to a doctor using FHIR R4 Communication resource (Community Resource Communication).
    Allows sending messages between health systems and community-based organizations about referral requests
    made through Continued Care & Services Coordination (CCSC) workflows.
    Uses HTTP POST {api_base}/Communication

    Args:
        patient_reference: FHIR Patient reference (e.g., 'Patient/12345')
        message_text: The message content to send
        recipient_reference: FHIR reference to the recipient (e.g., 'Practitioner/dr-margaret-reynolds')
        based_on_service_request: Optional ServiceRequest reference this message is based on (e.g., 'ServiceRequest/abc123')
        part_of_task: Optional Task reference this message is part of (e.g., 'Task/def456')
        in_response_to_communication: Optional Communication reference this message is responding to (e.g., 'Communication/ghi789')
        encounter_reference: Optional Encounter reference (e.g., 'Encounter/jkl012')
        sender_reference: Optional sender reference (e.g., 'Organization/mno345' or 'Practitioner/pqr678')
        status: Communication status (default: 'in-progress', options: 'preparation', 'in-progress', 'not-done', 'on-hold', 'stopped', 'completed', 'entered-in-error', 'unknown')

    Returns:
        A dict containing the created Communication resource with 'id' and other resource elements.
    """
    communication_resource = {
        "resourceType": "Communication",
        "status": status,
        "subject": {"reference": patient_reference},
        "payload": [{"contentString": message_text}],
    }
    if based_on_service_request:
        communication_resource["basedOn"] = [{"reference": based_on_service_request}]
    if part_of_task:
        communication_resource["partOf"] = [{"reference": part_of_task}]
    if in_response_to_communication:
        communication_resource["inResponseTo"] = [
            {"reference": in_response_to_communication}
        ]
    if encounter_reference:
        communication_resource["encounter"] = {"reference": encounter_reference}
    if sender_reference:
        communication_resource["sender"] = {"reference": sender_reference}
    if recipient_reference:
        communication_resource["recipient"] = [{"reference": recipient_reference}]
    from datetime import datetime, timezone

    communication_resource["sent"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return await client.create(communication_resource)


async def fhir_service_request_search(
    client: FHIRClient,
    patient: str | None = None,
    subject: str | None = None,
    status: str | None = None,
    intent: str | None = None,
    category: str | None = None,
    code: str | None = None,
    authored: str | None = None,
    requester: str | None = None,
    performer: str | None = None,
    encounter: str | None = None,
    count: int = 50,
    page_limit: int = 2,
) -> dict[str, Any]:
    """Query FHIR R4 ServiceRequest search for lab orders and other service requests.
    Retrieves service requests including lab orders, imaging orders, referrals, and procedures.
    Uses HTTP GET {api_base}/ServiceRequest?patient={patient}&status={status}&category={category}&...

    Args:
        patient: FHIR Patient id or reference (e.g., '12345')
        subject: FHIR subject reference (Patient/12345)
        status: ServiceRequest status (e.g., 'draft', 'active', 'completed', 'cancelled', 'on-hold', 'revoked', 'entered-in-error', 'unknown')
        intent: Intent of the request (e.g., 'proposal', 'plan', 'directive', 'order', 'original-order', 'reflex-order', 'filler-order', 'instance-order', 'option')
        category: FHIR 'category' token for service classification (e.g., 'laboratory', 'imaging', 'counselling')
        code: FHIR code filter for specific service (e.g., LOINC codes for lab tests)
        authored: Date when the request was authored (e.g., '2023-01-01' or date range like 'ge2023-01-01')
        requester: Reference to the ordering practitioner (e.g., 'Practitioner/12345')
        performer: Reference to the performer (e.g., 'Organization/lab123')
        encounter: Reference to the encounter (e.g., 'Encounter/12345')
        count: _count page size (server may cap)
        page_limit: max pages to follow via Bundle.link[rel=next]

    Returns:
        A dict with 'entries' (list of ServiceRequest resources), 'total' (if available),
            and 'pages' (how many Bundle pages retrieved).
    """
    params = {}
    if patient:
        params["patient"] = patient
    if subject:
        params["subject"] = subject
    if status:
        params["status"] = status
    if intent:
        params["intent"] = intent
    if category:
        params["category"] = category
    if code:
        params["code"] = code
    if authored:
        params["authored"] = authored
    if requester:
        params["requester"] = requester
    if performer:
        params["performer"] = performer
    if encounter:
        params["encounter"] = encounter
    if count:
        params["_count"] = count
    return await client.search("ServiceRequest", params, page_limit, decode_notes=False)


async def fhir_service_request_create(
    client: FHIRClient,
    patient_reference: str,
    code_code: str,
    code_display: str,
    requester_reference: str,
    category_code: str = "108252007",
    category_display: str = "Laboratory procedure",
    category_system: str = "http://snomed.info/sct",
    code_system: str = "http://loinc.org",
    status: str = "active",
    intent: str = "order",
    priority: str | None = None,
    encounter_reference: str | None = None,
    performer_reference: str | None = None,
    reason_code: str | None = None,
    reason_display: str | None = None,
    reason_system: str | None = None,
    note_text: str | None = None,
    occurrence_datetime: str | None = None,
    body_site_code: str | None = None,
    body_site_display: str | None = None,
    body_site_system: str | None = None,
) -> dict[str, Any]:
    """Create a service request using FHIR R4 ServiceRequest resource.
    Places orders for labs, imaging, procedures, referrals, therapy, and other non-medication services.
    Uses HTTP POST {api_base}/ServiceRequest

    Args:
        patient_reference: FHIR Patient reference (e.g., 'Patient/12345')
        code_code: The code for the service (e.g., LOINC '2345-7' for Glucose, CPT '71046' for chest X-ray)
        code_display: Display name for the service (e.g., 'Glucose [Mass/volume] in Serum or Plasma')
        requester_reference: FHIR reference to the ordering practitioner (e.g., 'Practitioner/dr-margaret-reynolds')
        category_code: SNOMED CT code for service category (default: '108252007' for Laboratory procedure)
        category_display: Display name for category (default: 'Laboratory procedure')
        category_system: Code system for category (default: 'http://snomed.info/sct')
        code_system: Code system for the service code (default: 'http://loinc.org')
        status: ServiceRequest status (default: 'active', options: 'draft', 'active', 'on-hold', 'revoked', 'completed', 'entered-in-error', 'unknown')
        intent: Intent of the request (default: 'order', options: 'proposal', 'plan', 'directive', 'order', 'original-order', 'reflex-order', 'filler-order', 'instance-order', 'option')
        priority: Priority of the request (options: 'routine', 'urgent', 'asap', 'stat')
        encounter_reference: Optional Encounter reference (e.g., 'Encounter/abc123')
        performer_reference: Optional reference to performing organization (e.g., 'Organization/lab456')
        reason_code: Optional reason code for the order (e.g., SNOMED CT code)
        reason_display: Optional display text for the reason
        reason_system: Optional code system for the reason (e.g., 'http://snomed.info/sct')
        note_text: Optional clinical note or special instructions
        occurrence_datetime: Optional date/time for when the service should occur (ISO format)
        body_site_code: Optional body site code for procedures/imaging (SNOMED CT)
        body_site_display: Optional body site display name
        body_site_system: Optional body site code system (default: 'http://snomed.info/sct')

    Returns:
        A dict containing the created ServiceRequest resource with 'id' and other resource elements.
    """
    service_request = {
        "resourceType": "ServiceRequest",
        "status": status,
        "intent": intent,
        "category": [
            {
                "coding": [
                    {
                        "system": category_system,
                        "code": category_code,
                        "display": category_display,
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {"system": code_system, "code": code_code, "display": code_display}
            ]
        },
        "subject": {"reference": patient_reference},
        "requester": {"reference": requester_reference},
    }
    if priority:
        service_request["priority"] = priority
    if encounter_reference:
        service_request["encounter"] = {"reference": encounter_reference}
    if performer_reference:
        service_request["performer"] = [{"reference": performer_reference}]
    if reason_code and reason_display:
        service_request["reasonCode"] = [
            {
                "coding": [
                    {
                        "system": reason_system or "http://snomed.info/sct",
                        "code": reason_code,
                        "display": reason_display,
                    }
                ]
            }
        ]
    if note_text:
        service_request["note"] = [{"text": note_text}]
    if occurrence_datetime:
        service_request["occurrenceDateTime"] = occurrence_datetime
    if body_site_code and body_site_display:
        service_request["bodySite"] = [
            {
                "coding": [
                    {
                        "system": body_site_system or "http://snomed.info/sct",
                        "code": body_site_code,
                        "display": body_site_display,
                    }
                ]
            }
        ]
    from datetime import datetime, timezone

    service_request["authoredOn"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return await client.create(service_request)


async def fhir_medication_request_create(
    client: FHIRClient,
    patient_reference: str,
    medication_display: str,
    requester_reference: str,
    medication_code: str | None = None,
    medication_system: str = "http://www.nlm.nih.gov/research/umls/rxnorm",
    status: str = "active",
    intent: str = "order",
    dose_value: float | None = None,
    dose_unit: str | None = None,
    frequency_text: str | None = None,
    route_code: str | None = None,
    route_display: str | None = None,
    duration_value: float | None = None,
    duration_unit: str | None = None,
    dispense_quantity: float | None = None,
    dispense_unit: str | None = None,
    num_refills: int | None = None,
    reason_code: str | None = None,
    reason_display: str | None = None,
    reason_system: str | None = None,
    note_text: str | None = None,
    encounter_reference: str | None = None,
    priority: str | None = None,
) -> dict[str, Any]:
    """Create a medication order using FHIR R4 MedicationRequest resource.
    Places prescription orders for medications including dose, frequency, route, and duration.
    Uses HTTP POST {api_base}/MedicationRequest

    Args:
        patient_reference: FHIR Patient reference (e.g., 'Patient/12345')
        medication_display: Display name of the medication (e.g., 'Escitalopram 10 MG Oral Tablet')
        requester_reference: FHIR reference to the ordering practitioner (e.g., 'Practitioner/dr-margaret-reynolds')
        medication_code: RxNorm or other code for the medication (e.g., '352741' for escitalopram 10mg)
        medication_system: Code system for the medication (default: RxNorm)
        status: MedicationRequest status (default: 'active', options: 'active', 'on-hold', 'cancelled', 'completed', 'entered-in-error', 'stopped', 'draft', 'unknown')
        intent: Intent of the request (default: 'order', options: 'proposal', 'plan', 'order', 'original-order', 'reflex-order', 'filler-order', 'instance-order', 'option')
        dose_value: Numeric dose amount (e.g., 10)
        dose_unit: Dose unit (e.g., 'mg')
        frequency_text: Human-readable frequency (e.g., 'Once daily', 'BID', 'Every 8 hours')
        route_code: SNOMED CT code for route (e.g., '26643006' for oral)
        route_display: Display name for route (e.g., 'Oral route')
        duration_value: Duration amount (e.g., 30)
        duration_unit: Duration unit (e.g., 'd' for days, 'wk' for weeks, 'mo' for months)
        dispense_quantity: Number of units to dispense (e.g., 30)
        dispense_unit: Unit for dispense quantity (e.g., 'tablets')
        num_refills: Number of refills authorized (e.g., 3)
        reason_code: Optional reason code for the order (e.g., ICD-10 or SNOMED CT)
        reason_display: Optional display text for the reason
        reason_system: Optional code system for the reason
        note_text: Optional clinical note or special instructions
        encounter_reference: Optional Encounter reference (e.g., 'Encounter/abc123')
        priority: Priority of the request (options: 'routine', 'urgent', 'asap', 'stat')

    Returns:
        A dict containing the created MedicationRequest resource with 'id' and other resource elements.
    """
    medication_concept = {"text": medication_display}
    if medication_code:
        medication_concept["coding"] = [
            {
                "system": medication_system,
                "code": medication_code,
                "display": medication_display,
            }
        ]
    medication_request = {
        "resourceType": "MedicationRequest",
        "status": status,
        "intent": intent,
        "medicationCodeableConcept": medication_concept,
        "subject": {"reference": patient_reference},
        "requester": {"reference": requester_reference},
    }
    dosage: dict[str, Any] = {}
    if frequency_text:
        dosage["text"] = frequency_text
        dosage["timing"] = {"code": {"text": frequency_text}}
    if dose_value is not None and dose_unit:
        dosage["doseAndRate"] = [
            {"doseQuantity": {"value": dose_value, "unit": dose_unit}}
        ]
    if route_code and route_display:
        dosage["route"] = {
            "coding": [
                {
                    "system": "http://snomed.info/sct",
                    "code": route_code,
                    "display": route_display,
                }
            ]
        }
    if dosage:
        medication_request["dosageInstruction"] = [dosage]
    dispense: dict[str, Any] = {}
    if duration_value is not None and duration_unit:
        dispense["expectedSupplyDuration"] = {
            "value": duration_value,
            "unit": duration_unit,
            "system": "http://unitsofmeasure.org",
            "code": duration_unit,
        }
    if dispense_quantity is not None and dispense_unit:
        dispense["quantity"] = {"value": dispense_quantity, "unit": dispense_unit}
    if num_refills is not None:
        dispense["numberOfRepeatsAllowed"] = num_refills
    if dispense:
        medication_request["dispenseRequest"] = dispense
    if priority:
        medication_request["priority"] = priority
    if encounter_reference:
        medication_request["encounter"] = {"reference": encounter_reference}
    if reason_code and reason_display:
        medication_request["reasonCode"] = [
            {
                "coding": [
                    {
                        "system": reason_system or "http://hl7.org/fhir/sid/icd-10-cm",
                        "code": reason_code,
                        "display": reason_display,
                    }
                ]
            }
        ]
    if note_text:
        medication_request["note"] = [{"text": note_text}]
    from datetime import datetime, timezone

    medication_request["authoredOn"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return await client.create(medication_request)


async def fhir_observation_search_social_history(
    client: FHIRClient,
    patient: str | None = None,
    subject: str | None = None,
    code: str | None = None,
    date: str | None = None,
    count: int = 50,
    page_limit: int = 2,
) -> dict[str, Any]:
    """Query FHIR R4 Observation search for Social History data.
    Returns social history observations including tobacco use, alcohol use,
    substance use, occupation, and other social determinants of health.
    Uses HTTP GET {api_base}/Observation?patient={patient}&category=social-history&code={code}&date={date}

    Args:
        patient: FHIR Patient id or reference (e.g., '12345')
        subject: FHIR subject reference (Patient/12345)
        code: FHIR code filter for specific social history type (e.g., LOINC codes):
        date: Date filter for observations (e.g., '2023-01-01' or date range)
        count: _count page size (server may cap)
        page_limit: max pages to follow via Bundle.link[rel=next]

    Returns:
        A dict with 'entries' (list of Observation resources for social history),
            'total' (if available), and 'pages' (how many Bundle pages retrieved).
    """
    params = {"category": "social-history"}
    if patient:
        params["patient"] = patient
    if subject:
        params["subject"] = subject
    if code:
        params["code"] = code
    if date:
        params["date"] = date
    if count:
        params["_count"] = max(1, min(count, 200))
    return await client.search("Observation", params, page_limit, decode_notes=False)


async def fhir_appointment_create(
    client: FHIRClient,
    patient_reference: str,
    practitioner_reference: str,
    description: str,
    start: str | None = None,
    end: str | None = None,
    status: str = "booked",
    based_on: str | None = None,
    appointment_type_code: str | None = None,
    appointment_type_display: str | None = None,
    appointment_type_system: str = "http://terminology.hl7.org/CodeSystem/v2-0276",
    reason_code: str | None = None,
    reason_display: str | None = None,
    reason_system: str | None = None,
    note_text: str | None = None,
    service_type_code: str | None = None,
    service_type_display: str | None = None,
    minutes_duration: int | None = None,
) -> dict[str, Any]:
    """Create a FHIR R4 Appointment resource to schedule a clinic visit or follow-up.
    Uses HTTP POST {api_base}/Appointment

    Args:
        patient_reference: FHIR Patient reference (e.g., 'Patient/12345')
        practitioner_reference: FHIR Practitioner reference (e.g., 'Practitioner/dr-margaret-reynolds')
        description: Human-readable description of the appointment (e.g., 'Follow-up with ortho fracture liaison PA')
        start: Start datetime in ISO 8601 format (e.g., '2024-03-26T09:00:00Z')
        end: End datetime in ISO 8601 format
        status: Appointment status (default: 'booked'; options: 'proposed', 'pending', 'booked', 'arrived', 'fulfilled', 'cancelled', 'noshow')
        based_on: Reference to the ServiceRequest that this appointment fulfills (e.g. 'ServiceRequest/123')
        appointment_type_code: Code for appointment type (e.g., 'FOLLOWUP')
        appointment_type_display: Display name for appointment type (e.g., 'Follow-up visit')
        appointment_type_system: Code system for appointment type
        reason_code: Reason code (e.g., SNOMED CT or ICD-10)
        reason_display: Reason display text
        reason_system: Reason code system
        note_text: Additional notes or instructions
        service_type_code: Service type code
        service_type_display: Service type display name
        minutes_duration: Duration of appointment in minutes

    Returns:
        A dict containing the created Appointment resource with 'id' and other elements.
    """
    _REFERRAL_KEYWORDS = [
        "referral",
        "consult",
        "refer to",
        "referring",
        "pulmonology",
        "cardiology",
        "oncology",
        "neurology",
        "dermatology",
        "endocrinology",
        "rheumatology",
        "nephrology",
        "gastroenterology",
        "hematology",
        "urology",
    ]
    desc_lower = description.lower()
    if any(kw in desc_lower for kw in _REFERRAL_KEYWORDS) and (not based_on):
        return {
            "error": "Appointment rejected: the description indicates a referral or consult, but no 'based_on' ServiceRequest reference was provided. Referrals must first be placed as a ServiceRequest, then an Appointment may be scheduled with based_on referencing that ServiceRequest."
        }
    appointment = {
        "resourceType": "Appointment",
        "status": status,
        "description": description,
        "participant": [
            {"actor": {"reference": patient_reference}, "status": "accepted"},
            {"actor": {"reference": practitioner_reference}, "status": "accepted"},
        ],
    }
    if based_on:
        appointment["basedOn"] = [{"reference": based_on}]
    if start:
        appointment["start"] = start
    if end:
        appointment["end"] = end
    if minutes_duration:
        appointment["minutesDuration"] = minutes_duration
    if appointment_type_code and appointment_type_display:
        appointment["appointmentType"] = {
            "coding": [
                {
                    "system": appointment_type_system,
                    "code": appointment_type_code,
                    "display": appointment_type_display,
                }
            ]
        }
    if reason_code and reason_display:
        appointment["reasonCode"] = [
            {
                "coding": [
                    {
                        "system": reason_system or "http://snomed.info/sct",
                        "code": reason_code,
                        "display": reason_display,
                    }
                ]
            }
        ]
    if service_type_code and service_type_display:
        appointment["serviceType"] = [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/service-type",
                        "code": service_type_code,
                        "display": service_type_display,
                    }
                ]
            }
        ]
    if note_text:
        appointment["comment"] = note_text
    from datetime import datetime, timezone

    appointment["created"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return await client.create(appointment)


NATIVE_FUNCTIONS = [
    fhir_condition_search_problems,
    fhir_observation_search_labs,
    fhir_observation_search_vitals,
    fhir_patient_search_demographics,
    fhir_procedure_search_orders,
    fhir_medication_request_search_orders,
    fhir_document_reference_search_clinical_notes,
    fhir_communication_create_message,
    fhir_service_request_search,
    fhir_service_request_create,
    fhir_medication_request_create,
    fhir_observation_search_social_history,
    fhir_appointment_create,
]


FHIR_TOOL_SCHEMAS = [{'name': 'fhir_condition_search_problems',
  'description': "Search FHIR Condition resources for problems on a patient's chart. Retrieves "
                 'problem list data across all encounters.',
  'parameters': {'type': 'object',
                 'properties': {'subject': {'type': 'string',
                                            'description': 'FHIR subject reference (e.g. '
                                                           'Patient/12345)'},
                                'patient': {'type': 'string',
                                            'description': "FHIR Patient id (e.g. '12345')"},
                                'clinical_status': {'type': 'string',
                                                    'description': "e.g. 'active', 'inactive', "
                                                                   "'resolved'"},
                                'category': {'type': 'string',
                                             'description': "e.g. 'problem-list-item'"},
                                'code': {'type': 'string',
                                         'description': 'Condition code filter (ICD-10, SNOMED, '
                                                        'etc.)'},
                                'onset_date': {'type': 'string',
                                               'description': 'The date a condition was noted'},
                                'count': {'type': 'integer',
                                          'description': 'Page size (_count). Default: 10',
                                          'default': 10},
                                'page_limit': {'type': 'integer',
                                               'description': 'Max pages to follow via '
                                                              'Bundle.link[rel=next]. Default: 3',
                                               'default': 3}}}},
 {'name': 'fhir_observation_search_labs',
  'description': 'Search FHIR Observation resources for lab results. Returns component-level data '
                 'for lab results.',
  'parameters': {'type': 'object',
                 'properties': {'patient': {'type': 'string', 'description': 'FHIR Patient id'},
                                'subject': {'type': 'string',
                                            'description': 'FHIR subject reference'},
                                'category': {'type': 'string', 'description': "e.g. 'laboratory'"},
                                'code': {'type': 'string', 'description': 'LOINC code filter'},
                                'date': {'type': 'string',
                                         'description': 'The date range for when the Observation '
                                                        'was taken.'},
                                'count': {'type': 'integer',
                                          'description': 'Page size (_count). Default: 50',
                                          'default': 50},
                                'page_limit': {'type': 'integer',
                                               'description': 'Max pages to follow via '
                                                              'Bundle.link[rel=next]. Default: 6',
                                               'default': 6}}}},
 {'name': 'fhir_observation_search_vitals',
  'description': 'Search FHIR Observation resources for vital signs. Retrieves vital sign data and '
                 'flowsheet data across all encounters.',
  'parameters': {'type': 'object',
                 'properties': {'patient': {'type': 'string', 'description': 'FHIR Patient id'},
                                'subject': {'type': 'string',
                                            'description': 'FHIR subject reference'},
                                'code': {'type': 'string',
                                         'description': 'LOINC code filter for vitals'},
                                'date': {'type': 'string',
                                         'description': 'The date range for when the Observation '
                                                        'was taken.'},
                                'category': {'type': 'string', 'description': "e.g. 'vital-signs'"},
                                'count': {'type': 'integer',
                                          'description': 'Page size (_count). Default: 50',
                                          'default': 50},
                                'page_limit': {'type': 'integer',
                                               'description': 'Max pages to follow via '
                                                              'Bundle.link[rel=next]. Default: 6',
                                               'default': 6}}}},
 {'name': 'fhir_patient_search_demographics',
  'description': 'Search FHIR Patient resources by demographic parameters. Retrieves patient '
                 'demographic information from patient charts.',
  'parameters': {'type': 'object',
                 'properties': {'address': {'type': 'string'},
                                'address_city': {'type': 'string'},
                                'address_postalcode': {'type': 'string'},
                                'address_state': {'type': 'string'},
                                'birthdate': {'type': 'string', 'description': "e.g. '1990-01-01'"},
                                'family': {'type': 'string', 'description': 'Family name'},
                                'gender': {'type': 'string',
                                           'description': "e.g. 'male', 'female'"},
                                'given': {'type': 'string', 'description': 'Given name'},
                                'identifier': {'type': 'string',
                                               'description': 'Patient identifier (e.g. MRN)'},
                                'name': {'type': 'string',
                                         'description': 'Name search (family and/or given)'},
                                'own_name': {'type': 'string'},
                                'own_prefix': {'type': 'string'},
                                'partner_name': {'type': 'string'},
                                'partner_prefix': {'type': 'string'},
                                'telecom': {'type': 'string'},
                                'legal_sex': {'type': 'string'},
                                'count': {'type': 'integer',
                                          'description': 'Page size (_count). Default: 50',
                                          'default': 50},
                                'page_limit': {'type': 'integer',
                                               'description': 'Max pages to follow via '
                                                              'Bundle.link[rel=next]. Default: 6',
                                               'default': 6}}}},
 {'name': 'fhir_procedure_search_orders',
  'description': 'Search FHIR Procedure resources for completed procedures. Includes surgeries, '
                 'endoscopies, biopsies, counseling, physiotherapy, etc.',
  'parameters': {'type': 'object',
                 'properties': {'date': {'type': 'string',
                                         'description': 'Date or period that the procedure was '
                                                        'performed, using the FHIR date parameter '
                                                        'format.'},
                                'patient': {'type': 'string', 'description': 'FHIR Patient id'},
                                'subject': {'type': 'string',
                                            'description': 'FHIR subject reference'},
                                'category': {'type': 'string'},
                                'count': {'type': 'integer',
                                          'description': 'Page size (_count). Default: 50',
                                          'default': 50},
                                'page_limit': {'type': 'integer',
                                               'description': 'Max pages to follow via '
                                                              'Bundle.link[rel=next]. Default: 6',
                                               'default': 6}}}},
 {'name': 'fhir_medication_request_search_orders',
  'description': 'Search FHIR MedicationRequest resources for medication orders. Returns '
                 'inpatient, clinic-administered, patient-reported, and reconciled medications.',
  'parameters': {'type': 'object',
                 'properties': {'patient': {'type': 'string', 'description': 'FHIR Patient id'},
                                'subject': {'type': 'string',
                                            'description': 'FHIR subject reference'},
                                'category': {'type': 'string'},
                                'status': {'type': 'string',
                                           'description': "e.g. 'active', 'completed', "
                                                          "'cancelled'"},
                                'authoredon': {'type': 'string',
                                               'description': 'Date prescription was authored'},
                                'date': {'type': 'string',
                                         'description': 'The medication administration date'},
                                'intent': {'type': 'string',
                                           'description': "e.g. 'order', 'plan', 'proposal'"},
                                'count': {'type': 'integer',
                                          'description': 'Page size (_count). Default: 50',
                                          'default': 50},
                                'page_limit': {'type': 'integer',
                                               'description': 'Max pages to follow via '
                                                              'Bundle.link[rel=next]. Default: 6',
                                               'default': 6}}}},
 {'name': 'fhir_document_reference_search_clinical_notes',
  'description': 'Search FHIR DocumentReference resources for clinical note metadata. Use '
                 'Binary.Read to retrieve actual note content from returned references.',
  'parameters': {'type': 'object',
                 'properties': {'category': {'type': 'string',
                                             'description': "e.g. 'clinical-note'"},
                                'date': {'type': 'string',
                                         'description': 'FHIR date search using prefix syntax. Use '
                                                        "'ge2022-01-01' for on/after, "
                                                        "'le2022-02-01' for on/before."},
                                'encounter': {'type': 'string',
                                              'description': 'Encounter reference'},
                                'patient': {'type': 'string', 'description': 'FHIR Patient id'},
                                'subject': {'type': 'string',
                                            'description': 'FHIR subject reference'},
                                'type': {'type': 'string',
                                         'description': 'LOINC code for note type'},
                                'count': {'type': 'integer',
                                          'description': 'Page size (_count). Default: 5',
                                          'default': 5},
                                'page_limit': {'type': 'integer',
                                               'description': 'Max pages to follow via '
                                                              'Bundle.link[rel=next]. Default: 2',
                                               'default': 2}}}},
 {'name': 'fhir_service_request_search',
  'description': 'Search FHIR ServiceRequest resources for lab orders, imaging orders, referrals, '
                 'and procedures.',
  'parameters': {'type': 'object',
                 'properties': {'patient': {'type': 'string', 'description': 'FHIR Patient id'},
                                'subject': {'type': 'string',
                                            'description': 'FHIR subject reference'},
                                'status': {'type': 'string',
                                           'description': "e.g. 'active', 'completed', "
                                                          "'cancelled'"},
                                'intent': {'type': 'string', 'description': "e.g. 'order', 'plan'"},
                                'category': {'type': 'string',
                                             'description': "e.g. 'laboratory', 'imaging'"},
                                'code': {'type': 'string', 'description': 'LOINC/CPT code'},
                                'authored': {'type': 'string',
                                             'description': 'Date authored using FHIR prefix '
                                                            "syntax. Use 'ge2022-01-01' for "
                                                            "on/after, 'le2022-02-01' for "
                                                            'on/before.'},
                                'requester': {'type': 'string',
                                              'description': 'Practitioner reference'},
                                'performer': {'type': 'string',
                                              'description': 'Organization reference'},
                                'encounter': {'type': 'string',
                                              'description': 'Encounter reference'},
                                'count': {'type': 'integer',
                                          'description': 'Page size (_count). Default: 50',
                                          'default': 50},
                                'page_limit': {'type': 'integer',
                                               'description': 'Max pages to follow via '
                                                              'Bundle.link[rel=next]. Default: 6',
                                               'default': 6}}}},
 {'name': 'fhir_observation_search_social_history',
  'description': 'Search FHIR Observation resources for social history data. Returns tobacco use, '
                 'alcohol use, substance use, occupation, and other social determinants. Always '
                 'queries category=social-history.',
  'parameters': {'type': 'object',
                 'properties': {'patient': {'type': 'string', 'description': 'FHIR Patient id'},
                                'subject': {'type': 'string',
                                            'description': 'FHIR subject reference'},
                                'code': {'type': 'string',
                                         'description': "LOINC code filter (e.g. '72166-2' Tobacco "
                                                        "smoking status, '11331-6' History of "
                                                        'Alcohol use)'},
                                'date': {'type': 'string',
                                         'description': 'FHIR date search using prefix syntax. Use '
                                                        "'ge2022-01-01' for on/after, "
                                                        "'le2022-02-01' for on/before."},
                                'count': {'type': 'integer',
                                          'description': 'Page size (_count). Default: 50',
                                          'default': 50},
                                'page_limit': {'type': 'integer',
                                               'description': 'Max pages to follow via '
                                                              'Bundle.link[rel=next]. Default: 6',
                                               'default': 6}}}},
 {'name': 'fhir_medication_request_create',
  'description': 'Create a medication order using FHIR R4 MedicationRequest. Places prescription '
                 'orders with dose, frequency, route, and duration.',
  'parameters': {'type': 'object',
                 'properties': {'patient_reference': {'type': 'string',
                                                      'description': "e.g. 'Patient/12345'"},
                                'medication_display': {'type': 'string',
                                                       'description': 'Display name of the '
                                                                      'medication'},
                                'requester_reference': {'type': 'string',
                                                        'description': "e.g. 'Practitioner/67890'"},
                                'medication_code': {'type': 'string', 'description': 'RxNorm code'},
                                'medication_system': {'type': 'string',
                                                      'default': 'http://www.nlm.nih.gov/research/umls/rxnorm'},
                                'status': {'type': 'string', 'default': 'active'},
                                'intent': {'type': 'string', 'default': 'order'},
                                'dose_value': {'type': 'number',
                                               'description': 'Numeric dose amount'},
                                'dose_unit': {'type': 'string', 'description': "e.g. 'mg'"},
                                'frequency_text': {'type': 'string',
                                                   'description': "e.g. 'Once daily', 'BID'"},
                                'route_code': {'type': 'string',
                                               'description': 'SNOMED CT route code'},
                                'route_display': {'type': 'string',
                                                  'description': "e.g. 'Oral route'"},
                                'duration_value': {'type': 'number'},
                                'duration_unit': {'type': 'string',
                                                  'description': "e.g. 'd', 'wk', 'mo'"},
                                'dispense_quantity': {'type': 'number'},
                                'dispense_unit': {'type': 'string'},
                                'num_refills': {'type': 'integer'},
                                'reason_code': {'type': 'string'},
                                'reason_display': {'type': 'string'},
                                'reason_system': {'type': 'string'},
                                'note_text': {'type': 'string'},
                                'encounter_reference': {'type': 'string'},
                                'priority': {'type': 'string',
                                             'description': "'routine', 'urgent', 'asap', 'stat'"}},
                 'required': ['patient_reference', 'medication_display', 'requester_reference']}},
 {'name': 'fhir_communication_create_message',
  'description': 'Create a FHIR Communication resource to send a secure message to a provider. '
                 'Used for messages about referral requests in care coordination workflows.',
  'parameters': {'type': 'object',
                 'properties': {'patient_reference': {'type': 'string',
                                                      'description': "e.g. 'Patient/12345'"},
                                'message_text': {'type': 'string',
                                                 'description': 'Message content to send'},
                                'recipient_reference': {'type': 'string',
                                                        'description': "e.g. 'Practitioner/67890'"},
                                'based_on_service_request': {'type': 'string'},
                                'part_of_task': {'type': 'string'},
                                'in_response_to_communication': {'type': 'string'},
                                'encounter_reference': {'type': 'string'},
                                'sender_reference': {'type': 'string'},
                                'status': {'type': 'string', 'default': 'in-progress'}},
                 'required': ['patient_reference', 'message_text']}},
 {'name': 'fhir_service_request_create',
  'description': 'Create a FHIR ServiceRequest to place orders for labs, imaging, procedures, '
                 'referrals, or other services.',
  'parameters': {'type': 'object',
                 'properties': {'patient_reference': {'type': 'string',
                                                      'description': "e.g. 'Patient/12345'"},
                                'code_code': {'type': 'string',
                                              'description': 'Service code (LOINC/CPT/SNOMED)'},
                                'code_display': {'type': 'string',
                                                 'description': 'Display name for the service'},
                                'requester_reference': {'type': 'string',
                                                        'description': 'Ordering practitioner '
                                                                       'reference'},
                                'category_code': {'type': 'string',
                                                  'default': '108252007',
                                                  'description': 'SNOMED category (108252007=Lab, '
                                                                 '363679005=Imaging, '
                                                                 '387713003=Surgical)'},
                                'category_display': {'type': 'string',
                                                     'default': 'Laboratory procedure'},
                                'category_system': {'type': 'string',
                                                    'default': 'http://snomed.info/sct'},
                                'code_system': {'type': 'string', 'default': 'http://loinc.org'},
                                'status': {'type': 'string', 'default': 'active'},
                                'intent': {'type': 'string', 'default': 'order'},
                                'priority': {'type': 'string',
                                             'description': "'routine', 'urgent', 'asap', 'stat'"},
                                'encounter_reference': {'type': 'string'},
                                'performer_reference': {'type': 'string'},
                                'reason_code': {'type': 'string'},
                                'reason_display': {'type': 'string'},
                                'reason_system': {'type': 'string'},
                                'note_text': {'type': 'string'},
                                'occurrence_datetime': {'type': 'string'},
                                'body_site_code': {'type': 'string'},
                                'body_site_display': {'type': 'string'},
                                'body_site_system': {'type': 'string'}},
                 'required': ['patient_reference',
                              'code_code',
                              'code_display',
                              'requester_reference']}},
 {'name': 'fhir_appointment_create',
  'description': 'Create a FHIR Appointment to schedule a clinic visit or follow-up. Requires a '
                 'preceding ServiceRequest or order reference via based_on when the appointment '
                 'originates from a clinical order or referral.',
  'parameters': {'type': 'object',
                 'properties': {'patient_reference': {'type': 'string',
                                                      'description': "e.g. 'Patient/12345'"},
                                'practitioner_reference': {'type': 'string',
                                                           'description': 'e.g. '
                                                                          "'Practitioner/67890'"},
                                'description': {'type': 'string',
                                                'description': 'Human-readable description'},
                                'start': {'type': 'string',
                                          'description': 'Start datetime ISO 8601'},
                                'end': {'type': 'string', 'description': 'End datetime ISO 8601'},
                                'status': {'type': 'string', 'default': 'booked'},
                                'based_on': {'type': 'string',
                                             'description': 'Reference to the ServiceRequest that '
                                                            'this appointment fulfills (e.g. '
                                                            "'ServiceRequest/123')"},
                                'appointment_type_code': {'type': 'string'},
                                'appointment_type_display': {'type': 'string'},
                                'appointment_type_system': {'type': 'string',
                                                            'default': 'http://terminology.hl7.org/CodeSystem/v2-0276'},
                                'reason_code': {'type': 'string'},
                                'reason_display': {'type': 'string'},
                                'reason_system': {'type': 'string'},
                                'note_text': {'type': 'string'},
                                'service_type_code': {'type': 'string'},
                                'service_type_display': {'type': 'string'},
                                'minutes_duration': {'type': 'integer'}},
                 'required': ['patient_reference', 'practitioner_reference', 'description']}}]
