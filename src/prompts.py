import json


ALLOWED_SECTIONS = """
Allowed sections:
- Lungs and Airways -> All the findins closely related to Lungs and airways should be placed under this.
- Pleura -> All the findings closely related to pleura should be placed under this.
- Cardiovascular -> All the findings closely related to heart, aorta, vascular congestion, and pulmonary vasculature should be placed under this.
- Hila and Mediastinum -> All the findings closely related to hilar enlargement, mediastinal widening, lymphadenopathy, tracheal deviation, and mediastinal shift should be placed under this.
- Tubes, Catheters, and Support Devices -> All the findings closely related to endotracheal tube, enteric tube, central venous catheter, chest tube, pacemaker, ICD, port, and other visible devices should be placed under this.
- Musculoskeletal and Chest Wall -> All the findings closely related toMusculoskeletal and Chest Wall should be placed under this.
- Abdominal -> All the findings closely related to upper abdominal findings visible on chest imaging, pneumoperitoneum, bowel gas, stomach bubble, and upper abdominal surgical clips should be placed under this.
- Other -> findings or comments that do not clearly fit a specific section should be placed under this.
""".strip()


main_prompt = """
Your task is to improve the formatting of a radiology report to a clear and concise radiology report with section headings.
        Guidelines:


        1. **Findings:**
        Describe all positive observations and any relevant negative observations for each organ or organ system under distinct headers. Start with the organ system name followed by a colon, then list observations.
        **Here is the corresponding template:**
        Organ 1:
        – Observation 1
        Organ 2:
        – Observation 1
        – Observation 2

        **use only necessary headers from the following headers for organ systems: **

        * Lungs and Airways
        * Pleura
        * Cardiovascular
        * Hila and Mediastinuma
        * Tubes, Catheters, and Support Devices
        * Musculoskeletal and Chest Wall
        * Abdominal
        * Other

         Example:
        **Free text :** No active infiltrate or effusion is seen. Only mild peribronchial thickening is noted. The heart is within normal limits in size. No bony abnormality is seen.

        **Structured Report:**

        FINDINGS:

        Lungs and Airways:

        - No active infiltrate is seen.

        - Only mild peribronchial thickening is noted.

        Pleura:

        - No effusion is seen.

        Cardiovascular:

        - The heart is within normal limits in size.

        Musculoskeletal and Chest Wall:

        - No bony abnormality is seen.

"""


def build_structuring_prompt(free_text: str) -> str:
    return f"""
  {main_prompt}

The radiology report to structure is the following:

{free_text}
""".strip()


def build_findings_judge_prompt(free_text: str, structured_report: str) -> str:
    return f"""\
You are an expert radiology report checker.

Your role is to verify that the STRUCTURED report is clinically faithful to the SOURCE report.

SOURCE is the original free-text radiology report.
STRUCTURED is a reformatted version of SOURCE organized under section headers.

Main goal:
- Ensure every clinical finding stated in SOURCE is represented in STRUCTURED.
- Ensure STRUCTURED does not contain findings unsupported by SOURCE.

Core rule:
- Use only the SOURCE text.
- Do not guess, infer from general radiology knowledge, or add routine normal findings.
- If a finding is uncertain or not clearly supported by SOURCE, leave it out.
- If you are not sure about a finding, do not include it in either list.

The structured report should use these allowed sections:
{ALLOWED_SECTIONS}

━━━ TASK 1: Find MISSING findings ━━━━━━━━━━━━━━━━━━━━━━━━
A missing finding is a finding that is written in SOURCE but is completely absent from STRUCTURED.

Missing finding criteria:
A SOURCE finding is missing only if its clinical meaning is absent from every section of STRUCTURED.

Rules:
- Check each SOURCE finding against all STRUCTURED sections.
- Split combined SOURCE sentences into separate findings by clinical meaning.
- Accept synonyms, paraphrases, abbreviations, and concise equivalents as present.
- Do not flag wrong-section placement as missing.
- If only part of a combined SOURCE finding is absent, report only the absent part.
- Only reportv a finding is mising when you are confident it is not present in any form in STRUCTURED.
- It is better to not report a finding as missing than to report it incorrectly.

Output: missing_findings
- "finding": copy the finding text from SOURCE.
- "suggested_section": the section where it should be added in STRUCTURED.

━━━ TASK 2: Find UNSUPPORTED findings ━━━━━━━━━━━━━━━━━━━━
An unsupported finding is a finding that is written in STRUCTURED but does not appear in SOURCE.

How to find unsupported findings:
- Read STRUCTURED and identify each finding listed in it.
- For each STRUCTURED finding, look for it in SOURCE.
- If SOURCE contains it (even with slightly different wording), it is NOT unsupported.
- Only report it as unsupported if it is truly absent from SOURCE.
- Being in the wrong section is NOT unsupported. Only flag it if SOURCE does not mention it at all.

Output: unsupported_findings
- "finding": copy the finding text from STRUCTURED.
- "current_section": the section it should be appeared in inside STRUCTURED.

━━━ Important rules for both tasks ━━━━━━━━━━━━━━━━━━━━━━━
- SOURCE is the only source of truth. Do not use radiology knowledge to add or infer findings.
- Accept loose wording matches: "No pneumothorax" and "No pneumothorax is noted" are the same finding.
- A finding cannot appear in both lists. If unsure, leave it out.

Before writing your answer, re-read STRUCTURED above.
For each finding you plan to mark as missing: confirm it does not appear anywhere in STRUCTURED.
If you can see it in STRUCTURED (even with slightly different wording), remove it from your list.

Return JSON only:
{{
  "missing_findings": [
    {{"finding": "", "suggested_section": ""}}
  ],
  "unsupported_findings": [
    {{"finding": "", "current_section": ""}}
  ]
}}

STRUCTURED:
{structured_report}

SOURCE:
{free_text}""".strip()


def build_anatomy_duplication_judge_prompt(structured_report: str) -> str:
    return f"""\
You are a radiology report structure checker.

STRUCTURED is a structured radiology report organized under section headers.

{ALLOWED_SECTIONS}

━━━ TASK 1: Find WRONG-SECTION findings ━━━━━━━━━━━━━━━━━━
A wrong-section finding is a finding that is placed under a section that does not match
the primary anatomical structure the finding describes.

How to find wrong-section findings:
- Read each finding in STRUCTURED and identify its primary anatomical subject.
- Ask: does the current section match that primary anatomical subject?
- If the current section is a reasonable match, do not flag it.
- Only flag it if another section is clearly more appropriate.
- If you are unsure or more than one section could apply, do not flag it.
- current_section and correct_section must always be different values.
  If they would be the same, do not include that entry.

Output: wrong_section_findings
- "finding": the finding text as it appears in STRUCTURED.
- "current_section": the section it is currently placed under.
- "correct_section": the section it should be moved to.

━━━ TASK 2: Find DUPLICATE findings ━━━━━━━━━━━━━━━━━━━━━━
A duplicate finding is the same clinical fact written more than once anywhere in STRUCTURED.

How to find duplicates:
- Read all findings across all sections of STRUCTURED.
- Identify any finding whose clinical meaning appears more than once, even with different wording.
- Keep the version in the most appropriate section. Flag the other copy for removal.
- Only flag it if the same fact is clearly repeated. Do not flag findings that are related but different.

Output: duplicate_findings
- "finding": the finding text of the copy that should be removed.
- "section_to_remove_the_finding_from": the section the copy should be removed from.

━━━ Important rules for both tasks ━━━━━━━━━━━━━━━━━━━━━━━
- Do not compare with any source report. Judge STRUCTURED only.
- Do not flag technique, image quality, or exam limitation statements as wrong-section
  if they are placed under Other.
- A finding that mentions multiple anatomical structures may reasonably stay in its current section.
  Do not flag it unless the placement is clearly wrong.
- If in doubt, leave it out.

Return JSON only:
{{
  "wrong_section_findings": [
    {{"finding": "", "current_section": "", "correct_section": ""}}
  ],
  "duplicate_findings": [
    {{"finding": "", "section_to_remove_the_finding_from": ""}}
  ]
}}

STRUCTURED:
{structured_report}""".strip()


def build_revision_prompt(
    free_text: str,
    current_report: str,
    findings_feedback: dict,
    anatomy_feedback: dict,
) -> str:
    return f"""\
You are a radiology report reviser.

You are given:
- SOURCE: the original free-text radiology report. This is the ground truth.
- CURRENT: the current structured report that needs to be revised.
- FINDINGS_FEEDBACK: lists missing findings and unsupported findings.
- ANATOMY_FEEDBACK: lists wrong-section findings and duplicate findings.

Your task is to apply the feedback to CURRENT and produce a corrected structured report.
SOURCE is the only ground truth. If any feedback contradicts SOURCE, ignore that feedback item.

{ALLOWED_SECTIONS}

━━━ TASK 1: Apply FINDINGS_FEEDBACK ━━━━━━━━━━━━━━━━━━━━━━━━

missing_findings — add each listed finding to CURRENT:
- Only add it if the finding is written in SOURCE.
- Only add it if the same finding is not already present in CURRENT.
- Add it once under the suggested section.

unsupported_findings — remove each listed finding from CURRENT:
- Only remove it if the finding is genuinely absent from SOURCE.
- If SOURCE mentions it, do not remove it regardless of what the feedback says.

━━━ TASK 2: Apply ANATOMY_FEEDBACK ━━━━━━━━━━━━━━━━━━━━━━━━━

wrong_section_findings — move each listed finding to the correct section:
- Remove the finding from current_section.
- Add it once under correct_section.
- The finding must appear in the new section only. Do not leave a copy in the old section.

duplicate_findings — remove the duplicate copy:
- * Must remove the finding from section_to_remove_the_finding_from.*
- Keep the copy in the other section untouched.

━━━ Important rules for all tasks ━━━━━━━━━━━━━━━━━━━━━━━━━━
- Each finding must appear exactly once in the final report. Never create duplicates.
- Use only the allowed section headers listed above.
- After all edits, remove any section that has no findings left.
- Never output an empty section header.
- Do not change any finding that is not mentioned in the feedback.
- If anatomy header already exists, add findings to it. If it does not exist, create it.
- After removing the findings, if an anatomy header has no findings left, remove that header as well.
- Don't add or remove findings based on your own knowledge. Use only the SOURCE text and the feedback lists.

Return ONLY the revised structured radiology report.
No explanations, notes, JSON, or change summaries.

SOURCE:
{free_text}

CURRENT:
{current_report}

FINDINGS_FEEDBACK:
{json.dumps(findings_feedback, indent=2, ensure_ascii=False)}

ANATOMY_FEEDBACK:
{json.dumps(anatomy_feedback, indent=2, ensure_ascii=False)}
""".strip()
