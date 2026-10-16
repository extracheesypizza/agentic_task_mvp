"""Validate model output without interpreting or changing clinical values."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Finding(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")

    parameter: str | None = None
    value: str | int | float | None = None
    unit: str | None = None
    reference_range: str | None = None
    status: Literal["below_normal", "normal", "above_normal", "unknown"] | None = "unknown"


class Medication(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")

    name: str | None = None
    dosage: str | None = None
    frequency: str | None = None


class Extraction(BaseModel):
    # Preserve additional metadata from existing model responses for API consumers.
    model_config = ConfigDict(strict=True, extra="allow")

    document_type: Literal["lab_result", "prescription", "discharge_summary", "unreadable", "other"]
    findings: list[Finding] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    patient_info: dict[str, Any] | None = None
    raw_notes: str | None = None

    @field_validator("findings", "medications", mode="before")
    @classmethod
    def normalize_absent_lists(cls, value):
        # The prompt permits null for absent fields; downstream code expects lists.
        return [] if value is None else value

    @model_validator(mode="after")
    def require_usable_result(self):
        if self.document_type == "unreadable":
            if self.findings or self.medications:
                raise ValueError("Unreadable documents must not contain extracted results")
        elif not (self.findings or self.medications):
            raise ValueError("Extraction contains no findings or medications")
        return self
