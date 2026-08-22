from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

class SegmentResult(BaseModel):
    cut1: float
    cut2: float
    
class AteResult(BaseModel):
    cls : int
    clsnum : int
    ate : float
    se : float
    ci_low : float
    ci_high : float
    

Scalar = str | int | float | bool
class DrcdfResult(BaseModel):
    seg: int
    c: int
    threshold: Scalar
    F1_dr: float
    F0_dr: float
    tau_c: float
    se_c: float
    ci_low: float
    ci_high: float
    
class HeiResult(BaseModel):
    score : float

class EstimateResponse(BaseModel):

    segment: SegmentResult

    ate: list[AteResult]

    drcdf: list[DrcdfResult]

    hei: HeiResult

class BinaryColumnTreatment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["binary_column"]
    column: str
    treated_value: Scalar = 1
    control_value: Scalar = 0

    @model_validator(mode="after")
    def validate_values(self):
        if self.treated_value == self.control_value:
            raise ValueError(
                "treated_valueとcontrol_valueは"
                "異なる値を指定してください。"
            )

        return self


class QuantileTreatment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["quantile"]
    source_column: str
    quantile: float = Field(
        gt=0,
        lt=1,
        allow_inf_nan=False,
    )
    treated_when: Literal[
        "ge",
        "gt",
        "le",
        "lt",
    ] = "ge"


TreatmentSpec = Annotated[
    BinaryColumnTreatment | QuantileTreatment,
    Field(discriminator="mode"),
]


class OutcomeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    levels: list[Scalar] = Field(
        min_length=2,
    )
    scores: list[float] = Field(
        min_length=2,
    )

    @model_validator(mode="after")
    def validate_outcome(self):
        if len(self.levels) != len(self.scores):
            raise ValueError(
                "levelsとscoresの長さを"
                "一致させてください。"
            )

        if len(set(self.levels)) != len(self.levels):
            raise ValueError(
                "levelsには重複しない値を"
                "指定してください。"
            )

        if any(
            left >= right
            for left, right in zip(
                self.scores,
                self.scores[1:],
            )
        ):
            raise ValueError(
                "scoresは昇順で指定してください。"
            )

        return self


class SegmentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    missing_values: list[Scalar] = Field(
        default_factory=list,
    )


class CovariateSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    columns: list[str] = Field(
        min_length=1,
    )
    categorical_columns: list[str] = Field(
        default_factory=list,
    )

    @model_validator(mode="after")
    def validate_categorical_columns(self):
        unknown = (
            set(self.categorical_columns)
            - set(self.columns)
        )

        if unknown:
            raise ValueError(
                "categorical_columnsはcolumnsに"
                "含まれる列を指定してください: "
                f"{sorted(unknown)}"
            )

        return self


class MissingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["drop", "zero", "fill"] = "drop"
    fill_values: dict[str, Scalar] = Field(
        default_factory=dict,
    )
    
    @model_validator(mode="after")
    def validate_missing(self):
        if self.strategy == "fill" and not self.fill_values:
            raise ValueError(
                "strategy='fill'ではfill_valuesを指定してください。"
            )

        if self.strategy != "fill" and self.fill_values:
            raise ValueError(
                "fill_valuesはstrategy='fill'の場合だけ指定できます。"
            )

        return self
    
class AnalysisRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    schema_version: Literal["1"] = "1"
    treatment: TreatmentSpec
    outcome: OutcomeSpec
    segment: SegmentSpec
    covariates: CovariateSpec
    missing: MissingSpec = Field(
        default_factory=MissingSpec,
    )

    @model_validator(mode="after")
    def validate_column_roles(self):
        if self.treatment.mode == "binary_column":
            treatment_column = self.treatment.column
        else:
            treatment_column = self.treatment.source_column

        main_roles = {
            "outcome": self.outcome.column,
            "segment": self.segment.column,
            "treatment": treatment_column,
        }

        roles_by_column = {}

        for role, column in main_roles.items():
            roles_by_column.setdefault(
                column,
                [],
            ).append(role)

        duplicated_roles = {
            column: roles
            for column, roles in roles_by_column.items()
            if len(roles) > 1
        }

        if duplicated_roles:
            raise ValueError(
                "outcome・segment・treatmentには"
                "異なる列を指定してください: "
                f"{duplicated_roles}"
            )

        overlap = (
            set(self.covariates.columns)
            & set(main_roles.values())
        )

        if overlap:
            raise ValueError(
                "covariatesにはoutcome・segment・"
                "treatmentの列を指定できません: "
                f"{sorted(overlap)}"
            )

        return self