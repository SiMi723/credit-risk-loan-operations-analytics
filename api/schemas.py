"""
Pydantic schemas for the Credit Risk API.
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator
import math


class ApplicationInput(BaseModel):
    """
    Input schema for a single loan application prediction request.
    Fields mirror the key columns from application_train.csv.
    """

    # Identity / contract type
    NAME_CONTRACT_TYPE: str = Field(..., description="Cash loans or Revolving loans")
    CODE_GENDER: str = Field(..., description="M, F, or XNA")
    FLAG_OWN_CAR: str = Field(..., description="Y or N")
    FLAG_OWN_REALTY: str = Field(..., description="Y or N")

    # Demographic
    CNT_CHILDREN: int = Field(..., ge=0, le=20)
    CNT_FAM_MEMBERS: float = Field(..., ge=1, le=25)
    DAYS_BIRTH: int = Field(..., description="Days relative to application (negative integer)")
    DAYS_EMPLOYED: float = Field(..., description="Days employed (negative) or 365243 for N/A")
    DAYS_REGISTRATION: float = Field(...)
    DAYS_ID_PUBLISH: int = Field(...)

    # Financial
    AMT_INCOME_TOTAL: float = Field(..., gt=0)
    AMT_CREDIT: float = Field(..., gt=0)
    AMT_ANNUITY: Optional[float] = Field(None, gt=0)
    AMT_GOODS_PRICE: Optional[float] = Field(None, gt=0)

    # Application details
    NAME_INCOME_TYPE: str = Field(...)
    NAME_EDUCATION_TYPE: str = Field(...)
    NAME_FAMILY_STATUS: str = Field(...)
    NAME_HOUSING_TYPE: str = Field(...)
    OCCUPATION_TYPE: Optional[str] = None
    ORGANIZATION_TYPE: Optional[str] = None

    # Region
    REGION_POPULATION_RELATIVE: float = Field(..., ge=0, le=1)
    REGION_RATING_CLIENT: int = Field(..., ge=1, le=3)
    REGION_RATING_CLIENT_W_CITY: int = Field(..., ge=1, le=3)

    # External sources
    EXT_SOURCE_1: Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_2: Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_3: Optional[float] = Field(None, ge=0, le=1)

    # Contact flags
    FLAG_MOBIL: int = Field(..., ge=0, le=1)
    FLAG_EMP_PHONE: int = Field(..., ge=0, le=1)
    FLAG_WORK_PHONE: int = Field(..., ge=0, le=1)
    FLAG_PHONE: int = Field(..., ge=0, le=1)
    FLAG_EMAIL: int = Field(..., ge=0, le=1)

    # Credit bureau enquiries
    AMT_REQ_CREDIT_BUREAU_HOUR: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_DAY: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_WEEK: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_MON: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_QRT: Optional[float] = None
    AMT_REQ_CREDIT_BUREAU_YEAR: Optional[float] = None

    # Social circle
    OBS_30_CNT_SOCIAL_CIRCLE: Optional[float] = None
    DEF_30_CNT_SOCIAL_CIRCLE: Optional[float] = None
    OBS_60_CNT_SOCIAL_CIRCLE: Optional[float] = None
    DEF_60_CNT_SOCIAL_CIRCLE: Optional[float] = None

    # Days flags
    DAYS_LAST_PHONE_CHANGE: Optional[float] = None

    # OWN_CAR_AGE
    OWN_CAR_AGE: Optional[float] = None

    # Weekday / hour
    WEEKDAY_APPR_PROCESS_START: Optional[str] = None
    HOUR_APPR_PROCESS_START: Optional[int] = Field(None, ge=0, le=23)

    # Region flags
    REG_REGION_NOT_LIVE_REGION: Optional[int] = Field(None, ge=0, le=1)
    REG_REGION_NOT_WORK_REGION: Optional[int] = Field(None, ge=0, le=1)
    LIVE_REGION_NOT_WORK_REGION: Optional[int] = Field(None, ge=0, le=1)
    REG_CITY_NOT_LIVE_CITY: Optional[int] = Field(None, ge=0, le=1)
    REG_CITY_NOT_WORK_CITY: Optional[int] = Field(None, ge=0, le=1)
    LIVE_CITY_NOT_WORK_CITY: Optional[int] = Field(None, ge=0, le=1)

    @field_validator("CODE_GENDER")
    @classmethod
    def validate_gender(cls, v):
        if v not in {"M", "F", "XNA"}:
            raise ValueError(f"CODE_GENDER must be one of M, F, XNA. Got: {v!r}")
        return v

    @field_validator("NAME_CONTRACT_TYPE")
    @classmethod
    def validate_contract_type(cls, v):
        if v not in {"Cash loans", "Revolving loans"}:
            raise ValueError(
                f"NAME_CONTRACT_TYPE must be 'Cash loans' or 'Revolving loans'. Got: {v!r}"
            )
        return v

    @field_validator("DAYS_BIRTH")
    @classmethod
    def validate_age(cls, v):
        age_years = -v / 365.0
        if age_years < 18 or age_years > 100:
            raise ValueError(
                f"DAYS_BIRTH corresponds to age {age_years:.1f} years; must be between 18 and 100."
            )
        return v

    model_config = {"extra": "allow"}  # allow auxiliary features in payload


class PredictionResponse(BaseModel):
    """Response schema for a credit-risk prediction."""
    default_probability: float = Field(..., description="Predicted probability of default [0, 1]")
    prediction: int = Field(..., description="Binary prediction (1=default risk, 0=low risk)")
    risk_segment: str = Field(..., description="LOW / MEDIUM / HIGH / VERY_HIGH")
    threshold: float = Field(..., description="Decision threshold used")
    model_version: str = Field(default="1.0.0")
    disclaimer: str = Field(
        default=(
            "This is a research prototype trained on the Home Credit public dataset. "
            "It is NOT a regulatory-approved credit decision tool."
        )
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str


class ErrorResponse(BaseModel):
    detail: str
    error_type: str


def classify_risk_segment(prob: float) -> str:
    """Map default probability to a risk segment label."""
    if prob < 0.10:
        return "LOW"
    elif prob < 0.25:
        return "MEDIUM"
    elif prob < 0.50:
        return "HIGH"
    else:
        return "VERY_HIGH"
