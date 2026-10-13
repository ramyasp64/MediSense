"""
Clinical alert generation — bridges technical anomaly detection to actionable
guidance for healthcare workers with limited medical training.

Thesis requirement: dual-level detection with human-readable alerts.
    Level 1: Sensor malfunction  → "Equipment problem detected"
    Level 2: Patient anomaly     → "URGENT: HR >120 BPM. Possible tachycardia."
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from ..data.base import VitalSigns


class AlertLevel(Enum):
    GREEN = "green"    # Normal — no action needed
    YELLOW = "yellow"  # Warning — monitor closely
    RED = "red"        # Urgent — assess patient immediately


class AlertType(Enum):
    NORMAL = "normal"
    SENSOR_MALFUNCTION = "sensor_malfunction"
    TACHYCARDIA = "tachycardia"
    BRADYCARDIA = "bradycardia"
    TACHYPNEA = "tachypnea"
    BRADYPNEA = "bradypnea"
    HYPOXEMIA = "hypoxemia"
    FEVER = "fever"
    HYPOTHERMIA = "hypothermia"
    MULTI_PARAMETER = "multi_parameter"  # multiple abnormalities


@dataclass
class ClinicalAlert:
    level: AlertLevel
    alert_type: AlertType
    title: str
    message: str
    action: str
    vitals: Optional[VitalSigns] = None

    def display(self) -> str:
        color_prefix = {"green": "✓", "yellow": "⚠", "red": "🔴"}[self.level.value]
        return (
            f"\n{color_prefix} [{self.level.value.upper()}] {self.title}\n"
            f"   {self.message}\n"
            f"   Action: {self.action}"
        )


# Clinical thresholds (WHO / standard clinical practice)
_THRESHOLDS = {
    "hr_tachy": 100,
    "hr_severe_tachy": 130,
    "hr_brady": 60,
    "hr_severe_brady": 40,
    "rr_tachy": 20,
    "rr_severe_tachy": 30,
    "rr_brady": 12,
    "rr_severe_brady": 8,
    "spo2_low": 95,
    "spo2_critical": 90,
    "temp_fever": 38.0,
    "temp_high_fever": 39.5,
    "temp_hypo": 36.0,
    "temp_severe_hypo": 35.0,
}


def generate_alert(
    sensor_anomaly_score: float,
    patient_anomaly_score: float,
    sensor_threshold: float,
    patient_threshold: float,
    vitals: Optional[VitalSigns] = None,
) -> ClinicalAlert:
    """
    Generate a clinical alert from dual-level anomaly scores.

    Decision tree:
        1. If sensor_anomaly_score ≥ threshold → Level 1 alert (equipment)
        2. Else if patient_anomaly_score ≥ threshold → Level 2 alert (patient)
        3. Else → Green / normal

    Args:
        sensor_anomaly_score: Reconstruction error from sensor-malfunction model.
        patient_anomaly_score: Reconstruction error from patient-anomaly model.
        sensor_threshold: Calibrated threshold for Level 1 detection.
        patient_threshold: Calibrated threshold for Level 2 detection.
        vitals: Ground-truth or estimated vital signs (for specific alert messages).

    Returns:
        ClinicalAlert with level, type, and human-readable guidance.
    """
    # Level 1: Sensor malfunction
    if sensor_anomaly_score >= sensor_threshold:
        return ClinicalAlert(
            level=AlertLevel.YELLOW,
            alert_type=AlertType.SENSOR_MALFUNCTION,
            title="Equipment Problem Detected",
            message=(
                "Sensor readings appear unreliable. "
                "Possible causes: sensor detached, obscured, or malfunctioning."
            ),
            action=(
                "Check sensor placement. Clean sensor contact area. "
                "Reattach and wait 30 seconds for stable readings."
            ),
        )

    # Level 2: Patient anomaly — use vitals if available for specific message
    if patient_anomaly_score >= patient_threshold:
        return _patient_alert(vitals)

    # Green: all normal
    return ClinicalAlert(
        level=AlertLevel.GREEN,
        alert_type=AlertType.NORMAL,
        title="Vital Signs Normal",
        message="All monitored parameters within normal range.",
        action="Continue monitoring. Check again in 15 minutes.",
        vitals=vitals,
    )


def _patient_alert(vitals: Optional[VitalSigns]) -> ClinicalAlert:
    """Generate specific patient anomaly alert based on which vitals are abnormal."""
    if vitals is None:
        return ClinicalAlert(
            level=AlertLevel.RED,
            alert_type=AlertType.MULTI_PARAMETER,
            title="URGENT: Abnormal Vital Signs Detected",
            message="Sensor pattern indicates abnormal patient status.",
            action="Assess patient immediately. Check pulse and breathing manually.",
        )

    alerts_found = []
    t = _THRESHOLDS

    hr = vitals.heart_rate_bpm
    rr = vitals.respiratory_rate_bpm
    spo2 = vitals.spo2_pct
    temp = vitals.temperature_c

    if hr is not None:
        if hr >= t["hr_severe_tachy"]:
            alerts_found.append(("RED", AlertType.TACHYCARDIA,
                f"HR {hr:.0f} BPM — severe tachycardia",
                "Possible sepsis, severe infection, or cardiac event. "
                "Assess patient urgently. Apply oxygen if available."))
        elif hr >= t["hr_tachy"]:
            alerts_found.append(("YELLOW", AlertType.TACHYCARDIA,
                f"HR {hr:.0f} BPM — elevated heart rate",
                "Monitor closely. Check for fever, pain, or dehydration."))
        elif hr <= t["hr_severe_brady"]:
            alerts_found.append(("RED", AlertType.BRADYCARDIA,
                f"HR {hr:.0f} BPM — severe bradycardia",
                "Critical: very slow heart rate. Assess patient immediately."))
        elif hr <= t["hr_brady"]:
            alerts_found.append(("YELLOW", AlertType.BRADYCARDIA,
                f"HR {hr:.0f} BPM — low heart rate",
                "Monitor closely. Reassess in 5 minutes."))

    if rr is not None:
        if rr >= t["rr_severe_tachy"]:
            alerts_found.append(("RED", AlertType.TACHYPNEA,
                f"RR {rr:.0f}/min — severe respiratory distress",
                "Possible pneumonia, sepsis, or respiratory failure. "
                "Assess breathing urgently."))
        elif rr >= t["rr_tachy"]:
            alerts_found.append(("YELLOW", AlertType.TACHYPNEA,
                f"RR {rr:.0f}/min — elevated breathing rate",
                "Monitor closely. Check for fever or respiratory infection."))
        elif rr <= t["rr_severe_brady"]:
            alerts_found.append(("RED", AlertType.BRADYPNEA,
                f"RR {rr:.0f}/min — dangerously slow breathing",
                "Critical: very slow breathing. Assess patient immediately."))

    if spo2 is not None:
        if spo2 < t["spo2_critical"]:
            alerts_found.append(("RED", AlertType.HYPOXEMIA,
                f"SpO2 {spo2:.0f}% — critical hypoxemia",
                "Apply supplemental oxygen immediately if available. "
                "Urgent clinical assessment required."))
        elif spo2 < t["spo2_low"]:
            alerts_found.append(("YELLOW", AlertType.HYPOXEMIA,
                f"SpO2 {spo2:.0f}% — low oxygen saturation",
                "Position patient upright. Apply oxygen if available. Monitor closely."))

    if temp is not None:
        if temp >= t["temp_high_fever"]:
            alerts_found.append(("RED", AlertType.FEVER,
                f"Temperature {temp:.1f}°C — high fever",
                "Assess for infection source. Administer antipyretics if available."))
        elif temp >= t["temp_fever"]:
            alerts_found.append(("YELLOW", AlertType.FEVER,
                f"Temperature {temp:.1f}°C — fever",
                "Monitor temperature. Check for infection signs."))
        elif temp <= t["temp_severe_hypo"]:
            alerts_found.append(("RED", AlertType.HYPOTHERMIA,
                f"Temperature {temp:.1f}°C — hypothermia",
                "Warm patient immediately. Prevent heat loss."))

    if not alerts_found:
        # Anomaly detected by model but vitals look borderline — flag for review
        return ClinicalAlert(
            level=AlertLevel.YELLOW,
            alert_type=AlertType.MULTI_PARAMETER,
            title="Warning: Unusual Signal Pattern",
            message="Sensor pattern is outside normal range but specific cause unclear.",
            action="Manually assess patient. Recheck vitals in 5 minutes.",
            vitals=vitals,
        )

    # Use the highest severity alert
    level_order = {"RED": 2, "YELLOW": 1}
    alerts_found.sort(key=lambda a: level_order.get(a[0], 0), reverse=True)
    top_level, top_type, top_msg, top_action = alerts_found[0]

    # Include all findings in the message
    all_findings = " | ".join(a[2] for a in alerts_found)

    return ClinicalAlert(
        level=AlertLevel.RED if top_level == "RED" else AlertLevel.YELLOW,
        alert_type=top_type,
        title=f"{'URGENT: ' if top_level == 'RED' else 'Warning: '}{top_msg}",
        message=all_findings,
        action=top_action,
        vitals=vitals,
    )
