from __future__ import annotations


class RiskAnalyzer:
    def assess(self, action: str) -> list[str]:
        risks: list[str] = []
        lowered = action.lower()
        if any(term in lowered for term in ("write", "delete", "execute", "buy", "send")):
            risks.append("Consequential action requires explicit authorization.")
        if "assumption" in lowered or "current evidence" in lowered:
            risks.append("Recommendation may be brittle if assumptions are wrong.")
        return risks or ["Low immediate risk; main risk is opportunity cost."]

