from __future__ import annotations

from pathlib import Path


def test_skill_is_catalog_friendly() -> None:
    path = Path("skills/ad-compliance-check/SKILL.md")
    if not path.is_file():
        path = Path(__file__).resolve().parents[1] / "skills/ad-compliance-check/SKILL.md"
    text = path.read_text(encoding="utf-8")
    assert text.lstrip().startswith("---")
    assert "name: ad-compliance-check" in text
    assert "description:" in text
    assert "ad-compliance check" in text
    assert "check_ad" in text
    assert "not legal advice" in text.lower()
    assert "Gambling Commission" in text
    # Catalogs name the skill after the folder that holds SKILL.md, not the repo root.
    assert path.parent.name == "ad-compliance-check"
