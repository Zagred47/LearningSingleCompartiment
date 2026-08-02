from pathlib import Path

from scripts.validate_research_program import validate


def test_research_program_cross_references():
    root = Path(__file__).resolve().parents[1]
    assert validate(root) == []
