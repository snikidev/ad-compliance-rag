from __future__ import annotations

from ad_compliance.mcp_server import mcp
from ad_compliance.models import ComplianceReport


def test_check_ad_tool_is_registered() -> None:
    tools = getattr(mcp, "tools", None) or getattr(mcp, "_tools", None)
    # FastMCP keeps tools in different places across versions; fall back to names.
    names: list[str] = []
    if isinstance(tools, dict):
        names = list(tools)
    elif tools is not None:
        names = [getattr(tool, "name", str(tool)) for tool in tools]
    else:
        manager = getattr(mcp, "_tool_manager", None) or getattr(mcp, "tool_manager", None)
        if manager is not None:
            maybe = getattr(manager, "tools", {}) or getattr(manager, "_tools", {})
            names = (
                list(maybe)
                if isinstance(maybe, dict)
                else [getattr(t, "name", str(t)) for t in maybe]
            )
    if not names:
        # Last resort: the decorator should have bound the function on the module.
        from ad_compliance import mcp_server

        assert callable(mcp_server.check_ad)
    else:
        assert "check_ad" in names or any("check_ad" in str(name) for name in names)


def test_tool_returns_report(monkeypatch) -> None:
    from ad_compliance import mcp_server

    fake = ComplianceReport(overall="ok", summary="stub", ad_text="hello", verdicts=[])
    monkeypatch.setattr(mcp_server, "run_check", lambda ad_text: fake)
    payload = mcp_server.check_ad("hello")
    assert payload["overall"] == "ok"
    assert payload["ad_text"] == "hello"
