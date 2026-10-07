"""
Writes the viewer of an analysed clip: analysis.json (the data) and a self-contained index.html
(the template web/telestrator.html with the data inlined). The page needs no server: it loads
clip.mp4 by a relative path and reads the data from a <script type="application/json"> block,
because fetch() of a local JSON file is blocked on file:// URLs.
"""
import html
import json
import logging
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Tuple

from . import config

logger = logging.getLogger(__name__)

PLACEHOLDER_DATA = "__ANALYSIS_JSON__"
PLACEHOLDER_VIDEO = "__VIDEO_SRC__"
PLACEHOLDER_TITLE = "__TITLE__"

ANALYSIS_FILE = "analysis.json"
HTML_FILE = "index.html"


def dump_json(analysis: Dict[str, Any]) -> str:
    """The compact JSON text of the analysis (UTF-8: the Vietnamese labels stay readable)."""
    return json.dumps(analysis, ensure_ascii=False, separators=(",", ":"))


SCRIPT_ESCAPES = {char: chr(92) + "u%04x" % ord(char) for char in ("<", chr(0x2028), chr(0x2029))}
"""JSON escapes (backslash u + 4 hex digits) of the characters that must not appear raw inside a <script>."""


def script_safe(json_text: str) -> str:
    """JSON text made safe inside a <script> element: every "<" escaped (so "</script>" and "<!--"
    cannot occur) and the line separators U+2028/U+2029 escaped (they end a line in older JavaScript)."""
    for char, escaped in SCRIPT_ESCAPES.items():
        json_text = json_text.replace(char, escaped)
    return json_text


def inline_json(analysis: Dict[str, Any]) -> str:
    return script_safe(dump_json(analysis))


def _fill(template_text: str, json_text: str, video_rel: str, title: str) -> str:
    """The template with the analysis JSON text, the clip path and the title filled in.

    video_rel must be a relative POSIX path (the page is moved around together with the clip, so
    an absolute or drive-letter path would break it).
    """
    if (not video_rel or video_rel.startswith(("/", "\\")) or "\\" in video_rel or ":" in video_rel
            or PurePosixPath(video_rel).is_absolute()):
        raise ValueError(f"video_rel must be a relative POSIX path, got {video_rel!r}")
    for placeholder in (PLACEHOLDER_DATA, PLACEHOLDER_VIDEO):
        if template_text.count(placeholder) != 1:
            raise ValueError(f"Template must contain {placeholder} exactly once")
    out = template_text.replace(PLACEHOLDER_TITLE, html.escape(title))
    out = out.replace(PLACEHOLDER_VIDEO, html.escape(video_rel, quote=True))
    # last, so a placeholder-looking string inside the data is never substituted
    return out.replace(PLACEHOLDER_DATA, script_safe(json_text))


def render_html(template_text: str, analysis: Dict[str, Any], video_rel: str, title: str = "") -> str:
    """The viewer page for `analysis` (see _fill)."""
    return _fill(template_text, dump_json(analysis), video_rel, title)


def write_clip_files(out_dir, analysis: Dict[str, Any], video_rel: str, title: str = "",
                     template_path=config.TELESTRATOR_TEMPLATE_PATH) -> Tuple[Path, Path]:
    """Write out_dir/analysis.json and out_dir/index.html (the analysis is serialised once).
    Returns (json_path, html_path)."""
    template_path = Path(template_path)
    if not template_path.exists():
        raise FileNotFoundError(f"Viewer template not found at {template_path} (expected backend/web/telestrator.html)")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_text = dump_json(analysis)
    json_path = out_dir / ANALYSIS_FILE
    json_path.write_text(json_text, encoding="utf-8")
    html_path = out_dir / HTML_FILE
    html_path.write_text(_fill(template_path.read_text(encoding="utf-8"), json_text, video_rel, title), encoding="utf-8")
    logger.info("Wrote %s and %s", json_path, html_path)
    return json_path, html_path
