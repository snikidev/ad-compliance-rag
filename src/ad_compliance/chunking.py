"""Structure-aware chunking of the CAP/BCAP under-18s gambling advertising guidance."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

DOC_ID = "cap-bcap-2022"
DOC_TITLE = "Gambling and lotteries advertising: protecting under-18s"
DEFAULT_PDF_NAME = "Protecting-children-and-young-people-gambling-guidance-2022.pdf"

# Numbered guidance sections are 1–30. Dotted citations like "17.16.2" are Code rules,
# not section headings.
_SECTION_HEADING = re.compile(r"^(?P<num>\d{1,2})\.(?P<gap>\s*)(?P<title>[A-Za-z“'‘].+)$")
_FRONT_HEADING = re.compile(
    r"^(Foreword|Protecting under-18s: checklist|First steps|"
    r"Principle: restricting appeal of ads to under-18s|"
    r"High risk ad content/\s*approaches|Activities of inherent .+appeal|"
    r"Use of persons and characters|Targeting exemption|Introduction)$",
    re.IGNORECASE,
)
_FOOTER = re.compile(r"^Advertising Guidance\s+\d+$")
_CODE_REF = re.compile(
    r"\b(?:CAP|BCAP)\s+Code\s+rules?\s+[0-9.]+(?:\s*/\s*[0-9.]+)?",
    re.IGNORECASE,
)
_WRAP_TAIL = re.compile(
    r"(?:\b(?:of|and|or|the|to|a|an|for|in|on|owing to)|"
    r"['‘’]strong['’]?|character['’]s|persons and)$",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z“\"‘])")

DEFAULT_MAX_CHARS = 1400
DEFAULT_OVERLAP = 180


@dataclass(frozen=True)
class GuidanceChunk:
    """A retrieval unit with stable identity and section metadata."""

    chunk_id: str
    doc_id: str
    text: str
    section_id: str
    section_title: str
    section_number: int | None
    page_start: int
    page_end: int
    chunk_index: int
    chunk_count: int
    code_refs: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class _Section:
    section_id: str
    title: str
    number: int | None
    pages: list[int]
    paragraphs: list[str]


def default_pdf_path(data_dir: Path | None = None) -> Path:
    """Locate the bundled guidance PDF, whether running from a checkout or a wheel."""
    candidates: list[Path] = []
    if data_dir is not None:
        candidates.append(data_dir / DEFAULT_PDF_NAME)
    candidates.extend(
        [
            Path.cwd() / "data" / DEFAULT_PDF_NAME,
            Path(__file__).resolve().parents[2] / "data" / DEFAULT_PDF_NAME,
            Path(__file__).resolve().parent / "data" / DEFAULT_PDF_NAME,
        ]
    )
    for path in candidates:
        if path.is_file():
            return path
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {DEFAULT_PDF_NAME}. Looked in: {searched}")


def extract_pages(pdf_path: Path) -> list[tuple[int, list[str]]]:
    """Return (1-based page number, cleaned lines) for every page in the PDF."""
    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, list[str]]] = []
    for index, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        lines = [_clean_line(line) for line in raw.splitlines()]
        lines = [line for line in lines if line and not _FOOTER.match(line)]
        pages.append((index, lines))
    return pages


def chunk_pdf(
    pdf_path: Path | str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
    doc_id: str = DOC_ID,
) -> list[GuidanceChunk]:
    """Split the guidance PDF into section-aware chunks with stable ids."""
    path = Path(pdf_path)
    sections = _parse_sections(extract_pages(path))
    chunks: list[GuidanceChunk] = []
    for section in sections:
        texts = _split_section(section, max_chars=max_chars, overlap=overlap)
        count = len(texts)
        page_start = min(section.pages) if section.pages else 1
        page_end = max(section.pages) if section.pages else page_start
        refs = _unique_refs(" ".join(section.paragraphs))
        for index, text in enumerate(texts):
            chunk_id = _stable_id(doc_id, section.section_id, text)
            chunks.append(
                GuidanceChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=text,
                    section_id=section.section_id,
                    section_title=section.title,
                    section_number=section.number,
                    page_start=page_start,
                    page_end=page_end,
                    chunk_index=index,
                    chunk_count=count,
                    code_refs=refs,
                )
            )
    return chunks


def _clean_line(line: str) -> str:
    line = line.replace("\xa0", " ")
    line = re.sub(r"[ \t]+", " ", line).strip()
    return line


def _parse_sections(pages: list[tuple[int, list[str]]]) -> list[_Section]:
    sections: list[_Section] = []
    current: _Section | None = None

    def start(title: str, number: int | None, page: int) -> None:
        nonlocal current
        if current is not None and (current.paragraphs or current.pages):
            sections.append(current)
        section_id = f"s{number:02d}" if number is not None else _slug(title)
        current = _Section(
            section_id=section_id,
            title=title,
            number=number,
            pages=[page],
            paragraphs=[],
        )

    # The title page has no conventional heading.
    start(DOC_TITLE, None, 1)

    pending_wrap: tuple[int, str] | None = None  # (section_number, title_so_far)

    for page_number, lines in pages:
        i = 0
        while i < len(lines):
            line = lines[i]
            next_line = lines[i + 1] if i + 1 < len(lines) else ""

            if pending_wrap is not None:
                number, title = pending_wrap
                if _is_title_continuation(title, next_line=line):
                    title = f"{title} {line}".strip()
                    start(title, number, page_number)
                    pending_wrap = None
                    i += 1
                    continue
                start(title, number, page_number)
                pending_wrap = None
                # fall through and reprocess this line

            heading = _match_heading(line, next_line)
            if heading is not None:
                number, title, consumed_next = heading
                if consumed_next:
                    i += 1
                if current is None or current.section_id != (
                    f"s{number:02d}" if number is not None else _slug(title)
                ):
                    # If the title still looks truncated and there is no next line on
                    # this page, wait for the following page.
                    if number is not None and _WRAP_TAIL.search(title) and not consumed_next:
                        pending_wrap = (number, title)
                    else:
                        start(title, number, page_number)
                i += 1
                continue

            # Checklist pages print the destination section number on its own line.
            if re.fullmatch(r"\d{1,2}", line):
                i += 1
                continue

            if current is None:
                start(DOC_TITLE, None, page_number)
            assert current is not None
            if page_number not in current.pages:
                current.pages.append(page_number)
            current.paragraphs.append(line)
            i += 1

    if pending_wrap is not None:
        number, title = pending_wrap
        start(title, number, pages[-1][0] if pages else 1)

    if current is not None:
        sections.append(current)

    # Drop empty sections (e.g. a title page placeholder that never received body
    # if we immediately replaced it — we keep the title page if it has text).
    return [section for section in sections if _section_text(section).strip()]


def _match_heading(line: str, next_line: str) -> tuple[int | None, str, bool] | None:
    numbered = _SECTION_HEADING.match(line)
    if numbered:
        number = int(numbered.group("num"))
        if number < 1 or number > 30:
            return None
        # "17.1 Marketing…" and "16.3.12" are Code citations, not section titles.
        title = numbered.group("title").strip()
        if title[:1].isdigit():
            return None
        consumed = False
        if _is_title_continuation(title, next_line=next_line):
            title = f"{title} {next_line}".strip()
            consumed = True
        return number, title, consumed

    # "Introduction" and checklist sub-headings.
    if _FRONT_HEADING.match(line):
        title = re.sub(r"\s+", " ", line).strip()
        return None, title, False
    return None


def _is_title_continuation(title: str, *, next_line: str) -> bool:
    if not next_line:
        return False
    if _SECTION_HEADING.match(next_line) or _FRONT_HEADING.match(next_line):
        return False
    if next_line.startswith(("•", "-", "o ")):
        return False
    if _WRAP_TAIL.search(title):
        return True
    # Short leftover fragments like "appeal to under-18s" or "characters".
    return bool(
        len(next_line) <= 60 and not next_line.endswith((".", ";", ":")) and next_line[0].islower()
    )


def _slug(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "section"


def _section_text(section: _Section) -> str:
    return _join_lines(section.paragraphs)


def _join_lines(lines: list[str]) -> str:
    if not lines:
        return ""
    parts: list[str] = [lines[0]]
    for line in lines[1:]:
        prev = parts[-1]
        if prev.endswith("-") and line[:1].isalnum():
            # Keep authored hyphens: "under-\n18s" -> "under-18s".
            parts[-1] = prev + line
        elif prev.endswith((".", ":", ";", "?", "!", "•")) or line.startswith(("•", "o ")):
            parts.append(line)
        else:
            parts[-1] = f"{prev} {line}".strip()
    text = "\n".join(parts)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_section(section: _Section, *, max_chars: int, overlap: int) -> list[str]:
    text = _section_text(section)
    header = _header(section)
    body_budget = max(max_chars - len(header) - 2, 400)
    if len(text) <= body_budget:
        return [f"{header}\n\n{text}".strip() if text else header]

    pieces = _window(text, max_chars=body_budget, overlap=overlap)
    return [f"{header}\n\n{piece}".strip() for piece in pieces]


def _header(section: _Section) -> str:
    if section.number is not None:
        return f"Section {section.number}. {section.title}"
    return section.title


def _window(text: str, *, max_chars: int, overlap: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    units: list[str] = []
    for para in paragraphs:
        # Treat bullet lists as split points so a long checklist cannot blow the budget.
        bullets = re.split(r"(?=\n• |\n- |\no )", para)
        for piece in bullets:
            piece = piece.strip()
            if not piece:
                continue
            if len(piece) <= max_chars:
                units.append(piece)
            else:
                units.extend(_split_sentences(piece, max_chars=max_chars))

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}".strip() if current else unit
        if len(candidate) <= max_chars or not current:
            current = candidate
            continue
        chunks.append(current)
        current = _overlap_prefix(current, overlap)
        current = f"{current}\n\n{unit}".strip() if current else unit
    if current:
        chunks.append(current)
    tight: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            tight.append(chunk)
        else:
            tight.extend(_split_sentences(chunk, max_chars=max_chars))
    return tight


def _split_sentences(text: str, *, max_chars: int) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE.split(text) if s.strip()]
    if not sentences:
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars or not current:
            current = candidate
            continue
        chunks.append(current)
        current = sentence
    if current:
        chunks.append(current)
    return chunks


def _overlap_prefix(text: str, overlap: int) -> str:
    if overlap <= 0 or len(text) <= overlap:
        return text
    snippet = text[-overlap:]
    # Start on a word boundary when we can.
    cut = snippet.find(" ")
    return snippet[cut + 1 :].strip() if cut != -1 else snippet


def _unique_refs(text: str) -> tuple[str, ...]:
    seen: list[str] = []
    for match in _CODE_REF.finditer(text):
        ref = re.sub(r"\s+", " ", match.group(0)).strip()
        if ref not in seen:
            seen.append(ref)
    return tuple(seen)


def _stable_id(doc_id: str, section_id: str, text: str) -> str:
    normalised = re.sub(r"\s+", " ", text).strip()
    digest = hashlib.sha1(normalised.encode("utf-8")).hexdigest()[:12]
    return f"{doc_id}:{section_id}:{digest}"
