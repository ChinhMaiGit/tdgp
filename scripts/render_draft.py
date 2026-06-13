#!/usr/bin/env python3
"""
Convert paper draft markdown files to LaTeX and compile.

Default behaviour: reads all .md and .txt files from drafts/final/,
generates drafts/render/{main.tex, sections/*.tex, references.tex},
runs pdflatex twice, reports page count.

Usage:
    python scripts/render_draft.py                 # render drafts/final/ only
    python scripts/render_draft.py --include-wip   # also include drafts/Section *.md not in final/
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINAL_DIR = PROJECT_ROOT / "drafts" / "final"
DRAFTS_DIR = PROJECT_ROOT / "drafts"
RENDER_DIR = PROJECT_ROOT / "drafts" / "render"
SECTIONS_DIR = RENDER_DIR / "sections"
FIGURES_SRC_DIR = PROJECT_ROOT / "drafts" / "figures"
FIGURES_OUT_DIR = RENDER_DIR / "figures"

# Environment used for bullet ("-"/"*") and indented-paragraph lists.
# Flipped to "enumerate" by --enumerate-lists so all lists render numbered.
LIST_ENV = "itemize"


# ---------------------------------------------------------------------------
# Markdown -> LaTeX conversion
# ---------------------------------------------------------------------------

def _strip_references_block(text: str) -> tuple[str, list[str]]:
    """Strip a trailing 'References' or '## References' block from a draft.

    Returns (body_without_refs, list_of_reference_lines).
    """
    lines = text.splitlines()
    ref_start = None
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        if stripped in {"references", "## references", "### references"}:
            ref_start = i
            break
    if ref_start is None:
        return text, []
    body = "\n".join(lines[:ref_start]).rstrip()
    refs = [ln for ln in lines[ref_start + 1:] if ln.strip()]
    return body, refs


def _convert_inline(text: str) -> str:
    """Apply inline conversions that are safe outside math/code blocks."""
    text = text.replace("—", "---").replace("–", "--")
    text = text.replace("≠", r"$\neq$")
    text = text.replace("≈", r"$\approx$")
    text = text.replace("≥", r"$\geq$")
    text = text.replace("≤", r"$\leq$")
    text = text.replace("→", r"$\rightarrow$")
    text = text.replace("∈", r"$\in$")
    text = text.replace("∉", r"$\notin$")
    text = text.replace("×", r"$\times$")
    text = text.replace("²", "$^2$")
    text = text.replace("Δ", r"$\Delta$")
    text = text.replace("σ", r"$\sigma$")
    text = text.replace("α", r"$\alpha$")
    text = text.replace("β", r"$\beta$")

    # Bold **text** -> \textbf{text}
    text = re.sub(r"\*\*([^*]+?)\*\*", r"\\textbf{\1}", text)
    # Italic *text* -> \textit{text}  (avoid touching already-converted **)
    text = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"\\textit{\1}", text)
    # Inline code `x` -> \code{x} (monospace + gray background, like Markdown).
    # Escape _ and ^ inside so they don't trigger math-mode errors.
    # The lookarounds skip double backticks: by this point straight quotes
    # have already become LaTeX ``...'' (in _escape_special_outside_math),
    # and without the guards two quoted strings on one line would be misread
    # as a code span between them.
    def _bt(m: re.Match) -> str:
        inner = m.group(1).replace("_", r"\_").replace("^", r"\^{}")
        return r"\code{" + inner + r"}"
    text = re.sub(r"(?<!`)`([^`]+?)`(?!`)", _bt, text)

    return text


def _escape_special_outside_math(line: str) -> str:
    """Escape % # & outside math segments. Leaves $...$ blocks intact."""
    parts = re.split(r"(\$[^$]*\$)", line)
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            out.append(part)
            continue
        # Straight double quotes become LaTeX curly quotes (``...'').
        # LaTeX renders a bare " as a closing quote, so "word" would get two
        # right-facing quotes; pairs are converted, an unpaired " is left alone.
        part = re.sub(r'"([^"]*)"', r"``\1''", part)
        # Outside math: escape special chars (but not _ to avoid messing up LaTeX commands)
        part = re.sub(r"(?<!\\)%", r"\\%", part)
        part = re.sub(r"(?<!\\)#", r"\\#", part)
        part = re.sub(r"(?<!\\)&", r"\\&", part)
        out.append(part)
    return "".join(out)


def _convert_display_math(block: str) -> str:
    r"""Convert $$...$$ block to \[...\] — or pass through if it contains its own math env."""
    inner = block.strip().strip("$").strip()
    # If the block contains its own equation environment (align*, equation, gather, etc.),
    # emit it directly — wrapping in \[...\] would be a fatal nesting error.
    # Only top-level display environments — NOT cases/aligned/array, which are
    # sub-environments that need to live inside \[...\] or another math env.
    env_pattern = re.compile(
        r"\\begin\{(align\*?|equation\*?|gather\*?|multline\*?|eqnarray\*?)\}"
    )
    if env_pattern.search(inner):
        return inner
    return "\\[\n" + inner + "\n\\]"


def _convert_table(rows: list[str]) -> str:
    """Convert a markdown pipe-table to a booktabs LaTeX longtable.

    Every table is a longtable so it can break across pages. Columns whose
    maximum cell length exceeds 35 characters get a paragraph column spec
    (p{<width>}) so the table doesn't overflow the line; otherwise short
    numeric tables use l/r columns. longtable centres itself by default.
    """
    parsed: list[list[str]] = []
    for row in rows:
        row = row.strip()
        if not row.startswith("|"):
            continue
        # Skip the alignment row (|---|---|...)
        if re.fullmatch(r"\|[\s\-:|]+\|", row):
            continue
        cells = [c.strip() for c in row.strip("|").split("|")]
        parsed.append(cells)
    if not parsed:
        return ""
    ncols = len(parsed[0])
    # Pad short rows up front so column-width analysis is correct
    for row in parsed:
        while len(row) < ncols:
            row.append("")

    # Compute column max lengths over ALL rows, including the header, so a
    # column is never narrower than its own title (e.g. "Observed" over short
    # numeric cells). Group-header rows (cols 2..n empty) are skipped so they
    # don't inflate the first column.
    max_lens = [0] * ncols
    for row in parsed:
        if all(c.strip() == "" for c in row[1:]):
            continue
        for j, cell in enumerate(row):
            max_lens[j] = max(max_lens[j], len(cell))

    LONG_THRESHOLD = 35
    long_cols = [j for j, ml in enumerate(max_lens) if ml > LONG_THRESHOLD]
    has_wide = bool(long_cols)
    n_data_rows = sum(
        1 for row in parsed[1:] if not all(c.strip() == "" for c in row[1:])
    )

    # Every table is emitted as a longtable so it can break across pages.
    small_font = False
    col_spec = ""  # longtable column spec

    if ncols >= 5 and has_wide:
        # Wide multi-column table: weight columns by content length to fill
        # \linewidth, footnotesize so it fits (e.g. Table 4.1).
        total = sum(max_lens) or 1
        # Floor keeps a short column wide enough for its own header text.
        raw = [max(ncols * ml / total, 0.75) for ml in max_lens]
        s = sum(raw) or 1
        weights = [round(w * ncols / s, 2) for w in raw]
        weights[-1] = round(ncols - sum(weights[:-1]), 2)
        col_spec = "".join(
            r">{\raggedright\arraybackslash}p{\dimexpr "
            + f"{round(w / ncols, 4)}" + r"\linewidth-2\tabcolsep\relax}"
            for w in weights
        )
        small_font = True
    elif has_wide:
        # Few columns, some carrying long text: paragraph columns for those.
        nlong = len(long_cols)
        per_col = round(0.7 / nlong, 2)
        col_spec = "".join(
            f"p{{{per_col}\\linewidth}}" if j in long_cols else "l"
            for j in range(ncols)
        )
    else:
        # Short numeric results table: left label + right-aligned numbers.
        col_spec = "l" + "r" * (ncols - 1)
        # A table whose summed cell content approaches the line width still
        # overflows at full size (e.g. a 7-column results table); drop to
        # footnotesize so the columns fit.
        if sum(max_lens) + 2 * ncols > 75:
            small_font = True

    def _cell(c: str) -> str:
        """Convert a table cell with two auto-formatting passes:
        1. Explicit <br> tags become LaTeX line breaks.
        2. 'N,NNN (X.X%)' patterns split count and percentage onto separate lines.
        """
        # Normalise explicit line-break tags first
        c = c.replace("<br>", "<BR>").replace("<BR>", "<BR>")
        # Put the percentage on its own line below the raw count
        c = re.sub(r"(\d[\d,]*)\s+(\(\d+\.?\d*%\))", r"\1<BR>\2", c)
        c = _convert_inline(_escape_special_outside_math(c))
        c = c.replace("<BR>", r"\newline ")
        return c

    header = parsed[0]
    header_line = " & ".join(_cell(c) for c in header) + r" \\"

    out: list[str] = []
    # All tables are longtables so they flow across pages; a longtable centres
    # itself by default (\LTleft = \LTright = \fill) and its header repeats on
    # every page it spans.
    out.append(r"{")
    if small_font:
        out.append(r"\footnotesize")
        out.append(r"\setlength{\tabcolsep}{5pt}")
    else:
        out.append(r"\setlength{\tabcolsep}{7pt}")
    out.append(r"\renewcommand{\arraystretch}{1.15}")
    out.append(r"\begin{longtable}{" + col_spec + "}")
    out.append(r"\toprule")
    out.append(header_line)
    out.append(r"\midrule")
    out.append(r"\endfirsthead")
    out.append(r"\toprule")
    out.append(header_line)
    out.append(r"\midrule")
    out.append(r"\endhead")
    out.append(r"\bottomrule")
    out.append(r"\endlastfoot")

    first_data_row = True
    for row in parsed[1:]:
        # Group header row: all cells after the first are empty.
        # Emit as a full-width multicolumn with a midrule above (between groups)
        # and a thin specialrule below to visually box each section.
        if all(c.strip() == "" for c in row[1:]):
            content = _cell(row[0])
            if not first_data_row:
                # Full-width rule between groups — acts as the box bottom of the
                # previous group and the box top of the new one.
                out.append(r"\midrule")
            out.append(
                r"\multicolumn{" + str(ncols) + r"}{l}{"
                + content
                + r"} \\[-2pt]"
            )
            # Thin rule below the header, separating it from its data rows.
            out.append(r"\specialrule{0.3pt}{3pt}{5pt}")
            # Keep a group header with at least its first data row on the page.
            out.append(r"\nopagebreak")
            continue
        out.append(" & ".join(_cell(c) for c in row) + r" \\")
        first_data_row = False

    out.append(r"\end{longtable}")
    out.append(r"}")
    return "\n".join(out)


def _img_width(alt: str) -> str:
    """Return the \\linewidth fraction for a figure, read from an optional
    width hint embedded in the image alt text: ``![Figure 4.2|0.66](path)``
    or ``![alt|width=0.66](path)``. Defaults to 0.95 when no hint is given."""
    m = re.search(r"\|\s*(?:width\s*=\s*)?([0-9]*\.?[0-9]+)", alt or "")
    return m.group(1) if m else "0.95"


def _img_placement(alt: str) -> str:
    """Return the LaTeX float placement specifier from an optional hint in
    the image alt text: ``![alt|0.98|H](path)`` yields ``H`` (exact placement
    via the float package); ``![alt|0.8|tbp](path)`` yields ``tbp``.
    Defaults to ``tbp`` when no hint is present.

    The hint must appear as a separate ``|``-delimited token that matches
    ``[HhTtBbPp!]+`` so it is distinguished from numeric width hints."""
    for token in (alt or "").split("|"):
        t = token.strip()
        if re.fullmatch(r"[HhTtBbPp!]+", t) and not re.fullmatch(r"[0-9]*\.?[0-9]+", t):
            return t
    return "tbp"


FIGURE_CAPTION_RE = re.compile(r"\*\s*Figure\s+(\d+)[\.\s_-]+(\d+)[\.\s]", re.IGNORECASE)

# Staged-figure provenance: dst basename -> first source path. Sources are
# scattered across analysis folders (data/, experiments/, inference/, ...) but
# staged flat into drafts/render/figures/, so two distinct files sharing a
# basename would silently overwrite each other (this happened once with
# fig_deployment.png, putting the wrong image in the PDF).
_STAGED_FIGURES: dict[str, Path] = {}


def _stage_figure(img_src: Path, dst: Path) -> None:
    """Copy a referenced figure into the render figures dir, warning loudly
    when a different source file already claimed the same basename."""
    prior = _STAGED_FIGURES.get(dst.name)
    if prior is not None and prior != img_src:
        print(
            f"[figures] WARNING: basename collision on {dst.name}: "
            f"{img_src} overwrites {prior}. Rename one of the sources.",
            file=sys.stderr,
        )
    _STAGED_FIGURES[dst.name] = img_src
    if img_src.exists():
        shutil.copy2(str(img_src), str(dst))


def _find_figure_asset(major: str, minor: str) -> Path | None:
    """Return path (relative to RENDER_DIR) to the figure asset for Figure
    major.minor, or None. Prefers raster/vector formats LaTeX can use directly:
    .pdf > .png > .jpg > .jpeg."""
    stem = f"figure{major}_{minor}"
    for ext in (".pdf", ".png", ".jpg", ".jpeg"):
        target = FIGURES_OUT_DIR / (stem + ext)
        if target.exists():
            return target.relative_to(RENDER_DIR)
    return None


def _emit_figure_or_placeholder(upcoming_lines: list[str], caption_line: str | None = None) -> str:
    """Emit a floating figure for a mermaid diagram.

    Looks ahead at the lines after the mermaid block for a 'Figure X.Y ...'
    caption to locate the image asset. The image is wrapped in a ``figure``
    float so it does not orphan a large blank gap when it cannot fit the
    current page; the caption (if supplied) is placed inside the float so it
    stays attached to the image."""
    content = None
    for ln in upcoming_lines[:5]:
        m = FIGURE_CAPTION_RE.search(ln)
        if m:
            asset_rel = _find_figure_asset(m.group(1), m.group(2))
            if asset_rel:
                content = (
                    r"\includegraphics[width=0.425\linewidth]{"
                    + str(asset_rel).replace("\\", "/") + r"}"
                )
            break
    if content is None:
        content = (
            r"\fbox{\parbox{0.85\linewidth}{\small \emph{[Figure placeholder "
            r"--- source not available for render]}}}"
        )
    parts = [r"\begin{figure}[tbp]", r"\centering", content]
    if caption_line:
        cap = caption_line.strip()
        if len(cap) > 1 and cap.startswith("*") and cap.endswith("*"):
            cap = cap[1:-1].strip()
        cap = _convert_inline(_escape_special_outside_math(cap))
        parts.append(r"\caption*{" + cap + r"}")
    parts.append(r"\end{figure}")
    return "\n".join(parts)


def convert_markdown_to_latex(text: str, registry: set[str] | None = None) -> str:
    """Convert a markdown draft body to LaTeX."""
    text, _ = _strip_references_block(text)
    # Apply citation linking on the raw markdown body so it isn't disturbed by
    # later LaTeX escaping (it only inserts \cit{...}{...} commands, whose
    # internals will be escaped/inline-converted normally).
    if registry:
        text = link_citations(text, registry)
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    in_code_fence = False
    code_fence_lang: str | None = None
    while i < len(lines):
        line = lines[i]
        # Code fences (including mermaid)
        m = re.match(r"^```(\w*)\s*$", line)
        if m:
            lang = m.group(1).lower() or None
            if not in_code_fence and lang == "mermaid":
                # Find the closing ```; emit a floating figure, pulling in the
                # following 'Figure X.Y ...' caption if present, then skip past
                # the whole block (and the consumed caption line).
                close_idx = i + 1
                while close_idx < len(lines) and not lines[close_idx].strip().startswith("```"):
                    close_idx += 1
                after = lines[close_idx + 1: close_idx + 6]
                caption_line = None
                skip_to = close_idx
                for off, ln in enumerate(after):
                    if FIGURE_CAPTION_RE.search(ln):
                        caption_line = ln
                        skip_to = close_idx + 1 + off
                        break
                    if ln.strip():
                        break
                out.append(_emit_figure_or_placeholder(after, caption_line))
                i = skip_to + 1
                continue
            # Generic (non-mermaid) code fence: toggle state and skip the marker
            in_code_fence = not in_code_fence
            code_fence_lang = lang if in_code_fence else None
            i += 1
            continue
        if in_code_fence:
            # Skip the body of code fences (incl. mermaid)
            i += 1
            continue

        # Display math block $$ ... $$  (may span multiple lines)
        if line.strip().startswith("$$"):
            block_lines = [line]
            # If $$ doesn't close on same line, scan until closing $$
            if line.count("$$") < 2:
                i += 1
                while i < len(lines) and "$$" not in lines[i]:
                    block_lines.append(lines[i])
                    i += 1
                if i < len(lines):
                    block_lines.append(lines[i])
            block = "\n".join(block_lines)
            out.append(_convert_display_math(block))
            i += 1
            continue

        # Bold figure caption immediately followed by a markdown image:
        #   **Figure N.M — caption text**
        #   ![alt](path)
        # Consumed together as a figure float with the caption inside.
        if re.match(r"^\*\*Figure\s+[\d.]+", line.strip()):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                img_m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", lines[j])
                if img_m:
                    cap_text = re.sub(r"^\*\*(.+)\*\*\s*$", r"\1", line.strip())
                    img_path_str = img_m.group(2)
                    width = _img_width(img_m.group(1))
                    img_src = (DRAFTS_DIR / img_path_str).resolve()
                    FIGURES_OUT_DIR.mkdir(parents=True, exist_ok=True)
                    dst = FIGURES_OUT_DIR / img_src.name
                    _stage_figure(img_src, dst)
                    rel = str(dst.relative_to(RENDER_DIR)).replace("\\", "/")
                    cap_latex = _convert_inline(_escape_special_outside_math(cap_text))
                    placement = _img_placement(img_m.group(1))
                    out.append(r"\begin{figure}[" + placement + r"]")
                    out.append(r"\centering")
                    out.append(r"\includegraphics[width=" + width + r"\linewidth]{" + rel + r"}")
                    out.append(r"\caption*{" + cap_latex + r"}")
                    out.append(r"\end{figure}")
                    i = j + 1
                    continue

        # Standalone markdown image not preceded by a bold caption:
        #   ![alt](path)
        img_m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", line)
        if img_m:
            img_path_str = img_m.group(2)
            width = _img_width(img_m.group(1))
            img_src = (DRAFTS_DIR / img_path_str).resolve()
            FIGURES_OUT_DIR.mkdir(parents=True, exist_ok=True)
            dst = FIGURES_OUT_DIR / img_src.name
            _stage_figure(img_src, dst)
            rel = str(dst.relative_to(RENDER_DIR)).replace("\\", "/")
            placement = _img_placement(img_m.group(1))
            out.append(r"\begin{figure}[" + placement + r"]")
            out.append(r"\centering")
            out.append(r"\includegraphics[width=" + width + r"\linewidth]{" + rel + r"}")
            out.append(r"\end{figure}")
            i += 1
            continue

        # Horizontal rules (--- / *** / ___) are visual separators in the
        # markdown drafts; dropped, since in LaTeX they would typeset as a
        # stray em-dash line.
        if re.fullmatch(r"\s*(-{3,}|\*{3,}|_{3,})\s*", line):
            i += 1
            continue

        # Headings. \FloatBarrier precedes each top-level section heading so
        # figures stay within their own section. Subsection headings carry no
        # barrier: a flushed float that cannot share a page with text produces
        # a figure-only page with large blank space, whereas letting figures
        # drift past a subsection boundary keeps every page filled (captions
        # carry the figure numbers, so placement stays unambiguous).
        if line.startswith("#### "):
            title = _convert_inline(line[5:].strip())
            title = re.sub(r"^\d+(\.\d+)*\s+", "", title)
            out.append(f"\\subsubsection*{{{title}}}")
            i += 1
            continue
        if line.startswith("### "):
            title = _convert_inline(line[4:].strip())
            # Drop leading section number if present (e.g., "2.2 ")
            title = re.sub(r"^\d+(\.\d+)*\s+", "", title)
            out.append(f"\\subsection{{{title}}}")
            i += 1
            continue
        if line.startswith("## "):
            title = _convert_inline(line[3:].strip())
            title = re.sub(r"^\d+(\.\d+)*\.?\s+", "", title)
            # \FloatBarrier before each major section flushes pending floats
            # without forcing a page break, so sections flow naturally onto
            # the same page rather than leaving section-end whitespace gaps.
            # Key figures with cross-section drift risk use [H] placement in
            # the source markdown to anchor them in-place.
            out.append(r"\FloatBarrier")
            out.append(f"\\section{{{title}}}")
            i += 1
            continue
        if line.startswith("# "):
            title = _convert_inline(line[2:].strip())
            out.append(r"\FloatBarrier")
            out.append(f"\\section{{{title}}}")
            i += 1
            continue

        # Bold table caption immediately followed by a markdown pipe table:
        #   **Table N.M — caption text**
        #   | ... |
        # Caption sits directly above the table as a paragraph (\nopagebreak
        # keeps it with the first rows); the table itself is a longtable that
        # flows across pages, so no float or minipage is used.
        if re.match(r"^\*\*Table\s+[\d.]+", line.strip()):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if (
                j + 1 < len(lines)
                and lines[j].strip().startswith("|")
                and re.fullmatch(r"\|[\s\-:|]+\|", lines[j + 1].strip())
            ):
                cap_latex = _convert_inline(_escape_special_outside_math(line.strip()))
                k = j
                rows = []
                while k < len(lines) and lines[k].strip().startswith("|"):
                    rows.append(lines[k])
                    k += 1
                table_latex = _convert_table(rows)
                # Caption directly above the table; the table itself is a
                # longtable that breaks across pages as needed. \nopagebreak
                # keeps the caption with the table's first rows.
                out.append(r"\par\medskip\noindent")
                # Match figure captions: small italic (bold label from **...**).
                out.append(r"{\small\itshape " + cap_latex + r"\par}")
                out.append(r"\nopagebreak\medskip\nopagebreak")
                out.append(table_latex)
                out.append(r"\par\medskip")
                i = k
                continue

        # Tables
        if line.strip().startswith("|") and i + 1 < len(lines) and re.fullmatch(r"\|[\s\-:|]+\|", lines[i + 1].strip()):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i])
                i += 1
            out.append(_convert_table(rows))
            continue

        # Numbered lists (1. 2. 3. ...)  ->  enumerate. Blank lines between
        # items do not end the list (otherwise each item restarts at 1.).
        if re.match(r"^\s*\d+\.\s+", line):
            out.append(r"\begin{enumerate}")
            while i < len(lines):
                if re.match(r"^\s*\d+\.\s+", lines[i]):
                    item = re.sub(r"^\s*\d+\.\s+", "", lines[i])
                    out.append(r"  \item " + _convert_inline(_escape_special_outside_math(item)))
                    i += 1
                elif not lines[i].strip():
                    j = i
                    while j < len(lines) and not lines[j].strip():
                        j += 1
                    if j < len(lines) and re.match(r"^\s*\d+\.\s+", lines[j]):
                        i = j
                    else:
                        break
                else:
                    break
            out.append(r"\end{enumerate}")
            continue

        # Bullet lists
        if re.match(r"^\s*[-*]\s+", line):
            out.append(r"\begin{" + LIST_ENV + "}")
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i]):
                item = re.sub(r"^\s*[-*]\s+", "", lines[i])
                out.append(r"  \item " + _convert_inline(_escape_special_outside_math(item)))
                i += 1
            out.append(r"\end{" + LIST_ENV + "}")
            continue

        # Indented-paragraph list: text indented with 2+ spaces, where the previous
        # non-blank line ends with ":". This handles drafts where bullets were
        # written as indented paragraphs (e.g., the Contributions section listing
        # the three conditions or the three TDGP stages).
        if (
            line.strip()
            and (line.startswith("  ") or line.startswith("\t"))
        ):
            prev_nonblank = None
            for k in range(i - 1, -1, -1):
                if lines[k].strip():
                    prev_nonblank = lines[k]
                    break
            if prev_nonblank is not None and prev_nonblank.rstrip().endswith(":"):
                out.append(r"\begin{" + LIST_ENV + "}")
                while i < len(lines):
                    # Skip blank lines between items
                    while i < len(lines) and not lines[i].strip():
                        i += 1
                    if i >= len(lines):
                        break
                    # End of list when we hit an unindented non-empty line
                    if not (lines[i].startswith("  ") or lines[i].startswith("\t")):
                        break
                    # Collect the item — a single paragraph that may wrap across
                    # multiple indented lines without a blank in between.
                    item_lines = [lines[i].lstrip()]
                    i += 1
                    while (
                        i < len(lines)
                        and lines[i].strip()
                        and (lines[i].startswith(" ") or lines[i].startswith("\t"))
                    ):
                        item_lines.append(lines[i].lstrip())
                        i += 1
                    item_text = " ".join(item_lines)
                    out.append(r"  \item " + _convert_inline(_escape_special_outside_math(item_text)))
                out.append(r"\end{" + LIST_ENV + "}")
                continue

        # Plain text (paragraph). Just convert inline and pass through.
        converted = _convert_inline(_escape_special_outside_math(line))
        out.append(converted)
        i += 1

    return "\n".join(out)


# ---------------------------------------------------------------------------
# File assembly
# ---------------------------------------------------------------------------

PREAMBLE = r"""\documentclass[11pt,a4paper]{article}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{geometry}
\geometry{margin=1in}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{xcolor}
\definecolor{linknavy}{RGB}{0, 70, 130}
\usepackage{hyperref}
\hypersetup{colorlinks=true, linkcolor=linknavy, citecolor=linknavy, urlcolor=linknavy, breaklinks=true,
  pdftitle={Causal Inference for Compliance-Gated Administrative Data: A Tested Data-Generating Process Approach with Application to Municipal Energy Benchmarking},
  pdfauthor={Chinh Mai}}
\usepackage{enumitem}
\usepackage{needspace}
\usepackage{placeins}
\usepackage{float}
\usepackage{longtable}
% Publication-quality figure captions: small text, 10 pt vertical skip.
% \caption*{} (starred) suppresses the automatic "Figure X." prefix so the
% manual "\textbf{Figure X.Y.} ..." embedded in each caption drives the label.
% singlelinecheck=false prevents single-line captions from being centred.
\usepackage[font=small,skip=10pt,justification=justified,
            singlelinecheck=false]{caption}

% Classic journal paragraph style: indented first lines, no inter-paragraph
% gap. (The draft renders previously used parskip's block paragraphs.)
\setlength{\parindent}{15pt}
\setlength{\parskip}{0pt plus 1pt}

% Keep paragraph-opening and -closing lines attached to their paragraph
% across page breaks (no widows/orphans in a submission-ready manuscript).
\widowpenalty=10000
\clubpenalty=10000

% Float placement: topfraction=0.72 means one wide figure + caption can sit
% at the top of a page (most figures are 200--400 pt scaled) but two cannot
% stack and cause overfull vboxes. floatpagefraction=0.65 is low enough that
% medium-tall figures (shap_combined ~514 pt, type_intercepts ~491 pt) reach
% a dedicated float page rather than being deferred to the document end.
% A \FloatBarrier is emitted at each \section{} boundary (not subsections) so
% figures stay inside their own section without creating mid-section gaps.
\renewcommand{\topfraction}{0.72}
\renewcommand{\bottomfraction}{0.60}
\renewcommand{\textfraction}{0.12}
\renewcommand{\floatpagefraction}{0.65}
\setcounter{topnumber}{2}
\setcounter{bottomnumber}{1}
\setcounter{totalnumber}{3}

% Top-align dedicated float pages: pack figures from the top and let any
% leftover space collect at the bottom, instead of spreading gaps between them.
\makeatletter
\setlength{\@fptop}{0pt}
\setlength{\@fpsep}{14pt plus 0fil}
\setlength{\@fpbot}{0pt plus 1fil}
\makeatother

\definecolor{citecolor}{RGB}{0, 90, 156}
\newcommand{\cit}[2]{\hyperlink{#1}{\textcolor{citecolor}{#2}}}

% Inline code: monospace with light gray background, mimicking Markdown rendering.
\newcommand{\code}[1]{{\setlength{\fboxsep}{2pt}\colorbox{gray!12}{\strut\texttt{#1}}}}

\newtheorem{definition}{Definition}[section]
\newtheorem{proposition}{Proposition}[section]

\title{Causal Inference for Compliance-Gated Administrative Data:\\
A Tested Data-Generating Process Approach with Application to Municipal Energy Benchmarking}
\author{Chinh Mai\\[2pt]
{\small\texttt{chinhmai.work@gmail.com}}}
\date{June 2026}

\begin{document}
\maketitle

% Loose typesetting for draft renders -- allow longer inter-word spaces to
% avoid overfull boxes on long math expressions and URLs.
\sloppy
"""

POSTAMBLE = r"""
\clearpage
\section*{References}
\input{sections/references}

\end{document}
"""

# ---------------------------------------------------------------------------
# Vietnamese variant preamble/postamble
# ---------------------------------------------------------------------------
VI_PREAMBLE = """\
\\documentclass[11pt,a4paper]{article}

\\usepackage[utf8]{inputenc}
\\usepackage[T5]{fontenc}
\\usepackage[vietnamese]{babel}
\\usepackage{lmodern}
\\usepackage{amsmath, amssymb, amsthm}
\\usepackage{booktabs}
\\usepackage{tabularx}
\\usepackage{geometry}
\\geometry{margin=1in}
\\usepackage{microtype}
\\usepackage{graphicx}
\\usepackage{xcolor}
\\definecolor{linknavy}{RGB}{0, 70, 130}
\\usepackage{hyperref}
\\hypersetup{colorlinks=true, linkcolor=linknavy, citecolor=linknavy, urlcolor=linknavy, breaklinks=true,
  pdftitle={Suy Diễn Nhân Quả cho Dữ Liệu Hành Chính Có Cổng Tuân Thủ},
  pdfauthor={Chinh Mai}}
\\usepackage{enumitem}
\\usepackage{needspace}
\\usepackage{placeins}
\\usepackage{float}
\\usepackage{longtable}
\\usepackage[font=small,skip=10pt,justification=justified,
            singlelinecheck=false]{caption}

\\setlength{\\parindent}{15pt}
\\setlength{\\parskip}{0pt plus 1pt}

\\widowpenalty=10000
\\clubpenalty=10000

\\renewcommand{\\topfraction}{0.72}
\\renewcommand{\\bottomfraction}{0.60}
\\renewcommand{\\textfraction}{0.12}
\\renewcommand{\\floatpagefraction}{0.65}
\\setcounter{topnumber}{2}
\\setcounter{bottomnumber}{1}
\\setcounter{totalnumber}{3}

\\makeatletter
\\setlength{\\@fptop}{0pt}
\\setlength{\\@fpsep}{14pt plus 0fil}
\\setlength{\\@fpbot}{0pt plus 1fil}
\\makeatother

\\definecolor{citecolor}{RGB}{0, 90, 156}
\\newcommand{\\cit}[2]{\\hyperlink{#1}{\\textcolor{citecolor}{#2}}}
\\newcommand{\\code}[1]{{\\setlength{\\fboxsep}{2pt}\\colorbox{gray!12}{\\strut\\texttt{#1}}}}

\\newtheorem{definition}{Định nghĩa}[section]
\\newtheorem{proposition}{Mệnh đề}[section]

\\title{Suy Diễn Nhân Quả cho Dữ Liệu Hành Chính Có Cổng Tuân Thủ:\\\\
Phương pháp Quy Trình Tạo Dữ Liệu Được Kiểm tra\\\\
với Ứng dụng trong Đánh giá Chuẩn Năng lượng Đô thị}
\\author{Chinh Mai\\\\[2pt]
{\\small\\texttt{chinhmai.work@gmail.com}}}
\\date{Tháng 6, 2026}

\\begin{document}
\\maketitle

\\sloppy
"""

VI_POSTAMBLE = r"""
\clearpage
\section*{Tài liệu tham khảo}
\input{sections/references}

\end{document}
"""


# Compact layout for length-constrained renders: 10pt, tighter margins,
# indented paragraphs (no inter-paragraph skip), and tight lists/section
# spacing. Same packages and macros as PREAMBLE so all content renders
# identically -- only the density changes.
DENSE_PREAMBLE = r"""\documentclass[10pt,a4paper]{article}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{geometry}
\geometry{a4paper, top=1.8cm, bottom=1.8cm, left=1.9cm, right=1.9cm}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{hyperref}
\hypersetup{colorlinks=true, linkcolor=blue, citecolor=blue, urlcolor=blue, breaklinks=true}
\usepackage{enumitem}
\setlist{nosep, topsep=2pt, leftmargin=1.4em}
\usepackage{xcolor}
\usepackage{needspace}
\usepackage{placeins}
\usepackage{float}
\usepackage{longtable}
\usepackage[font=small,skip=8pt,justification=justified,singlelinecheck=false]{caption}

\setlength{\parindent}{1.1em}
\setlength{\parskip}{1.5pt plus 0.5pt}
\linespread{0.98}

\usepackage{titlesec}
\titlespacing*{\section}{0pt}{1.4ex plus 0.3ex minus 0.2ex}{0.8ex plus 0.2ex}
\titlespacing*{\subsection}{0pt}{1.1ex plus 0.3ex minus 0.2ex}{0.5ex plus 0.2ex}

% Float placement: let figures fill more of a page and avoid near-empty
% dedicated float pages, so a figure shares its page with surrounding text.
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.8}
\renewcommand{\textfraction}{0.06}
\renewcommand{\floatpagefraction}{0.8}
\setcounter{topnumber}{2}
\setcounter{totalnumber}{4}

% Top-align dedicated float pages: pack figures from the top and let any
% leftover space collect at the bottom, instead of spreading gaps between them.
\makeatletter
\setlength{\@fptop}{0pt}
\setlength{\@fpsep}{12pt plus 0fil}
\setlength{\@fpbot}{0pt plus 1fil}
\makeatother

\definecolor{citecolor}{RGB}{0, 90, 156}
\newcommand{\cit}[2]{\hyperlink{#1}{\textcolor{citecolor}{#2}}}

% Inline code: monospace with light gray background, mimicking Markdown rendering.
\newcommand{\code}[1]{{\setlength{\fboxsep}{2pt}\colorbox{gray!12}{\strut\texttt{#1}}}}

\newtheorem{definition}{Definition}[section]
\newtheorem{proposition}{Proposition}[section]

\title{\vspace{-1.2cm}Causal Inference for Compliance-Gated Administrative Data:\\
A Tested Data-Generating Process Approach with Application to Municipal Energy Benchmarking}
\author{Draft (current progress)}
\date{\today}

\begin{document}
\maketitle
\vspace{-0.6cm}
\sloppy
"""


@dataclass
class DraftFile:
    path: Path
    order: int
    label: str  # section group: "intro", "section2", "section3", ...
    sub_order: int  # ordering within group

    def read(self) -> str:
        return self.path.read_text(encoding="utf-8")


def discover_drafts(include_wip: bool = False) -> list[DraftFile]:
    """Discover draft files from drafts/final/ (and optionally drafts/)."""
    files: list[DraftFile] = []
    candidates = list(FINAL_DIR.glob("*.md")) + list(FINAL_DIR.glob("*.txt"))
    if include_wip:
        # Add Section *.md files from drafts/ that aren't superseded by something in final/
        seen_roots: set[str] = set()
        for p in DRAFTS_DIR.glob("Section *.md"):
            stem_root = re.sub(r"\s*v\d+\s*$", "", p.stem).strip()
            # Process each section root once, even if several versions exist in drafts/.
            if stem_root in seen_roots:
                continue
            seen_roots.add(stem_root)
            # Take the highest-version file per section root that has no final/ counterpart
            if not any(stem_root in f.stem for f in (FINAL_DIR.glob("*.md"))):
                # Pick the highest v-number we have in drafts/
                same = sorted(
                    [q for q in DRAFTS_DIR.glob(f"{stem_root}*v*.md")],
                    key=lambda q: int(re.search(r"v(\d+)", q.stem).group(1)),
                    reverse=True,
                )
                if same:
                    candidates.append(same[0])

    for path in candidates:
        name = path.stem.lower()
        if name.startswith("abstract"):
            # The abstract is rendered separately into the title page
            # (render_abstract), not as a body section.
            continue
        if name.startswith("introduction"):
            m = re.search(r"-\s*(\d+)", path.stem)
            sub = int(m.group(1)) if m else 99
            files.append(DraftFile(path=path, order=1, label="intro", sub_order=sub))
        elif re.match(r"section\s+\d+", name):
            # Handles any section number: Section 2, 3, 4, 5, ...
            # "Section 4 - empirical demonstration" => 4.0
            # "Section 2.2 - selection bias"        => 2.2
            m = re.match(r"section\s+(\d+)(?:\.(\d+))?", name)
            if m:
                major = int(m.group(1))
                minor = int(m.group(2)) if m.group(2) else 0
                files.append(DraftFile(path=path, order=major, label=f"section{major}", sub_order=minor))
        else:
            # Catch-all
            files.append(DraftFile(path=path, order=99, label="misc", sub_order=0))

    files.sort(key=lambda f: (f.order, f.sub_order))
    return files


def render_abstract(registry: set[str] | None = None) -> str:
    """Render drafts/final/Abstract*.md into an abstract environment for the
    title page. Returns "" when no abstract draft exists.

    A trailing line starting with "Keywords:" is set below the abstract body
    in the journal style (bold label, italic keyword list)."""
    candidates = sorted(
        list(FINAL_DIR.glob("Abstract*.md")) + list(FINAL_DIR.glob("Abstract*.txt"))
    )
    if not candidates:
        return ""
    body, _ = _strip_references_block(candidates[-1].read_text(encoding="utf-8"))
    if registry:
        body = link_citations(body, registry)

    paragraphs: list[str] = []
    keywords: str | None = None
    for raw in body.split("\n\n"):
        chunk = " ".join(ln.strip() for ln in raw.splitlines() if ln.strip())
        if not chunk:
            continue
        m = re.match(r"^\**(?:Keywords|Từ\s+khóa)\s*:?\**\s*(.+)$", chunk, re.IGNORECASE)
        if m:
            keywords = _convert_inline(_escape_special_outside_math(m.group(1)))
            continue
        paragraphs.append(_convert_inline(_escape_special_outside_math(chunk)))

    out = [r"\begin{abstract}", r"\noindent " + "\n\n".join(paragraphs)]
    if keywords:
        out.append(r"\par\bigskip\noindent\textbf{Keywords:} \textit{" + keywords + r"}")
    out.append(r"\end{abstract}")
    return "\n".join(out)


def render_intro(files: list[DraftFile], registry: set[str] | None = None, flat: bool = False) -> str:
    # The Introduction renders as one continuous section: the per-part headers
    # (The Problem / Literature Gap / Contributions / Paper Outline) are dropped,
    # since the problem -> gap -> contributions -> outline arc reads without
    # signposts and the headers only added vertical space. The `flat` parameter
    # is retained for call-site compatibility but the intro is always flat now.
    intro_files = [f for f in files if f.label == "intro"]
    if not intro_files:
        return ""
    out = [r"\section{Introduction}"]
    for f in intro_files:
        body = f.read()
        # Strip up to the first 3 short non-empty title-like lines at the top of the file
        # (e.g., "Introduction\nThe Problem\n\nMandatory...").
        body_lines = body.splitlines()
        stripped_count = 0
        while body_lines and stripped_count < 3:
            first = body_lines[0]
            # A title-like line is short, non-empty, no period, no math, no markdown markup
            if (
                first.strip()
                and len(first.strip()) < 50
                and "." not in first
                and "$" not in first
                and not first.startswith(("#", "-", "*", "|"))
            ):
                body_lines.pop(0)
                stripped_count += 1
                # Also drop a single blank line following the title
                if body_lines and not body_lines[0].strip():
                    body_lines.pop(0)
            else:
                break
        body = "\n".join(body_lines)
        out.append(convert_markdown_to_latex(body, registry=registry))
        out.append("")
    return "\n".join(out)


def render_section(files: list[DraftFile], section_num: int, registry: set[str] | None = None) -> str:
    sec_files = [f for f in files if f.label == f"section{section_num}"]
    if not sec_files:
        return ""
    out: list[str] = []
    for f in sec_files:
        body = f.read()
        out.append(convert_markdown_to_latex(body, registry=registry))
        out.append("")
    return "\n".join(out)


def collect_references(files: list[DraftFile]) -> list[str]:
    """Collect, deduplicate, and sort reference lines from all drafts.

    Deduplication is by the same citation key used in the body (``_ref_key``):
    single-author and multi-author references with the same first surname and
    year therefore stay distinct. When two reference lines share a key
    exactly (the same work, transcribed slightly differently across drafts),
    the longer entry wins so we keep the richer metadata.
    """
    refs: dict[str, str] = {}
    for f in files:
        _, file_refs = _strip_references_block(f.read())
        for line in file_refs:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Skip bracketed annotation lines ([Verified: ...], [TO VERIFY: ...], etc.)
            if line.startswith("["):
                continue
            key = _ref_key(line) or re.sub(r"\s+", " ", line)[:80].lower()
            existing = refs.get(key)
            if existing is None or len(line) > len(existing):
                refs[key] = line

    def _sort_key(line: str) -> str:
        # Alphabetise on letters/digits only, so punctuation does not distort
        # the order ("Pearl, J., & Mackenzie" would otherwise sort before
        # "Pearl, J., Glymour" because '&' < 'G' in ASCII). Whitespace is
        # collapsed so removed punctuation leaves no double-space artefacts.
        key = re.sub(r"[^a-z0-9 ]", "", _strip_accents(line).lower())
        return re.sub(r"\s+", " ", key)

    return sorted(refs.values(), key=_sort_key)


# ---------------------------------------------------------------------------
# Citation linking
# ---------------------------------------------------------------------------

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))


def _norm_surname(s: str) -> str:
    return _strip_accents(s).lower()


def _ref_key(line: str) -> str | None:
    """Derive a citation key from a reference line.

    Format: ``<surname1><surname2|etal><year>`` (lowercase, accents stripped).
    Examples:
      - ``Pearl, J. (2009). ...`` -> ``pearl2009``
      - ``Pearl, J., & Mackenzie, D. (2018). ...`` -> ``pearlmackenzie2018``
      - ``Pearl, J., Glymour, M., & Jewell, N. P. (2016). ...`` -> ``pearletal2016``
      - ``Hernán, M. A., & Robins, J. M. (2020). ...`` -> ``hernanrobins2020``
      - ``U.S. Securities and Exchange Commission. (n.d.). ...`` -> ``securitiesnd``
    """
    bare = re.sub(r"[*_`]", "", line).strip()
    y_match = re.search(r"\((\d{4}[a-z]?|n\.d\.[a-z]?)\)", bare)
    if not y_match:
        return None
    year = y_match.group(1).lower().replace(".", "")
    pre_year = bare[: y_match.start()]

    # Standard author format: "Surname, F." pattern
    surnames = re.findall(r"([A-Z][\w\-']+),\s+[A-Z]\.", pre_year)
    if surnames:
        first = _norm_surname(surnames[0])
        if len(surnames) >= 3:
            return f"{first}etal{year}"
        if len(surnames) == 2:
            return f"{first}{_norm_surname(surnames[1])}{year}"
        return f"{first}{year}"

    # Organisational author: use first capitalised word with 3+ letters
    for w in re.findall(r"[A-Z][\w]{2,}", pre_year):
        return f"{_norm_surname(w)}{year}"
    return None


def _cite_key(authors: str, year: str) -> str | None:
    """Derive a citation key from an in-line citation's author chunk and year."""
    year = year.lower().replace(".", "")
    has_etal = bool(re.search(r"\bet\s+al\.?", authors, re.IGNORECASE))
    cleaned = re.sub(r"\bet\s+al\.?", "", authors, flags=re.IGNORECASE).strip()
    cleaned = cleaned.rstrip(",")

    if has_etal:
        m = re.match(r"([A-Z][\w\-']+)", cleaned)
        if m:
            return f"{_norm_surname(m.group(1))}etal{year}"

    # Multi-author written-out forms, separated by commas / & / "and":
    #   "Hernán & Robins"          -> hernanrobins<year>
    #   "Pearl, Glymour & Jewell"  -> pearletal<year>  (3+ authors)
    # Mirrors _ref_key, which keys 3+ authors as "<first>etal<year>".
    tokens = [t for t in re.split(r"\s*(?:,|&|\\&|\band\b)\s*", cleaned) if t]
    surname_tokens = [t for t in tokens if re.fullmatch(r"[A-Z][\w\-']+", t)]
    if len(surname_tokens) >= 3:
        return f"{_norm_surname(surname_tokens[0])}etal{year}"
    if len(surname_tokens) == 2:
        return f"{_norm_surname(surname_tokens[0])}{_norm_surname(surname_tokens[1])}{year}"

    # Single surname
    m = re.match(r"^([A-Z][\w\-']+)\s*$", cleaned)
    if m:
        return f"{_norm_surname(m.group(1))}{year}"

    # Organisational author
    for w in re.findall(r"[A-Z][\w]{2,}", cleaned):
        return f"{_norm_surname(w)}{year}"
    return None


_PAREN_BLOCK_RE = re.compile(r"\(([^()]+?)\)")
_PAREN_CITATION_PART_RE = re.compile(
    r"^(.+?),\s*(\d{4}[a-z]?|n\.d\.[a-z]?)(?:\s*,\s*[^,]+)?$"
)
_NARRATIVE_RE = re.compile(
    r"\b([A-Z][\w\-']+(?:\s+(?:&|and)\s+[A-Z][\w\-']+)?(?:\s+et\s+al\.?)?)\s*"
    r"\((\d{4}[a-z]?)(?:,\s*[^)]+)?\)"
)


def link_citations(text: str, registry: set[str]) -> str:
    """Wrap in-line citations matching keys in `registry` with \\cit{key}{text}."""
    if not registry:
        return text

    # Narrative form first: ``Author (YEAR)`` / ``Author and Author (YEAR)``
    def narrative_repl(m: re.Match) -> str:
        key = _cite_key(m.group(1).strip(), m.group(2))
        if key and key in registry:
            return f"\\cit{{{key}}}{{{m.group(0)}}}"
        return m.group(0)

    text = _NARRATIVE_RE.sub(narrative_repl, text)

    # Parenthetical form: ``(Author, YEAR)``, possibly multi-cite via ``;``
    def paren_repl(m: re.Match) -> str:
        inside = m.group(1)
        if not re.search(r"\d{4}|n\.d\.", inside):
            return m.group(0)
        # Don't reprocess if we're already inside a \cit{...}{...} (the inner (YYYY) would re-match)
        if not re.search(r"[A-Za-z]", inside):
            return m.group(0)
        parts = re.split(r"\s*;\s*", inside)
        linked_parts = []
        any_linked = False
        for part in parts:
            stripped = part.strip()
            am = _PAREN_CITATION_PART_RE.match(stripped)
            if am:
                authors = am.group(1).strip()
                year = am.group(2)
                key = _cite_key(authors, year)
                if key and key in registry:
                    linked_parts.append(f"\\cit{{{key}}}{{{stripped}}}")
                    any_linked = True
                    continue
            linked_parts.append(stripped)
        if not any_linked:
            return m.group(0)
        return "(" + "; ".join(linked_parts) + ")"

    text = _PAREN_BLOCK_RE.sub(paren_repl, text)
    return text


# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://[^\s)]+")


def _wrap_urls(text: str) -> str:
    r"""Wrap bare URLs in \url{} so they can break across lines."""
    return _URL_RE.sub(lambda m: r"\url{" + m.group(0) + r"}", text)


def write_references_tex(refs: list[str]) -> str:
    out = [r"\begin{description}\setlength{\itemsep}{0pt}"]
    for r in refs:
        key = _ref_key(r)
        converted = _convert_inline(_escape_special_outside_math(r))
        # URL-wrap after the other conversions so we don't double-process inside \url{}
        converted = _wrap_urls(converted)
        if key:
            out.append(rf"\item[] \hypertarget{{{key}}}{{}}" + converted)
        else:
            out.append(r"\item[] " + converted)
    out.append(r"\end{description}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Figure conversion
# ---------------------------------------------------------------------------

def convert_figures() -> list[Path]:
    """Stage figure assets from drafts/figures/ into drafts/render/figures/.

    Per stem, prefer assets in this order: .pdf > .png > .jpg/.jpeg > .svg.
    PDF/PNG/JPG are copied directly. SVG is converted to PDF via svglib.
    Returns the list of resulting paths in drafts/render/figures/.
    """
    if not FIGURES_SRC_DIR.exists():
        return []

    FIGURES_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Group sources by stem
    stems: dict[str, dict[str, Path]] = {}
    for p in FIGURES_SRC_DIR.iterdir():
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in {".pdf", ".png", ".jpg", ".jpeg", ".svg"}:
            stems.setdefault(p.stem, {})[ext] = p

    out_paths: list[Path] = []
    for stem, files in stems.items():
        chosen_ext = next(
            (e for e in (".pdf", ".png", ".jpg", ".jpeg", ".svg") if e in files),
            None,
        )
        if chosen_ext is None:
            continue
        src = files[chosen_ext]

        if chosen_ext in {".pdf", ".png", ".jpg", ".jpeg"}:
            # Copy directly
            dst = FIGURES_OUT_DIR / src.name
            if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
                shutil.copy2(src, dst)
                print(f"[figures] copied {src.name} -> {dst.relative_to(PROJECT_ROOT)}")
            out_paths.append(dst)
            continue

        # SVG fallback
        try:
            from svglib.svglib import svg2rlg
            from reportlab.graphics import renderPDF
        except ImportError as exc:
            print(f"[figures] svglib not available ({exc}); SVG {src.name} skipped.", file=sys.stderr)
            continue
        dst = FIGURES_OUT_DIR / (stem + ".pdf")
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            out_paths.append(dst)
            continue
        try:
            drawing = svg2rlg(str(src))
            if drawing is None:
                print(f"[figures] could not parse {src.name}; skipping.", file=sys.stderr)
                continue
            renderPDF.drawToFile(drawing, str(dst))
            out_paths.append(dst)
            print(f"[figures] converted {src.name} -> {dst.relative_to(PROJECT_ROOT)}")
        except Exception as exc:
            print(f"[figures] failed to convert {src.name}: {exc}", file=sys.stderr)
    return out_paths


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def compile_pdf() -> tuple[bool, int | None]:
    """Run pdflatex twice. Return (success, pages)."""
    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "main.tex",
    ]
    for run_idx in range(2):
        result = subprocess.run(cmd, cwd=RENDER_DIR, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"pdflatex failed on run {run_idx + 1}", file=sys.stderr)
            print(result.stdout[-2000:], file=sys.stderr)
            return False, None
    pdf = RENDER_DIR / "main.pdf"
    if not pdf.exists():
        return False, None
    # Parse page count from log
    log = (RENDER_DIR / "main.log").read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"Output written on main\.pdf \((\d+) pages?", log)
    pages = int(m.group(1)) if m else None
    return True, pages


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Render paper drafts to PDF.")
    parser.add_argument(
        "--include-wip",
        action="store_true",
        help="Also include in-progress Section *.md files from drafts/ that aren't yet in final/.",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Generate .tex files but skip pdflatex compilation.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Source directory of draft files (default: drafts/final).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output render directory (default: drafts/render).",
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help="Use the compact layout (10pt, tight margins/spacing) for length-constrained renders.",
    )
    parser.add_argument(
        "--flat-intro",
        action="store_true",
        help="Render the Introduction as one continuous section (suppress the per-part subsection headers).",
    )
    parser.add_argument(
        "--enumerate-lists",
        action="store_true",
        help="Render all bullet lists as numbered (enumerate) lists.",
    )
    parser.add_argument(
        "--vi",
        action="store_true",
        help="Vietnamese output: T5 fontenc, babel[vietnamese], translated document headings.",
    )
    args = parser.parse_args()

    global FINAL_DIR, RENDER_DIR, SECTIONS_DIR, FIGURES_OUT_DIR, LIST_ENV
    if args.source:
        FINAL_DIR = Path(args.source).resolve()
    if args.out:
        RENDER_DIR = Path(args.out).resolve()
        SECTIONS_DIR = RENDER_DIR / "sections"
        FIGURES_OUT_DIR = RENDER_DIR / "figures"
    if args.enumerate_lists:
        LIST_ENV = "enumerate"

    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    SECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Convert SVG figures to PDF up front; needed before sections are rendered
    # so the mermaid-block handler can find them.
    convert_figures()

    files = discover_drafts(include_wip=args.include_wip)
    if not files:
        print("No draft files found in drafts/final/.", file=sys.stderr)
        return 1

    print(f"Discovered {len(files)} draft files:")
    for f in files:
        print(f"  - {f.path.relative_to(PROJECT_ROOT)} (group: {f.label}, sub: {f.sub_order})")

    # Build citation registry up front so body-text rendering can link citations
    refs = collect_references(files)
    registry = {_ref_key(r) for r in refs}
    registry.discard(None)
    print(f"\nCitation registry: {len(registry)} keys derived from {len(refs)} references.")

    # Build main.tex
    if args.vi:
        chosen_preamble = VI_PREAMBLE
        chosen_postamble = VI_POSTAMBLE
    elif args.dense:
        chosen_preamble = DENSE_PREAMBLE
        chosen_postamble = POSTAMBLE
    else:
        chosen_preamble = PREAMBLE
        chosen_postamble = POSTAMBLE
    main_tex = [chosen_preamble]
    abstract = render_abstract(registry=registry)
    if abstract:
        (SECTIONS_DIR / "00_abstract.tex").write_text(abstract + "\n", encoding="utf-8")
        main_tex.append(r"\input{sections/00_abstract}")
    intro = render_intro(files, registry=registry, flat=args.flat_intro)
    if intro:
        (SECTIONS_DIR / "01_introduction.tex").write_text(intro + "\n", encoding="utf-8")
        main_tex.append(r"\input{sections/01_introduction}")

    for sec_num in (2, 3, 4, 5, 6, 7, 8):
        sec = render_section(files, sec_num, registry=registry)
        if sec:
            (SECTIONS_DIR / f"0{sec_num}_section.tex").write_text(sec + "\n", encoding="utf-8")
            main_tex.append(r"\input{sections/0" + str(sec_num) + r"_section}")

    main_tex.append(chosen_postamble)
    (RENDER_DIR / "main.tex").write_text("\n".join(main_tex), encoding="utf-8")

    # References with hypertargets
    (SECTIONS_DIR / "references.tex").write_text(write_references_tex(refs) + "\n", encoding="utf-8")

    print(f"\nGenerated:")
    print(f"  - {(RENDER_DIR / 'main.tex').relative_to(PROJECT_ROOT)}")
    for p in sorted(SECTIONS_DIR.glob("*.tex")):
        print(f"  - {p.relative_to(PROJECT_ROOT)}")

    if args.no_compile:
        return 0

    print("\nCompiling with pdflatex (2 passes)...")
    ok, pages = compile_pdf()
    if not ok:
        print("Compilation failed. See drafts/render/main.log for details.", file=sys.stderr)
        return 2

    print(f"\nOK -- {(RENDER_DIR / 'main.pdf').relative_to(PROJECT_ROOT)}  ({pages} pages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
