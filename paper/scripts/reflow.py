"""
Reflow prose in a .tex file to one sentence per line.
Handles:
  - Regular prose paragraphs
  - Prose after \paragraph{...} on the same line, plus continuation lines
  - Prose after \item on the same line, plus continuation lines
  - Multi-line \caption{...} contents
Skips: \begin/\end environments, section headers, table rows, comments,
preamble commands, bare braces / brackets.

Usage:  python reflow.py main.tex
Writes: main.tex.reflowed   (review the diff, then rename to overwrite)
"""
import re
import sys
from pathlib import Path

SENTENCE_SPLIT = re.compile(r'(?<=[.!?])[ \t]+(?=[A-Z\\])')

# Lines that should never be reflowed or joined into a prose run
STRUCT_LINE_RE = re.compile(
    r'^\s*('
    r'\\begin\{|\\end\{|'
    r'\\section\b|\\subsection\b|\\subsubsection\b|'
    r'\\label\b|\\toprule|\\midrule|\\bottomrule|'
    r'\\centering\b|\\small\b|'
    r'\\vskip\b|\\vspace\b|\\hspace\b|\\setlength\b|'
    r'\\twocolumn|\\onecolumn|\\newpage|\\appendix|'
    r'\\icml|\\printAffil|\\bibliography\b|\\bibliographystyle\b|'
    r'\\newcommand|\\usepackage|\\documentclass|'
    r'\\theoremstyle|\\newtheorem|\\setcounter|'
    r'%'
    r')'
    r'|^\s*[\[\]\{\}]\s*$'
    r'|\\\\\s*$'      # line ends with \\ (table row)
    r'| & '           # table cell separator
)

# Structural-prefix-with-inline-prose patterns: prefix consumes the LaTeX command,
# the remaining text is prose to be joined with following continuation lines.
PARAGRAPH_PREFIX = re.compile(r'^(\s*)(\\paragraph\{[^}]*\}\.?)\s*(.*)$')
ITEM_PREFIX      = re.compile(r'^(\s*)(\\item\b)\s*(.*)$')
CAPTION_OPEN     = re.compile(r'^(\s*)\\caption\{(.*)$')

def is_structural(line: str) -> bool:
    return bool(STRUCT_LINE_RE.search(line))

def split_sentences(text: str) -> list:
    return [s for s in SENTENCE_SPLIT.split(text.strip()) if s]

def collect_prose_run(lines, start):
    """Collect continuation lines starting at index `start`.
    Stops at blank line, structural line, or a new inline-prefix line."""
    parts = []
    j = start
    while j < len(lines):
        ln = lines[j]
        if not ln.strip():
            break
        if is_structural(ln):
            break
        if PARAGRAPH_PREFIX.match(ln) or ITEM_PREFIX.match(ln) or CAPTION_OPEN.match(ln):
            break
        parts.append(ln.strip())
        j += 1
    return parts, j

def reflow_caption(lines, i):
    """Process \\caption{...} starting at line i; the opening brace may not close on this line.
    Tracks brace depth to find the matching }. Returns (output_lines, new_i)."""
    m = CAPTION_OPEN.match(lines[i])
    indent, first_content = m.groups()
    # Collect all content lines until brace depth returns to 0.
    # Start at depth 1 (the \caption opening brace), then add/subtract per char.
    buf = [first_content]
    depth = 1 + first_content.count('{') - first_content.count('}')
    j = i + 1
    while depth > 0 and j < len(lines):
        buf.append(lines[j])
        depth += lines[j].count('{') - lines[j].count('}')
        j += 1
    # Join all content; the final character should be the closing } of \caption
    joined = ' '.join(p.strip() for p in buf if p.strip())
    # Peel off the trailing '}' that closes \caption
    if joined.endswith('}'):
        inner = joined[:-1].rstrip()
        tail  = '}'
    else:
        inner = joined
        tail  = ''
    sentences = split_sentences(inner)
    if not sentences:
        return [f'{indent}\\caption{{{tail}'], j
    out = [f'{indent}\\caption{{{sentences[0]}']
    for s in sentences[1:-1]:
        out.append(f'{indent}  {s}')
    if len(sentences) > 1:
        out.append(f'{indent}  {sentences[-1]}{tail}')
    else:
        out[-1] = f'{indent}\\caption{{{sentences[0]}{tail}'
    return out, j

def process(text: str) -> str:
    lines = text.split('\n')
    out = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if not line.strip():
            out.append(line); i += 1; continue

        if CAPTION_OPEN.match(line):
            chunk, i = reflow_caption(lines, i)
            out.extend(chunk)
            continue

        if is_structural(line):
            out.append(line); i += 1; continue

        m = PARAGRAPH_PREFIX.match(line)
        if m:
            indent, prefix, rest = m.groups()
            rest = rest.strip()
            cont, i = collect_prose_run(lines, i + 1)
            joined = ' '.join([rest] + cont).strip()
            sentences = split_sentences(joined) if joined else []
            if not sentences:
                out.append(f'{indent}{prefix}')
            else:
                out.append(f'{indent}{prefix} {sentences[0]}')
                for s in sentences[1:]:
                    out.append(f'{indent}  {s}')
            continue

        m = ITEM_PREFIX.match(line)
        if m:
            indent, prefix, rest = m.groups()
            rest = rest.strip()
            cont, i = collect_prose_run(lines, i + 1)
            joined = ' '.join([rest] + cont).strip()
            sentences = split_sentences(joined) if joined else []
            if not sentences:
                out.append(f'{indent}{prefix}')
            else:
                out.append(f'{indent}{prefix} {sentences[0]}')
                for s in sentences[1:]:
                    out.append(f'{indent}    {s}')
            continue

        # Regular prose line — start of a prose run
        first_indent = re.match(r'^(\s*)', line).group(1)
        parts = [line.strip()]
        cont, i = collect_prose_run(lines, i + 1)
        parts.extend(cont)
        joined = ' '.join(parts)
        sentences = split_sentences(joined)
        for s in sentences:
            out.append(f'{first_indent}{s}')

    return '\n'.join(out)

def main(path: str) -> None:
    raw = Path(path).read_bytes()
    use_crlf = b'\r\n' in raw
    text = raw.decode('utf-8').replace('\r\n', '\n').replace('\r', '\n')
    result = process(text)
    if use_crlf:
        result = result.replace('\n', '\r\n')
    Path(path + '.reflowed').write_bytes(result.encode('utf-8'))
    print(f'Output: {path}.reflowed  ({"CRLF" if use_crlf else "LF"})')

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'main.tex')
