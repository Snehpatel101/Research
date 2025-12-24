"""
Formatting utilities for Phase 1 reports.
Handles markdown to HTML conversion and other formatting tasks.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def markdown_to_html(md_text: str) -> str:
    """
    Convert markdown text to HTML with styling.

    Args:
        md_text: Markdown text content

    Returns:
        HTML string with embedded CSS
    """
    html = _get_html_header()
    html += _convert_markdown_body(md_text)
    html += _get_html_footer()

    # Style status indicators
    html = _apply_status_styling(html)

    return html


def save_html_report(md_path: Path, output_path: Path) -> None:
    """
    Generate HTML version of a markdown report.

    Args:
        md_path: Path to markdown file
        output_path: Output path for HTML file
    """
    logger.info(f"Generating HTML report: {output_path}")

    with open(md_path) as f:
        md_content = f.read()

    html_content = markdown_to_html(md_content)

    with open(output_path, 'w') as f:
        f.write(html_content)

    logger.info(f"  HTML report saved to {output_path}")


def _get_html_header() -> str:
    """Get HTML document header with CSS styles."""
    return """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Phase 1 Summary Report</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f6f8fa;
        }
        .container {
            background-color: white;
            padding: 40px;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }
        h1 { color: #24292e; border-bottom: 2px solid #e1e4e8; padding-bottom: 10px; }
        h2 { color: #24292e; border-bottom: 1px solid #e1e4e8; padding-bottom: 8px; margin-top: 32px; }
        h3 { color: #24292e; margin-top: 24px; }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 16px 0;
        }
        table th, table td {
            border: 1px solid #dfe2e5;
            padding: 8px 12px;
            text-align: left;
        }
        table th {
            background-color: #f6f8fa;
            font-weight: 600;
        }
        table tr:nth-child(even) {
            background-color: #f6f8fa;
        }
        code {
            background-color: #f6f8fa;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
            font-size: 85%;
        }
        pre {
            background-color: #f6f8fa;
            padding: 16px;
            border-radius: 6px;
            overflow-x: auto;
        }
        pre code {
            background-color: transparent;
            padding: 0;
        }
        .pass { color: #22863a; font-weight: bold; }
        .fail { color: #d73a49; font-weight: bold; }
        .warning { color: #b08800; font-weight: bold; }
        img { max-width: 100%; height: auto; margin: 16px 0; }
    </style>
</head>
<body>
    <div class="container">
"""


def _get_html_footer() -> str:
    """Get HTML document footer."""
    return """
    </div>
</body>
</html>
"""


def _convert_markdown_body(md_text: str) -> str:
    """
    Convert markdown text to HTML body content.

    This is a basic implementation that handles common markdown elements.
    For production use, consider using a proper markdown library.
    """
    lines = md_text.split('\n')
    result_lines = []
    in_code_block = False

    for i, line in enumerate(lines):
        # Code blocks
        if line.startswith('```'):
            if in_code_block:
                result_lines.append('</code></pre>')
                in_code_block = False
            else:
                result_lines.append('<pre><code>')
                in_code_block = True
            continue

        if in_code_block:
            result_lines.append(line)
            continue

        # Headers
        if line.startswith('#### '):
            result_lines.append(f"<h4>{line[5:]}</h4>")
        elif line.startswith('### '):
            result_lines.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith('## '):
            result_lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith('# '):
            result_lines.append(f"<h1>{line[2:]}</h1>")
        # Horizontal rules
        elif line.strip() == '---':
            result_lines.append('<hr>')
        # Images
        elif line.startswith('!['):
            result_lines.append(_convert_image(line))
        # Tables
        elif line.strip().startswith('|'):
            result_lines.append(_convert_table_row(line, i, lines))
        # List items (unordered)
        elif line.strip().startswith('- '):
            result_lines.append(_convert_unordered_list_item(line, i, lines))
        # List items (ordered)
        elif _is_ordered_list_item(line):
            result_lines.append(_convert_ordered_list_item(line, i, lines))
        # Regular paragraphs with inline formatting
        elif line.strip():
            formatted = _apply_inline_formatting(line)
            result_lines.append(f'<p>{formatted}</p>')
        else:
            result_lines.append('')

    return '\n'.join(result_lines)


def _convert_image(line: str) -> str:
    """Convert markdown image syntax to HTML."""
    alt_end = line.find(']')
    url_start = line.find('(')
    url_end = line.find(')')
    if alt_end > 0 and url_start > 0:
        alt = line[2:alt_end]
        url = line[url_start + 1:url_end]
        return f'<img src="{url}" alt="{alt}">'
    return line


def _convert_table_row(line: str, index: int, lines: list) -> str:
    """Convert markdown table row to HTML."""
    # Skip separator rows
    if '---' in line:
        return ''

    cells = [c.strip() for c in line.split('|')[1:-1]]

    # Check if this is a header row (next row is separator)
    is_header = False
    if index + 1 < len(lines):
        next_line = lines[index + 1]
        if '---' in next_line and next_line.strip().startswith('|'):
            is_header = True

    # Check if we need to start/end table
    prev_is_table = index > 0 and lines[index - 1].strip().startswith('|')

    result = ''
    if not prev_is_table:
        result += '<table>\n'

    tag = 'th' if is_header else 'td'
    result += '<tr>'
    for cell in cells:
        formatted_cell = _apply_inline_formatting(cell)
        result += f'<{tag}>{formatted_cell}</{tag}>'
    result += '</tr>'

    # Check if we need to close table
    next_is_table = (
        index + 1 < len(lines) and
        lines[index + 1].strip().startswith('|')
    )
    if not next_is_table:
        result += '\n</table>'

    return result


def _convert_unordered_list_item(line: str, index: int, lines: list) -> str:
    """Convert markdown unordered list item to HTML."""
    content = line.strip()[2:]
    formatted = _apply_inline_formatting(content)

    # Check if previous line is also a list item
    prev_is_list = (
        index > 0 and
        lines[index - 1].strip().startswith('- ')
    )

    # Check if next line is also a list item
    next_is_list = (
        index + 1 < len(lines) and
        lines[index + 1].strip().startswith('- ')
    )

    result = ''
    if not prev_is_list:
        result += '<ul>\n'
    result += f'<li>{formatted}</li>'
    if not next_is_list:
        result += '\n</ul>'

    return result


def _is_ordered_list_item(line: str) -> bool:
    """Check if line is an ordered list item."""
    stripped = line.strip()
    if not stripped:
        return False
    if not stripped[0].isdigit():
        return False
    return '. ' in stripped


def _convert_ordered_list_item(line: str, index: int, lines: list) -> str:
    """Convert markdown ordered list item to HTML."""
    content = line.strip().split('. ', 1)[1]
    formatted = _apply_inline_formatting(content)

    # Check if previous line is also an ordered list item
    prev_is_list = index > 0 and _is_ordered_list_item(lines[index - 1])

    # Check if next line is also an ordered list item
    next_is_list = index + 1 < len(lines) and _is_ordered_list_item(lines[index + 1])

    result = ''
    if not prev_is_list:
        result += '<ol>\n'
    result += f'<li>{formatted}</li>'
    if not next_is_list:
        result += '\n</ol>'

    return result


def _apply_inline_formatting(text: str) -> str:
    """Apply inline markdown formatting (bold, code, etc.)."""
    # Bold
    while '**' in text:
        text = text.replace('**', '<strong>', 1)
        if '**' in text:
            text = text.replace('**', '</strong>', 1)

    # Inline code
    while '`' in text:
        text = text.replace('`', '<code>', 1)
        if '`' in text:
            text = text.replace('`', '</code>', 1)

    return text


def _apply_status_styling(html: str) -> str:
    """Apply CSS classes to status indicators."""
    replacements = [
        ('PASS', '<span class="pass">PASS</span>'),
        ('FAIL', '<span class="fail">FAIL</span>'),
        ('WARNING', '<span class="warning">WARNING</span>'),
        ('PASSED', '<span class="pass">PASSED</span>'),
        ('FAILED', '<span class="fail">FAILED</span>'),
    ]

    for old, new in replacements:
        # Only replace in table cells to avoid replacing in prose
        html = html.replace(f'<td>{old}</td>', f'<td>{new}</td>')
        html = html.replace(f'| {old} |', f'| {new} |')

    return html
