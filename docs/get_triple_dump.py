import re
import time
import requests
import argparse
import urllib3


def is_valid_uri(uri: str) -> bool:
    if uri.count('http://') + uri.count('https://') > 1:
        return False
    if re.search(r'\[|\]', uri):
        return False
    if '__' in uri:
        return False
    stripped = re.sub(r'^https?://', '', uri)
    after_colon = re.search(r':(\S+)', stripped)
    if after_colon and not after_colon.group(1).isdigit():
        return False
    return True


def has_illegal_literal(line: str) -> bool:
    """Check if any literal contains line-break control chars."""
    for m in re.finditer(r'"((?:[^"\\]|\\.)*)"', line):
        literal = m.group(1)
        if '\n' in literal or '\r' in literal:
            return True
        # escaped \r/\n are legal N-Triples, but some Turtle conversion paths
        # materialize them as raw control chars and fail parse; drop proactively.
        if "\\r" in literal or "\\n" in literal:
            return True
    return False


def quote_state_after(text: str, in_quote: bool = False) -> bool:
    """Return quote state after scanning text, honoring escapes."""
    escaped = False
    for ch in text:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_quote = not in_quote
    return in_quote


def is_complete_ntriple_statement(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#") and stripped.endswith(".")


def fix_ntriples_line(line: str) -> str | None:
    # Drop lines with raw newlines/CR inside literals
    if has_illegal_literal(line):
        return None

    def encode_uri(m):
        u = m.group(1)
        if not is_valid_uri(u):
            return None
        u = u.replace('"',  '%22')
        u = u.replace(' ',  '%20')
        u = u.replace('\\', '%5C')
        u = u.replace('^',  '%5E')
        u = u.replace('`',  '%60')
        u = u.replace('{',  '%7B')
        u = u.replace('}',  '%7D')
        u = u.replace('|',  '%7C')
        u = u.replace('[',  '%5B')
        u = u.replace(']',  '%5D')
        u = ''.join(c if ord(c) < 128 else f'%{ord(c):02X}' for c in u)
        return f'<{u}>'

    drop = False
    def encode_or_flag(m):
        nonlocal drop
        result = encode_uri(m)
        if result is None:
            drop = True
            return m.group(0)
        return result

    fixed = re.sub(r'<([^>]*)>', encode_or_flag, line)
    if drop:
        return None

    return fixed


def download_triples(
    endpoint: str,
    output_file: str,
    limit: int,
    headers: dict,
    with_specific_named_graph: str = None,
    verify_tls: bool | str = True,
):
    offset = 0
    total_triples = 0
    total_skipped = 0

    with open(output_file, "w", encoding="utf-8") as f:
        while True:
            if with_specific_named_graph:
                query = f"CONSTRUCT {{ ?s ?p ?o }} FROM <{with_specific_named_graph}> WHERE {{ ?s ?p ?o }} LIMIT {limit} OFFSET {offset}"
                print(f"Downloading triples from specific named graph: {with_specific_named_graph}")
            else:
                query = f"CONSTRUCT {{ ?s ?p ?o }} WHERE {{ ?s ?p ?o }} LIMIT {limit} OFFSET {offset}"
            print(f"Downloading triples at offset {offset}...")

            try:
                response = requests.post(
                    endpoint,
                    data={"query": query},
                    headers=headers,
                    timeout=300,
                    verify=verify_tls,
                )
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                print(f"Error at offset {offset}: {e}")
                print("Retrying in 10 seconds...")
                time.sleep(10)
                continue

            raw_text = response.content.decode("utf-8", errors="replace")
            raw_text = raw_text.replace('\r\n', '\n').replace('\r', '\n').strip()

            if not raw_text:
                print("Empty response. Done.")
                break

            raw_lines = raw_text.splitlines()
            statement_lines = []
            skipped_multiline = 0
            in_open_literal = False

            for raw_line in raw_lines:
                if not raw_line.strip():
                    continue

                if in_open_literal:
                    in_open_literal = quote_state_after(raw_line, in_open_literal)
                    continue

                if quote_state_after(raw_line):§
                    skipped_multiline += 1
                    in_open_literal = True
                    continue

                if is_complete_ntriple_statement(raw_line):
                    statement_lines.append(raw_line)
                else:
                    skipped_multiline += 1

            if in_open_literal:
                skipped_multiline += 1

            cleaned_lines = [fix_ntriples_line(line) for line in statement_lines]
            skipped = cleaned_lines.count(None) + skipped_multiline
            cleaned_lines = [l for l in cleaned_lines if l is not None]

            if skipped:
                total_skipped += skipped
                print(f"  Dropped {skipped} malformed lines at offset {offset}")

            cleaned = "\n".join(cleaned_lines)
            count = len(cleaned_lines)

            f.write(cleaned + "\n")
            f.flush()

            total_triples += count
            print(f"Got {count} triples | Total so far: {total_triples}")

            if len(raw_lines) < limit:
                print("Reached end of dataset.")
                break

            offset += limit

    print(f"\nDone! Total triples saved : {total_triples}")
    print(f"Total dropped             : {total_skipped}")
    print(f"Output file               : {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint",    type=str, default="https://lod.ehri-project-test.eu/sparql")
    parser.add_argument("--output-file", type=str, default="ehri_dump_cleaned.nt")
    parser.add_argument("--limit",       type=int, default=100000)
    parser.add_argument("--with-specific-named-graph", type=str, default=None)
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification (use only for trusted endpoints).",
    )
    parser.add_argument(
        "--ca-bundle",
        type=str,
        default=None,
        help="Path to a custom CA bundle file to verify TLS certificates.",
    )
    args = parser.parse_args()

    if args.ca_bundle:
        verify_tls = args.ca_bundle
    else:
        verify_tls = not args.insecure

    if args.insecure and not args.ca_bundle:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        print("WARNING: TLS certificate verification is disabled (--insecure).")

    headers = {"Accept": "application/n-triples"}
    download_triples(
        args.endpoint,
        args.output_file,
        args.limit,
        headers,
        args.with_specific_named_graph,
        verify_tls=verify_tls,
    )
    print(f"Downloaded to {args.output_file}")
