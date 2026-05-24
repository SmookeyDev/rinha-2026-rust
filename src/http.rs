// Minimal HTTP/1.1 parser. Only supports POST /fraud-score and GET /ready
// with keep-alive. No chunked encoding, no multipart.

#[derive(Debug, PartialEq)]
pub enum RequestKind {
    FraudScore,
    Ready,
    NotFound,
    NeedMore,
    BadRequest,
}

pub struct ParsedRequest<'a> {
    pub kind: RequestKind,
    pub body: &'a [u8],
    pub consumed: usize,
    pub keep_alive: bool,
}

pub fn parse_request<'a>(buf: &'a [u8]) -> ParsedRequest<'a> {
    let header_end = match find_seq(buf, b"\r\n\r\n") {
        Some(i) => i,
        None => return ParsedRequest { kind: RequestKind::NeedMore, body: &[], consumed: 0, keep_alive: false },
    };

    let first_line_end = match find_crlf(buf, 0) {
        Some(i) => i,
        None => return ParsedRequest { kind: RequestKind::BadRequest, body: &[], consumed: 0, keep_alive: false },
    };

    let line = &buf[..first_line_end];
    let mut sp1 = None;
    let mut sp2 = None;
    for (i, &b) in line.iter().enumerate() {
        if b == b' ' {
            if sp1.is_none() { sp1 = Some(i); } else { sp2 = Some(i); break; }
        }
    }
    let (sp1, sp2) = match (sp1, sp2) {
        (Some(a), Some(b)) => (a, b),
        _ => return ParsedRequest { kind: RequestKind::BadRequest, body: &[], consumed: 0, keep_alive: false },
    };

    let kind = match (&line[..sp1], &line[sp1 + 1..sp2]) {
        (b"POST", b"/fraud-score") => RequestKind::FraudScore,
        (b"GET", b"/ready") => RequestKind::Ready,
        _ => RequestKind::NotFound,
    };

    let mut content_length = 0usize;
    let mut keep_alive = true;
    let mut i = first_line_end + 2;
    while i < header_end {
        let line_end = find_crlf(buf, i).unwrap_or(header_end);
        let header_line = &buf[i..line_end];
        if let Some(colon) = header_line.iter().position(|&b| b == b':') {
            let name = &header_line[..colon];
            let mut value = &header_line[colon + 1..];
            while !value.is_empty() && (value[0] == b' ' || value[0] == b'\t') {
                value = &value[1..];
            }
            if eq_ascii_case(name, b"content-length") {
                content_length = parse_usize(value).unwrap_or(0);
            } else if eq_ascii_case(name, b"connection") && eq_ascii_case(value, b"close") {
                keep_alive = false;
            }
        }
        i = line_end + 2;
    }

    let body_start = header_end + 4;
    let total = body_start + content_length;
    if buf.len() < total {
        return ParsedRequest { kind: RequestKind::NeedMore, body: &[], consumed: 0, keep_alive };
    }
    ParsedRequest {
        kind,
        body: &buf[body_start..total],
        consumed: total,
        keep_alive,
    }
}

#[inline]
fn eq_ascii_case(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| x.to_ascii_lowercase() == y.to_ascii_lowercase())
}

#[inline]
fn find_seq(buf: &[u8], needle: &[u8]) -> Option<usize> {
    if buf.len() < needle.len() { return None; }
    buf.windows(needle.len()).position(|w| w == needle)
}

#[inline]
fn find_crlf(buf: &[u8], start: usize) -> Option<usize> {
    let mut i = start;
    while i + 2 <= buf.len() {
        if buf[i] == b'\r' && buf[i + 1] == b'\n' { return Some(i); }
        i += 1;
    }
    None
}

#[inline]
fn parse_usize(s: &[u8]) -> Option<usize> {
    let mut n = 0usize;
    let mut any = false;
    for &b in s {
        match b {
            b'0'..=b'9' => {
                n = n.checked_mul(10)?.checked_add((b - b'0') as usize)?;
                any = true;
            }
            b' ' | b'\t' => continue,
            _ => return if any { Some(n) } else { None },
        }
    }
    if any { Some(n) } else { None }
}
