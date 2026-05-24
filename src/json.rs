// JSON parser specialised for the POST /fraud-score payload. Not generic.
// Zero-alloc: string slices point back into the original buffer. Tolerates
// arbitrary key ordering and basic whitespace.

const MAX_KNOWN_MERCHANTS: usize = 16;

#[derive(Debug, Default, Clone, Copy)]
pub struct Payload<'a> {
    pub amount: f32,
    pub installments: i32,
    pub requested_at: &'a [u8],
    pub customer_avg: f32,
    pub tx_count_24h: i32,
    pub known_merchants: [&'a [u8]; MAX_KNOWN_MERCHANTS],
    pub known_n: usize,
    pub merchant_id: &'a [u8],
    pub merchant_mcc: &'a [u8],
    pub merchant_avg: f32,
    pub is_online: bool,
    pub card_present: bool,
    pub terminal_km_from_home: f32,
    pub last_transaction: Option<LastTx<'a>>,
}

#[derive(Debug, Clone, Copy)]
pub struct LastTx<'a> {
    pub timestamp: &'a [u8],
    pub km_from_current: f32,
}

#[derive(Debug)]
pub enum ParseErr {
    Unexpected,
    EndOfInput,
}

struct Parser<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Parser<'a> {
    #[inline(always)]
    fn skip_ws(&mut self) {
        while self.p < self.b.len() {
            match self.b[self.p] {
                b' ' | b'\t' | b'\n' | b'\r' => self.p += 1,
                _ => break,
            }
        }
    }
    #[inline(always)]
    fn peek(&self) -> Option<u8> {
        if self.p < self.b.len() { Some(self.b[self.p]) } else { None }
    }
    #[inline(always)]
    fn consume(&mut self, ch: u8) -> Result<(), ParseErr> {
        self.skip_ws();
        if self.peek() == Some(ch) { self.p += 1; Ok(()) } else { Err(ParseErr::Unexpected) }
    }
    // Strings in this challenge's payloads never contain escape sequences
    // (plain ASCII IDs and timestamps), so a `\` is treated as a fatal error.
    #[inline]
    fn parse_str(&mut self) -> Result<&'a [u8], ParseErr> {
        self.skip_ws();
        if self.peek() != Some(b'"') { return Err(ParseErr::Unexpected); }
        self.p += 1;
        let start = self.p;
        while self.p < self.b.len() {
            let c = self.b[self.p];
            if c == b'"' {
                let slice = &self.b[start..self.p];
                self.p += 1;
                return Ok(slice);
            }
            if c == b'\\' {
                return Err(ParseErr::Unexpected);
            }
            self.p += 1;
        }
        Err(ParseErr::EndOfInput)
    }
    #[inline]
    fn parse_number(&mut self) -> Result<f64, ParseErr> {
        self.skip_ws();
        let start = self.p;
        if self.peek() == Some(b'-') { self.p += 1; }
        while self.p < self.b.len() {
            let c = self.b[self.p];
            if !((c >= b'0' && c <= b'9') || c == b'.' || c == b'e' || c == b'E' || c == b'+' || c == b'-') {
                break;
            }
            self.p += 1;
        }
        if self.p == start { return Err(ParseErr::Unexpected); }
        let s = unsafe { std::str::from_utf8_unchecked(&self.b[start..self.p]) };
        s.parse::<f64>().map_err(|_| ParseErr::Unexpected)
    }
    #[inline]
    fn parse_bool(&mut self) -> Result<bool, ParseErr> {
        self.skip_ws();
        if self.b[self.p..].starts_with(b"true") {
            self.p += 4;
            Ok(true)
        } else if self.b[self.p..].starts_with(b"false") {
            self.p += 5;
            Ok(false)
        } else {
            Err(ParseErr::Unexpected)
        }
    }
    #[inline]
    fn parse_null(&mut self) -> Result<bool, ParseErr> {
        self.skip_ws();
        if self.b[self.p..].starts_with(b"null") {
            self.p += 4;
            Ok(true)
        } else {
            Ok(false)
        }
    }
    #[inline]
    fn skip_value(&mut self) -> Result<(), ParseErr> {
        self.skip_ws();
        let c = self.peek().ok_or(ParseErr::EndOfInput)?;
        match c {
            b'"' => { self.parse_str()?; }
            b'{' => self.skip_object()?,
            b'[' => self.skip_array()?,
            b't' | b'f' => { self.parse_bool()?; }
            b'n' => { self.parse_null()?; }
            _ => { self.parse_number()?; }
        }
        Ok(())
    }
    #[inline]
    fn skip_array(&mut self) -> Result<(), ParseErr> {
        self.consume(b'[')?;
        loop {
            self.skip_ws();
            if self.peek() == Some(b']') { self.p += 1; return Ok(()); }
            self.skip_value()?;
            self.skip_ws();
            if self.peek() == Some(b',') { self.p += 1; continue; }
            self.consume(b']')?;
            return Ok(());
        }
    }
    #[inline]
    fn skip_object(&mut self) -> Result<(), ParseErr> {
        self.consume(b'{')?;
        loop {
            self.skip_ws();
            if self.peek() == Some(b'}') { self.p += 1; return Ok(()); }
            self.parse_str()?;
            self.consume(b':')?;
            self.skip_value()?;
            self.skip_ws();
            if self.peek() == Some(b',') { self.p += 1; continue; }
            self.consume(b'}')?;
            return Ok(());
        }
    }
}

pub fn parse_payload<'a>(body: &'a [u8]) -> Result<Payload<'a>, ParseErr> {
    let mut p = Payload::default();
    let mut parser = Parser { b: body, p: 0 };
    parser.consume(b'{')?;
    loop {
        parser.skip_ws();
        if parser.peek() == Some(b'}') { break; }
        let key = parser.parse_str()?;
        parser.consume(b':')?;
        match key {
            b"id" => { parser.parse_str()?; }
            b"transaction" => parse_transaction(&mut parser, &mut p)?,
            b"customer" => parse_customer(&mut parser, &mut p)?,
            b"merchant" => parse_merchant(&mut parser, &mut p)?,
            b"terminal" => parse_terminal(&mut parser, &mut p)?,
            b"last_transaction" => parse_last_tx(&mut parser, &mut p)?,
            _ => parser.skip_value()?,
        }
        parser.skip_ws();
        if parser.peek() == Some(b',') {
            parser.p += 1;
        }
    }
    Ok(p)
}

fn parse_transaction<'a>(parser: &mut Parser<'a>, p: &mut Payload<'a>) -> Result<(), ParseErr> {
    parser.consume(b'{')?;
    loop {
        parser.skip_ws();
        if parser.peek() == Some(b'}') { parser.p += 1; return Ok(()); }
        let key = parser.parse_str()?;
        parser.consume(b':')?;
        match key {
            b"amount" => { p.amount = parser.parse_number()? as f32; }
            b"installments" => { p.installments = parser.parse_number()? as i32; }
            b"requested_at" => { p.requested_at = parser.parse_str()?; }
            _ => parser.skip_value()?,
        }
        parser.skip_ws();
        if parser.peek() == Some(b',') { parser.p += 1; continue; }
    }
}

fn parse_customer<'a>(parser: &mut Parser<'a>, p: &mut Payload<'a>) -> Result<(), ParseErr> {
    parser.consume(b'{')?;
    loop {
        parser.skip_ws();
        if parser.peek() == Some(b'}') { parser.p += 1; return Ok(()); }
        let key = parser.parse_str()?;
        parser.consume(b':')?;
        match key {
            b"avg_amount" => { p.customer_avg = parser.parse_number()? as f32; }
            b"tx_count_24h" => { p.tx_count_24h = parser.parse_number()? as i32; }
            b"known_merchants" => {
                parser.consume(b'[')?;
                let mut n = 0;
                loop {
                    parser.skip_ws();
                    if parser.peek() == Some(b']') { parser.p += 1; break; }
                    let s = parser.parse_str()?;
                    if n < MAX_KNOWN_MERCHANTS {
                        p.known_merchants[n] = s;
                        n += 1;
                    }
                    parser.skip_ws();
                    if parser.peek() == Some(b',') { parser.p += 1; continue; }
                }
                p.known_n = n;
            }
            _ => parser.skip_value()?,
        }
        parser.skip_ws();
        if parser.peek() == Some(b',') { parser.p += 1; continue; }
    }
}

fn parse_merchant<'a>(parser: &mut Parser<'a>, p: &mut Payload<'a>) -> Result<(), ParseErr> {
    parser.consume(b'{')?;
    loop {
        parser.skip_ws();
        if parser.peek() == Some(b'}') { parser.p += 1; return Ok(()); }
        let key = parser.parse_str()?;
        parser.consume(b':')?;
        match key {
            b"id" => { p.merchant_id = parser.parse_str()?; }
            b"mcc" => { p.merchant_mcc = parser.parse_str()?; }
            b"avg_amount" => { p.merchant_avg = parser.parse_number()? as f32; }
            _ => parser.skip_value()?,
        }
        parser.skip_ws();
        if parser.peek() == Some(b',') { parser.p += 1; continue; }
    }
}

fn parse_terminal<'a>(parser: &mut Parser<'a>, p: &mut Payload<'a>) -> Result<(), ParseErr> {
    parser.consume(b'{')?;
    loop {
        parser.skip_ws();
        if parser.peek() == Some(b'}') { parser.p += 1; return Ok(()); }
        let key = parser.parse_str()?;
        parser.consume(b':')?;
        match key {
            b"is_online" => { p.is_online = parser.parse_bool()?; }
            b"card_present" => { p.card_present = parser.parse_bool()?; }
            b"km_from_home" => { p.terminal_km_from_home = parser.parse_number()? as f32; }
            _ => parser.skip_value()?,
        }
        parser.skip_ws();
        if parser.peek() == Some(b',') { parser.p += 1; continue; }
    }
}

fn parse_last_tx<'a>(parser: &mut Parser<'a>, p: &mut Payload<'a>) -> Result<(), ParseErr> {
    parser.skip_ws();
    if parser.parse_null()? {
        p.last_transaction = None;
        return Ok(());
    }
    parser.consume(b'{')?;
    let mut last = LastTx { timestamp: b"", km_from_current: 0.0 };
    loop {
        parser.skip_ws();
        if parser.peek() == Some(b'}') { parser.p += 1; break; }
        let key = parser.parse_str()?;
        parser.consume(b':')?;
        match key {
            b"timestamp" => { last.timestamp = parser.parse_str()?; }
            b"km_from_current" => { last.km_from_current = parser.parse_number()? as f32; }
            _ => parser.skip_value()?,
        }
        parser.skip_ws();
        if parser.peek() == Some(b',') { parser.p += 1; continue; }
    }
    p.last_transaction = Some(last);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parse_legit_example() {
        let body = br#"{
            "id":"tx-1329056812",
            "transaction":{"amount":41.12,"installments":2,"requested_at":"2026-03-11T18:45:53Z"},
            "customer":{"avg_amount":82.24,"tx_count_24h":3,"known_merchants":["MERC-003","MERC-016"]},
            "merchant":{"id":"MERC-016","mcc":"5411","avg_amount":60.25},
            "terminal":{"is_online":false,"card_present":true,"km_from_home":29.23},
            "last_transaction":null
        }"#;
        let p = parse_payload(body).unwrap();
        assert!((p.amount - 41.12).abs() < 1e-3);
        assert_eq!(p.installments, 2);
        assert_eq!(p.requested_at, b"2026-03-11T18:45:53Z");
        assert!((p.customer_avg - 82.24).abs() < 1e-3);
        assert_eq!(p.tx_count_24h, 3);
        assert_eq!(p.known_n, 2);
        assert_eq!(p.known_merchants[0], b"MERC-003");
        assert_eq!(p.known_merchants[1], b"MERC-016");
        assert_eq!(p.merchant_id, b"MERC-016");
        assert_eq!(p.merchant_mcc, b"5411");
        assert!((p.merchant_avg - 60.25).abs() < 1e-3);
        assert!(!p.is_online);
        assert!(p.card_present);
        assert!((p.terminal_km_from_home - 29.23).abs() < 1e-3);
        assert!(p.last_transaction.is_none());
    }
}
