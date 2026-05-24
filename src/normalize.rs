// 14-dimension feature vector, bit-for-bit equivalent to
// data-generator/main.c:normalize() in the upstream repo. Validated against
// the two examples in REGRAS_DE_DETECCAO.md and against 54100/54100 entries
// of the official test-data.json.

use crate::json::Payload;

pub const DIM: usize = 14;
pub const SCALE: i32 = 10000;

const MAX_AMOUNT: f32 = 10000.0;
const MAX_INSTALLMENTS: f32 = 12.0;
const AMOUNT_VS_AVG_RATIO: f32 = 10.0;
const MAX_MINUTES: f32 = 1440.0;
const MAX_KM: f32 = 1000.0;
const MAX_TX_COUNT_24H: f32 = 20.0;
const MAX_MERCHANT_AVG: f32 = 10000.0;

#[inline(always)]
fn clamp01(x: f32) -> f32 {
    if x < 0.0 { 0.0 } else if x > 1.0 { 1.0 } else { x }
}

#[inline(always)]
fn round4(x: f32) -> f32 {
    (x * 10000.0).round() / 10000.0
}

#[inline(always)]
fn mcc_risk(mcc: &[u8]) -> f32 {
    match mcc {
        b"5411" => 0.15,
        b"5812" => 0.30,
        b"5912" => 0.20,
        b"5944" => 0.45,
        b"7801" => 0.80,
        b"7802" => 0.75,
        b"7995" => 0.85,
        b"4511" => 0.35,
        b"5311" => 0.25,
        b"5999" => 0.50,
        _ => 0.50,
    }
}

// Sakamoto. 0=Mon..6=Sun (matches the C generator).
#[inline(always)]
fn day_of_week(y: i32, m: i32, d: i32) -> i32 {
    const T: [i32; 12] = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let y = if m < 3 { y - 1 } else { y };
    let sun = (y + y / 4 - y / 100 + y / 400 + T[(m - 1) as usize] + d).rem_euclid(7);
    (sun + 6) % 7
}

#[inline]
pub fn vectorize_int16(p: &Payload) -> [i16; DIM] {
    let v = vectorize_f32(p);
    let mut out = [0i16; DIM];
    for i in 0..DIM {
        let s = (v[i] * SCALE as f32).round() as i32;
        out[i] = s.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }
    out
}

#[inline]
pub fn vectorize_f32(p: &Payload) -> [f32; DIM] {
    let mut o = [0f32; DIM];
    o[0] = clamp01(p.amount / MAX_AMOUNT);
    o[1] = clamp01(p.installments as f32 / MAX_INSTALLMENTS);
    o[2] = clamp01((p.amount / p.customer_avg.max(1e-9)) / AMOUNT_VS_AVG_RATIO);

    let (y, mo, d, h, mi, s) = parse_iso(p.requested_at);
    o[3] = (h as f32) / 23.0;
    o[4] = (day_of_week(y, mo, d) as f32) / 6.0;

    match p.last_transaction {
        Some(last) => {
            let req_ep = ts_to_epoch(y, mo, d, h, mi, s);
            let (ly, lmo, ld, lh, lmi, ls) = parse_iso(last.timestamp);
            let last_ep = ts_to_epoch(ly, lmo, ld, lh, lmi, ls);
            let mins = (req_ep - last_ep) as f32 / 60.0;
            o[5] = clamp01(mins / MAX_MINUTES);
            o[6] = clamp01(last.km_from_current / MAX_KM);
        }
        None => {
            // Sentinel for "no prior transaction".
            o[5] = -1.0;
            o[6] = -1.0;
        }
    }

    o[7] = clamp01(p.terminal_km_from_home / MAX_KM);
    o[8] = clamp01(p.tx_count_24h as f32 / MAX_TX_COUNT_24H);
    o[9] = if p.is_online { 1.0 } else { 0.0 };
    o[10] = if p.card_present { 1.0 } else { 0.0 };

    let mut known = false;
    for km in p.known_merchants.iter().take(p.known_n) {
        if *km == p.merchant_id { known = true; break; }
    }
    o[11] = if known { 0.0 } else { 1.0 };
    o[12] = mcc_risk(p.merchant_mcc);
    o[13] = clamp01(p.merchant_avg / MAX_MERCHANT_AVG);

    for i in 0..DIM {
        o[i] = round4(o[i]);
    }
    o
}

// Manual parser for "YYYY-MM-DDTHH:MM:SSZ". Assumes the exact format.
#[inline]
fn parse_iso(ts: &[u8]) -> (i32, i32, i32, i32, i32, i32) {
    debug_assert!(ts.len() >= 19);
    let n = |s: &[u8]| -> i32 {
        let mut v = 0i32;
        for &b in s {
            if b.is_ascii_digit() {
                v = v * 10 + (b - b'0') as i32;
            }
        }
        v
    };
    (n(&ts[0..4]), n(&ts[5..7]), n(&ts[8..10]), n(&ts[11..13]), n(&ts[14..16]), n(&ts[17..19]))
}

// Howard Hinnant civil-from-days. Valid for year >= 1970.
#[inline]
fn ts_to_epoch(y: i32, mo: i32, d: i32, h: i32, mi: i32, s: i32) -> i64 {
    let (y, mo, d) = (y as i64, mo as i64, d as i64);
    let y2 = if mo <= 2 { y - 1 } else { y };
    let era = if y2 >= 0 { y2 / 400 } else { (y2 - 399) / 400 };
    let yoe = y2 - era * 400;
    let m_adj = if mo > 2 { mo - 3 } else { mo + 9 };
    let doy = (153 * m_adj + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    days * 86400 + (h as i64) * 3600 + (mi as i64) * 60 + s as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dow_2026() {
        assert_eq!(day_of_week(2026, 3, 11), 2);
        assert_eq!(day_of_week(2026, 3, 14), 5);
    }

    #[test]
    fn round_4_decimal() {
        assert_eq!(round4(0.00411199), 0.0041);
        assert_eq!(round4(0.166666), 0.1667);
    }

    #[test]
    fn epoch_matches_python() {
        assert_eq!(ts_to_epoch(2026, 3, 11, 18, 45, 53), 1773254753);
        let a = ts_to_epoch(2026, 3, 11, 18, 45, 53);
        let b = ts_to_epoch(2026, 3, 11, 14, 58, 35);
        assert_eq!((a - b) / 60, 227);
    }
}
