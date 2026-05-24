"""14-dimension feature vector. Mirrors data-generator/main.c:normalize()."""
from datetime import datetime, timezone

MAX_AMOUNT = 10000.0
MAX_INSTALLMENTS = 12.0
AMOUNT_VS_AVG_RATIO = 10.0
MAX_MINUTES = 1440.0
MAX_KM = 1000.0
MAX_TX_COUNT_24H = 20.0
MAX_MERCHANT_AVG = 10000.0

MCC_RISK = {
    "5411": 0.15, "5812": 0.30, "5912": 0.20, "5944": 0.45,
    "7801": 0.80, "7802": 0.75, "7995": 0.85, "4511": 0.35,
    "5311": 0.25, "5999": 0.50,
}


def clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def round4(x):
    return round(x * 10000.0) / 10000.0


def parse_iso(ts):
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def normalize(payload):
    tx = payload["transaction"]
    customer = payload["customer"]
    merchant = payload["merchant"]
    terminal = payload["terminal"]
    last = payload.get("last_transaction")

    ts = parse_iso(tx["requested_at"])
    out = [0.0] * 14
    out[0] = clamp01(tx["amount"] / MAX_AMOUNT)
    out[1] = clamp01(tx["installments"] / MAX_INSTALLMENTS)
    out[2] = clamp01((tx["amount"] / customer["avg_amount"]) / AMOUNT_VS_AVG_RATIO)
    out[3] = ts.hour / 23.0
    out[4] = ts.weekday() / 6.0
    if last is None:
        out[5] = -1.0
        out[6] = -1.0
    else:
        last_ts = parse_iso(last["timestamp"])
        out[5] = clamp01(((ts - last_ts).total_seconds() / 60.0) / MAX_MINUTES)
        out[6] = clamp01(last["km_from_current"] / MAX_KM)
    out[7] = clamp01(terminal["km_from_home"] / MAX_KM)
    out[8] = clamp01(customer["tx_count_24h"] / MAX_TX_COUNT_24H)
    out[9] = 1.0 if terminal["is_online"] else 0.0
    out[10] = 1.0 if terminal["card_present"] else 0.0
    out[11] = 0.0 if merchant["id"] in customer["known_merchants"] else 1.0
    out[12] = MCC_RISK.get(merchant["mcc"], 0.5)
    out[13] = clamp01(merchant["avg_amount"] / MAX_MERCHANT_AVG)
    return out


# Examples lifted from the official REGRAS_DE_DETECCAO.md document. Used to
# check that the normalization matches the C generator bit for bit.
EXAMPLE_LEGIT = {
    "id": "tx-1329056812",
    "transaction": {"amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z"},
    "customer": {"avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": ["MERC-003", "MERC-016"]},
    "merchant": {"id": "MERC-016", "mcc": "5411", "avg_amount": 60.25},
    "terminal": {"is_online": False, "card_present": True, "km_from_home": 29.23},
    "last_transaction": None,
}
EXPECTED_LEGIT = [0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006]

EXAMPLE_FRAUD = {
    "id": "tx-3330991687",
    "transaction": {"amount": 9505.97, "installments": 10, "requested_at": "2026-03-14T05:15:12Z"},
    "customer": {"avg_amount": 81.28, "tx_count_24h": 20, "known_merchants": ["MERC-008", "MERC-007", "MERC-005"]},
    "merchant": {"id": "MERC-068", "mcc": "7802", "avg_amount": 54.86},
    "terminal": {"is_online": False, "card_present": True, "km_from_home": 952.27},
    "last_transaction": None,
}
EXPECTED_FRAUD = [0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055]


if __name__ == "__main__":
    for payload, expected, name in [
        (EXAMPLE_LEGIT, EXPECTED_LEGIT, "legit"),
        (EXAMPLE_FRAUD, EXPECTED_FRAUD, "fraud"),
    ]:
        actual = [round4(v) for v in normalize(payload)]
        assert actual == expected, f"{name}: got {actual}"
        print(f"{name}: ok")
