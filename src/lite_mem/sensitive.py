"""Secrets that must not become memories (design §5, S5b): passwords and PINs, card numbers,
national ID numbers, API keys. Found by rules in code — a prompt instruction is not a control (the
S5 eval: every system, mem0 included, stored a bank-card PIN).

The rules prefer precision:
- a credential needs a keyword and a value holding a digit ("password hunter2", "PIN 是 4829 1038"),
  so "forgot her password" and "密码学" pass;
- a card number must start like a card network's, pass the Luhn check and not be labelled as an
  order, tracking or invoice number;
- an ID number must pass its check digit.
A digit-free password after a keyword is missed — the trade-off for not dropping ordinary facts."""

import re
from dataclasses import dataclass
from typing import Literal

Kind = Literal["credential", "card", "national_id", "api_key"]
Policy = Literal["drop", "mask", "keep"]
MASK = "[redacted]"

_API_KEY = re.compile(
    r"sk-[A-Za-z0-9_-]{16,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|gh[pousr]_[A-Za-z0-9]{36,}"
    r"|xox[abprs]-[A-Za-z0-9-]{10,}"
    r"|stw_[A-Za-z0-9]+_[A-Za-z0-9]+"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
)
_NATIONAL_ID = re.compile(r"(?<![0-9Xx])\d{17}[0-9Xx](?![0-9Xx])")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
# Visa, Mastercard, Amex, Discover, JCB, Diners, UnionPay.
_CARD_NETWORK = re.compile(r"4|5[1-5]|2[2-7]|3[47]|6011|65|64[4-9]|35|3[068]|62")
# A label just before the number that makes it something else.
_NOT_A_CARD = re.compile(
    r"(?:订单号?|单号|快递|运单|发票|order|tracking|invoice|ref(?:erence)?)\s*(?:no\.?|number|#)?"
    r"\s*[:：#]?\s*$",
    re.IGNORECASE,
)
_KEYWORD = (
    r"(?:支付密码|密码|口令|验证码|安全码|pin码"
    r"|\b(?:password|passwd|passcode|pwd|pin|cvv|cvc|otp|security code)\b)"
)
_CONNECTOR = r"[ \t:：=]*(?:(?:是|为|is|was|are)[ \t:：=]*)?"
_TOKEN = r"[^\s，。,;；、]+"
# The value: one token, then up to three more that hold digits ("4829 1038", "tiger 77").
_CREDENTIAL = re.compile(
    _KEYWORD + _CONNECTOR + f"({_TOKEN}(?:[ \\t]+(?=\\S*\\d){_TOKEN}){{0,3}})", re.IGNORECASE
)
_ID_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
_LABEL_WINDOW = 20  # characters before a number searched for an order/tracking label


@dataclass(frozen=True)
class Finding:
    kind: Kind
    start: int  # the secret value's span in the text
    end: int


def find(text: str) -> list[Finding]:
    """Every secret in `text`, in order of position. Findings may overlap (a card number inside a
    credential value): `mask` hides their union, so no part of a longer match stays visible."""
    found: list[Finding] = []

    def take(kind: Kind, start: int, end: int) -> None:
        found.append(Finding(kind, start, end))

    for m in _API_KEY.finditer(text):
        take("api_key", m.start(), m.end())
    for m in _NATIONAL_ID.finditer(text):
        if _id_ok(m.group()):
            take("national_id", m.start(), m.end())
    for m in _CARD.finditer(text):
        if _card(m.group(), text[max(0, m.start() - _LABEL_WINDOW) : m.start()]):
            take("card", m.start(), m.end())
    for m in _CREDENTIAL.finditer(text):
        value = m.group(1)
        if len(value) >= 4 and any(c.isdigit() for c in value):
            take("credential", m.start(1), m.end(1))
    return sorted(found, key=lambda f: (f.start, -f.end))


def mask(text: str) -> str:
    """`text` with each secret replaced by [redacted]; overlapping secrets become one."""
    spans: list[list[int]] = []
    for f in find(text):
        if spans and f.start <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], f.end)
        else:
            spans.append([f.start, f.end])
    parts, last = [], 0
    for start, end in spans:
        parts += [text[last:start], MASK]
        last = end
    return "".join(parts) + text[last:]


def _card(number: str, before: str) -> bool:
    digits = "".join(c for c in number if c.isdigit())
    return (
        _CARD_NETWORK.match(digits) is not None
        and _luhn(digits)
        and _NOT_A_CARD.search(before) is None
    )


def _luhn(digits: str) -> bool:
    values = [int(c) for c in reversed(digits)]
    total = sum(d if i % 2 == 0 else (d * 2 - 9 if d > 4 else d * 2) for i, d in enumerate(values))
    return total % 10 == 0


def _id_ok(number: str) -> bool:
    total = sum(int(d) * w for d, w in zip(number[:17], _ID_WEIGHTS, strict=True))
    return "10X98765432"[total % 11] == number[17].upper()
