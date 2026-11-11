"""S5b: secrets are found by rules in code, never by asking the model (design §5)."""

import time

import pytest

from lite_mem.sensitive import find, mask


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("银行卡密码是 482915", "credential"),
        ("我的密码：hunter2", "credential"),
        ("Bank card PIN: 4829", "credential"),
        ("User's bank card password is 482915", "credential"),
        ("the wifi passcode is tiger-77", "credential"),
        ("验证码 683921", "credential"),
        ("我的 PIN 是 4829 1038", "credential"),  # a value split by spaces (Codex S5b)
        ("password is tiger 77", "credential"),
        ("card number 4111 1111 1111 1111", "card"),
        ("卡号 6222-0212-3456-7890-128", "card"),  # 19 digits, Luhn-valid
        ("身份证号 11010519491231002X", "national_id"),
        ("key sk-proj-Ab3dEf6hIj9kLm2nOp5q", "api_key"),
        ("AKIAIOSFODNN7EXAMPLE was leaked", "api_key"),
        ("ghp_abcdefghijklmnopqrstuvwxyz0123456789", "api_key"),
        ("-----BEGIN RSA PRIVATE KEY-----", "api_key"),
    ],
)
def test_secrets_are_found(text: str, kind: str) -> None:
    assert [f.kind for f in find(text)] == [kind]


@pytest.mark.parametrize(
    "text",
    [
        "住在上海，2026 年搬家",
        "电话 13812345678，分机 4021",
        "order 1234 5678 9012 3456 shipped",  # 16 digits, fails the Luhn check
        "order 4111 1111 1111 1111 shipped",  # Luhn-valid, but an order id (Codex S5b)
        "订单号 6222021234567890128 已发货",
        "tracking 9400111202555842761326",  # no card network starts with 94
        "Luhn-valid 1000000000000008 but no card network prefix",
        "身份证号 110105194912310021",  # wrong check digit
        "Emma forgot her password yesterday",  # the word alone, no value
        "密码学是她的研究方向",  # "cryptography", no value
        "Works the night shift, 22:00 to 06:00",
    ],
)
def test_ordinary_text_is_left_alone(text: str) -> None:
    assert find(text) == []


def test_many_matches_stay_linear() -> None:
    """Overlap tracking must not rescan earlier findings: 20 000 matches take well under a
    second when linear, and minutes when each match compares with every earlier one."""
    text = "card 4111 1111 1111 1111 " * 20_000
    started = time.perf_counter()
    assert len(find(text)) == 20_000
    assert time.perf_counter() - started < 5


def test_mask_hides_only_the_secret_value() -> None:
    assert mask("银行卡密码是 482915，记一下") == "银行卡密码是 [redacted]，记一下"
    assert mask("card 4111 1111 1111 1111 expires soon") == "card [redacted] expires soon"
    assert mask("no secrets here") == "no secrets here"
    assert mask("我的 PIN 是 4829 1038，别忘了") == "我的 PIN 是 [redacted]，别忘了"
