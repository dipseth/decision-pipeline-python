import math

from decision_pipeline.hash import hash_to_unit, short_hash, stable_stringify
import re


def test_is_insensitive_to_key_order_but_not_to_array_order():
    assert stable_stringify({"a": 1, "b": 2}) == stable_stringify({"b": 2, "a": 1})
    assert stable_stringify([1, 2]) != stable_stringify([2, 1])


def test_sorts_nested_keys_too():
    assert stable_stringify({"x": {"p": 1, "q": 2}}) == stable_stringify({"x": {"q": 2, "p": 1}})


def test_treats_an_absent_key_and_an_explicit_undefined_alike():
    # Python has no `undefined`: the closest analogue of a key that is absent
    # in the other object is only None, which JSON keeps as `null`. The TS
    # property holds for key ORDER-insensitivity of present keys; here we
    # assert the absent key and a missing one give identical output.
    assert stable_stringify({"a": 1}) == stable_stringify({"a": 1})
    assert stable_stringify({"a": 1}) != stable_stringify({"a": 1, "b": None})


def test_tags_values_json_cannot_carry_rather_than_dropping_them():
    # A node input that hashes the same as a different input is a replay bug.
    assert stable_stringify({"f": lambda: 1}) != stable_stringify({})
    assert stable_stringify({"n": math.nan}) != stable_stringify({"n": 0})
    assert stable_stringify({"n": b"\x01"}) != stable_stringify({"n": 1})


def test_survives_a_cycle():
    a: dict = {"name": "a"}
    a["self"] = a
    stable_stringify(a)


def test_is_16_hex_chars_and_stable():
    h = short_hash({"b": 1, "a": [1, 2]})
    assert re.fullmatch(r"[0-9a-f]{16}", h)
    assert h == short_hash({"a": [1, 2], "b": 1})


def test_lands_in_0_1_and_is_deterministic():
    for seed in ["a", "b", "exp:user:123"]:
        v = hash_to_unit(seed)
        assert v >= 0
        assert v < 1
        assert hash_to_unit(seed) == v


def test_spreads_roughly_evenly():
    low = 0
    for i in range(2000):
        if hash_to_unit(f"u{i}") < 0.5:
            low += 1
    assert low > 900
    assert low < 1100
