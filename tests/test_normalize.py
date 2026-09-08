"""Tests for the deterministic normalisers.

These carry real weight: the reconciliation engine's verdicts are only as
trustworthy as the unit and period resolution underneath them. Several cases
below are lifted verbatim from the starter documents.
"""

from datetime import date

import pytest

from factlayer.normalize.entities import entity_key, same_entity
from factlayer.normalize.numbers import parse_number, plausible_grouping
from factlayer.normalize.periods import infer_fiscal_year_end, parse_period
from factlayer.normalize.units import build_quantity, magnitude_from_header


class TestNumbers:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("₹8,142 Cr", 8142.0),
            ("81,415.38", 81415.38),
            ("(4,516.08)", -4516.08),          # parenthesised negative
            ("₹(452) Cr", -452.0),
            ("1,23,456", 123456.0),            # Indian digit grouping
            ("6.4 per cent", 6.4),
            ("-2,491.86", -2491.86),
        ],
    )
    def test_values(self, text, expected):
        parsed = parse_number(text)
        assert parsed is not None
        assert parsed.value == pytest.approx(expected)

    def test_approximation_is_flagged(self):
        assert parse_number("about 740 Mn").approximate
        assert not parse_number("740 Mn").approximate

    def test_implausible_grouping_rejected(self):
        # Mangled table cells must not be silently turned into numbers.
        assert not plausible_grouping("1,23,45,6")
        assert plausible_grouping("1,234,567")
        assert plausible_grouping("12,34,567")


class TestUnits:
    def test_crore_and_million_reconcile(self):
        """The core case-1 mechanic: same fact, two unit systems."""
        deck = build_quantity("₹8,142 Cr")
        report = build_quantity(
            "81,415.38", unit_hint="(₹ in Million)",
            magnitude_hint=magnitude_from_header("(₹ in Million)"),
        )
        assert deck.canonical_unit == report.canonical_unit == "INR"
        rel = abs(deck.canonical_value - report.canonical_value) / abs(report.canonical_value)
        assert rel < 0.001

    def test_percent(self):
        q = build_quantity("6.4 per cent")
        assert q.kind == "percent" and q.canonical_value == pytest.approx(6.4)

    def test_negative_currency_from_header(self):
        q = build_quantity(
            "(4,516.08)", unit_hint="₹ in Million",
            magnitude_hint=magnitude_from_header("₹ in Million"),
        )
        assert q.canonical_value == pytest.approx(-4.51608e9)

    def test_no_cross_currency_conversion(self):
        inr = build_quantity("₹100 Cr")
        usd = build_quantity("USD 100 million")
        assert inr.canonical_unit != usd.canonical_unit
        assert not inr.comparable_to(usd)


class TestPeriods:
    @pytest.mark.parametrize(
        "text,label,start,end",
        [
            ("FY24", "FY2024", date(2023, 4, 1), date(2024, 3, 31)),
            ("2023-24", "FY2024", date(2023, 4, 1), date(2024, 3, 31)),
            ("FY2025/26", "FY2026", date(2025, 4, 1), date(2026, 3, 31)),
            ("Q4 FY24", "Q4 FY2024", date(2024, 1, 1), date(2024, 3, 31)),
            ("H2:2024-25", "H2 FY2025", date(2024, 10, 1), date(2025, 3, 31)),
            ("CY2024", "CY2024", date(2024, 1, 1), date(2024, 12, 31)),
        ],
    )
    def test_windows(self, text, label, start, end):
        p = parse_period(text)
        assert p is not None
        assert (p.label, p.start, p.end) == (label, start, end)

    def test_q2_label_collision(self):
        """'Q2 of FY25' and '2025Q2' both read as Q2 and are nine months apart.

        This exact pair appears in the macro starter data with a 6x value gap;
        matching on labels would report a spurious contradiction.
        """
        fiscal = parse_period("Q2 of FY25")
        calendar = parse_period("2025Q2")
        assert fiscal.kind == "fiscal_quarter"
        assert calendar.kind == "calendar_quarter"
        assert fiscal.start == date(2024, 7, 1)
        assert calendar.start == date(2025, 4, 1)
        assert fiscal.overlaps(calendar) is False
        assert fiscal.same_window(calendar) is False

    def test_fiscal_year_end_is_inferred_not_assumed(self):
        assert infer_fiscal_year_end("for the year ended March 31, 2024") == 3
        assert infer_fiscal_year_end("fiscal year ended September 30, 2024") == 9
        assert infer_fiscal_year_end("no statement here") is None

    def test_non_march_fiscal_year_resolves_differently(self):
        march = parse_period("FY24", fy_end_month=3)
        september = parse_period("FY24", fy_end_month=9)
        assert march.end == date(2024, 3, 31)
        assert september.end == date(2024, 9, 30)


class TestEntities:
    def test_corporate_suffixes_folded(self):
        assert entity_key("Delhivery Limited") == entity_key("Delhivery Ltd.") == "delhivery"

    def test_honorifics_folded(self):
        assert entity_key("Mr. Suvir Suren Sujan") == "suvir suren sujan"

    def test_same_entity_variants(self):
        assert same_entity("Delhivery Limited", "Delhivery")
        assert same_entity("Mr. Suvir Suren Sujan", "S. S. Sujan")
        assert not same_entity("Delhivery Limited", "Blue Dart Express Limited")
