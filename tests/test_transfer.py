"""Cross-card transfer experiment on the committed FLA / mk3 sweeps (opceil.model.transfer)."""
from opceil.model import transfer


def test_measured_sibling_ratios_beat_spec_and_model_ratios():
    for a, b in (transfer.CARDS, transfer.CARDS[::-1]):
        r = transfer.run(a, b)
        k, g = r["kda_fla"], r["gdn_fla"]
        best_spec = min(k["spec: compute peak ratio"]["median_pct"], k["spec: HBM bandwidth ratio"]["median_pct"])
        assert k["FLA GDN ratio, whole op (other operator)"]["median_pct"] < best_spec / 2
        assert k["FLA GDN ratio, whole op (other operator)"]["median_pct"] < \
            k["tool model ratio (T_split p82, per shape)"]["median_pct"]
        assert g["FLA KDA ratio, whole op"]["median_pct"] < 12.0
