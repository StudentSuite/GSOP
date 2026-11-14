import pytest

from gsop import doctor
from gsop.earth import gee_auth


class FakeEE:
    def __init__(self, fail=None):
        self.fail, self.calls = fail, []

    def Initialize(self, **kwargs):  # noqa: N802
        self.calls.append(kwargs)
        if self.fail:
            raise self.fail


def test_project_comes_from_env_then_config(monkeypatch):
    monkeypatch.delenv(gee_auth.ENV_PROJECT, raising=False)
    assert gee_auth.ee_project() is None  # config/run.yaml ships with no project
    monkeypatch.setenv(gee_auth.ENV_PROJECT, "  my-proj ")
    assert gee_auth.ee_project() == "my-proj"


def test_no_project_id_is_committed():
    from pathlib import Path

    run = (Path(__file__).resolve().parents[1] / "config" / "run.yaml").read_text(encoding="utf-8")
    assert "project: null" in run


def test_ensure_ee_success_passes_project(monkeypatch):
    fake = FakeEE()
    monkeypatch.setattr(gee_auth, "_import_ee", lambda: fake)
    assert gee_auth.ensure_ee(project="p1") is fake
    assert fake.calls == [{"project": "p1"}]


def test_ensure_ee_failure_prints_steps_and_exits_once(monkeypatch, capsys):
    fake = FakeEE(fail=RuntimeError("Please authorize access\nmore"))
    monkeypatch.setattr(gee_auth, "_import_ee", lambda: fake)
    with pytest.raises(SystemExit) as exc:
        gee_auth.ensure_ee(project="p1")
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "earthengine authenticate" in err and "GSOP_EE_PROJECT" in err and "Please authorize access" in err
    assert len(fake.calls) == 1  # never loops


def test_check_ee_when_package_missing(monkeypatch):
    def boom():
        raise ImportError("no ee")

    monkeypatch.setattr(gee_auth, "_import_ee", boom)
    ok, reason = gee_auth.check_ee()
    assert not ok and "cannot be imported" in reason


def test_doctor_reports_authenticated_and_hides_project(monkeypatch):
    monkeypatch.setattr(gee_auth, "_import_ee", lambda: FakeEE())
    monkeypatch.setenv(gee_auth.ENV_PROJECT, "secret-proj")
    info = doctor.collect()
    assert info["earthengine"]["authenticated"] is True
    text = doctor.render(info)
    assert "earthengine:   authenticated" in text
    assert "secret-proj" not in text and "secret-proj" not in str(info)


def test_doctor_not_authenticated_has_no_traceback(monkeypatch):
    monkeypatch.setattr(gee_auth, "_import_ee", lambda: FakeEE(fail=RuntimeError("nope")))
    text = doctor.render(doctor.collect())
    assert "NOT authenticated" in text and "Traceback" not in text
