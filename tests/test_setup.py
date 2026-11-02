"""First-run / setup plumbing: config file, folder resolution, verify command."""

import os
import shutil
import stat

from typer.testing import CliRunner

from djtagger import cli, config, models

runner = CliRunner()
FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def test_config_round_trip_is_private(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_FILE", str(tmp_path / "djtagger" / "config.json"))
    assert config.load_user_config() == {}
    config.save_user_config({"music_path": "/music"})
    config.save_user_config({"lastfm_api_key": "abc"})  # merges, keeps music_path
    assert config.load_user_config() == {"music_path": "/music", "lastfm_api_key": "abc"}
    assert stat.S_IMODE(os.stat(config.CONFIG_FILE).st_mode) == 0o600


def test_clean_path_answer_handles_drag_and_drop():
    assert cli._clean_path_answer("'/Volumes/My Music/' ") == "/Volumes/My Music"
    assert cli._clean_path_answer("/Volumes/My\\ Music") == "/Volumes/My Music"
    assert cli._clean_path_answer("~/Music").startswith(os.path.expanduser("~"))


def test_command_without_folder_asks_instead_of_using_cwd(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_FILE", str(tmp_path / "none.json"))
    monkeypatch.delenv("DJTAGGER_MUSIC_PATH", raising=False)
    result = runner.invoke(cli.app, ["verify", ""])
    assert result.exit_code == 1
    assert "no folder given" in result.output


def test_saved_folder_is_used(tmp_path, monkeypatch):
    music = tmp_path / "music"
    music.mkdir()
    monkeypatch.setattr(config, "CONFIG_FILE", str(tmp_path / "c.json"))
    monkeypatch.delenv("DJTAGGER_MUSIC_PATH", raising=False)
    config.save_user_config({"music_path": str(music)})
    result = runner.invoke(cli.app, ["verify", ""])
    assert result.exit_code == 0
    assert "Total tracks: 0" in result.output


def test_verify_flags_untagged_file(tmp_path):
    shutil.copy(os.path.join(FIXTURES, "silent.flac"), tmp_path / "A - B.flac")
    result = runner.invoke(cli.app, ["verify", str(tmp_path)])
    assert result.exit_code == 1
    assert "no/invalid role" in result.output


def test_missing_models_reported(tmp_path):
    assert models.missing_models(str(tmp_path), required_only=True)
    try:
        models.require_models(str(tmp_path))
    except models.ModelsMissingError as ex:
        assert "djtagger setup" in str(ex)
    else:
        raise AssertionError("expected ModelsMissingError")


def test_tag_without_models_noninteractive_gives_setup_hint(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "MODEL_DIR", str(tmp_path / "models"))
    result = runner.invoke(cli.app, ["tag", str(tmp_path)])  # CliRunner stdin is not a TTY
    assert result.exit_code == 1
    assert "djtagger setup" in result.output
