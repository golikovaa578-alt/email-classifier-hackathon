import os
import shutil
import tempfile
import pytest

from src.reader import read_email
from src.classifier import classify


@pytest.fixture
def temp_inbox():
    tmpdir = tempfile.mkdtemp()
    inbox = os.path.join(tmpdir, "inbox")
    os.makedirs(inbox)
    yield inbox
    shutil.rmtree(tmpdir)


def write_email_file(inbox_dir, filename, content):
    path = os.path.join(inbox_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


class TestEmailReader:
    def test_read_txt_letter(self, temp_inbox):
        content = "Subject: Тест\nFrom: user@corp.ru\n\nТело письма"
        filepath = write_email_file(temp_inbox, "mail_001.txt", content)

        result = read_email(filepath)

        assert result["subject"] == "Тест"
        assert "user@corp.ru" in result["from_addr"]

    def test_empty_file(self, temp_inbox):
        filepath = write_email_file(temp_inbox, "mail_empty.txt", "")

        result = read_email(filepath)

        assert isinstance(result, dict)


class TestEmailClassifier:
    def test_classify_critical(self):
        email_data = {
            "subject": "Критический инцидент — сервер недоступен",
            "body": "Работа полностью остановлена",
        }
        result = classify(email_data)
        assert result == "it_issue"

    def test_classify_spam(self):
        email_data = {
            "subject": "Ваш аккаунт будет заблокирован",
            "body": "Подтвердите данные",
        }
        result = classify(email_data)
        assert result == "other"

    def test_classify_unknown(self):
        email_data = {"subject": "Привет", "body": "Как дела?"}
        result = classify(email_data)
        assert result == "other"

    @pytest.mark.parametrize(
        "subject,expected",
        [
            ("Срочно! Система не работает", "urgent"),
            ("Запрос доступа к VPN", "it_issue"),
            ("Рекламная акция", "spam"),
            ("Привет", "other"),
        ],
    )
    def test_parametrized(self, subject, expected):
        email_data = {"subject": subject, "body": ""}
        result = classify(email_data)
        assert result == expected

    def test_unknown_for_new_type(self, classifier):
        email = make_email(subject="Вопрос про зарплату", body="Когда будут выплаты?")
        result = classifier.classify(email)
        assert result.category == UNKNOWN

    def test_empty_email_unknown(self, classifier):
        email = make_email("", "", "")
        result = classifier.classify(email)
        assert result.category == UNKNOWN

    def test_very_long_text_no_crash(self, classifier):
        long_body = "a" * 5000
        email = make_email(body=long_body)
        result = classifier.classify(email)
        assert result is not None
