import os
import shutil
import tempfile
import pytest
from email_reader import EmailReader, Email
from classifier import EmailClassifier, UNKNOWN, CRITICAL, SPAM, REQUESTS, MONITORING, INFO


# --- helpers ---

@pytest.fixture
def temp_inbox():
    tmpdir = tempfile.mkdtemp()
    inbox = os.path.join(tmpdir, "inbox")
    os.makedirs(inbox)
    yield inbox
    shutil.rmtree(tmpdir)


@pytest.fixture
def classifier():
    return EmailClassifier()


def write_email_file(inbox_dir, filename, content):
    path = os.path.join(inbox_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def make_email(subject="", sender="", body=""):
    return Email(filepath="test.txt", subject=subject, sender=sender, body=body, raw_text="")


# --- TestEmailReader ---

class TestEmailReader:
    def test_read_txt_letter(self, temp_inbox):
        content = "Subject: Тест\nFrom: user@corp.ru\n\nТело письма"
        write_email_file(temp_inbox, "mail_001.txt", content)

        emails = EmailReader().read_all(temp_inbox)

        assert len(emails) == 1
        assert emails[0].subject == "Тест"
        assert "user@corp.ru" in emails[0].sender

    def test_empty_file(self, temp_inbox):
        write_email_file(temp_inbox, "mail_empty.txt", "")

        emails = EmailReader().read_all(temp_inbox)

        assert len(emails) == 1
        assert isinstance(emails[0], Email)


# --- TestEmailClassifier ---

class TestEmailClassifier:
    def test_classify_critical(self, classifier):
        email = make_email(subject="Критический инцидент — сервер недоступен", body="Работа полностью остановлена")
        result = classifier.classify(email)
        assert result.category == CRITICAL

    def test_classify_spam(self, classifier):
        email = make_email(subject="Ваш аккаунт будет заблокирован", body="Подтвердите данные")
        result = classifier.classify(email)
        assert result.category == SPAM

    def test_classify_unknown(self, classifier):
        email = make_email(subject="Привет", body="Как дела?")
        result = classifier.classify(email)
        assert result.category == UNKNOWN

    @pytest.mark.parametrize("subject,expected", [
        ("Срочно! Система не отвечает", CRITICAL),
        ("Запрос доступа к VPN", REQUESTS),
        ("Ваш аккаунт будет заблокирован", SPAM),
        ("Привет", UNKNOWN),
    ])
    def test_parametrized(self, classifier, subject, expected):
        email = make_email(subject=subject, body="")
        result = classifier.classify(email)
        assert result.category == expected

    def test_unknown_for_new_type(self, classifier):
        email = make_email(subject="Вопрос про зарплату", body="Когда будут выплаты?")
        result = classifier.classify(email)
        assert result.category == UNKNOWN

    def test_empty_email_unknown(self, classifier):
        email = make_email("", "", "")
        result = classifier.classify(email)
        assert result.category == UNKNOWN

    def test_very_long_text_no_crash(self, classifier):
        email = make_email(body="a" * 5000)
        result = classifier.classify(email)
        assert result is not None