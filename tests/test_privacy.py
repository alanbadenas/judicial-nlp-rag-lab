from src.privacy import redact_deterministic


def test_structured_identifiers_are_redacted():
    text = (
        "Processo 1234567-89.2024.8.13.0001, CPF 123.456.789-00, "
        "CNPJ 12.345.678/0001-90, email pessoa@example.com, "
        "telefone (41) 99999-0000 em 24/09/2026."
    )
    result = redact_deterministic(text)
    assert "1234567-89.2024.8.13.0001" not in result
    assert "123.456.789-00" not in result
    assert "12.345.678/0001-90" not in result
    assert "pessoa@example.com" not in result
    assert "99999-0000" not in result
    assert "24/09/2026" not in result
    assert "[ANON_PROCESS]" in result
    assert "[ANON_CPF]" in result
    assert "[ANON_CNPJ]" in result
    assert "[ANON_EMAIL]" in result
