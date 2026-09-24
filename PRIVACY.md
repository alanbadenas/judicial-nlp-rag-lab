# Privacy and data-handling policy

This public repository contains **no real judicial case records** and no derived artifacts produced from them.

Excluded by design:

- raw or OCR-extracted case documents;
- case numbers and party identifiers;
- CPF/CNPJ, e-mail addresses, phone numbers, addresses, and professional registration identifiers;
- API keys and local `.env` files;
- embeddings, trained models, indexes, cluster assignments, and plots derived from non-public working data;
- notebook outputs or logs that may reproduce source text.

The repository ships only synthetic examples. The deterministic redaction layer in `src/privacy.py` is a first-pass safeguard and is **not** a guarantee of complete anonymization for real-world legal documents. Production use would require a formal privacy review, stronger entity detection, validation, access control, and a documented data-retention policy.
