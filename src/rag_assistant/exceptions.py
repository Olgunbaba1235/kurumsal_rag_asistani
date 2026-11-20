"""Uygulamaya özgü hata hiyerarşisi."""


class RAGError(Exception):
    """Tüm uygulama hatalarının temel sınıfı."""


class UnsupportedFileTypeError(RAGError):
    """Desteklenmeyen dosya uzantısı."""


class DocumentLoadError(RAGError):
    """Doküman okunamadı ya da metin çıkarılamadı."""


class ConfigurationError(RAGError):
    """Eksik ya da hatalı yapılandırma (ör. API anahtarı yok)."""


class LLMError(RAGError):
    """Dil modeli / embedding servisinden hata döndü."""
