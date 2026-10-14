import os
import sys

# Polyfill sys.set_int_max_str_digits / sys.get_int_max_str_digits if missing
# (e.g. Python 3.11.0rc1)
if not hasattr(sys, "set_int_max_str_digits"):
    def set_int_max_str_digits(maxdigits: int) -> None:
        pass
    sys.set_int_max_str_digits = set_int_max_str_digits

if not hasattr(sys, "get_int_max_str_digits"):
    def get_int_max_str_digits() -> int:
        return 4300
    sys.get_int_max_str_digits = get_int_max_str_digits

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "sse_api.tests.settings_django")
try:
    from langchain_community import document_loaders
    if not hasattr(document_loaders, "HTMLLoader"):
        from langchain_community.document_loaders.html_bs import BSHTMLLoader
        document_loaders.HTMLLoader = BSHTMLLoader
except Exception:
    pass

try:
    import django
    django.setup()
except Exception:
    pass
