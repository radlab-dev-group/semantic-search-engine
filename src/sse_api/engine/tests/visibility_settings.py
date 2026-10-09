from sse_api.authorization.tests.settings import *
import sys
from unittest.mock import MagicMock

sys.modules["sentence_transformers"] = MagicMock()
sys.modules["sentence_transformers.cross_encoder"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["radlab_data.text.reader"] = MagicMock()
sys.modules["radlab_data.text.document"] = MagicMock()

INSTALLED_APPS = INSTALLED_APPS + ["system", "data", "engine", "chat"]
MIGRATION_MODULES = {app: None for app in ("authorization", "system", "data", "engine", "chat")}
DEFAULT_APP_LANGUAGE = "en"
USE_TZ = True
REST_FRAMEWORK = {"DEFAULT_AUTHENTICATION_CLASSES": [], "DEFAULT_PERMISSION_CLASSES": []}