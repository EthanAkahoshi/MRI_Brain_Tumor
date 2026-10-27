# "v2" preprocessing was adopted as the shared model.preprocess(); re-exported so
# train.py --prep v2 and its caches keep working unchanged.
from model import preprocess as preprocess_v2  # noqa: F401
