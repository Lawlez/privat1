try:
    from packaging.version import parse as parse_version
except ImportError:
    # Fallback for very minimal environments
    def parse_version(v):
        import re
        return [int(x) if x.isdigit() else x for x in re.findall(r"(\d+|[a-zA-Z]+)", v)]
