"""Path rules for MR patches and their source evidence."""

from collections.abc import Iterable
from fnmatch import fnmatchcase

# PR-Agent 2f7d159 default language filter and generated artifacts.
LOCK_FILES = {
    "package-lock.json",
    "pnpm-lock.yaml",
    "go.sum",
    ".terraform.lock.hcl",
    "packages.lock.json",
    "Package.resolved",
}
GENERATED_SUFFIXES = (".min.js", ".min.css", ".js.map", ".ts.map", ".css.map")
EXCLUDED_EXTENSIONS = frozenset(
    """
    app bin bmp bz2 class csv dat db dll dylib egg eot exe gif gitignore glif gz
    ico jar jpeg jpg lo lock log mp3 mp4 nar o ogg otf p pdf png pickle pkl pyc
    pyd pyo rkt so ss svg tar tgz tsv ttf war webm woff woff2 xz zip zst snap lockb
    """.split()
)
GENERATED_FRAMEWORK_GLOBS = {
    "protobuf": (
        "**/*.pb.go",
        "**/*.pb.cc",
        "**/*_pb2.py",
        "**/*.pb.swift",
        "**/*.pb.rb",
        "**/*.pb.php",
        "**/*.pb.h",
    ),
    "openapi": ("**/__generated__/**", "**/openapi_client/**", "**/openapi_server/**"),
    "swagger": ("**/swagger.json", "**/swagger.yaml"),
    "graphql": ("**/*.graphql.ts", "**/*.generated.ts", "**/*.graphql.js"),
    "grpc_python": ("**/*_grpc.py",),
    "grpc_java": ("**/*Grpc.java",),
    "grpc_csharp": ("**/*Grpc.cs",),
    "grpc_typescript": ("**/*_grpc.ts", "**/*_grpc.js"),
    "go_gen": ("**/*_gen.go", "**/*generated.go"),
}


def excluded_from_review(path: str, *, frameworks: Iterable[str] = ()) -> bool:
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    return (
        path.startswith("vendor/")
        or name in LOCK_FILES
        or name.lower().endswith(GENERATED_SUFFIXES)
        or path.rsplit(".", 1)[-1].lower() in EXCLUDED_EXTENSIONS
        or any(
            # Prefix a slash so the leading **/ also matches root files.
            fnmatchcase(f"/{path}", pattern)
            for framework in frameworks
            for pattern in GENERATED_FRAMEWORK_GLOBS.get(framework, ())
        )
    )
