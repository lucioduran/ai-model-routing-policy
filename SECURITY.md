# Security

This is a local experimental tool, not an exposed inference service. Use trusted
fixtures and local model files. The CLI does not execute model repository code and
loads safetensors rather than pickle weights. Input files are loaded in memory;
untrusted input needs external size and resource limits. Report vulnerabilities
through the repository's GitHub security reporting options when available, or open
an issue without secrets or exploit details. No independent security audit is claimed.
