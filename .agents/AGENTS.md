# Agent Rules & Behavioral Guidelines

## File Deletion Protocol
- **DO NOT** arbitrarily use OS-level deletion commands (like `rm`, `del`, `os.remove`) to delete files within the workspace, even if they appear to be junk or temporary artifacts.
- **ALWAYS** perform a proper inventory check before proposing a file for deletion. This includes checking `git log`, `git status`, and project references/imports.
- **ALWAYS** present the findings (with the git log history) to the user and request explicit approval before executing any file deletion.
