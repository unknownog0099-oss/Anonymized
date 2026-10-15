# Upload the anonymous repository

Use this folder's contents as the repository root. Use an anonymous GitHub
account/repository owner if reviewer-facing anonymity is required: a personal
account or repository URL can identify you even when every file is anonymous.

Create an empty `provguard` repository on that account, then run these commands
in this folder (replace only the anonymous account placeholder):

```powershell
git init -b main
git config --local user.name "Anonymous Authors"
git config --local user.email "anonymous@users.noreply.github.com"
git config --local commit.gpgsign false
git config --local core.autocrlf false
git add .
git status --short
git commit -m "Release anonymous ProvGuard code and experiment artifacts"
git remote add origin https://github.com/ANONYMOUS_ACCOUNT/provguard.git
git push -u origin main
```

The repo has no inherited Git history. The commit identity above is local to
this repo. Authentication still uses the account that owns the remote.
Do not put credentials or access tokens in the URL or tracked files.

Use the Git CLI for this release. The persisted sparse leakage representation
is about 84 MiB; it is below GitHub's 100 MiB blocking limit but exceeds the
25 MiB browser upload limit and triggers the ordinary 50 MiB size warning.
[GitHub's file-size documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

The ZIP beside this folder is for transfer/extraction. Upload the extracted
contents, rather than adding the ZIP to the repository. The original workspace,
environments, model weights, logs, caches, temporary files, and old Git metadata
are not part of this release.
