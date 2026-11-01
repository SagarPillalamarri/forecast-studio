# Upload only the necessary project files to GitHub

Create a new **empty repository** on GitHub. Do not add a README, license, or .gitignore during repository creation because this project already supplies its own files.

Open the VS Code terminal in this project directory and run:

```bash
git init
git branch -M main
git add .
git status --short
git commit -m "Add Forecast Studio dashboard"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Replace `YOUR_USERNAME` and `YOUR_REPOSITORY` with your GitHub details. Authenticate with GitHub if prompted. If `origin` already exists, use `git remote set-url origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git` instead of `git remote add origin`.

The root `.gitignore` excludes `.venv`, `node_modules`, compiled frontend output, raw downloads, caches, generated PDFs, previews, archives, and local secrets. Cleaned CSVs, source code, tests, dependency files, and setup instructions are included.

Before committing, `git status --short` lets you review what will be uploaded. `.gitignore` applies to untracked files; it does not automatically untrack files committed in an older repository.

For later changes:

```bash
git add .
git commit -m "Update Forecast Studio"
git push
```
