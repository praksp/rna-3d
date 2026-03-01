# Publish to a private Git repository

The project is already a Git repo with an initial commit. To publish it as a **private** repository:

## Option 1: GitHub (web)

1. **Create a new repository on GitHub**
   - Go to [github.com/new](https://github.com/new)
   - Repository name: e.g. `kaggle-rna-3d-folding`
   - Choose **Private**
   - Do **not** add a README, .gitignore, or license (they already exist locally)

2. **Add the remote and push** (replace `YOUR_USERNAME` and `REPO_NAME` with your values):

   ```bash
   cd "/Users/prakash/ml-learning/cursor/kaggle rna 3d"
   git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

   If GitHub prompts for credentials, use a [Personal Access Token](https://github.com/settings/tokens) (not your password) for HTTPS.

## Option 2: GitHub CLI

If you install [GitHub CLI](https://cli.github.com/) and run `gh auth login`:

```bash
cd "/Users/prakash/ml-learning/cursor/kaggle rna 3d"
gh repo create kaggle-rna-3d-folding --private --source=. --push
```

## Option 3: GitLab / other host

Create a **private** repo on your host, then:

```bash
git remote add origin <your-repo-url>
git push -u origin main
```

---

**Note:** `data/`, `output/`, `viz/`, and `kaggle.json` are in `.gitignore` and are not pushed (keeps data and credentials local).
