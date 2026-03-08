# Push this repo to your GitHub

Follow these steps to push the MedGemma web app to **your** GitHub (the current `origin` is Hugging Face, which is read-only for you).

## Step 1: Create a new repo on GitHub

1. Go to [https://github.com/new](https://github.com/new).
2. Choose a name (e.g. `medgemma-webapp`).
3. Leave it **empty** (no README, no .gitignore).
4. Create the repo and copy the repo URL (e.g. `https://github.com/YOUR_USERNAME/medgemma-webapp.git`).

## Step 2: Add your GitHub repo as a remote and push

Open a terminal in this folder and run (replace with your URL):

```bash
cd "c:\Users\nitim\Downloads\medgama1\medgemma-1.5-4b-it"

# Add your GitHub repo as remote "github"
git remote add github https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push the branch (no large model files; under GitHub limits)
git push -u github github-main:main
```

Use your actual GitHub username and repo name in the URL.

## Step 3: Done

Your code is on GitHub. The large `.safetensors` model files are **not** in the repo (GitHub limit). Anyone who clones should download them from [Hugging Face](https://huggingface.co/google/medgemma-1.5-4b-it) and place them in the project folder—see **WEBAPP_README.md**.

## If you already added "github" remote

If you added the remote before, set the URL:

```bash
git remote set-url github https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u github github-main:main
```
