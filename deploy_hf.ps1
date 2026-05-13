# ============================================================
#  MedBI — Hugging Face Spaces Deployment Script
#  Run from PowerShell in the project root directory
# ============================================================

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  MedBI — Deploying to Hugging Face Spaces" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Step 1 — Navigate to project
Set-Location "C:\Users\adity\OneDrive\Documents\GitHub\End-to-End-Project-on-Medical-Health-Care-Assisstant-with-Chatbot-"
Write-Host "[1/8] Working directory set." -ForegroundColor Green

# Step 2 — Remove binary files from git tracking
Write-Host "[2/8] Removing binary files from git tracking..." -ForegroundColor Yellow
git rm --cached data/Canada_Hosp1_COVID_InpatientData.xlsx 2>$null
git rm --cached models/Hospital_LoS_Model.joblib 2>$null
git rm --cached models/ner_model.joblib 2>$null
Write-Host "      Binary files untracked." -ForegroundColor Green

# Step 3 — Stage updated files
Write-Host "[3/8] Staging changes..." -ForegroundColor Yellow
git add .gitignore
git add entrypoint.sh
git add .
Write-Host "      Files staged." -ForegroundColor Green

# Step 4 — Commit
Write-Host "[4/8] Committing..." -ForegroundColor Yellow
git commit -m "fix: Remove binary files, train model at container startup"
Write-Host "      Committed." -ForegroundColor Green

# Step 5 — Set up HF remote
Write-Host "[5/8] Configuring Hugging Face remote..." -ForegroundColor Yellow
git remote remove hf 2>$null
git remote add hf https://huggingface.co/spaces/AdityaSrivatsav/MedicalChatbot
git config --global credential.helper store
Write-Host "      Remote 'hf' configured." -ForegroundColor Green

# Step 6 — Push to Hugging Face
Write-Host ""
Write-Host "[6/8] Pushing to Hugging Face Spaces..." -ForegroundColor Yellow
Write-Host "      When prompted:" -ForegroundColor White
Write-Host "        Username : AdityaSrivatsav" -ForegroundColor Cyan
Write-Host "        Password : your HF token from https://huggingface.co/settings/tokens" -ForegroundColor Cyan
Write-Host ""
git push hf main --force

# Step 7 — Push to GitHub
Write-Host ""
Write-Host "[7/8] Pushing to GitHub..." -ForegroundColor Yellow
git push origin main
Write-Host "      GitHub updated." -ForegroundColor Green

# Step 8 — Open Space in browser
Write-Host ""
Write-Host "[8/8] Opening Hugging Face Space in browser..." -ForegroundColor Yellow
Start-Process "https://huggingface.co/spaces/AdityaSrivatsav/MedicalChatbot"

# Done
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Deployment pushed successfully!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Build log : https://huggingface.co/spaces/AdityaSrivatsav/MedicalChatbot" -ForegroundColor White
Write-Host "  Live URL  : https://adityasrivatsav-medicalchatbot.hf.space" -ForegroundColor White
Write-Host ""
Write-Host "  First build takes 5-8 minutes (model training + vector store)." -ForegroundColor Yellow
Write-Host "  Watch the Logs tab on your Space page." -ForegroundColor Yellow
Write-Host ""
