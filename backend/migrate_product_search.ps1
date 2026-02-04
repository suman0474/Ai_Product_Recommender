# Product Search Workflow Migration Script
# Removes duplicate agentic/workflows/product_search and updates all imports

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Product Search Workflow Migration" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$backendPath = "D:\AI PR\AIPR\backend"
$wrapperPath = Join-Path $backendPath "agentic\workflows\product_search"
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$backupPath = Join-Path $backendPath ".backup\product_search_$timestamp"

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Backend Path: $backendPath"
Write-Host "  Wrapper Path: $wrapperPath"
Write-Host "  Backup Path:  $backupPath"
Write-Host ""

# Step 1: Create Backup
Write-Host "[1/5] Creating backup..." -ForegroundColor Yellow
if (Test-Path $wrapperPath) {
    $backupDir = Split-Path $backupPath
    if (!(Test-Path $backupDir)) {
        New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
    }
    Copy-Item -Path $wrapperPath -Destination $backupPath -Recurse -Force
    Write-Host "  [OK] Backup created: $backupPath" -ForegroundColor Green
} else {
    Write-Host "  [SKIP] Wrapper folder not found." -ForegroundColor Yellow
}

# Step 2: Update Import in deep_agentic_workflow.py
Write-Host ""
Write-Host "[2/5] Updating deep_agentic_workflow.py..." -ForegroundColor Yellow
$file1 = Join-Path $backendPath "agentic\deep_agent\workflows\deep_agentic_workflow.py"
if (Test-Path $file1) {
    $content = Get-Content $file1 -Raw -Encoding UTF8
    $originalContent = $content
    
    # Replace imports
    $content = $content -replace 'from agentic\.workflows\.product_search import ValidationTool', 'from product_search_workflow import ValidationTool'
    $content = $content -replace 'from agentic\.workflows\.product_search import AdvancedParametersTool', 'from product_search_workflow import AdvancedParametersTool'
    $content = $content -replace 'from agentic\.workflows\.product_search import VendorAnalysisTool', 'from product_search_workflow import VendorAnalysisTool'
    $content = $content -replace 'from agentic\.workflows\.product_search import RankingTool', 'from product_search_workflow import RankingTool'
    
    if ($content -ne $originalContent) {
        Set-Content $file1 -Value $content -Encoding UTF8 -NoNewline
        Write-Host "  [OK] Updated: deep_agentic_workflow.py (4 imports changed)" -ForegroundColor Green
    } else {
        Write-Host "  [SKIP] No changes needed in deep_agentic_workflow.py" -ForegroundColor Gray
    }
} else {
    Write-Host "  [ERROR] File not found: $file1" -ForegroundColor Red
}

# Step 3: Update Import in main_api.py
Write-Host ""
Write-Host "[3/5] Updating main_api.py..." -ForegroundColor Yellow
$file2 = Join-Path $backendPath "agentic\infrastructure\api\main_api.py"
if (Test-Path $file2) {
    $content = Get-Content $file2 -Raw -Encoding UTF8
    $originalContent = $content
    
    # Replace imports
    $content = $content -replace 'from agentic\.workflows\.product_search import AdvancedParametersTool', 'from product_search_workflow import AdvancedParametersTool'
    $content = $content -replace 'from agentic\.workflows\.product_search import SalesAgentTool', 'from product_search_workflow import SalesAgentTool'
    
    if ($content -ne $originalContent) {
        Set-Content $file2 -Value $content -Encoding UTF8 -NoNewline
        Write-Host "  [OK] Updated: main_api.py (3 imports changed)" -ForegroundColor Green
    } else {
        Write-Host "  [SKIP] No changes needed in main_api.py" -ForegroundColor Gray
    }
} else {
    Write-Host "  [ERROR] File not found: $file2" -ForegroundColor Red
}

# Step 4: Remove Wrapper Folder
Write-Host ""
Write-Host "[4/5] Removing wrapper folder..." -ForegroundColor Yellow
if (Test-Path $wrapperPath) {
    Remove-Item -Path $wrapperPath -Recurse -Force
    Write-Host "  [OK] Removed: $wrapperPath" -ForegroundColor Green
} else {
    Write-Host "  [SKIP] Wrapper folder already removed or not found" -ForegroundColor Yellow
}

# Step 5: Validation
Write-Host ""
Write-Host "[5/5] Validation..." -ForegroundColor Yellow

# Check for remaining old imports
Write-Host "  Checking for remaining old imports..." -ForegroundColor Gray
$searchPath = Join-Path $backendPath "agentic"
$pattern = "from agentic\.workflows\.product_search"
$remainingCount = 0

if (Test-Path $searchPath) {
    $pyFiles = Get-ChildItem -Path $searchPath -Filter "*.py" -Recurse -ErrorAction SilentlyContinue
    foreach ($file in $pyFiles) {
        $matches = Select-String -Path $file.FullName -Pattern $pattern -ErrorAction SilentlyContinue
        if ($matches) {
            Write-Host "    [WARN] Found in: $($file.FullName)" -ForegroundColor Red
            $remainingCount += $matches.Count
        }
    }
}

if ($remainingCount -eq 0) {
    Write-Host "  [OK] No remaining old imports found" -ForegroundColor Green
} else {
    Write-Host "  [WARN] Found $remainingCount remaining old import(s)" -ForegroundColor Red
}

# Verify wrapper folder is gone
if (!(Test-Path $wrapperPath)) {
    Write-Host "  [OK] Wrapper folder successfully removed" -ForegroundColor Green
} else {
    Write-Host "  [WARN] Wrapper folder still exists" -ForegroundColor Red
}

# Verify standalone package exists
$standalonePath = Join-Path $backendPath "product_search_workflow"
if (Test-Path $standalonePath) {
    Write-Host "  [OK] Standalone package exists" -ForegroundColor Green
} else {
    Write-Host "  [ERROR] Standalone package not found!" -ForegroundColor Red
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Migration Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Summary:" -ForegroundColor Yellow
Write-Host "  Files Updated: 2"
Write-Host "  Imports Changed: ~7"
Write-Host "  Folder Removed: 1"
Write-Host "  Backup Location: $backupPath"
Write-Host ""

Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Review changes: git diff" -ForegroundColor White
Write-Host "  2. Run tests: pytest backend/tests/ -v" -ForegroundColor White
Write-Host "  3. Manual testing of product search workflow" -ForegroundColor White
Write-Host "  4. Commit changes" -ForegroundColor White
Write-Host ""
