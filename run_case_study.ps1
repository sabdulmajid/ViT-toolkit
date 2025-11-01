#!/usr/bin/env pwsh
# Vision Transformer Case Study - Windows
# Runs ViT-Base attribution analysis on sample images

$ErrorActionPreference = "Stop"
Write-Host "=== ViT Attribution Case Study ===" -ForegroundColor Cyan

# Setup directories
$CaseStudyDir = "case_study_output"
$ImagesDir = "$CaseStudyDir/images"
New-Item -ItemType Directory -Force -Path $ImagesDir | Out-Null

# Download sample images (common ImageNet objects)
Write-Host "`nDownloading sample images..." -ForegroundColor Yellow
$Images = @(
    @{url="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/481px-Cat03.jpg"; name="cat.jpg"; class="Cat"},
    @{url="https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/640px-Cute_dog.jpg"; name="dog.jpg"; class="Dog"},
    @{url="https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Domestic_goat_kid_in_capeweed.jpg/440px-Domestic_goat_kid_in_capeweed.jpg"; name="goat.jpg"; class="Goat"}
)

foreach ($img in $Images) {
    $outPath = "$ImagesDir/$($img.name)"
    if (-not (Test-Path $outPath)) {
        Write-Host "  - Downloading $($img.class)..."
        Invoke-WebRequest -Uri $img.url -OutFile $outPath -UseBasicParsing
    }
}

# Run attribution methods on all images
Write-Host "`nRunning ViT-Base attribution analysis..." -ForegroundColor Yellow
$ImagePaths = (Get-ChildItem "$ImagesDir/*.jpg" | ForEach-Object { $_.FullName }) -join " "

$Methods = @("vanilla_gradients", "integrated_gradients", "gradcam")
foreach ($method in $Methods) {
    Write-Host "  - Running $method..."
    python scripts/run_attribution.py `
        --model-name vit_base_patch16_224 `
        --image-paths $ImagePaths.Split() `
        --method $method `
        --output-dir "$CaseStudyDir/$method" `
        --device cpu
}

Write-Host "`n=== Case Study Complete ===" -ForegroundColor Green
Write-Host "Results saved to: $CaseStudyDir/" -ForegroundColor Green
Write-Host "`nGenerated visualizations:" -ForegroundColor Cyan
Get-ChildItem -Path $CaseStudyDir -Recurse -Filter "*.png" | ForEach-Object {
    Write-Host "  - $($_.FullName)"
}
