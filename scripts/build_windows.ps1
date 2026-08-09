# Build the Windows desktop package with PyInstaller.
# Run from repo root on Windows:
#   powershell -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "==> Installing package and build dependencies"
python -m pip install --upgrade pip
python -m pip install -e ".[build]"

$Dist = Join-Path $Root "dist"
$Build = Join-Path $Root "build"
if (Test-Path $Dist) { Remove-Item -Recurse -Force $Dist }
if (Test-Path $Build) { Remove-Item -Recurse -Force $Build }

Write-Host "==> Running PyInstaller"
python -m PyInstaller `
  --noconfirm `
  --clean `
  --workpath $Build `
  --distpath $Dist `
  (Join-Path $Root "packaging\wb-analyzer.spec")

$AppDir = Join-Path $Dist "WB-Analyzer"
$ExePath = Join-Path $AppDir "WB-Analyzer.exe"
if (-not (Test-Path $ExePath)) {
  throw "Build failed: $ExePath not found"
}

$ZipPath = Join-Path $Dist "WB-Analyzer-Windows.zip"
if (Test-Path $ZipPath) { Remove-Item -Force $ZipPath }
Compress-Archive -Path $AppDir -DestinationPath $ZipPath

Write-Host ""
Write-Host "Build complete."
Write-Host "  App folder: $AppDir"
Write-Host "  Launcher:   $ExePath"
Write-Host "  Zip:        $ZipPath"
Write-Host ""
Write-Host "Distribute the zip, or run WB-Analyzer.exe inside the folder."
